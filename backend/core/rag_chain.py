import logging
import re
from typing import AsyncIterator

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from backend.core.feedback import log_query, save_failed_query
from backend.core.metadata import format_sources
from backend.rag.retriever import (
    BROAD_KEYWORDS,
    GENERIC_QUERY_TERMS,
    build_source_catalog,
    humanize_source_name,
    infer_source_candidates,
    retrieve,
    tokenize,
)

logger = logging.getLogger(__name__)

UNAVAILABLE_RESPONSE = "This information is not available in the provided SOP."

SYSTEM_PROMPT = """\
You are Prakriya AI, a STRICT internal SOP assistant.

NON-NEGOTIABLE RULES:
1. Answer ONLY from the provided CONTEXT.
2. Never use outside knowledge, assumptions, or unstated company practice.
3. If the context is unrelated or does not support any part of the answer, say EXACTLY:
"This information is not available in the provided SOP."
4. Prefer one SOP at a time. Do not merge procedures from different SOPs unless the context explicitly connects them.
5. Preserve exact SOP terminology for roles, Jira issue types, workflow labels, approvals, statuses, forms, and system names.
6. When the context contains steps, responsibilities, conditions, or checklists, include all relevant items completely.
7. Cite supporting page numbers inline when available, for example: [Page 3].
8. Do not say "etc." or invent missing steps.
9. If the context supports only part of the question, answer only that supported part and clearly say the remaining details are not available in the SOP.
10. Be concise, but never omit required SOP details."""

QUERY_REWRITE_PROMPT = """\
Rewrite the user's question into a standalone search query for retrieving SOP passages.

Rules:
- Preserve exact file names, SOP names, abbreviations, role titles, issue types, and system names.
- If the question depends on prior context, incorporate only the missing details needed from the conversation history.
- If an active SOP is provided, keep that SOP context when the user is clearly continuing the same topic.
- If the user asks for a list, complete process, workflow, responsibilities, or full details, preserve that breadth.
- Do not broaden the request or replace precise nouns with generic paraphrases.
- If the question is already standalone, return it unchanged.
- Output ONLY the rewritten search query, nothing else.


Active SOP:
{active_sop}

Conversation history:
{history}

User question: {question}"""

ANSWER_PROMPT = """\
You will receive retrieved SOP context.

Instructions:
- Answer the QUESTION using only the CONTEXT.
- Use the CONVERSATION HISTORY only to resolve references like "that process", "same SOP", or "this step".
- If the CONTEXT does not support any part of the QUESTION, respond with:
"This information is not available in the provided SOP."
- If the CONTEXT supports only part of the QUESTION, provide only that supported part and then clearly state what further detail is not available in the SOP.
- Prefer the most relevant SOP. If the retrieved chunks appear to be from unrelated SOPs, do not guess.
- Preserve exact workflow names, role names, Jira issue types, approvals, statuses, and numbered steps from the SOP.
- Use bullet points or numbered lists when the SOP content is procedural.
- Include inline page references such as [Page 2] when the supporting chunk provides a page number.
- End your response with a new line in the format: FOLLOWUP: question or NONE
- Only answer if the context explicitly contains the exact information requested.
- Do not interpret roles, labels, or workflow names as identity or definition unless explicitly stated.

CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

QUESTION:
{question}
"""

ANSWER_MODE_INSTRUCTIONS = {
    "brief": "Give a SHORT answer in 2-3 sentences maximum. Be extremely concise.",
    "detailed": "Give a thorough, comprehensive answer with all relevant details from the SOP.",
    "checklist": "Format your answer as a checklist with checkboxes (- [ ] item). Include all actionable items.",
    "step-by-step": "Format your answer as numbered steps. Each step should be clear and actionable.",
    "only-responsibilities": "ONLY list the responsibilities/duties mentioned in the SOP. Nothing else.",
    "only-objective": "ONLY state the objective/purpose mentioned in the SOP. Nothing else.",
}

SUGGESTIONS_PROMPT = """\
Based on the SOP context below, generate exactly 3 short follow-up questions a user might ask next.

STRICT RULES:
- Each question MUST be directly answerable using ONLY the provided context.
- Do NOT introduce new roles, entities, or concepts not explicitly present in the context.
- Do NOT assume relationships unless clearly stated.
- Questions must map to explicit information available in the SOP.
- If sufficient information is not available, return: NONE

Output ONLY the questions, one per line.

SOP Title: {sop_title}
Context: {context}
Previous question: {question}
"""

COMPARE_PROMPT = """\
You will receive context from two different SOPs. Compare them based on the user's question.

Instructions:
- Use ONLY the provided context to answer.
- Structure your comparison clearly with sections for each SOP.
- Highlight similarities and differences.
- Use a table format if comparing specific attributes like responsibilities, workflows, or objectives.
- If a detail is not available in the context for one SOP, say so explicitly.
- End with a brief summary of key differences.

CONTEXT FROM {sop_a_title}:
{context_a}

CONTEXT FROM {sop_b_title}:
{context_b}

QUESTION:
{question}
"""

CONVERSATIONAL_PATTERNS = {
    "greetings": {
        "words": {"hi", "hello", "hey", "hii", "howdy"},
        "response": "Hello! I'm Prakriya AI, your SOP assistant. Ask me anything about company policies, procedures, or workflows.",
    },
    "closings": {
        "words": {
            "bye",
            "goodbye",
            "see you",
            "cya",
            "thanks",
            "thank you",
            "thx",
            "ty",
        },
        "response": "Goodbye! Feel free to return anytime you have SOP-related questions.",
    },
    "acknowledgements": {
        "words": {"ok", "okay", "cool", "great", "got it", "noted", "alright"},
        "response": "Glad to help! Feel free to ask anything else about the SOPs.",
    },
}

GIBBERISH = {
    "k",
    "kk",
    "hmm",
    "hm",
    "lol",
    "lmao",
    "haha",
    "hehe",
    "ohh",
    "ohk",
    "umm",
    "uh",
    "wtf",
    "omg",
}


class RAGChain:
    def __init__(self, llm: BaseChatModel, vectorstore: Chroma):
        self.llm = llm
        self.vectorstore = vectorstore
        self.source_catalog = build_source_catalog(vectorstore)

    def check_conversational(self, question: str) -> str | None:
        q = question.strip().lower()
        if q in GIBBERISH or (len(q) == 1 and q not in {"y", "n"}):
            return "I didn't quite get that. Could you ask a specific SOP question?"
        for pattern in CONVERSATIONAL_PATTERNS.values():
            if q in pattern["words"]:
                return pattern["response"]
        return None

    def _needs_rewrite(
        self,
        question: str,
        history: list[dict],
        active_sop: str | None,
    ) -> bool:
        if not history and not active_sop:
            return False

        return self._is_context_dependent_followup(question)

    def _effective_active_sop(
        self,
        question: str,
        history: list[dict],
        active_sop: str | None,
        *,
        source_locked: bool,
    ) -> str | None:
        if not active_sop:
            return None
        if source_locked:
            return active_sop

        if self._is_context_dependent_followup(question):
            return active_sop

        return None

    async def rewrite_query(
        self,
        question: str,
        history: list[dict],
        active_sop: str | None = None,
    ) -> str:
        if not self._needs_rewrite(question, history, active_sop):
            return question

        history_text = "\n".join(
            f"{msg['role'].title()}: {msg['content']}" for msg in history[-6:]
        )
        prompt = QUERY_REWRITE_PROMPT.format(
            history=history_text or "No previous conversation.",
            question=question,
            active_sop=active_sop or "None",
        )

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        rewritten = response.content.strip()

        if not rewritten or len(rewritten) < 4:
            return question
        return rewritten

    def retrieve_docs(
        self,
        query: str,
        active_sop: str | None = None,
    ) -> tuple[list[Document], str | None]:
        return retrieve(
            self.vectorstore,
            query,
            active_sop=active_sop,
            source_catalog=self.source_catalog,
        )

    def _source_title(self, source: str) -> str:
        info = self.source_catalog.get(source, {})
        title = str(info.get("title") or "").strip()
        if not title:
            return humanize_source_name(source)

        normalized_title = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
        title_terms = set(normalized_title.split())
        generic_title_terms = GENERIC_QUERY_TERMS | BROAD_KEYWORDS | {
            "annexure",
            "document",
            "guide",
            "introduction",
            "manual",
            "overview",
            "purpose",
            "scope",
            "section",
            "summary",
            "uat",
        }
        looks_like_heading = bool(re.match(r"^\d+(\.\d+)*\b", title.strip()))
        title_is_generic = (
            not title_terms
            or title_terms.issubset(generic_title_terms)
        )
        if looks_like_heading or title_is_generic:
            return humanize_source_name(source)

        return title

    def _specific_terms(self, question: str) -> list[str]:
        return [
            term
            for term in sorted(tokenize(question), key=len, reverse=True)
            if term not in GENERIC_QUERY_TERMS and term not in BROAD_KEYWORDS
        ]

    def _is_context_dependent_followup(self, question: str) -> bool:
        q = question.strip().lower()
        reference_patterns = [
            r"\b(it|this|that|those|these|same|above|previous|earlier)\b",
            r"\bwhat about\b",
            r"\bhow about\b",
            r"\bthen\b",
            r"\bmore\b",
            r"\bcontinue\b",
            r"\bnext\b",
            r"\balso\b",
        ]
        if any(re.search(pattern, q) for pattern in reference_patterns):
            return True

        terms = tokenize(question)
        if not terms:
            return True

        specific_terms = self._specific_terms(question)
        if not specific_terms:
            return True

        if len(terms) <= 3 and all(
            term in GENERIC_QUERY_TERMS or term in BROAD_KEYWORDS
            for term in terms
        ):
            return True

        return False

    def _clarification_prompt(self, question: str) -> str:
        terms = tokenize(question)
        if any(term in {"role", "roles"} or term.startswith("responsibil") for term in terms):
            return "Which role do you mean?"
        if any(
            term in {"process", "procedure", "procedures", "workflow", "objective", "objectives"}
            for term in terms
        ):
            return "Which SOP or workflow do you mean?"
        return "Which SOP or role do you mean?"

    def _clarification_options(
        self,
        candidates: list[tuple[str, float]],
    ) -> list[str]:
        options: list[str] = []
        for source, _ in candidates:
            title = self._source_title(source)
            if title not in options:
                options.append(title)
            if len(options) >= 4:
                break
        return options

    def _should_clarify(
        self,
        question: str,
        active_sop: str | None,
    ) -> tuple[bool, list[tuple[str, float]], list[str]]:
        if active_sop:
            return False, [], []

        candidates = infer_source_candidates(question, self.source_catalog, limit=6)
        specific_terms = self._specific_terms(question)
        clarification_options = self._clarification_options(candidates)

        if len(clarification_options) < 2:
            return False, candidates, []

        if not specific_terms:
            return True, candidates, clarification_options

        if len(specific_terms) == 1 and len(candidates) >= 2:
            top_score = candidates[0][1]
            second_score = candidates[1][1]
            if abs(top_score - second_score) <= 0.12:
                return True, candidates, clarification_options

        return False, candidates, []

    def _confidence_level(
        self,
        question: str,
        docs: list[Document],
        detected_sop: str | None,
        candidates: list[tuple[str, float]],
        active_sop: str | None,
    ) -> str:
        if not docs or not detected_sop:
            return "low"

        if active_sop and detected_sop == active_sop and not self._specific_terms(question):
            return "high"

        candidate_scores = {source: score for source, score in candidates}
        source_score = candidate_scores.get(detected_sop, 0.0)
        next_score = next(
            (score for source, score in candidates if source != detected_sop),
            0.0,
        )
        supporting_sections = sum(
            1
            for doc in docs
            if doc.metadata.get("section_title") not in {"Document Profile", "Document Guide"}
        )

        if source_score >= 2.0 and (source_score - next_score >= 0.2 or next_score == 0) and supporting_sections >= 2:
            return "high"
        if source_score >= 1.2 or supporting_sections >= 2:
            return "medium"
        return "low"

    def _apply_confidence_notice(self, answer: str, confidence: str) -> str:
        if confidence != "low":
            return answer
        if answer.lower().startswith("low confidence:"):
            return answer
        if answer == UNAVAILABLE_RESPONSE:
            return answer
        return f"Low confidence: {answer}"

    def _page_label(self, doc: Document) -> str | None:
        page_label = doc.metadata.get("page_label")
        if page_label:
            return str(page_label)

        page = doc.metadata.get("page")
        if isinstance(page, int):
            return str(page + 1)
        if isinstance(page, str) and page and page != "unknown":
            return page
        return None

    def _format_context(self, docs: list[Document]) -> str:
        formatted: list[str] = []
        for doc in docs:
            header = f"[Source: {doc.metadata.get('source', 'SOP')}"
            page_label = self._page_label(doc)
            if page_label:
                header += f" | Page: {page_label}"
            header += "]"
            formatted.append(f"{header}\n{doc.page_content}")
        return "\n\n".join(formatted)

    def _format_history(self, history: list[dict]) -> str:
        if not history:
            return "No previous conversation."
        lines = []
        for msg in history[-6:]:
            lines.append(f"{msg['role'].title()}: {msg['content']}")
        return "\n".join(lines)

    def _parse_response(self, raw: str) -> tuple[str, str | None]:
        followup = None
        answer = raw

        match = re.search(r"FOLLOWUP:\s*(.+)", raw, re.IGNORECASE)
        if match:
            followup_text = match.group(1).strip()
            if followup_text.startswith("<") and followup_text.endswith(">"):
                followup_text = followup_text[1:-1].strip()
            if followup_text.upper() != "NONE":
                followup = followup_text
            answer = raw[: match.start()].strip()

        answer = re.sub(r"^ANSWER:\s*", "", answer, flags=re.IGNORECASE).strip()
        return answer, followup

    def _normalize_answer(self, answer: str) -> str:
        cleaned = answer.strip()
        lowered = re.sub(r"^low confidence:\s*", "", cleaned, flags=re.IGNORECASE).strip().lower()
        unavailable_cues = (
            "not available",
            "not explicitly mentioned",
            "not explicitly stated",
            "not explicitly provided",
            "not mentioned",
            "not specified",
            "not stated",
            "not described",
            "not provided",
        )
        if any(cue in lowered for cue in unavailable_cues):
            return UNAVAILABLE_RESPONSE
        return cleaned

    def _grounding_terms(self, text: str) -> set[str]:
        return {
            term for term in tokenize(text)
            if term not in GENERIC_QUERY_TERMS and term not in BROAD_KEYWORDS
        }

    def _is_grounded_answer(self, question: str, answer: str, context: str) -> bool:
        if answer == UNAVAILABLE_RESPONSE:
            return True

        answer_terms = tokenize(answer)
        context_terms = tokenize(context)
        if not answer_terms or not context_terms:
            return False

        overlap = len(answer_terms & context_terms)
        required_overlap = max(3, int(len(answer_terms) * 0.18))
        if overlap < required_overlap:
            return False

        question_specific_terms = set(self._specific_terms(question))
        if question_specific_terms:
            supported_question_terms = question_specific_terms & context_terms
            if not supported_question_terms:
                return False

            answer_question_terms = question_specific_terms & answer_terms
            required_question_terms = min(2, len(supported_question_terms))
            if len(answer_question_terms) < required_question_terms:
                return False

        answer_specific_terms = self._grounding_terms(answer)
        if answer_specific_terms:
            supported_answer_specific_terms = answer_specific_terms & context_terms
            minimum_supported_specific_terms = 1 if len(answer_specific_terms) <= 2 else 2
            if len(supported_answer_specific_terms) < minimum_supported_specific_terms:
                return False

            unsupported_answer_specific_terms = answer_specific_terms - context_terms
            if (
                len(unsupported_answer_specific_terms) >= 2
                and len(unsupported_answer_specific_terms) > len(supported_answer_specific_terms)
            ):
                return False

        return True

    def _is_image_relevant(self, question: str, image_path: str) -> bool:
        q = question.lower()
        name = re.sub(
            r"\.(png|jpg|jpeg|gif)$",
            "",
            image_path.split("/")[-1].split("\\")[-1].lower(),
        )
        keywords = re.split(r"[_\-\s.]+", name)
        return sum(1 for kw in keywords if len(kw) > 2 and kw in q) >= 1

    async def _generate_suggestions(
        self,
        question: str,
        context: str,
        detected_sop: str | None,
    ) -> list[str]:
        """Generate context-aware follow-up suggestions from the same SOP."""
        if not detected_sop:
            return []
        sop_title = self._source_title(detected_sop)
        prompt = SUGGESTIONS_PROMPT.format(
            sop_title=sop_title,
            context=context[:1500],
            question=question,
        )
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            lines = [
                line.strip()
                for line in response.content.strip().split("\n")
                if line.strip() and len(line.strip()) > 10
            ]
            return lines[:3]
        except Exception:
            return []

    async def compare_sops(
        self,
        question: str,
        sop_a: str,
        sop_b: str,
    ) -> dict:
        """Compare two SOPs side by side."""
        docs_a, _ = self.retrieve_docs(question, active_sop=sop_a)
        docs_b, _ = self.retrieve_docs(question, active_sop=sop_b)

        if not docs_a and not docs_b:
            return {
                "answer": "Could not find relevant information in either SOP.",
                "sources": None,
                "confidence": "low",
            }

        context_a = self._format_context(docs_a) if docs_a else "No relevant context found."
        context_b = self._format_context(docs_b) if docs_b else "No relevant context found."
        title_a = self._source_title(sop_a)
        title_b = self._source_title(sop_b)

        prompt = COMPARE_PROMPT.format(
            sop_a_title=title_a,
            sop_b_title=title_b,
            context_a=context_a,
            context_b=context_b,
            question=question,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = await self.llm.ainvoke(messages)
        answer = response.content.strip()

        sources_a = format_sources(docs_a)
        sources_b = format_sources(docs_b)

        return {
            "answer": answer,
            "sources": sources_a,
            "sources_b": sources_b,
            "sop_a_title": title_a,
            "sop_b_title": title_b,
            "confidence": "medium" if docs_a and docs_b else "low",
        }

    def get_all_sop_titles(self) -> list[dict[str, str]]:
        """Return list of all available SOPs with their source keys."""
        titles = []
        for source, info in self.source_catalog.items():
            title = self._source_title(source)
            titles.append({"source": source, "title": title})
        return sorted(titles, key=lambda x: x["title"])

    async def query(
        self,
        message: str,
        history: list[dict],
        active_sop: str | None = None,
        *,
        answer_mode: str = "detailed",
        source_locked: bool = False,
        llm_provider: str = "",
    ) -> dict:
        conv = self.check_conversational(message)
        if conv:
            return {
                "answer": conv,
                "sources": None,
                "followup": None,
                "active_sop": active_sop,
                "image": None,
                "confidence": "high",
                "suggestions": None,
            }

        effective_sop = self._effective_active_sop(
            message,
            history,
            active_sop,
            source_locked=source_locked,
        )

        search_query = await self.rewrite_query(message, history, effective_sop)
        should_clarify, candidates, clarification_options = self._should_clarify(
            search_query,
            effective_sop,
        )
        if should_clarify and not source_locked:
            log_query(message, active_sop, None, "low", was_clarification=True,
                      answer_mode=answer_mode, llm_provider=llm_provider)
            return {
                "answer": self._clarification_prompt(message),
                "sources": None,
                "followup": None,
                "active_sop": active_sop,
                "image": None,
                "confidence": "low",
                "suggestions": clarification_options,
            }

        docs, detected_sop = self.retrieve_docs(search_query, effective_sop)

        if not docs:
            save_failed_query(message, "low", active_sop)
            log_query(message, active_sop, None, "low",
                      answer_mode=answer_mode, llm_provider=llm_provider)
            return {
                "answer": UNAVAILABLE_RESPONSE,
                "sources": None,
                "followup": None,
                "active_sop": None,
                "image": None,
                "confidence": "low",
                "suggestions": None,
            }

        image = None
        for doc in docs:
            if doc.metadata.get("type") == "image" and self._is_image_relevant(
                message, doc.metadata.get("path", "")
            ):
                image = doc.metadata["path"]
                break

        context = self._format_context(docs)
        history_text = self._format_history(history)

        # Apply answer mode instruction
        mode_instruction = ANSWER_MODE_INSTRUCTIONS.get(answer_mode, "")
        answer_prompt = ANSWER_PROMPT.format(
            context=context,
            history=history_text,
            question=message,
        )
        if mode_instruction:
            answer_prompt = f"ANSWER MODE: {mode_instruction}\n\n{answer_prompt}"

        confidence = self._confidence_level(
            search_query,
            docs,
            detected_sop,
            candidates,
            effective_sop,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=answer_prompt),
        ]
        response = await self.llm.ainvoke(messages)
        answer, followup = self._parse_response(response.content)
        answer = self._normalize_answer(answer)

        if answer == UNAVAILABLE_RESPONSE:
            docs = []
            followup = None
            image = None
            confidence = "low"
            detected_sop = None

        if not self._is_grounded_answer(message, answer, context):
            answer = UNAVAILABLE_RESPONSE
            docs = []
            followup = None
            image = None
            confidence = "low"
            detected_sop = None

        answer = self._apply_confidence_notice(answer, confidence)

        # Generate suggestions from same SOP
        suggestions = await self._generate_suggestions(message, context, detected_sop) if docs else None

        sources = format_sources(docs)

        # Log query for analytics
        log_query(message, active_sop, detected_sop, confidence,
                  answer_mode=answer_mode, llm_provider=llm_provider)
        if confidence == "low" and docs:
            save_failed_query(message, confidence, active_sop, answer[:300])

        return {
            "answer": answer,
            "sources": sources,
            "followup": followup,
            "active_sop": detected_sop,
            "image": image,
            "confidence": confidence,
            "suggestions": suggestions or None,
        }

    async def stream_query(
        self,
        message: str,
        history: list[dict],
        active_sop: str | None = None,
        *,
        answer_mode: str = "detailed",
        source_locked: bool = False,
        llm_provider: str = "",
    ) -> AsyncIterator[dict]:
        conv = self.check_conversational(message)
        if conv:
            yield {"type": "token", "content": conv}
            yield {
                "type": "done",
                "sources": None,
                "followup": None,
                "active_sop": active_sop,
                "image": None,
                "confidence": "high",
                "suggestions": None,
                "full_answer": conv,
            }
            return

        effective_sop = self._effective_active_sop(
            message,
            history,
            active_sop,
            source_locked=source_locked,
        )

        search_query = await self.rewrite_query(message, history, effective_sop)
        should_clarify, candidates, clarification_options = self._should_clarify(
            search_query,
            effective_sop,
        )
        if should_clarify and not source_locked:
            clarification_text = self._clarification_prompt(message)
            log_query(message, active_sop, None, "low", was_clarification=True,
                      answer_mode=answer_mode, llm_provider=llm_provider)
            yield {"type": "token", "content": clarification_text}
            yield {
                "type": "done",
                "sources": None,
                "followup": None,
                "active_sop": active_sop,
                "image": None,
                "confidence": "low",
                "suggestions": clarification_options,
                "full_answer": clarification_text,
            }
            return

        docs, detected_sop = self.retrieve_docs(search_query, effective_sop)

        if not docs:
            msg = UNAVAILABLE_RESPONSE
            save_failed_query(message, "low", active_sop)
            log_query(message, active_sop, None, "low",
                      answer_mode=answer_mode, llm_provider=llm_provider)
            yield {"type": "token", "content": msg}
            yield {
                "type": "done",
                "sources": None,
                "followup": None,
                "active_sop": None,
                "image": None,
                "confidence": "low",
                "suggestions": None,
                "full_answer": msg,
            }
            return

        image = None
        for doc in docs:
            if doc.metadata.get("type") == "image" and self._is_image_relevant(
                message, doc.metadata.get("path", "")
            ):
                image = doc.metadata["path"]
                break

        context = self._format_context(docs)
        history_text = self._format_history(history)

        # Apply answer mode instruction
        mode_instruction = ANSWER_MODE_INSTRUCTIONS.get(answer_mode, "")
        answer_prompt = ANSWER_PROMPT.format(
            context=context,
            history=history_text,
            question=message,
        )
        if mode_instruction:
            answer_prompt = f"ANSWER MODE: {mode_instruction}\n\n{answer_prompt}"

        confidence = self._confidence_level(
            search_query,
            docs,
            detected_sop,
            candidates,
            effective_sop,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=answer_prompt),
        ]

        buffer = ""
        followup_marker = "FOLLOWUP:"
        marker_detected = False
        sent_up_to = 0
        if confidence == "low":
            prefix = "Low confidence: "
            buffer += prefix
            sent_up_to = len(buffer)
            yield {"type": "token", "content": prefix}

        async for chunk in self.llm.astream(messages):
            token = chunk.content or ""
            buffer += token

            if marker_detected:
                continue

            marker_pos = buffer.upper().find(followup_marker)
            if marker_pos != -1:
                marker_detected = True
                unsent = buffer[sent_up_to:marker_pos].rstrip("\n ")
                if unsent:
                    yield {"type": "token", "content": unsent}
                continue

            safe_end = len(buffer)
            for i in range(1, len(followup_marker)):
                if buffer.upper().endswith(followup_marker[:i]):
                    safe_end = len(buffer) - i
                    break

            unsent = buffer[sent_up_to:safe_end]
            if unsent:
                yield {"type": "token", "content": unsent}
                sent_up_to = safe_end

        if not marker_detected:
            unsent = buffer[sent_up_to:]
            if unsent:
                yield {"type": "token", "content": unsent}

        answer_text, followup = self._parse_response(buffer)
        answer_text = re.sub(
            r"^ANSWER:\s*",
            "",
            answer_text,
            flags=re.IGNORECASE,
        ).strip()
        answer_text = self._normalize_answer(answer_text)

        if answer_text == UNAVAILABLE_RESPONSE:
            docs = []
            followup = None
            image = None
            confidence = "low"
            detected_sop = None

        if not self._is_grounded_answer(message, answer_text, context):
            answer_text = UNAVAILABLE_RESPONSE
            docs = []
            followup = None
            image = None
            confidence = "low"
            detected_sop = None

        answer_text = self._apply_confidence_notice(answer_text, confidence)

        # Generate suggestions from same SOP
        suggestions = await self._generate_suggestions(message, context, detected_sop) if docs else None

        sources = format_sources(docs)

        # Log for analytics
        log_query(message, active_sop, detected_sop, confidence,
                  answer_mode=answer_mode, llm_provider=llm_provider)
        if confidence == "low" and docs:
            save_failed_query(message, confidence, active_sop, answer_text[:300])

        yield {
            "type": "done",
            "sources": sources,
            "followup": followup,
            "active_sop": detected_sop,
            "image": image,
            "full_answer": answer_text,
            "confidence": confidence,
            "suggestions": suggestions or None,
        }
