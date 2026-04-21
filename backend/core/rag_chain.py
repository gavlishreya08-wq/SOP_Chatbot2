import logging
import re
from dataclasses import dataclass
from typing import AsyncIterator, Literal

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
    normalize_query,
    retrieve,
    tokenize,
)

logger = logging.getLogger(__name__)

UNAVAILABLE_RESPONSE = "This information is not available in the provided SOP."
SHOW_MORE_PAGE_SIZE = 15
SHOW_MORE_MARKER = "[SHOW_MORE_AVAILABLE]"


@dataclass
class ParsedTable:
    columns: list[str]
    rows: list[dict[str, str | list[str]]]


AnswerQuality = Literal["full", "partial", "no_answer"]


@dataclass
class ResponseValidation:
    answer: str
    quality: AnswerQuality

SYSTEM_PROMPT = """\
You are Prakriya AI, a STRICT internal SOP assistant.

NON-NEGOTIABLE RULES:
1. Answer ONLY from the provided CONTEXT.
2. Never use outside knowledge, assumptions, or unstated company practice.
3. If the context is unrelated and no relevant answer can be generated, say EXACTLY:
"This information is not available in the provided SOP."
4. Prefer one SOP at a time. Do not merge procedures from different SOPs unless the context explicitly connects them.
5. Preserve exact SOP terminology for roles, Jira issue types, workflow labels, approvals, statuses, forms, and system names.
6. When the context contains steps, responsibilities, conditions, or checklists, include all relevant items completely.
7. Cite supporting page numbers inline when available, for example: [Page 3].
8. Do not say "etc." or invent missing steps.
9. If the context supports only part of the question, answer only the supported part. Do not append an unavailable/disclaimer statement when any meaningful answer is provided.
10. Be concise, but never omit required SOP details.
11. Ensure numbered lists are continuous and do not restart from 1 unless a clearly separate nested list begins.
12. Never combine a meaningful answer with phrases like "not available", "not mentioned", or "not in the provided context"."""

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
- If the CONTEXT does not support any part of the QUESTION and no relevant answer can be generated, respond with:
"This information is not available in the provided SOP."
- If the CONTEXT supports only part of the QUESTION, provide only that supported part. Do not add any unavailable/disclaimer sentence for missing details.
- If related SOP context can answer the user's intent partially or indirectly, return that related answer without a disclaimer.
- Prefer the most relevant SOP. If the retrieved chunks appear to be from unrelated SOPs, do not guess.
- Preserve exact workflow names, role names, Jira issue types, approvals, statuses, and numbered steps from the SOP.
- Use bullet points or numbered lists when the SOP content is procedural.
- Ensure numbered lists are continuous and do not restart from 1.
- Include inline page references such as [Page 2] when the supporting chunk provides a page number.
- End your response with a new line in the format: FOLLOWUP: question or NONE
- Answer when the context contains exact, partial, or closely related information that addresses the user's intent.
- Do not interpret roles, labels, or workflow names as identity or definition unless explicitly stated.
- Never output a meaningful answer together with "not available", "not mentioned", "not specified", "not found", or similar disclaimer language.

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

ASSISTANT_IDENTITY_RESPONSE = (
    "I'm Prakriya AI, your internal SOP assistant. I can help you find, "
    "summarize, and explain information from the available SOP documents "
    "across Development, Testing, Database, Deployment, and Reports. "
    "I provide SOP-based details about roles and responsibilities, job objectives, "
    "workflows, procedures, standards, guidelines, training requirements, reporting "
    "authority, and source references."
)

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
        self.memory: dict[str, str | None] = {
            "last_entity": None,
            "last_sop": None,
            "last_question": None,
        }

    def _remember_context(self, question: str, detected_sop: str | None) -> None:
        if detected_sop:
            self.memory["last_sop"] = detected_sop
        specific = self._specific_terms(question)
        if specific:
            self.memory["last_entity"] = " ".join(specific[:4])
        self.memory["last_question"] = question

    def _inject_memory(self, question: str, active_sop: str | None) -> tuple[str, str | None]:
        normalized = normalize_query(question)
        entity = self.memory.get("last_entity")
        sop = active_sop or self.memory.get("last_sop")
        if entity and re.search(r"\b(his|her|their|its|it|that|this|same)\b", normalized):
            normalized = f"{normalized} {entity}"
        return normalized, sop

    def check_conversational(self, question: str) -> str | None:
        q = question.strip().lower()
        normalized_q = re.sub(r"[^\w\s]", " ", q)
        normalized_q = re.sub(r"\s+", " ", normalized_q).strip()
        normalized_q = re.sub(r"\bu\b", "you", normalized_q)
        normalized_q = re.sub(r"\bur\b", "your", normalized_q)
        normalized_q = re.sub(r"\br\b", "are", normalized_q)

        if normalized_q in GIBBERISH or (len(normalized_q) == 1 and normalized_q not in {"y", "n"}):
            return "I didn't quite get that. Could you ask a specific SOP question?"
        for pattern in CONVERSATIONAL_PATTERNS.values():
            if q in pattern["words"] or normalized_q in pattern["words"]:
                return pattern["response"]

        identity_patterns = [
            r"^(who|what) are you$",
            r"^who you are$",
            r"^what can you do$",
            r"^do you know what you can do$",
            r"^do you know who you are$",
            r"^do you know your purpose$",
            r"^what do you do$",
            r"^how can you help( me)?$",
            r"^how do you help( me)?$",
            r"^what is your purpose$",
            r"^what is the purpose of you$",
            r"^what information (do you|can you) provide$",
            r"^what kind of information (do you|can you) provide$",
            r"^what information is available$",
            r"^what sop information (do you|can you) provide$",
            r"^tell me about yourself$",
        ]
        if any(re.fullmatch(pattern, normalized_q) for pattern in identity_patterns):
            return ASSISTANT_IDENTITY_RESPONSE

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

    def _context_body(self, doc: Document) -> str:
        content = doc.page_content or ""
        if "CONTENT:" in content:
            return content.split("CONTENT:", 1)[1].strip()
        return content.strip()

    def _strip_context_numbering(self, line: str) -> str:
        return re.sub(r"^(\s*)(?:[-*]\s*)?\d+[\).]\s+(.+)$", r"\1\2", line)

    def _is_table_metadata_line(self, line: str) -> bool:
        return bool(
            re.match(
                r"^\s*(source title|source file|source kind|aliases|search tags|key sections|summary|page range|section):",
                line,
                flags=re.IGNORECASE,
            )
        )

    def _is_table_separator_line(self, line: str) -> bool:
        stripped = line.strip()
        if "|" in stripped:
            cells = [cell.strip() for cell in stripped.strip("|").split("|") if cell.strip()]
            return bool(cells) and all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells)
        return bool(re.fullmatch(r"[-=_\s|:]+", stripped))

    def _split_table_cells(self, line: str) -> list[str] | None:
        stripped = line.strip()
        if not stripped or stripped.startswith("[Source:") or self._is_table_metadata_line(stripped):
            return None
        if self._is_table_separator_line(stripped):
            return None

        if "|" in stripped:
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            return cells if len([cell for cell in cells if cell]) >= 2 else None
        elif "\t" in stripped:
            cells = [cell.strip() for cell in stripped.split("\t")]
        else:
            cells = [cell.strip() for cell in re.split(r"\s{2,}", stripped)]

        cells = [cell for cell in cells if cell]
        if len(cells) < 2:
            return None
        if all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells):
            return None
        return cells

    def _normalize_table_columns(self, cells: list[str]) -> list[str]:
        columns: list[str] = []
        seen: dict[str, int] = {}
        for index, cell in enumerate(cells, start=1):
            column = re.sub(r"\s+", " ", cell).strip(" :-") or f"Column {index}"
            key = column.lower()
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > 1:
                column = f"{column} {seen[key]}"
            columns.append(column)
        return columns

    def _row_from_cells(self, columns: list[str], cells: list[str]) -> dict[str, str | list[str]]:
        if len(cells) > len(columns):
            cells = [*cells[: len(columns) - 1], " ".join(cells[len(columns) - 1:])]
        padded = [*cells, *([""] * (len(columns) - len(cells)))]
        return {column: padded[index].strip() for index, column in enumerate(columns)}

    def _append_table_continuation(self, columns: list[str], row: dict[str, str | list[str]], line: str) -> None:
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[\).])\s*", "", line).strip()
        if not cleaned:
            return
        target_column = columns[-1]
        current = row.get(target_column, "")
        if isinstance(current, list):
            if cleaned not in current:
                current.append(cleaned)
            return
        if current:
            values = [current]
            if cleaned not in values:
                values.append(cleaned)
            row[target_column] = values
        else:
            row[target_column] = cleaned

    def _parse_table_block(self, block: list[str]) -> ParsedTable | None:
        header_cells = self._split_table_cells(block[0])
        if not header_cells:
            return None

        columns = self._normalize_table_columns(header_cells)
        rows: list[dict[str, str | list[str]]] = []
        seen_rows: set[tuple[str, ...]] = set()

        for raw_line in block[1:]:
            cells = self._split_table_cells(raw_line)
            if cells:
                if len(cells) < len(columns) and rows:
                    self._append_table_continuation(columns, rows[-1], " ".join(cells))
                    continue
                row = self._row_from_cells(columns, cells)
                row_key = tuple(
                    " | ".join(value) if isinstance(value, list) else value
                    for value in row.values()
                )
                if row_key in seen_rows:
                    continue
                seen_rows.add(row_key)
                rows.append(row)
                continue

            if rows:
                self._append_table_continuation(columns, rows[-1], raw_line)

        if len(columns) < 2 or not rows:
            return None
        return self._group_table_rows(ParsedTable(columns=columns, rows=rows))

    def _table_time_columns(self, columns: list[str]) -> tuple[str | None, str | None, str | None]:
        start_column = next((column for column in columns if re.search(r"\bstart\b", column, re.I) and self._is_time_column(column)), None)
        end_column = next((column for column in columns if re.search(r"\bend\b", column, re.I) and self._is_time_column(column)), None)
        duration_column = next((column for column in columns if self._is_duration_column(column)), None)
        return start_column, end_column, duration_column

    def _value_items(self, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            values = value
        else:
            values = re.split(r"\s*(?:;|\n|(?:\s+-\s+))\s*", value)
        return [item.strip(" -") for item in values if item.strip(" -")]

    def _merge_table_value(self, existing: str | list[str], incoming: str | list[str]) -> str | list[str]:
        merged: list[str] = []
        for item in [*self._value_items(existing), *self._value_items(incoming)]:
            if item and item not in merged:
                merged.append(item)
        if not merged:
            return ""
        if len(merged) == 1:
            return merged[0]
        return merged

    def _group_table_rows(self, table: ParsedTable) -> ParsedTable:
        start_column, end_column, duration_column = self._table_time_columns(table.columns)
        if not start_column or not end_column:
            return table

        grouped: list[dict[str, str | list[str]]] = []
        row_by_slot: dict[tuple[str, str, str], dict[str, str | list[str]]] = {}

        for row in table.rows:
            key = (
                str(row.get(start_column, "")).strip().lower(),
                str(row.get(end_column, "")).strip().lower(),
                str(row.get(duration_column, "")).strip().lower() if duration_column else "",
            )
            if not key[0] or not key[1]:
                grouped.append(row)
                continue

            existing = row_by_slot.get(key)
            if existing is None:
                row_by_slot[key] = row
                grouped.append(row)
                continue

            for column in table.columns:
                if column in {start_column, end_column, duration_column}:
                    continue
                existing[column] = self._merge_table_value(existing.get(column, ""), row.get(column, ""))

        return ParsedTable(columns=table.columns, rows=grouped)

    def _extract_table_blocks(self, docs: list[Document]) -> list[list[str]]:
        blocks: list[list[str]] = []
        current: list[str] = []

        def flush() -> None:
            nonlocal current
            table_lines = [line for line in current if self._split_table_cells(line)]
            if len(table_lines) >= 2:
                blocks.append(current)
            current = []

        for doc in docs:
            for line in self._context_body(doc).splitlines():
                if self._split_table_cells(line):
                    current.append(line)
                    continue
                if current and self._is_table_separator_line(line):
                    current.append(line)
                    continue
                if current and re.match(r"^\s*(?:[-*]|\d+[\).])\s+\S+", line):
                    current.append(line)
                    continue
                flush()
            flush()

        return blocks

    def _parse_tables(self, docs: list[Document]) -> list[ParsedTable]:
        tables: list[ParsedTable] = []
        table_docs = [doc for doc in docs if doc.metadata.get("content_type") != "profile"]
        for block in self._extract_table_blocks(table_docs):
            parsed = self._parse_table_block(block)
            if parsed:
                tables.append(parsed)
        return tables

    def _is_time_column(self, column: str) -> bool:
        normalized = normalize_query(column)
        return "time" in normalized

    def _is_duration_column(self, column: str) -> bool:
        normalized = normalize_query(column)
        return "duration" in normalized or normalized in {"hrs", "hours", "mins", "minutes"}

    def _is_row_number_column(self, column: str) -> bool:
        normalized = normalize_query(column)
        return normalized in {"", "#", "no", "sr", "sr no", "serial", "serial no", "row"}

    def _format_cell_value(self, value: str | list[str]) -> tuple[str, list[str]]:
        parts = self._value_items(value)
        if len(parts) > 1:
            return parts[0], parts[1:]
        return parts[0] if parts else "", []

    def _format_table(self, table: ParsedTable) -> str:
        start_column, end_column, duration_column = self._table_time_columns(table.columns)
        time_based = bool(start_column and end_column)

        lines: list[str] = []
        for index, row in enumerate(table.rows, start=1):
            if time_based:
                start_value = str(row.get(start_column or "", "")).strip()
                end_value = str(row.get(end_column or "", "")).strip()
                duration_value = str(row.get(duration_column or "", "")).strip() if duration_column else ""
                time_range = f"{start_value} - {end_value}" if end_value else start_value
                heading = f"{index}. {time_range}".strip()
                if duration_value:
                    heading = f"{heading} ({duration_value})"
                lines.append(heading)

                activity_items: list[str] = []
                for column in table.columns:
                    if column in {start_column, end_column, duration_column} or self._is_row_number_column(column):
                        continue
                    for item in self._value_items(row.get(column, "")):
                        if item not in activity_items:
                            activity_items.append(item)

                if len(activity_items) == 1:
                    lines.append(f"   {activity_items[0]}")
                else:
                    for item in activity_items:
                        lines.append(f"   - {item}")
                continue

            primary_column = table.columns[0]
            primary_value, primary_items = self._format_cell_value(row.get(primary_column, ""))
            lines.append(f"{index}. {primary_value or 'Row ' + str(index)}")
            for item in primary_items:
                lines.append(f"   - {item}")
            for column in table.columns[1:]:
                value, subitems = self._format_cell_value(row.get(column, ""))
                if value:
                    lines.append(f"   - {column}: {value}")
                for item in subitems:
                    lines.append(f"     - {item}")

        return "\n".join(lines).strip()

    def _format_tables(self, tables: list[ParsedTable]) -> str:
        formatted = [self._format_table(table) for table in tables if table.rows]
        return "\n\n".join(part for part in formatted if part).strip()

    def _table_answer_from_docs(self, docs: list[Document]) -> str | None:
        tables = self._parse_tables(docs)
        if not tables:
            return None
        answer = self._format_tables(tables)
        return self._normalize_answer(answer) if answer else None

    def _section_sort_key(self, doc: Document) -> tuple[int, int, int, str]:
        section_index = doc.metadata.get("section_index")
        chunk_in_section = doc.metadata.get("chunk_in_section")
        page = doc.metadata.get("page")
        page_label = str(doc.metadata.get("page_label", ""))
        if not isinstance(section_index, int):
            section_index = 10**6
        if not isinstance(chunk_in_section, int):
            chunk_in_section = 10**6
        if not isinstance(page, int):
            page = 10**6
        return (section_index, chunk_in_section, page, page_label)

    def _requested_section_titles(self, question: str) -> set[str]:
        terms = tokenize(question)
        normalized = normalize_query(question)
        requested: set[str] = set()
        phrase_map = {
            "job objective": {"job objectives", "objective", "objectives"},
            "training requirement": {"training requirement", "training requirements"},
            "reporting authority": {"reporting authority"},
            "minimum qualification": {"minimum qualification", "minimum qualifications"},
            "minimum experience": {"minimum experience", "minimum experiences"},
            "best practice": {"best practices"},
            "revision history": {"revision history"},
            "review history": {"review history"},
        }
        for phrase, titles in phrase_map.items():
            if phrase in normalized:
                requested.update(titles)
        if any(term.startswith("responsibil") for term in terms):
            requested.add("responsibilities")
        if "standard" in terms or "standards" in terms:
            requested.add("standards")
        if "guideline" in terms or "guidelines" in terms:
            requested.add("guidelines")
        if "workflow" in terms:
            requested.add("workflow")
        if "process" in terms:
            requested.add("process")
        if "procedure" in terms:
            requested.add("procedure")
        if "objective" in terms or "objectives" in terms:
            requested.update({"objective", "objectives", "job objectives"})
        return requested

    def _section_matches_request(self, section_title: str, requested_titles: set[str]) -> bool:
        if not requested_titles:
            return True
        normalized = normalize_query(section_title)
        if normalized in requested_titles:
            return True
        requested_normalized = {normalize_query(title) for title in requested_titles}
        if "responsibilities" in requested_normalized:
            return bool(re.search(r"\b(?:roles?\s+and\s+)?responsibilities\b$", normalized))
        return False

    def _source_section_docs(self, detected_sop: str | None) -> list[Document]:
        if not detected_sop:
            return []
        try:
            records = self.vectorstore.get(where={"source": detected_sop}, include=["documents", "metadatas"])
        except TypeError:
            records = self.vectorstore.get(include=["documents", "metadatas"])
        except Exception:
            return []

        docs: list[Document] = []
        for content, metadata in zip(records.get("documents") or [], records.get("metadatas") or []):
            if not metadata or metadata.get("source") != detected_sop:
                continue
            if metadata.get("content_type") != "section":
                continue
            docs.append(Document(page_content=content or "", metadata=metadata))
        return sorted(docs, key=self._section_sort_key)

    def _doc_identity(self, doc: Document) -> str:
        return str(
            doc.metadata.get("chunk_id")
            or f"{doc.metadata.get('source')}::{doc.metadata.get('section_index')}::{doc.metadata.get('chunk_in_section')}::{hash(doc.page_content)}"
        )

    def _top_level_markers_in_doc(self, doc: Document) -> list[int]:
        return [
            marker
            for line in self._context_body(doc).splitlines()
            if (marker := self._top_level_numbered_marker(line)) is not None
        ]

    def _refetch_numbered_continuation_docs(
        self,
        question: str,
        docs: list[Document],
        detected_sop: str | None,
    ) -> list[Document]:
        if not detected_sop:
            return []
        if not (self._wants_complete_section_answer(question) or self._requested_item_number(question) is not None):
            return []

        all_docs = self._source_section_docs(detected_sop)
        if not all_docs:
            return []

        requested_titles = self._requested_section_titles(question)
        selected_keys = {
            (
                doc.metadata.get("section_index"),
                normalize_query(str(doc.metadata.get("section_title", ""))),
            )
            for doc in docs
            if doc.metadata.get("content_type") == "section"
            and isinstance(doc.metadata.get("section_index"), int)
        }

        seed_ids: set[str] = set()
        for doc in all_docs:
            section_title = str(doc.metadata.get("section_title", ""))
            key = (
                doc.metadata.get("section_index"),
                normalize_query(section_title),
            )
            if requested_titles and self._section_matches_request(section_title, requested_titles):
                seed_ids.add(self._doc_identity(doc))
            elif not requested_titles and key in selected_keys:
                seed_ids.add(self._doc_identity(doc))
            elif requested_titles and key in selected_keys:
                seed_ids.add(self._doc_identity(doc))

        if not seed_ids:
            return []

        expanded: list[Document] = []
        expanded_ids: set[str] = set()
        highest_marker = 0
        first_seed_index = next(
            (index for index, doc in enumerate(all_docs) if self._doc_identity(doc) in seed_ids),
            None,
        )
        if first_seed_index is None:
            return []

        for index, doc in enumerate(all_docs):
            doc_id = self._doc_identity(doc)
            markers = self._top_level_markers_in_doc(doc)
            is_seed = doc_id in seed_ids
            is_continuation = (
                index > first_seed_index
                and bool(markers)
                and highest_marker > 0
                and min(markers) <= highest_marker + 1
                and max(markers) > highest_marker
            )
            if not (is_seed or is_continuation):
                continue
            if doc_id not in expanded_ids:
                expanded.append(doc)
                expanded_ids.add(doc_id)
            if markers:
                highest_marker = max(highest_marker, max(markers))

        return sorted(expanded, key=self._section_sort_key)

    def _refetch_requested_section_docs(self, question: str, detected_sop: str | None) -> list[Document]:
        requested_titles = self._requested_section_titles(question)
        if not detected_sop or not requested_titles:
            return []
        docs = []
        for doc in self._source_section_docs(detected_sop):
            if not self._section_matches_request(str(doc.metadata.get("section_title", "")), requested_titles):
                continue
            docs.append(doc)
        return sorted(docs, key=self._section_sort_key)

    def _refetch_selected_section_docs(self, docs: list[Document], detected_sop: str | None) -> list[Document]:
        if not detected_sop:
            return []

        section_keys = {
            (
                doc.metadata.get("section_index"),
                normalize_query(str(doc.metadata.get("section_title", ""))),
            )
            for doc in docs
            if doc.metadata.get("content_type") == "section"
            and isinstance(doc.metadata.get("section_index"), int)
        }
        if not section_keys:
            return []

        try:
            records = self.vectorstore.get(where={"source": detected_sop}, include=["documents", "metadatas"])
        except TypeError:
            records = self.vectorstore.get(include=["documents", "metadatas"])
        except Exception:
            return []

        expanded: list[Document] = []
        for content, metadata in zip(records.get("documents") or [], records.get("metadatas") or []):
            if not metadata or metadata.get("source") != detected_sop:
                continue
            if metadata.get("content_type") != "section":
                continue
            key = (
                metadata.get("section_index"),
                normalize_query(str(metadata.get("section_title", ""))),
            )
            if key in section_keys:
                expanded.append(Document(page_content=content or "", metadata=metadata))
        return sorted(expanded, key=self._section_sort_key)

    def _merge_docs(self, docs: list[Document], extra_docs: list[Document]) -> list[Document]:
        by_id: dict[str, Document] = {}
        for doc in [*docs, *extra_docs]:
            by_id[self._doc_identity(doc)] = doc
        return sorted(by_id.values(), key=self._section_sort_key)

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

    def _is_manual_continuation_request(self, question: str) -> bool:
        normalized = normalize_query(question)
        return bool(
            re.fullmatch(
                r"(more|show more|continue|next|next point|next points|remaining|remaining points|rest)",
                normalized,
            )
        )

    def _top_level_numbered_marker(self, line: str) -> int | None:
        match = re.match(r"^\s*(\d+)[\).]\s+\S+", line)
        if not match:
            return None
        return int(match.group(1))

    def _shown_item_count_from_text(self, text: str) -> int:
        numbered_markers = [
            marker
            for line in text.splitlines()
            if (marker := self._top_level_numbered_marker(line)) is not None
        ]
        if numbered_markers:
            return max(numbered_markers)
        return sum(1 for line in text.splitlines() if self._structured_marker(line))

    def _manual_continuation_context(
        self,
        question: str,
        history: list[dict],
    ) -> tuple[str, int] | None:
        if not self._is_manual_continuation_request(question):
            return None

        original_index: int | None = None
        original_question: str | None = None
        for index in range(len(history) - 1, -1, -1):
            message = history[index]
            if message.get("role") != "user":
                continue
            content = str(message.get("content") or "").strip()
            if not content or self._is_manual_continuation_request(content):
                continue
            original_index = index
            original_question = content
            break

        if original_index is None or not original_question:
            last_question = self.memory.get("last_question")
            if isinstance(last_question, str) and last_question.strip():
                return last_question, 0
            return None

        offset = 0
        for message in history[original_index + 1:]:
            if message.get("role") != "assistant":
                continue
            offset = max(offset, self._shown_item_count_from_text(str(message.get("content") or "")))

        return original_question, offset

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

    def _format_context(self, docs: list[Document], *, strip_numbering: bool = True) -> str:
        formatted: list[str] = []
        grouped: dict[tuple[str, object, str], dict[str, object]] = {}
        ungrouped: list[Document] = []
        for doc in docs:
            if doc.metadata.get("content_type") == "section":
                key = (
                    str(doc.metadata.get("source", "SOP")),
                    doc.metadata.get("section_index"),
                    str(doc.metadata.get("section_title", "Section")),
                )
                group = grouped.setdefault(key, {"pages": [], "lines": []})
                page_label = self._page_label(doc)
                if page_label and page_label not in group["pages"]:
                    group["pages"].append(page_label)
                seen = {line.strip() for line in group["lines"] if str(line).strip()}
                for line in self._context_body(doc).splitlines():
                    if strip_numbering:
                        line = self._strip_context_numbering(line)
                    if line.strip() and line.strip() in seen:
                        continue
                    group["lines"].append(line)
                    if line.strip():
                        seen.add(line.strip())
                continue
            ungrouped.append(doc)
        for (source, _, section_title), group in grouped.items():
            header = f"[Source: {source} | Section: {section_title}"
            pages = group["pages"]
            if pages:
                header += f" | Page: {', '.join(pages)}"
            header += "]"
            formatted.append(f"{header}\n" + "\n".join(group["lines"]).strip())
        for doc in ungrouped:
            header = f"[Source: {doc.metadata.get('source', 'SOP')}"
            page_label = self._page_label(doc)
            if page_label:
                header += f" | Page: {page_label}"
            header += "]"
            body = "\n".join(
                self._strip_context_numbering(line) if strip_numbering else line
                for line in self._context_body(doc).splitlines()
            )
            formatted.append(f"{header}\n{body}")
        return "\n\n".join(formatted)

    def _context_sections(self, context: str) -> list[tuple[str, list[str]]]:
        sections: list[tuple[str, list[str]]] = []
        current_title: str | None = None
        current_lines: list[str] = []
        for raw_line in context.splitlines():
            line = raw_line.rstrip()
            if re.match(r"^\[Source: .*?\]$", line):
                if current_title and current_lines:
                    sections.append((current_title, current_lines))
                match = re.search(r"\| Section: ([^\]|]+)", line)
                current_title = match.group(1).strip() if match else "SOP Content"
                current_lines = []
                continue
            if current_title:
                current_lines.append(line)
        if current_title and current_lines:
            sections.append((current_title, current_lines))
        return sections

    def _structured_marker(self, line: str) -> str | None:
        stripped = line.strip()
        if not stripped:
            return None
        match = re.match(
            r"^(?:[-*]\s*)?(?:(\d+(?:\.\d+)*)(?:[\).]|\s+)|\(?([a-zA-Z])\)|([a-zA-Z])[\).])",
            stripped,
        )
        if not match:
            return None
        marker = next((group for group in match.groups() if group), None)
        return marker.lower() if marker and marker.isalpha() else marker

    def _wants_complete_section_answer(self, question: str) -> bool:
        normalized = normalize_query(question)
        terms = tokenize(normalized)
        completeness_terms = {
            "all",
            "complete",
            "details",
            "entire",
            "every",
            "everything",
            "exhaustive",
            "full",
            "list",
            "points",
        }
        if terms & completeness_terms:
            return True
        return bool(
            re.search(
                r"\b(give|show|provide|tell)\s+(me\s+)?(the\s+)?"
                r"(procedure|process|workflow|steps|responsibilities|checklist|guidelines|standards)\b",
                normalized,
            )
        )

    def _requested_item_number(self, question: str) -> int | None:
        normalized = normalize_query(question)
        patterns = (
            r"\b(?:point|item|step|responsibility|role|row|no|number)\s*(?:no\.?|number|#)?\s*(\d{1,3})\b",
            r"\b(\d{1,3})(?:st|nd|rd|th)?\s+(?:point|item|step|responsibility|role|row)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                number = int(match.group(1))
                return number if number > 0 else None
        return None

    def _specific_structured_item_answer(self, question: str, answer: str) -> str | None:
        item_number = self._requested_item_number(question)
        if not item_number:
            return None

        lines = answer.splitlines()
        top_level_indexes = [
            (index, marker)
            for index, line in enumerate(lines)
            if (marker := self._top_level_numbered_marker(line)) is not None
        ]
        for position, (index, marker) in enumerate(top_level_indexes):
            if marker != item_number:
                continue
            end = top_level_indexes[position + 1][0] if position + 1 < len(top_level_indexes) else len(lines)
            header_lines = [line for line in lines[:top_level_indexes[0][0]] if line.strip()]
            return "\n".join([*header_lines, *lines[index:end]]).strip()

        marker_indexes = [
            index
            for index, line in enumerate(lines)
            if self._structured_marker(line)
        ]
        if not marker_indexes or item_number > len(marker_indexes):
            return None

        start = marker_indexes[item_number - 1]
        end = marker_indexes[item_number] if item_number < len(marker_indexes) else len(lines)
        header_lines = [line for line in lines[:marker_indexes[0]] if line.strip()]
        item_lines = lines[start:end]
        return "\n".join([*header_lines, *item_lines]).strip()

    def _extractive_section_answer(self, question: str, context: str) -> str | None:
        requested_titles = self._requested_section_titles(question)
        allow_selected_sections = (
            self._wants_complete_section_answer(question)
            or self._requested_item_number(question) is not None
        )
        if not requested_titles and not allow_selected_sections:
            return None
        blocks: list[str] = []
        highest_marker = 0
        started_requested_section = False
        for title, lines in self._context_sections(context):
            if title == "SOP Content":
                continue
            markers = [
                marker
                for line in lines
                if (marker := self._top_level_numbered_marker(line)) is not None
            ]
            matches_requested = self._section_matches_request(title, requested_titles)
            continues_requested = (
                allow_selected_sections
                and requested_titles
                and started_requested_section
                and bool(markers)
                and highest_marker > 0
                and min(markers) <= highest_marker + 1
                and max(markers) > highest_marker
            )
            if requested_titles and not (matches_requested or continues_requested):
                continue
            cleaned: list[str] = []
            seen: set[str] = set()
            for line in lines:
                stripped = line.rstrip()
                if stripped and stripped in seen:
                    continue
                cleaned.append(stripped)
                if stripped:
                    seen.add(stripped)
            while cleaned and cleaned[-1] == "":
                cleaned.pop()
            if cleaned:
                blocks.append(f"{title}:")
                blocks.extend(cleaned)
                if requested_titles:
                    started_requested_section = True
                if markers:
                    highest_marker = max(highest_marker, max(markers))
        if not blocks:
            return None
        return "\n".join(blocks).strip()

    def _paginate_answer(self, answer: str, offset: int = 0, limit: int = SHOW_MORE_PAGE_SIZE) -> tuple[str, bool, int]:
        lines = answer.splitlines()
        marker_indexes = [
            index
            for index, line in enumerate(lines)
            if self._top_level_numbered_marker(line) is not None
        ]
        if not marker_indexes:
            marker_indexes = [index for index, line in enumerate(lines) if self._structured_marker(line)]
        if not marker_indexes:
            return answer, False, 0
        if offset >= len(marker_indexes):
            return "", False, len(marker_indexes)
        end_marker = min(offset + limit, len(marker_indexes))
        start_line = marker_indexes[offset]
        end_line = marker_indexes[end_marker] if end_marker < len(marker_indexes) else len(lines)
        header_lines = [line for line in lines[:marker_indexes[0]] if line.strip()]
        page = "\n".join([*header_lines, *lines[start_line:end_line]]).strip()
        return page, end_marker < len(marker_indexes), end_marker

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
        cleaned = self._remove_unavailable_lines_from_valid_answer(cleaned)
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
        if self._is_unavailable_only_answer(lowered, unavailable_cues):
            return UNAVAILABLE_RESPONSE
        return self._renumber_numbered_lists(self._remove_duplicate_answer_lines(cleaned))

    def _validate_response(self, question: str, answer: str, context: str) -> ResponseValidation:
        normalized = self._normalize_answer(answer)
        had_disclaimer = self._contains_disclaimer(answer)

        if not self._has_answer_content(normalized):
            return ResponseValidation(answer=UNAVAILABLE_RESPONSE, quality="no_answer")

        if not self._is_grounded_answer(question, normalized, context):
            return ResponseValidation(answer=UNAVAILABLE_RESPONSE, quality="no_answer")

        quality: AnswerQuality = "partial" if had_disclaimer else "full"
        return ResponseValidation(answer=normalized, quality=quality)

    def _is_unavailable_only_answer(self, lowered: str, unavailable_cues: tuple[str, ...]) -> bool:
        if not lowered:
            return True
        normalized_unavailable = UNAVAILABLE_RESPONSE.lower()
        if lowered == normalized_unavailable:
            return True

        without_unavailable = lowered
        for cue in unavailable_cues:
            without_unavailable = without_unavailable.replace(cue, " ")
        without_unavailable = re.sub(r"[^a-z0-9]+", " ", without_unavailable).strip()
        generic_words = {
            "this",
            "information",
            "is",
            "in",
            "the",
            "provided",
            "sop",
            "context",
            "document",
            "details",
            "are",
            "were",
            "explicitly",
            "mentioned",
            "stated",
            "specified",
        }
        remaining = [word for word in without_unavailable.split() if word not in generic_words]
        return any(cue in lowered for cue in unavailable_cues) and len(remaining) <= 2

    def _remove_unavailable_lines_from_valid_answer(self, answer: str) -> str:
        if not answer.strip():
            return answer
        if not self._contains_disclaimer(answer):
            return answer

        lines = answer.splitlines()
        cleaned_lines = [self._remove_disclaimer_sentences(line) for line in lines]
        if not any(line.strip() for line in cleaned_lines):
            return answer

        return "\n".join(line for line in cleaned_lines if line.strip()).strip()

    def _contains_disclaimer(self, answer: str) -> bool:
        return any(self._is_disclaimer_text(part.strip()) for part in re.split(r"[\n.]+", answer) if part.strip())

    def _is_disclaimer_text(self, text: str) -> bool:
        lowered = text.lower().strip()
        if not lowered:
            return False

        normalized_unavailable = UNAVAILABLE_RESPONSE.lower().rstrip(".")
        lowered_without_period = lowered.rstrip(".")
        if lowered_without_period == normalized_unavailable:
            return True

        has_missing_cue = bool(
            re.search(
                r"\b("
                r"not\s+(?:available|mentioned|specified|stated|provided|described|found|included)"
                r"|no\s+(?:information|details?|relevant\s+information)"
                r"|does\s+not\s+(?:mention|specify|state|provide|describe|include)"
                r"|cannot\s+(?:find|determine|answer)"
                r"|unable\s+to\s+(?:find|determine|answer)"
                r")\b",
                lowered,
            )
        )
        if not has_missing_cue:
            return False

        return bool(
            re.search(
                r"\b("
                r"sop|provided\s+context|given\s+context|retrieved\s+context|available\s+context|"
                r"context|document|source|provided\s+information"
                r")\b",
                lowered,
            )
            or re.search(r"\b(this|that|the)\s+(information|detail|answer)\b", lowered)
            or re.search(r"\binformation\s+(?:is\s+)?not\s+(?:available|mentioned|provided|found)\b", lowered)
        )

    def _remove_disclaimer_sentences(self, text: str) -> str:
        if not text.strip():
            return text
        if not self._contains_disclaimer(text):
            return text

        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        if len(parts) <= 1 and self._is_disclaimer_text(text.strip()):
            return ""
        kept = [part for part in parts if part.strip() and not self._is_disclaimer_text(part.strip())]
        return " ".join(kept).strip()

    def _has_answer_content(self, answer: str) -> bool:
        if not answer or answer == UNAVAILABLE_RESPONSE:
            return False
        words = [word for word in tokenize(answer) if word not in GENERIC_QUERY_TERMS]
        return len(words) >= 3 or len(answer.strip()) >= 24

    def _log_response_decision(
        self,
        *,
        query: str,
        docs_count: int,
        confidence: str,
        decision: str,
        answer: str,
    ) -> None:
        logger.info(
            "RAG response decision: decision=%s docs=%d confidence=%s answer_chars=%d query=%r",
            decision,
            docs_count,
            confidence,
            len(answer or ""),
            query[:160],
        )

    def _remove_duplicate_answer_lines(self, answer: str) -> str:
        cleaned: list[str] = []
        previous_key = ""
        previous_blank = False
        in_code_block = False

        for line in answer.splitlines():
            if line.lstrip().startswith("```"):
                in_code_block = not in_code_block
                cleaned.append(line)
                previous_blank = False
                continue

            if in_code_block:
                cleaned.append(line)
                continue

            key = line.strip()
            if not key:
                if not previous_blank:
                    cleaned.append("")
                previous_blank = True
                previous_key = ""
                continue

            previous_blank = False
            if key == previous_key:
                continue
            previous_key = key
            cleaned.append(line)

        return "\n".join(cleaned).strip()

    def _renumber_numbered_lists(self, answer: str) -> str:
        lines = answer.splitlines()
        counters: dict[int, int] = {}
        fixed: list[str] = []
        in_code_block = False
        changed = False

        for line in lines:
            if line.lstrip().startswith("```"):
                in_code_block = not in_code_block
                fixed.append(line)
                continue

            if in_code_block:
                fixed.append(line)
                continue

            match = re.match(r"^(\s*)(\d+)([\).])\s+(.+)$", line)
            if not match:
                fixed.append(line)
                continue

            indent, original_number, _delimiter, body = match.groups()
            indent_level = len(indent.expandtabs(4))
            counters[indent_level] = counters.get(indent_level, 0) + 1

            for deeper_indent in [level for level in counters if level > indent_level]:
                del counters[deeper_indent]

            new_number = counters[indent_level]
            if int(original_number) != new_number:
                changed = True
            fixed.append(f"{indent}{new_number}. {body}")

        normalized = "\n".join(fixed).strip()
        if not changed and len(re.findall(r"(?m)^\s*1[\).]\s+", normalized)) <= 1:
            return normalized
        return normalized

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
                answer_specific_terms = self._grounding_terms(answer)
                supported_answer_specific_terms = answer_specific_terms & context_terms
                strong_answer_overlap = overlap >= max(4, int(len(answer_terms) * 0.30))
                if not (strong_answer_overlap and len(supported_answer_specific_terms) >= 2):
                    return False

            answer_question_terms = question_specific_terms & answer_terms
            required_question_terms = min(2, len(supported_question_terms))
            if supported_question_terms and len(answer_question_terms) < required_question_terms:
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
        answer = self._renumber_numbered_lists(response.content.strip())

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
        cursor_offset: int = 0,
        page_limit: int = SHOW_MORE_PAGE_SIZE,
    ) -> dict:
        continuation = self._manual_continuation_context(message, history)
        if continuation:
            continued_question, inferred_offset = continuation
            message = continued_question
            if cursor_offset <= 0:
                cursor_offset = inferred_offset

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
                "has_more": False,
                "next_offset": None,
            }

        message, memory_sop = self._inject_memory(message, active_sop)
        effective_sop = self._effective_active_sop(
            message,
            history,
            memory_sop,
            source_locked=source_locked,
        )

        search_query = message if cursor_offset > 0 else await self.rewrite_query(message, history, effective_sop)
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
        if not docs and cursor_offset > 0:
            detected_sop = effective_sop
            docs = self._refetch_requested_section_docs(message, detected_sop)

        if not docs:
            self._log_response_decision(
                query=search_query,
                docs_count=0,
                confidence="low",
                decision="fallback:no_docs",
                answer=UNAVAILABLE_RESPONSE,
            )
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
                "has_more": False,
                "next_offset": None,
            }

        refetched_docs = self._refetch_requested_section_docs(message, detected_sop)
        if refetched_docs:
            docs = self._merge_docs(docs, refetched_docs)
        if self._wants_complete_section_answer(message) or self._requested_item_number(message) is not None:
            selected_section_docs = self._refetch_selected_section_docs(docs, detected_sop)
            if selected_section_docs:
                docs = self._merge_docs(docs, selected_section_docs)
            continuation_docs = self._refetch_numbered_continuation_docs(message, docs, detected_sop)
            if continuation_docs:
                docs = self._merge_docs(docs, continuation_docs)
        retrieved_docs_count = len(docs)

        image = None
        for doc in docs:
            if doc.metadata.get("type") == "image" and self._is_image_relevant(
                message, doc.metadata.get("path", "")
            ):
                image = doc.metadata["path"]
                break

        context = self._format_context(docs)
        extractive_context = self._format_context(docs, strip_numbering=False)
        history_text = self._format_history(history)

        table_answer = self._table_answer_from_docs(docs)
        if table_answer:
            confidence = self._confidence_level(search_query, docs, detected_sop, candidates, effective_sop)
            sources = format_sources(docs)
            answer = self._apply_confidence_notice(table_answer, confidence)
            self._log_response_decision(
                query=search_query,
                docs_count=retrieved_docs_count,
                confidence=confidence,
                decision="answer:table",
                answer=answer,
            )
            self._remember_context(message, detected_sop)
            log_query(message, active_sop, detected_sop, confidence,
                      answer_mode=answer_mode, llm_provider=llm_provider)
            return {
                "answer": answer,
                "sources": sources,
                "followup": None,
                "active_sop": detected_sop,
                "image": image,
                "confidence": confidence,
                "suggestions": None,
                "has_more": False,
                "next_offset": None,
            }

        extractive_answer = self._extractive_section_answer(message, extractive_context)
        if extractive_answer:
            extractive_answer = self._renumber_numbered_lists(extractive_answer)
            specific_answer = self._specific_structured_item_answer(message, extractive_answer)
            if specific_answer:
                page_answer = specific_answer
                has_more = False
                next_offset = 0
            elif self._wants_complete_section_answer(message):
                page_answer = extractive_answer
                has_more = False
                next_offset = 0
            else:
                page_answer, has_more, next_offset = self._paginate_answer(
                    extractive_answer,
                    offset=max(0, cursor_offset),
                    limit=max(1, page_limit),
                )
                if has_more:
                    page_answer = f"{page_answer}\n{SHOW_MORE_MARKER}"
                page_answer = page_answer.replace(SHOW_MORE_MARKER, "").rstrip()
            confidence = self._confidence_level(search_query, docs, detected_sop, candidates, effective_sop)
            sources = format_sources(docs)
            final_answer = self._apply_confidence_notice(page_answer, confidence)
            self._log_response_decision(
                query=search_query,
                docs_count=retrieved_docs_count,
                confidence=confidence,
                decision="answer:extractive",
                answer=final_answer,
            )
            self._remember_context(message, detected_sop)
            log_query(message, active_sop, detected_sop, confidence,
                      answer_mode=answer_mode, llm_provider=llm_provider)
            return {
                "answer": final_answer,
                "sources": sources,
                "followup": None,
                "active_sop": detected_sop,
                "image": image,
                "confidence": confidence,
                "suggestions": None,
                "has_more": has_more,
                "next_offset": next_offset if has_more else None,
            }

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
        validation = self._validate_response(message, answer, context)
        answer = validation.answer

        decision = f"answer:llm:{validation.quality}"
        if validation.quality == "no_answer":
            decision = "fallback:no_answer"
            docs = []
            followup = None
            image = None
            confidence = "low"
            detected_sop = None

        answer = self._apply_confidence_notice(answer, confidence)
        self._log_response_decision(
            query=search_query,
            docs_count=retrieved_docs_count,
            confidence=confidence,
            decision=decision,
            answer=answer,
        )

        # Generate suggestions from same SOP
        suggestions = await self._generate_suggestions(message, context, detected_sop) if docs else None

        sources = format_sources(docs)

        # Log query for analytics
        log_query(message, active_sop, detected_sop, confidence,
                  answer_mode=answer_mode, llm_provider=llm_provider)
        if confidence == "low" and docs:
            save_failed_query(message, confidence, active_sop, answer[:300])
        self._remember_context(message, detected_sop)

        return {
            "answer": answer,
            "sources": sources,
            "followup": followup,
            "active_sop": detected_sop,
            "image": image,
            "confidence": confidence,
            "suggestions": suggestions or None,
            "has_more": False,
            "next_offset": None,
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
        cursor_offset: int = 0,
        page_limit: int = SHOW_MORE_PAGE_SIZE,
    ) -> AsyncIterator[dict]:
        continuation = self._manual_continuation_context(message, history)
        if continuation:
            continued_question, inferred_offset = continuation
            message = continued_question
            if cursor_offset <= 0:
                cursor_offset = inferred_offset

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
                "has_more": False,
                "next_offset": None,
            }
            return

        message, memory_sop = self._inject_memory(message, active_sop)
        effective_sop = self._effective_active_sop(
            message,
            history,
            memory_sop,
            source_locked=source_locked,
        )

        search_query = message if cursor_offset > 0 else await self.rewrite_query(message, history, effective_sop)
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
        if not docs and cursor_offset > 0:
            detected_sop = effective_sop
            docs = self._refetch_requested_section_docs(message, detected_sop)

        if not docs:
            msg = UNAVAILABLE_RESPONSE
            self._log_response_decision(
                query=search_query,
                docs_count=0,
                confidence="low",
                decision="fallback:no_docs",
                answer=msg,
            )
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
                "has_more": False,
                "next_offset": None,
            }
            return

        refetched_docs = self._refetch_requested_section_docs(message, detected_sop)
        if refetched_docs:
            docs = self._merge_docs(docs, refetched_docs)
        if self._wants_complete_section_answer(message) or self._requested_item_number(message) is not None:
            selected_section_docs = self._refetch_selected_section_docs(docs, detected_sop)
            if selected_section_docs:
                docs = self._merge_docs(docs, selected_section_docs)
            continuation_docs = self._refetch_numbered_continuation_docs(message, docs, detected_sop)
            if continuation_docs:
                docs = self._merge_docs(docs, continuation_docs)
        retrieved_docs_count = len(docs)

        image = None
        for doc in docs:
            if doc.metadata.get("type") == "image" and self._is_image_relevant(
                message, doc.metadata.get("path", "")
            ):
                image = doc.metadata["path"]
                break

        context = self._format_context(docs)
        extractive_context = self._format_context(docs, strip_numbering=False)
        history_text = self._format_history(history)

        table_answer = self._table_answer_from_docs(docs)
        if table_answer:
            confidence = self._confidence_level(search_query, docs, detected_sop, candidates, effective_sop)
            answer_text = self._apply_confidence_notice(table_answer, confidence)
            yield {"type": "token", "content": answer_text}
            sources = format_sources(docs)
            self._log_response_decision(
                query=search_query,
                docs_count=retrieved_docs_count,
                confidence=confidence,
                decision="answer:table",
                answer=answer_text,
            )
            self._remember_context(message, detected_sop)
            log_query(message, active_sop, detected_sop, confidence,
                      answer_mode=answer_mode, llm_provider=llm_provider)
            yield {
                "type": "done",
                "sources": sources,
                "followup": None,
                "active_sop": detected_sop,
                "image": image,
                "full_answer": answer_text,
                "confidence": confidence,
                "suggestions": None,
                "has_more": False,
                "next_offset": None,
            }
            return

        extractive_answer = self._extractive_section_answer(message, extractive_context)
        if extractive_answer:
            extractive_answer = self._renumber_numbered_lists(extractive_answer)
            specific_answer = self._specific_structured_item_answer(message, extractive_answer)
            if specific_answer:
                page_answer = specific_answer
                has_more = False
                next_offset = 0
            elif self._wants_complete_section_answer(message):
                page_answer = extractive_answer
                has_more = False
                next_offset = 0
            else:
                page_answer, has_more, next_offset = self._paginate_answer(
                    extractive_answer,
                    offset=max(0, cursor_offset),
                    limit=max(1, page_limit),
                )
            confidence = self._confidence_level(search_query, docs, detected_sop, candidates, effective_sop)
            answer_text = self._apply_confidence_notice(page_answer, confidence)
            yield {"type": "token", "content": answer_text}
            sources = format_sources(docs)
            self._log_response_decision(
                query=search_query,
                docs_count=retrieved_docs_count,
                confidence=confidence,
                decision="answer:extractive",
                answer=answer_text,
            )
            self._remember_context(message, detected_sop)
            log_query(message, active_sop, detected_sop, confidence,
                      answer_mode=answer_mode, llm_provider=llm_provider)
            yield {
                "type": "done",
                "sources": sources,
                "followup": None,
                "active_sop": detected_sop,
                "image": image,
                "full_answer": answer_text,
                "confidence": confidence,
                "suggestions": None,
                "has_more": has_more,
                "next_offset": next_offset if has_more else None,
            }
            return

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
        async for chunk in self.llm.astream(messages):
            token = chunk.content or ""
            buffer += token

        answer_text, followup = self._parse_response(buffer)
        answer_text = re.sub(
            r"^ANSWER:\s*",
            "",
            answer_text,
            flags=re.IGNORECASE,
        ).strip()
        validation = self._validate_response(message, answer_text, context)
        answer_text = validation.answer

        decision = f"answer:llm:{validation.quality}"
        if validation.quality == "no_answer":
            decision = "fallback:no_answer"
            docs = []
            followup = None
            image = None
            confidence = "low"
            detected_sop = None

        answer_text = self._apply_confidence_notice(answer_text, confidence)
        yield {"type": "token", "content": answer_text}
        self._log_response_decision(
            query=search_query,
            docs_count=retrieved_docs_count,
            confidence=confidence,
            decision=decision,
            answer=answer_text,
        )

        # Generate suggestions from same SOP
        suggestions = await self._generate_suggestions(message, context, detected_sop) if docs else None

        sources = format_sources(docs)

        # Log for analytics
        log_query(message, active_sop, detected_sop, confidence,
                  answer_mode=answer_mode, llm_provider=llm_provider)
        if confidence == "low" and docs:
            save_failed_query(message, confidence, active_sop, answer_text[:300])
        self._remember_context(message, detected_sop)

        yield {
            "type": "done",
            "sources": sources,
            "followup": followup,
            "active_sop": detected_sop,
            "image": image,
            "full_answer": answer_text,
            "confidence": confidence,
            "suggestions": suggestions or None,
            "has_more": False,
            "next_offset": None,
        }
