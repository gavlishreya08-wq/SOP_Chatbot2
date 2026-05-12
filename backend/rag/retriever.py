from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from difflib import get_close_matches
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "describe",
    "do",
    "explain",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "show",
    "tell",
    "the",
    "to",
    "what",
    "this",
    "these",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

BROAD_KEYWORDS = {
    "all",
    "complete",
    "details",
    "entire",
    "everything",
    "explain",
    "full",
    "guidelines",
    "list",
    "overview",
    "process",
    "responsibilities",
    "sections",
    "steps",
    "workflow",
}

COMPLETENESS_KEYWORDS = {
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
    "standards",
    "sub",
}

SMALL_SECTION_PHRASES = {
    "job objective",
    "job objectives",
    "training requirement",
    "training requirements",
    "reporting authority",
    "minimum qualification",
    "minimum qualifications",
    "minimum experience",
    "minimum experiences",
}

GENERIC_QUERY_TERMS = {
    "calendar",
    "company",
    "job",
    "meeting",
    "office",
    "objective",
    "objectives",
    "policy",
    "procedure",
    "process",
    "responsibility",
    "responsibilities",
    "role",
    "roles",
    "sop",
    "standard",
    "standards",
    "step",
    "steps",
    "workflow",
    "year",
}

VISUAL_KEYWORDS = {
    "chart",
    "diagram",
    "flow",
    "flowchart",
    "hierarchy",
    "visual",
    "workflow",
}

PROCEDURE_SECTION_TITLES = {
    "agenda",
    "checklist",
    "guidelines",
    "overview",
    "procedure",
    "process",
    "responsibilities",
    "steps",
    "workflow",
}

QUERY_PHRASE_REWRITES = {
    "source code management": "source code management gitlab code repository source control",
    "manage source code": "source code management gitlab code repository",
    "code management": "source code management code repository",
    "code repository": "code repository gitlab source control",
    "source control": "source code management gitlab repository",
    "git repository": "gitlab code repository",
    "repo setup": "repository setup gitlab",
}


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("&", " and ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw_token in normalize_text(text).split():
        if len(raw_token) <= 1 or raw_token in STOPWORDS:
            continue
        token = raw_token
        if token.endswith("ies") and len(token) > 4:
            token = f"{token[:-3]}y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        tokens.add(token)
        tokens.add(raw_token)
    return tokens


def normalize_query(text: str) -> str:
    normalized = normalize_text(text)
    for phrase, replacement in QUERY_PHRASE_REWRITES.items():
        normalized = re.sub(rf"\b{re.escape(phrase)}\b", replacement, normalized)
    replacements = {
        "responsibilites": "responsibilities",
        "responsibilty": "responsibility",
        "resposibilities": "responsibilities",
        "autority": "authority",
        "authorty": "authority",
        "work flow": "workflow",
        "jra": "jira",
        "git": "gitlab",
        "repos": "repository",
        "repo": "repository",
        "sop": "sop",
    }
    words = []
    vocabulary = GENERIC_QUERY_TERMS | BROAD_KEYWORDS | {
        "authority",
        "database",
        "deployment",
        "development",
        "engineer",
        "lead",
        "qualification",
        "reporting",
        "requirement",
        "technical",
        "testing",
        "training",
    }
    for word in normalized.split():
        replacement = replacements.get(word)
        if replacement:
            words.append(replacement)
            continue
        close = get_close_matches(word, vocabulary, n=1, cutoff=0.86)
        words.append(close[0] if close else word)
    return " ".join(words)


def _terms_overlap(query_terms: set[str], target_terms: set[str]) -> set[str]:
    matches: set[str] = set()

    for query_term in query_terms:
        for target_term in target_terms:
            if query_term == target_term:
                matches.add(query_term)
                break

            if len(query_term) >= 4 and len(target_term) >= 4:
                if query_term.startswith(target_term) or target_term.startswith(query_term):
                    matches.add(query_term)
                    break
                if query_term[:5] == target_term[:5]:
                    matches.add(query_term)
                    break

    return matches


def _strict_terms_overlap(query_terms: set[str], target_terms: set[str]) -> set[str]:
    if not query_terms or not target_terms:
        return set()
    return query_terms & target_terms


def humanize_source_name(source: str) -> str:
    stem = Path(source).stem
    stem = stem.replace("&", " and ")
    stem = re.sub(r"([a-z])([A-Z])", r"\1 \2", stem)
    stem = re.sub(r"[_\-.()/]+", " ", stem)
    stem = re.sub(r"\b(ver|version|v)\s*\d+(\.\d+)?\b", " ", stem, flags=re.I)
    stem = re.sub(r"\b\d{8,}\b", " ", stem)
    return re.sub(r"\s+", " ", stem).strip()


def keyword_overlap_score(
    query_terms: set[str],
    target_terms: set[str],
    *,
    allow_fuzzy: bool = True,
) -> float:
    if not query_terms or not target_terms:
        return 0.0
    overlap = (
        _terms_overlap(query_terms, target_terms)
        if allow_fuzzy
        else _strict_terms_overlap(query_terms, target_terms)
    )
    if not overlap:
        return 0.0
    coverage = len(overlap) / len(query_terms)
    bonus = 0.1 if len(overlap) >= 2 else 0.0
    return coverage + bonus


def source_match_score(
    query: str,
    source: str,
    *,
    source_terms: set[str] | None = None,
    normalized_source: str | None = None,
) -> float:
    query_terms = tokenize(query)
    source_title = humanize_source_name(source)
    source_terms = source_terms or tokenize(source_title)
    normalized_query = normalize_text(query)
    normalized_source = normalized_source or normalize_text(source_title)

    substring_bonus = 0.0
    if normalized_query and normalized_source:
        if normalized_query in normalized_source or normalized_source in normalized_query:
            substring_bonus = 0.35

    phrase_bonus = 0.0
    filtered_words = [
        word for word in normalized_query.split() if word and word not in STOPWORDS
    ]
    for size, bonus in ((3, 0.45), (2, 0.35)):
        if len(filtered_words) < size:
            continue
        phrases = [
            " ".join(filtered_words[index : index + size])
            for index in range(len(filtered_words) - size + 1)
        ]
        if any(phrase in normalized_source for phrase in phrases):
            phrase_bonus = bonus
            break

    return keyword_overlap_score(query_terms, source_terms) + substring_bonus + phrase_bonus


def get_k_for_question(question: str) -> int:
    question = normalize_query(question)
    q_terms = tokenize(question)
    if _is_small_targeted_section_query(question):
        return 3
    if _wants_complete_structured_answer(question):
        return 18
    if len(q_terms) >= 6 or any(word in q_terms for word in BROAD_KEYWORDS):
        return 12
    return 6


def get_candidate_pool_size(question: str) -> int:
    question = normalize_query(question)
    q_terms = tokenize(question)
    if _is_small_targeted_section_query(question):
        return 40
    if _wants_complete_structured_answer(question):
        return 180
    if len(q_terms) >= 6 or any(word in q_terms for word in BROAD_KEYWORDS):
        return 100
    return 60


def _is_small_targeted_section_query(query: str) -> bool:
    terms = tokenize(query)
    normalized = normalize_text(query)
    if terms & COMPLETENESS_KEYWORDS:
        return False
    return any(phrase in normalized for phrase in SMALL_SECTION_PHRASES)


def _wants_complete_structured_answer(query: str) -> bool:
    terms = tokenize(query)
    normalized = normalize_text(query)
    if re.search(r"\b\d+(?:\.\d+)+\b", normalized):
        return True
    if terms & COMPLETENESS_KEYWORDS:
        return True
    structured_terms = {
        "checklist",
        "guideline",
        "guidelines",
        "meeting",
        "practice",
        "procedure",
        "process",
        "responsibilities",
        "responsibility",
        "role",
        "standard",
        "standards",
        "step",
        "workflow",
    }
    return bool(terms & structured_terms and terms & BROAD_KEYWORDS)


def build_source_catalog(vectorstore: Chroma) -> dict[str, dict[str, str | set[str]]]:
    records = vectorstore.get(include=["metadatas", "documents"])
    metadata = records.get("metadatas", [])
    documents = records.get("documents", [])
    catalog: dict[str, dict[str, str | set[str]]] = {}

    for index, entry in enumerate(metadata):
        source = entry.get("source")
        if not source or source in catalog:
            continue

        title = entry.get("source_title") or humanize_source_name(source)
        aliases = entry.get("source_aliases", "")
        intents = entry.get("source_intents", "")
        summary = entry.get("source_summary", "")
        section_titles = entry.get("source_section_titles", "")
        snippet = ""
        if index < len(documents) and documents[index]:
            snippet = str(documents[index])[:300]
        title_alias_text = f"{title} {aliases} {intents}"
        search_text = f"{title} {aliases} {intents} {summary} {section_titles} {snippet}"
        combined_tokens = tokenize(search_text)
        catalog[source] = {
            "title": title,
            "normalized_title_alias_text": normalize_text(title_alias_text),
            "normalized_title": normalize_text(title),
            "normalized_search_text": normalize_text(search_text),
            "title_alias_tokens": tokenize(title_alias_text),
            "tokens": combined_tokens,
        }

    return catalog


def infer_source_candidates(
    query: str,
    source_catalog: dict[str, dict[str, str | set[str]]],
    limit: int = 3,
) -> list[tuple[str, float]]:
    query_terms = tokenize(query)
    primary_terms = {
        term for term in query_terms
        if term not in GENERIC_QUERY_TERMS
    }
    ranked: list[tuple[str, float]] = []

    for source, info in source_catalog.items():
        broad_score = source_match_score(
            query,
            source,
            source_terms=info.get("tokens"),
            normalized_source=info.get("normalized_search_text"),
        )
        title_alias_score = source_match_score(
            query,
            source,
            source_terms=info.get("title_alias_tokens"),
            normalized_source=info.get("normalized_title_alias_text"),
        )

        score = broad_score + (title_alias_score * 0.45)
        if primary_terms:
            title_overlap = _terms_overlap(
                primary_terms,
                info.get("title_alias_tokens", set()),
            )
            if title_overlap:
                score += 0.25 * (len(title_overlap) / len(primary_terms))
            else:
                score -= 0.4

        if score > 0:
            ranked.append((source, score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return [
        (source, score)
        for source, score in ranked[:limit]
        if score >= 0.55
    ]


def is_visual_query(question: str) -> bool:
    terms = tokenize(question)
    return any(term in VISUAL_KEYWORDS for term in terms)


def _primary_query_terms(query_terms: set[str]) -> list[str]:
    return [
        term for term in sorted(query_terms, key=len, reverse=True)
        if term not in GENERIC_QUERY_TERMS
    ]


def _specific_query_terms(query_terms: set[str]) -> list[str]:
    return [
        term for term in _primary_query_terms(query_terms)
        if term not in BROAD_KEYWORDS
    ]


def _active_source_shifted(
    active_sop: str | None,
    query_terms: set[str],
    inferred_sources: list[tuple[str, float]],
    source_catalog: dict[str, dict[str, str | set[str]]],
) -> bool:
    if not active_sop or not inferred_sources:
        return False

    target_source, target_score = inferred_sources[0]
    if target_source == active_sop:
        return False

    specific_terms = set(_specific_query_terms(query_terms))
    if not specific_terms:
        return False

    active_info = source_catalog.get(active_sop, {})
    target_info = source_catalog.get(target_source, {})
    active_overlap = _terms_overlap(
        specific_terms,
        active_info.get("title_alias_tokens", set()),
    )
    target_overlap = _terms_overlap(
        specific_terms,
        target_info.get("title_alias_tokens", set()),
    )
    active_title_score = 0.0
    active_title_text = str(active_info.get("normalized_title_alias_text") or "")
    if active_title_text:
        active_title_score = keyword_overlap_score(specific_terms, set(active_title_text.split()))

    target_title_score = 0.0
    target_title_text = str(target_info.get("normalized_title_alias_text") or "")
    if target_title_text:
        target_title_score = keyword_overlap_score(specific_terms, set(target_title_text.split()))

    if target_score >= 1.05 and len(target_overlap) > len(active_overlap):
        return True

    if target_overlap and not active_overlap and target_score >= 0.85:
        return True

    if target_title_score >= 0.6 and (target_title_score - active_title_score) >= 0.25:
        return True

    return False


def _page_text(doc: Document) -> str:
    return normalize_text(doc.page_content[:1600])


def _context_body(doc: Document) -> str:
    content = doc.page_content or ""
    if "CONTENT:" in content:
        return content.split("CONTENT:", 1)[1].strip()
    return content.strip()


def _document_query_terms(doc: Document) -> set[str]:
    return (
        tokenize(doc.page_content)
        | tokenize(str(doc.metadata.get("source", "")))
        | tokenize(str(doc.metadata.get("source_title", "")))
        | tokenize(str(doc.metadata.get("source_aliases", "")))
        | tokenize(str(doc.metadata.get("section_title", "")))
    )


def _structured_line_count(doc: Document) -> int:
    return sum(
        1
        for line in _context_body(doc).splitlines()
        if re.match(r"^\s*(?:[-*]\s+|\d+(?:\.\d+)*[\).]\s+|[a-zA-Z][\).]\s+)", line)
    )


def _content_length_score(doc: Document) -> float:
    content = _context_body(doc)
    if not content:
        return -0.1
    length = len(content)
    if length < 120:
        return -0.2
    if length < 260:
        return -0.08
    if length > 1200:
        return 0.14
    if length > 700:
        return 0.1
    return 0.04


def _required_primary_matches(primary_terms: list[str]) -> int:
    if not primary_terms:
        return 0
    if len(primary_terms) == 1:
        return 1
    return 2


def _fallback_doc_is_relevant(
    doc: Document,
    query: str,
    query_terms: set[str],
    inferred_sources: list[tuple[str, float]],
) -> bool:
    if doc.metadata.get("type") == "image" and not is_visual_query(query):
        return False

    primary_terms = _primary_query_terms(query_terms)
    doc_terms = _document_query_terms(doc)
    matched_primary_terms = [
        term for term in primary_terms
        if term in _strict_terms_overlap({term}, doc_terms)
    ]
    required_primary_matches = _required_primary_matches(primary_terms)
    source_score = max(
        source_match_score(query, str(doc.metadata.get("source", ""))),
        source_match_score(query, str(doc.metadata.get("source_title", ""))),
    )
    content_score = keyword_overlap_score(
        query_terms,
        tokenize(doc.page_content),
        allow_fuzzy=False,
    )
    top_inferred_score = inferred_sources[0][1] if inferred_sources else 0.0

    if required_primary_matches and len(matched_primary_terms) < required_primary_matches:
        if top_inferred_score < 1.1 or len(matched_primary_terms) == 0:
            return False

    if not inferred_sources and source_score < 0.85 and content_score < 0.45:
        return False

    if inferred_sources and source_score < 0.55 and content_score < 0.25 and top_inferred_score < 1.1:
        return False

    return True


def _score_document(
    doc: Document,
    distance: float,
    query: str,
    query_terms: set[str],
    preferred_sources: dict[str, float],
    active_sop: str | None,
) -> float:
    source = doc.metadata.get("source", "")
    source_title = doc.metadata.get("source_title", source)
    normalized_query = normalize_text(query)
    wants_complete = _wants_complete_structured_answer(query)
    semantic_score = 1.0 / (1.0 + max(distance, 0.0))
    content_score = keyword_overlap_score(
        query_terms,
        tokenize(doc.page_content),
        allow_fuzzy=False,
    )
    title_score = max(
        source_match_score(query, source),
        source_match_score(query, source_title),
    )

    exact_phrase_bonus = 0.0
    if normalized_query and normalized_query in _page_text(doc):
        exact_phrase_bonus = 0.15

    preferred_source_bonus = preferred_sources.get(source, 0.0) * 0.35
    active_source_bonus = 0.18 if active_sop and source == active_sop else 0.0

    section_title = normalize_text(str(doc.metadata.get("section_title", "")))
    content_type = str(doc.metadata.get("content_type", ""))
    source_kind = str(doc.metadata.get("source_kind", ""))
    section_bonus = 0.0
    history_penalty = 0.0
    role_query = any(
        term in {"role", "roles"} or term.startswith("responsibil")
        for term in query_terms
    )
    process_query = any(
        term in {"objective", "objectives", "process", "procedure", "procedures", "workflow"}
        for term in query_terms
    )
    if role_query and source_kind == "role":
        if "responsibil" in section_title:
            section_bonus += 0.22
        elif section_title in {"job objectives", "reporting authority", "training requirement"}:
            section_bonus += 0.12
        if content_type == "focus":
            section_bonus += 0.12
    elif process_query:
        if section_title in {"objective", "objectives", "overview", "process", "procedure", "workflow"}:
            section_bonus += 0.14
        if content_type == "focus":
            section_bonus += 0.08

    if not any(term in {"history", "revision", "change"} for term in query_terms):
        if section_title in {"revision history", "review history"}:
            history_penalty = -0.18
        elif any(term in section_title for term in {"addition", "updation"}):
            history_penalty = -0.12

    image_penalty = 0.0
    if doc.metadata.get("type") == "image" and not is_visual_query(query):
        image_penalty = -0.25

    richness_score = _content_length_score(doc)
    structured_bonus = 0.0
    structured_line_count = _structured_line_count(doc)
    if structured_line_count >= 3:
        structured_bonus += min(0.2, structured_line_count * 0.02)

    generic_chunk_penalty = 0.0
    if wants_complete:
        if content_type == "profile":
            generic_chunk_penalty -= 0.5
        elif content_type == "focus":
            generic_chunk_penalty -= 0.18
        if content_type == "section":
            richness_score += 0.08
        elif content_type:
            richness_score -= 0.06
        if section_title in PROCEDURE_SECTION_TITLES:
            structured_bonus += 0.12
        if structured_line_count == 0 and content_type != "section":
            generic_chunk_penalty -= 0.08
    elif content_type == "profile":
        generic_chunk_penalty -= 0.22
    elif content_type == "focus":
        generic_chunk_penalty -= 0.04

    return (
        semantic_score
        + (content_score * 0.85)
        + (title_score * 0.55)
        + exact_phrase_bonus
        + preferred_source_bonus
        + active_source_bonus
        + section_bonus
        + history_penalty
        + image_penalty
        + richness_score
        + structured_bonus
        + generic_chunk_penalty
    )


def _dedupe_results(results: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
    deduped: dict[str, tuple[Document, float]] = {}

    for doc, distance in results:
        chunk_id = doc.metadata.get("chunk_id") or (
            f"{doc.metadata.get('source')}::{doc.metadata.get('page')}::{hash(doc.page_content)}"
        )
        existing = deduped.get(chunk_id)
        if existing is None or distance < existing[1]:
            deduped[chunk_id] = (doc, distance)

    return list(deduped.values())


SECTION_INTENT_MAP: dict[str, list[str]] = {
    "objective": ["Objective", "Objectives", "Overview", "Job Objectives"],
    "responsibilities": ["Responsibilities", "Role", "Roles"],
    "workflow": ["Workflow", "Process", "Procedure"],
    "checklist": ["Checklist", "Best Practices", "Guidelines"],
    "participants": ["Participants", "Inputs", "Outputs"],
    "steps": ["Workflow", "Process", "Procedure", "Checklist"],
}


def _detect_section_intent(query: str) -> list[str]:
    """Detect which SOP sections the user likely wants based on query terms."""
    terms = tokenize(query)
    preferred: list[str] = []
    for intent_key, section_names in SECTION_INTENT_MAP.items():
        if any(term.startswith(intent_key[:5]) for term in terms):
            preferred.extend(section_names)
    return preferred


def _bm25_score(
    query_terms: set[str],
    doc_terms: set[str],
    doc_len: int,
    avg_doc_len: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Simple BM25-style score for keyword fallback."""
    if not query_terms or not doc_terms:
        return 0.0
    overlap = _strict_terms_overlap(query_terms, doc_terms)
    if not overlap:
        return 0.0
    score = 0.0
    for _ in overlap:
        tf = 1.0
        idf = 1.0
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / max(avg_doc_len, 1)))
        score += idf * (numerator / denominator)
    return score


def bm25_fallback_search(
    vectorstore: Chroma,
    query: str,
    *,
    active_sop: str | None = None,
    top_k: int = 6,
) -> list[tuple[Document, float]]:
    """Keyword-based BM25 fallback when semantic search is weak.

    Fetches a broad pool from the vectorstore and re-ranks by BM25 keyword
    overlap, preferring documents that match query terms in title or content.
    """
    query_terms = tokenize(query)
    if not query_terms:
        return []

    pool_size = max(100, get_candidate_pool_size(query))
    results: list[tuple[Document, float]] = vectorstore.similarity_search_with_score(
        query, k=pool_size
    )
    if active_sop:
        results.extend(
            vectorstore.similarity_search_with_score(
                query, k=20, filter={"source": active_sop}
            )
        )

    results = _dedupe_results(results)
    avg_len = sum(len(doc.page_content.split()) for doc, _ in results) / max(len(results), 1)

    scored: list[tuple[float, Document, float]] = []
    for doc, distance in results:
        if doc.metadata.get("type") == "image" and not is_visual_query(query):
            continue
        content_terms = tokenize(doc.page_content)
        title_terms = (
            tokenize(str(doc.metadata.get("source", "")))
            | tokenize(str(doc.metadata.get("source_title", "")))
            | tokenize(str(doc.metadata.get("source_aliases", "")))
        )
        doc_len = len(doc.page_content.split())
        content_score = _bm25_score(query_terms, content_terms, doc_len, avg_len)
        title_score = keyword_overlap_score(query_terms, title_terms)
        active_bonus = 0.3 if active_sop and doc.metadata.get("source") == active_sop else 0.0
        total = content_score + title_score * 0.6 + active_bonus
        scored.append((total, doc, distance))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [(doc, distance) for _, doc, distance in scored[:top_k] if _ > 0.3]


def _section_sort_key(doc: Document) -> tuple[int, int, int, str]:
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


def _vectorstore_documents_for_source(vectorstore: Chroma, source: str) -> list[Document]:
    try:
        records = vectorstore.get(where={"source": source}, include=["documents", "metadatas"])
    except TypeError:
        records = vectorstore.get(include=["documents", "metadatas"])
    except Exception:
        return []
    documents = records.get("documents") or []
    metadatas = records.get("metadatas") or []
    docs = []
    for content, metadata in zip(documents, metadatas):
        if not metadata or metadata.get("source") != source:
            continue
        docs.append(Document(page_content=content or "", metadata=metadata))
    return docs


def _matching_section_titles(query: str, preferred_sections: list[str]) -> set[str]:
    terms = tokenize(query)
    titles = {normalize_text(title) for title in preferred_sections}
    if any(term.startswith("responsibil") for term in terms):
        titles.add("responsibilities")
    if "objective" in terms or "objectives" in terms:
        titles.update({"objective", "objectives", "job objectives"})
    if "training" in terms:
        titles.add("training requirement")
    if "reporting" in terms or "authority" in terms:
        titles.add("reporting authority")
    if "qualification" in terms or "qualifications" in terms:
        titles.add("minimum qualification")
    if "experience" in terms:
        titles.add("minimum experience")
    if "standard" in terms or "standards" in terms:
        titles.add("standards")
    if "guideline" in terms or "guidelines" in terms:
        titles.add("guidelines")
    if "workflow" in terms or "process" in terms or "procedure" in terms or "step" in terms:
        titles.update({"workflow", "process", "procedure", "steps"})
    return titles


def _expand_requested_sections(
    vectorstore: Chroma,
    selected_docs: list[Document],
    source: str,
    query: str,
    preferred_sections: list[str],
) -> list[Document]:
    wanted_titles = _matching_section_titles(query, preferred_sections)
    if not wanted_titles and not _wants_complete_structured_answer(query):
        return selected_docs
    source_docs = _vectorstore_documents_for_source(vectorstore, source)
    if not source_docs:
        return selected_docs
    selected_indexes = {
        doc.metadata.get("section_index")
        for doc in selected_docs
        if isinstance(doc.metadata.get("section_index"), int)
        and doc.metadata.get("content_type") == "section"
    }
    include_indexes: set[int] = set()
    for doc in source_docs:
        if doc.metadata.get("content_type") != "section":
            continue
        section_index = doc.metadata.get("section_index")
        if not isinstance(section_index, int):
            continue
        title = normalize_text(str(doc.metadata.get("section_title", "")))
        if title in wanted_titles or (not wanted_titles and section_index in selected_indexes):
            include_indexes.add(section_index)
    if not include_indexes:
        return selected_docs
    expanded = [
        doc for doc in source_docs
        if doc.metadata.get("content_type") == "section"
        and doc.metadata.get("section_index") in include_indexes
    ]
    by_id: dict[str, Document] = {}
    for doc in [*selected_docs, *expanded]:
        chunk_id = str(doc.metadata.get("chunk_id") or id(doc))
        by_id[chunk_id] = doc
    return sorted(by_id.values(), key=_section_sort_key)


def generate_query_variants(query: str) -> list[str]:
    """Generate rewritten query variants for multi-query retrieval."""
    normalized_query = normalize_query(query)
    terms = tokenize(normalized_query)
    variants = [normalized_query]

    specific = [t for t in terms if t not in GENERIC_QUERY_TERMS and t not in BROAD_KEYWORDS]
    generic = [t for t in terms if t in GENERIC_QUERY_TERMS or t in BROAD_KEYWORDS]

    if specific and generic:
        variants.append(" ".join(specific))

    phrase_variants = [
        ("source code management", "gitlab code repository procedure"),
        ("manage source code", "gitlab code repository workflow"),
        ("code repository", "repository setup gitlab"),
        ("source control", "gitlab repository workflow"),
    ]
    for phrase, replacement in phrase_variants:
        if phrase in normalized_query:
            variants.append(normalized_query.replace(phrase, replacement))

    synonym_map = {
        "process": "workflow procedure steps",
        "workflow": "process procedure steps",
        "procedure": "process workflow steps",
        "procedures": "procedure workflow steps",
        "responsibilities": "roles duties accountabilities",
        "roles": "responsibilities duties",
        "objective": "purpose goal overview",
        "steps": "process workflow procedure",
        "checklist": "steps list guidelines",
        "approval": "review sign off",
        "issue": "ticket defect bug",
        "repository": "repo codebase gitlab",
        "gitlab": "repository source control gitlab",
    }

    for term in specific[:3]:
        synonyms = synonym_map.get(term)
        if not synonyms:
            continue
        variants.append(f"{normalized_query} {' '.join(synonyms.split()[:2])}".strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        cleaned = " ".join(variant.split()).strip()
        if not cleaned or cleaned in seen:
            continue
        deduped.append(cleaned)
        seen.add(cleaned)
    return deduped[:5]


def _select_docs_for_best_source(
    docs: list[tuple[float, Document]],
    query: str,
    *,
    limit: int,
) -> list[Document]:
    wants_complete = _wants_complete_structured_answer(query)
    selected: list[Document] = []
    non_section_quota = 1 if not wants_complete else 0
    non_section_count = 0

    for _, doc in docs:
        content_type = str(doc.metadata.get("content_type", ""))
        if wants_complete and content_type == "profile":
            continue
        if content_type != "section":
            if non_section_count >= non_section_quota:
                continue
            non_section_count += 1
        selected.append(doc)
        if len(selected) >= limit:
            break

    if selected:
        return selected
    return [doc for _, doc in docs[:limit]]


def retrieve(
    vectorstore: Chroma,
    query: str,
    *,
    active_sop: str | None = None,
    source_catalog: dict[str, dict[str, str | set[str]]] | None = None,
) -> tuple[list[Document], str | None]:
    query = normalize_query(query)
    query_terms = tokenize(query)
    if not query_terms:
        return [], active_sop

    source_catalog = source_catalog or {}
    inferred_sources = infer_source_candidates(query, source_catalog)
    explicit_source_shift = _active_source_shifted(
        active_sop,
        query_terms,
        inferred_sources,
        source_catalog,
    )
    preferred_sources = {source: score for source, score in inferred_sources}

    if active_sop and not explicit_source_shift and active_sop not in preferred_sources:
        preferred_sources[active_sop] = 0.4

    candidate_pool_size = get_candidate_pool_size(query)
    active_source_results: list[tuple[Document, float]] = []
    if active_sop and not explicit_source_shift:
        active_source_results = vectorstore.similarity_search_with_score(
            query,
            k=max(8, get_k_for_question(query) * 2),
            filter={"source": active_sop},
        )

    # Hybrid retrieval: semantic variants plus keyword/BM25 candidates are always merged.
    results: list[tuple[Document, float]] = []
    query_variants = generate_query_variants(query)
    for variant in query_variants:
        results.extend(
            vectorstore.similarity_search_with_score(
                variant,
                k=candidate_pool_size if variant == query else candidate_pool_size // 2,
            )
        )
    results.extend(bm25_fallback_search(vectorstore, query, active_sop=active_sop, top_k=get_candidate_pool_size(query)))

    semantic_sources: list[str] = []
    for doc, _ in results[:10]:
        source = doc.metadata.get("source")
        if source and source not in semantic_sources:
            semantic_sources.append(source)
        if len(semantic_sources) >= 4:
            break

    filtered_sources = [source for source, _ in inferred_sources[:2]]
    filtered_sources.extend(
        source for source in semantic_sources if source not in filtered_sources
    )
    if active_sop and not explicit_source_shift:
        filtered_sources = [active_sop] + [
            source for source in filtered_sources if source != active_sop
        ]

    for source in filtered_sources:
        results.extend(
            vectorstore.similarity_search_with_score(
                query,
                k=max(8, get_k_for_question(query) * 2),
                filter={"source": source},
            )
        )

    if active_sop and active_source_results:
        generic_followup = not _specific_query_terms(query_terms) and (
            len(query_terms) <= 4
            or all(term in GENERIC_QUERY_TERMS or term in BROAD_KEYWORDS for term in query_terms)
        )
        if generic_followup:
            selected_docs = [doc for doc, _ in _dedupe_results(active_source_results)[: get_k_for_question(query)]]
            if selected_docs:
                return selected_docs, active_sop

    # Section-first retrieval: boost docs matching intent sections
    preferred_sections = _detect_section_intent(query)

    ranked: list[tuple[float, Document, float]] = []
    for doc, distance in _dedupe_results(results):
        score = _score_document(
            doc,
            distance,
            query,
            query_terms,
            preferred_sources,
            active_sop,
        )
        # Section-first bonus
        if preferred_sections:
            doc_section = str(doc.metadata.get("section_title", ""))
            if doc_section in preferred_sections:
                score += 0.15
        ranked.append((score, doc, distance))

    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return [], None

    top_score, top_doc, top_distance = ranked[0]
    top_source_score = max(
        source_match_score(query, top_doc.metadata.get("source", "")),
        source_match_score(query, top_doc.metadata.get("source_title", "")),
    )
    top_content_score = keyword_overlap_score(
        query_terms,
        tokenize(top_doc.page_content),
        allow_fuzzy=False,
    )
    top_doc_terms = tokenize(top_doc.page_content) | tokenize(top_doc.metadata.get("source", ""))
    top_overlap = _strict_terms_overlap(query_terms, top_doc_terms)
    primary_terms = _primary_query_terms(query_terms)

    strong_semantic_match = top_distance <= 1.08
    strong_lexical_match = top_content_score >= 0.2 or top_source_score >= 0.75
    if top_score < 0.95 or not (strong_semantic_match or strong_lexical_match):
        # Fallback: try BM25 keyword search before giving up
        fallback_results = bm25_fallback_search(
            vectorstore, query, active_sop=active_sop
        )
        if fallback_results:
            fallback_docs = [doc for doc, _ in fallback_results]
            fallback_source = fallback_docs[0].metadata.get("source")
            if _fallback_doc_is_relevant(
                fallback_docs[0],
                query,
                query_terms,
                inferred_sources,
            ):
                return fallback_docs[:get_k_for_question(query)], fallback_source
        return [], None
    if primary_terms:
        required_primary_matches = _required_primary_matches(primary_terms)
        matched_primary_terms = [
            term for term in primary_terms
            if term in top_overlap
        ]
        if len(matched_primary_terms) < required_primary_matches:
            # Fallback: try BM25 keyword search before giving up
            fallback_results = bm25_fallback_search(
                vectorstore, query, active_sop=active_sop
            )
            if fallback_results:
                fallback_docs = [doc for doc, _ in fallback_results]
                fallback_source = fallback_docs[0].metadata.get("source")
                if _fallback_doc_is_relevant(
                    fallback_docs[0],
                    query,
                    query_terms,
                    inferred_sources,
                ):
                    return fallback_docs[:get_k_for_question(query)], fallback_source
            return [], None

    source_buckets: dict[str, list[tuple[float, Document]]] = defaultdict(list)
    for score, doc, _ in ranked:
        source = doc.metadata.get("source") or "unknown"
        source_buckets[source].append((score, doc))

    top_inferred = inferred_sources[:2]
    if (
        top_inferred
        and top_inferred[0][1] >= 1.1
        and top_inferred[0][0] in source_buckets
        and top_doc.metadata.get("source") == top_inferred[0][0]
        and (
            len(top_inferred) == 1
            or (top_inferred[0][1] - top_inferred[1][1]) >= 0.2
        )
    ):
        best_source = top_inferred[0][0]
    elif active_sop and active_sop in source_buckets and not explicit_source_shift:
        best_source = active_sop
    else:
        source_scores = {
            source: sum(score for score, _ in docs[:3])
            for source, docs in source_buckets.items()
        }
        best_source = max(source_scores, key=source_scores.get)

    selected_docs = _select_docs_for_best_source(
        source_buckets[best_source],
        query,
        limit=get_k_for_question(query),
    )
    if not selected_docs:
        return [], None
    selected_docs = _expand_requested_sections(
        vectorstore,
        selected_docs,
        best_source,
        query,
        preferred_sections,
    )
    return selected_docs, best_source
