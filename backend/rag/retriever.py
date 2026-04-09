from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
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

GENERIC_QUERY_TERMS = {
    "job",
    "meeting",
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


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("&", " and ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in normalize_text(text).split()
        if len(token) > 1 and token not in STOPWORDS
    }


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


def humanize_source_name(source: str) -> str:
    stem = Path(source).stem
    stem = stem.replace("&", " and ")
    stem = re.sub(r"([a-z])([A-Z])", r"\1 \2", stem)
    stem = re.sub(r"[_\-.()/]+", " ", stem)
    stem = re.sub(r"\b(ver|version|v)\s*\d+(\.\d+)?\b", " ", stem, flags=re.I)
    stem = re.sub(r"\b\d{8,}\b", " ", stem)
    return re.sub(r"\s+", " ", stem).strip()


def keyword_overlap_score(query_terms: set[str], target_terms: set[str]) -> float:
    if not query_terms or not target_terms:
        return 0.0
    overlap = _terms_overlap(query_terms, target_terms)
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
    q_terms = tokenize(question)
    if len(q_terms) >= 6 or any(word in q_terms for word in BROAD_KEYWORDS):
        return 6
    return 4


def get_candidate_pool_size(question: str) -> int:
    q_terms = tokenize(question)
    if len(q_terms) >= 6 or any(word in q_terms for word in BROAD_KEYWORDS):
        return 60
    return 40


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
    if target_source == active_sop or target_score < 1.05:
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

    return len(target_overlap) > len(active_overlap)


def _page_text(doc: Document) -> str:
    return normalize_text(doc.page_content[:1600])


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
    semantic_score = 1.0 / (1.0 + max(distance, 0.0))
    content_score = keyword_overlap_score(query_terms, tokenize(doc.page_content))
    title_score = max(
        source_match_score(query, source),
        source_match_score(query, source_title),
    )

    exact_phrase_bonus = 0.0
    normalized_query = normalize_text(query)
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
    overlap = _terms_overlap(query_terms, doc_terms)
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

    pool_size = 80
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
        content_terms = tokenize(doc.page_content)
        title_terms = tokenize(doc.metadata.get("source", ""))
        doc_len = len(doc.page_content.split())
        content_score = _bm25_score(query_terms, content_terms, doc_len, avg_len)
        title_score = keyword_overlap_score(query_terms, title_terms)
        active_bonus = 0.3 if active_sop and doc.metadata.get("source") == active_sop else 0.0
        total = content_score + title_score * 0.6 + active_bonus
        scored.append((total, doc, distance))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [(doc, distance) for _, doc, distance in scored[:top_k] if _ > 0.3]


def generate_query_variants(query: str) -> list[str]:
    """Generate rewritten query variants for multi-query retrieval."""
    terms = tokenize(query)
    variants = [query]

    specific = [t for t in terms if t not in GENERIC_QUERY_TERMS and t not in BROAD_KEYWORDS]
    generic = [t for t in terms if t in GENERIC_QUERY_TERMS or t in BROAD_KEYWORDS]

    if specific and generic:
        variants.append(" ".join(specific))

    synonym_map = {
        "process": "workflow procedure",
        "workflow": "process procedure",
        "procedure": "process workflow steps",
        "responsibilities": "roles duties",
        "roles": "responsibilities duties",
        "objective": "purpose goal overview",
        "steps": "process workflow procedure",
        "checklist": "steps list guidelines",
        "approval": "review sign-off",
        "issue": "ticket bug defect",
        "lead": "manager head",
    }

    for term in specific[:2]:
        if term in synonym_map:
            synonym_query = query.lower().replace(term, synonym_map[term].split()[0])
            if synonym_query != query.lower():
                variants.append(synonym_query)

    return variants[:3]


def retrieve(
    vectorstore: Chroma,
    query: str,
    *,
    active_sop: str | None = None,
    source_catalog: dict[str, dict[str, str | set[str]]] | None = None,
) -> tuple[list[Document], str | None]:
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

    # Multi-query retrieval: search with original + rewritten variants
    results: list[tuple[Document, float]] = []
    query_variants = generate_query_variants(query)
    for variant in query_variants:
        results.extend(
            vectorstore.similarity_search_with_score(
                variant,
                k=candidate_pool_size if variant == query else candidate_pool_size // 2,
            )
        )

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
    top_content_score = keyword_overlap_score(query_terms, tokenize(top_doc.page_content))
    top_doc_terms = tokenize(top_doc.page_content) | tokenize(top_doc.metadata.get("source", ""))
    top_overlap = _terms_overlap(query_terms, top_doc_terms)
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
            return fallback_docs[:get_k_for_question(query)], fallback_source
        return [], None
    if primary_terms:
        required_primary_matches = 1 if len(primary_terms) == 1 else 2
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

    selected_docs = [doc for _, doc in source_buckets[best_source][: get_k_for_question(query)]]
    if not selected_docs:
        return [], None

    return selected_docs, best_source
