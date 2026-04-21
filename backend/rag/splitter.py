from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.rag.preprocess import (
    canonical_section_title,
    clean_lines,
    normalize_text,
)

FOCUS_SECTION_ORDER = {
    "role": (
        "Job Objectives",
        "Responsibilities",
        "Reporting Authority",
        "Minimum Qualification",
        "Minimum Experience",
        "Training Requirement",
    ),
    "workflow": (
        "Objective",
        "Objectives",
        "Overview",
        "Process",
        "Procedure",
        "Workflow",
        "Responsibilities",
    ),
    "meeting": (
        "Objective",
        "Objectives",
        "Agenda",
        "Participants",
        "Procedure",
        "Workflow",
    ),
    "standards": (
        "Objective",
        "Objectives",
        "Overview",
        "Guidelines",
        "Best Practices",
        "Checklist",
    ),
    "document": (
        "Overview",
        "Objective",
        "Process",
        "Procedure",
        "Responsibilities",
    ),
}


def _page_sort_key(doc: Document) -> tuple[int, str]:
    page = doc.metadata.get("page")
    if isinstance(page, int):
        return (page, str(page))

    page_label = str(doc.metadata.get("page_label", ""))
    if page_label.isdigit():
        return (int(page_label) - 1, page_label)

    return (10**6, page_label)


def _page_range(pages: list[str]) -> str:
    if not pages:
        return "unknown"
    if len(pages) == 1:
        return pages[0]
    return f"{pages[0]}-{pages[-1]}"


def _parse_multi_value(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(" | ") if part.strip()]


def _search_tags(metadata: dict, section_title: str | None = None) -> str:
    tags: list[str] = []

    def add(value: str | None) -> None:
        if not value:
            return
        cleaned = " ".join(str(value).split())
        if cleaned and cleaned not in tags:
            tags.append(cleaned)

    add(metadata.get("source_title"))
    add(metadata.get("source_kind"))
    if section_title:
        add(section_title)

    for value in _parse_multi_value(metadata.get("source_aliases"))[:8]:
        add(value)
    for value in _parse_multi_value(metadata.get("source_intents"))[:8]:
        add(value)

    source_title = str(metadata.get("source_title", "")).strip()
    source_kind = str(metadata.get("source_kind", "")).strip().lower()
    if source_title and source_kind == "role":
        add(f"{source_title} role")
        add(f"{source_title} responsibilities")
        add(f"roles and responsibilities of {source_title}")
    elif source_title and source_kind in {"workflow", "meeting"}:
        add(f"{source_title} SOP")
        add(f"{source_title} procedure")

    return " | ".join(tags[:20])


def _profile_chunk(source: str, docs: list[Document]) -> Document:
    first = docs[0]
    section_titles: list[str] = []
    for doc in docs:
        for title in _parse_multi_value(doc.metadata.get("source_section_titles")):
            if title not in section_titles:
                section_titles.append(title)

    profile_text = "\n".join(
        [
            "DOCUMENT PROFILE",
            f"SOURCE TITLE: {first.metadata.get('source_title', source)}",
            f"SOURCE FILE: {source}",
            f"SOURCE KIND: {first.metadata.get('source_kind', 'document')}",
            f"ALIASES: {first.metadata.get('source_aliases', '')}",
            f"SEARCH TAGS: {_search_tags(first.metadata)}",
            f"KEY SECTIONS: {' | '.join(section_titles[:12])}",
            f"SUMMARY: {first.metadata.get('source_summary', '')}",
        ]
    ).strip()

    metadata = dict(first.metadata)
    metadata.update(
        {
            "content_type": "profile",
            "section_title": "Document Profile",
            "page_label": metadata.get("page_label", "1"),
            "chunk_id": f"{source}::profile",
        }
    )

    return Document(page_content=profile_text, metadata=metadata)


def _outline_sections(source_docs: list[Document]) -> list[dict]:
    sections: list[dict] = []
    source_title = normalize_text(str(source_docs[0].metadata.get("source_title", "")))
    current_title = "Overview"
    current_lines: list[str] = []
    current_pages: list[str] = []
    current_meta = dict(source_docs[0].metadata)

    def flush() -> None:
        nonlocal current_lines, current_pages, current_meta, current_title
        content = "\n".join(current_lines).strip()
        if not content:
            return
        sections.append(
            {
                "title": current_title,
                "content": content,
                "pages": list(current_pages),
                "metadata": dict(current_meta),
            }
        )

    for doc in source_docs:
        page_label = str(
            doc.metadata.get(
                "page_label",
                doc.metadata.get("page", "unknown"),
            )
        )
        lines = clean_lines(doc.page_content)
        if not lines:
            continue

        for line in lines:
            line_key = normalize_text(line)
            if source_title and line_key == source_title:
                continue

            heading = canonical_section_title(line)
            if heading and normalize_text(heading) != source_title:
                if current_lines and normalize_text(current_title) != normalize_text(heading):
                    flush()
                    current_lines = []
                    current_pages = []
                current_title = heading
                current_meta = dict(doc.metadata)
                if page_label not in current_pages:
                    current_pages.append(page_label)
                continue

            if page_label not in current_pages:
                current_pages.append(page_label)
            current_lines.append(line)

    flush()
    return sections


def _focus_sections(source_kind: str, sections: list[dict]) -> list[dict]:
    preferred = FOCUS_SECTION_ORDER.get(source_kind, FOCUS_SECTION_ORDER["document"])
    normalized_preferred = [normalize_text(title) for title in preferred]
    selected: list[dict] = []

    for preferred_title in normalized_preferred:
        for section in sections:
            if normalize_text(section["title"]) == preferred_title and section not in selected:
                selected.append(section)

    if selected:
        return selected
    return sections[:3]


def _focus_chunk(source: str, docs: list[Document], sections: list[dict]) -> Document | None:
    first = docs[0]
    source_kind = str(first.metadata.get("source_kind", "document"))
    selected_sections = _focus_sections(source_kind, sections)
    if not selected_sections:
        return None

    blocks = []
    for section in selected_sections:
        page_range = _page_range(section["pages"])
        blocks.append(
            "\n".join(
                [
                    f"[{section['title']} | Pages {page_range}]",
                    section["content"],
                ]
            )
        )

    focus_text = "\n\n".join(
        [
            "DOCUMENT GUIDE",
            f"SOURCE TITLE: {first.metadata.get('source_title', source)}",
            f"SOURCE FILE: {source}",
            f"SOURCE KIND: {source_kind}",
            f"SEARCH TAGS: {_search_tags(first.metadata)}",
            "",
            *blocks,
        ]
    ).strip()

    metadata = dict(first.metadata)
    metadata.update(
        {
            "content_type": "focus",
            "section_title": "Document Guide",
            "page_label": _page_range(selected_sections[0]["pages"]),
            "page_range": _page_range(
                sorted(
                    {
                        page
                        for section in selected_sections
                        for page in section["pages"]
                    }
                )
            ),
            "chunk_id": f"{source}::focus",
        }
    )

    return Document(page_content=focus_text, metadata=metadata)


def _section_chunk_text(metadata: dict, section_title: str, page_range: str, content: str) -> str:
    return "\n".join(
        [
            f"SOURCE TITLE: {metadata.get('source_title', metadata.get('source'))}",
            f"SOURCE FILE: {metadata.get('source')}",
            f"SOURCE KIND: {metadata.get('source_kind', 'document')}",
            f"ALIASES: {metadata.get('source_aliases', '')}",
            f"SEARCH TAGS: {_search_tags(metadata, section_title)}",
            f"PAGE RANGE: {page_range}",
            f"SECTION: {section_title}",
            "CONTENT:",
            content.strip(),
        ]
    ).strip()


def _split_synthetic_chunk(
    splitter: RecursiveCharacterTextSplitter,
    chunk: Document,
    *,
    prefix: str,
) -> list[Document]:
    pieces = splitter.split_text(chunk.page_content)
    results: list[Document] = []
    for index, piece in enumerate(pieces, start=1):
        metadata = dict(chunk.metadata)
        metadata["chunk_id"] = f"{prefix}::c{index}"
        results.append(Document(page_content=piece, metadata=metadata))
    return results


def split_docs(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", "; ", " "],
    )

    grouped: dict[str, list[Document]] = defaultdict(list)
    for doc in docs:
        grouped[doc.metadata.get("source", "unknown")].append(doc)

    chunks: list[Document] = []

    for source, source_docs in grouped.items():
        source_docs = sorted(source_docs, key=_page_sort_key)
        first = source_docs[0]

        if first.metadata.get("type") == "image":
            image_meta = dict(first.metadata)
            image_meta.update(
                {
                    "content_type": "image",
                    "section_title": "Workflow",
                    "chunk_id": f"{source}::image::{first.metadata.get('filename', 'unknown')}",
                }
            )
            chunks.append(Document(page_content=first.page_content, metadata=image_meta))
            continue

        chunks.append(_profile_chunk(source, source_docs))

        outline = _outline_sections(source_docs)
        focus_chunk = _focus_chunk(source, source_docs, outline)
        if focus_chunk is not None:
            chunks.extend(
                _split_synthetic_chunk(
                    splitter,
                    focus_chunk,
                    prefix=f"{source}::focus",
                )
            )

        if not outline:
            outline = [
                {
                    "title": "Overview",
                    "content": "\n".join(doc.page_content for doc in source_docs),
                    "pages": [
                        str(doc.metadata.get("page_label", doc.metadata.get("page", "unknown")))
                        for doc in source_docs
                    ],
                    "metadata": dict(first.metadata),
                }
            ]

        for section_index, section in enumerate(outline, start=1):
            page_range = _page_range(section["pages"])
            if normalize_text(section["title"]).lower().startswith("extracted table"):
                section_pieces = [section["content"]]
            else:
                section_pieces = splitter.split_text(section["content"])
            if not section_pieces:
                continue

            for piece_index, piece in enumerate(section_pieces, start=1):
                metadata = dict(section["metadata"])
                metadata.update(
                    {
                        "content_type": "section",
                        "section_title": section["title"],
                        "section_index": section_index,
                        "chunk_in_section": piece_index,
                        "page_label": page_range,
                        "page_range": page_range,
                        "chunk_id": f"{source}::s{section_index}::c{piece_index}",
                    }
                )
                chunks.append(
                    Document(
                        page_content=_section_chunk_text(
                            metadata,
                            section["title"],
                            page_range,
                            piece,
                        ),
                        metadata=metadata,
                    )
                )

    return chunks
