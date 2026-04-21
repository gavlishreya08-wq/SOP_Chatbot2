import json
import logging
import io
from pathlib import Path
from typing import Optional
from contextlib import redirect_stdout

import fitz
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from backend.config import settings
from backend.rag.preprocess import build_source_profile, clean_lines, humanize_source_name

logger = logging.getLogger(__name__)


def _load_sop_metadata() -> dict:
    path = Path(settings.sop_metadata_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


SOP_META = _load_sop_metadata()


def match_metadata(file_name: str) -> dict:
    normalized = (
        file_name.lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
    )
    for key, value in SOP_META.items():
        candidate = (
            key.lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
            .replace(".", "")
        )
        if candidate in normalized or normalized in candidate:
            return value
    return {}


def load_pdfs(
    filepaths: Optional[list[str]] = None,
    directories: Optional[list[str]] = None,
) -> list[Document]:
    if directories is None:
        directories = [settings.sop_documents_dir, settings.img_txt_dir]

    docs: list[Document] = []

    if filepaths:
        for filepath in filepaths:
            path = Path(filepath)
            if not path.exists():
                logger.warning("File not found: %s", filepath)
                continue
            if path.suffix.lower() == ".pdf":
                docs.extend(_load_single_pdf(path))
            elif path.suffix.lower() == ".txt":
                docs.extend(_load_single_txt(path))
        return docs

    for folder in directories:
        directory = Path(folder)
        if not directory.exists():
            continue

        for pdf in directory.rglob("*.pdf"):
            docs.extend(_load_single_pdf(pdf))

        for txt in directory.rglob("*.txt"):
            docs.extend(_load_single_txt(txt))

    flowchart_dir = Path(settings.flowcharts_dir)
    if flowchart_dir.exists():
        for image in flowchart_dir.glob("*.*"):
            if image.suffix.lower() in (".png", ".jpg", ".jpeg"):
                docs.append(_create_flowchart_doc(image))

    return docs


def _base_source_metadata(path: Path) -> dict:
    meta = match_metadata(path.name)
    return {
        "source": path.name,
        "pdf_link": meta.get("link", ""),
        "version": meta.get("version", "NA"),
        "created_date": meta.get("created_date", "NA"),
    }


def _load_single_pdf(path: Path) -> list[Document]:
    try:
        page_docs = _load_pdf_with_pymupdf(path)
        if page_docs:
            return page_docs
        logger.warning("PyMuPDF returned no text for %s, falling back to PyPDFLoader", path)
        return _load_pdf_with_pypdf(path)
    except Exception:
        logger.exception("Failed to load PDF: %s", path)
        return []


def _load_pdf_with_pymupdf(path: Path) -> list[Document]:
    pdf = fitz.open(path)
    try:
        raw_pages = []
        for page in pdf:
            page_text = page.get_text("text", sort=True) or ""
            table_text = _extract_page_tables(page)
            raw_pages.append("\n\n".join(part for part in [page_text, table_text] if part.strip()))
        non_empty_pages = [page for page in raw_pages if page.strip()]
        if not non_empty_pages:
            return []

        source_profile = build_source_profile(path.name, "\n".join(raw_pages))
        base_meta = _base_source_metadata(path)
        docs: list[Document] = []

        for index, raw_text in enumerate(raw_pages):
            lines = clean_lines(raw_text)
            if not lines:
                continue

            docs.append(
                Document(
                    page_content="\n".join(lines),
                    metadata={
                        **base_meta,
                        "type": "text",
                        "page": index,
                        "page_label": str(index + 1),
                        "total_pages": len(raw_pages),
                        "source_title": source_profile["source_title"],
                        "source_kind": source_profile["source_kind"],
                        "source_aliases": " | ".join(source_profile["source_aliases"]),
                        "source_intents": " | ".join(source_profile["source_intents"]),
                        "source_section_titles": " | ".join(source_profile["section_titles"]),
                        "source_summary": source_profile["summary_text"],
                    },
                )
            )

        return docs
    finally:
        pdf.close()


def _clean_table_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ")
    return " ".join(text.split()).strip()


def _looks_like_table_header(row: list[str]) -> bool:
    normalized = " ".join(row).lower()
    if "activity" in normalized and ("start" in normalized or "end" in normalized or "duration" in normalized):
        return True
    filled = [cell for cell in row if cell]
    return len(filled) >= 2 and not filled[0].isdigit()


def _merge_extracted_table_rows(rows: list[list[str]]) -> tuple[list[str], list[list[str]]] | None:
    cleaned_rows = [[_clean_table_cell(cell) for cell in row] for row in rows]
    header_index = next(
        (index for index, row in enumerate(cleaned_rows) if _looks_like_table_header(row)),
        None,
    )
    if header_index is None:
        return None

    columns = cleaned_rows[header_index]
    if len([column for column in columns if column]) < 2:
        return None

    merged_rows: list[list[str]] = []
    for row in cleaned_rows[header_index + 1:]:
        if not any(row):
            continue
        row = [*row, *([""] * (len(columns) - len(row)))]
        row = row[: len(columns)]
        starts_new_row = bool(row[0])
        if starts_new_row or not merged_rows:
            merged_rows.append(row)
            continue

        target = merged_rows[-1]
        for index, value in enumerate(row):
            if not value:
                continue
            if target[index]:
                target[index] = f"{target[index]}; {value}"
            else:
                target[index] = value

    merged_rows = [row for row in merged_rows if any(cell for cell in row)]
    if not merged_rows:
        return None
    return columns, merged_rows


def _extract_page_tables(page) -> str:
    try:
        with redirect_stdout(io.StringIO()):
            found = page.find_tables()
    except Exception:
        return ""

    tables: list[str] = []
    for table in getattr(found, "tables", []):
        parsed = _merge_extracted_table_rows(table.extract())
        if not parsed:
            continue
        columns, rows = parsed
        lines = [
            "EXTRACTED TABLE:",
            " | ".join(columns),
            " | ".join("---" for _ in columns),
        ]
        lines.extend(" | ".join(row) for row in rows)
        tables.append("\n".join(lines))
    return "\n\n".join(tables)


def _load_pdf_with_pypdf(path: Path) -> list[Document]:
    loader = PyPDFLoader(str(path))
    pdf_docs = loader.load()
    if not pdf_docs:
        return []

    source_profile = build_source_profile(
        path.name,
        "\n".join(doc.page_content for doc in pdf_docs),
    )
    base_meta = _base_source_metadata(path)

    docs: list[Document] = []
    for doc in pdf_docs:
        lines = clean_lines(doc.page_content)
        if not lines:
            continue
        doc.metadata.update(
            {
                **base_meta,
                "type": "text",
                "source_title": source_profile["source_title"],
                "source_kind": source_profile["source_kind"],
                "source_aliases": " | ".join(source_profile["source_aliases"]),
                "source_intents": " | ".join(source_profile["source_intents"]),
                "source_section_titles": " | ".join(source_profile["section_titles"]),
                "source_summary": source_profile["summary_text"],
            }
        )
        doc.page_content = "\n".join(lines)
        docs.append(doc)

    return docs


def _load_single_txt(path: Path) -> list[Document]:
    try:
        raw_text = path.read_text(encoding="utf-8")
        lines = clean_lines(raw_text)
        if not lines:
            return []

        source_profile = build_source_profile(path.name, raw_text)
        return [
            Document(
                page_content="\n".join(lines),
                metadata={
                    "type": "text",
                    "source": path.name,
                    "page": 0,
                    "page_label": "1",
                    "total_pages": 1,
                    "source_title": source_profile["source_title"],
                    "source_kind": source_profile["source_kind"],
                    "source_aliases": " | ".join(source_profile["source_aliases"]),
                    "source_intents": " | ".join(source_profile["source_intents"]),
                    "source_section_titles": " | ".join(source_profile["section_titles"]),
                    "source_summary": source_profile["summary_text"],
                },
            )
        ]
    except Exception:
        logger.exception("Failed to load TXT: %s", path)
        return []


def _create_flowchart_doc(image: Path) -> Document:
    clean_name = humanize_source_name(image.name)
    description = (
        f"Official flowchart diagram for the {clean_name} process. "
        f"Shows workflow, hierarchy, approval flow, and process structure."
    )
    return Document(
        page_content=description,
        metadata={
            "type": "image",
            "path": str(image),
            "source": "flowchart",
            "filename": image.name,
            "page": "unknown",
            "page_label": "unknown",
            "source_title": clean_name,
            "source_kind": "flowchart",
            "source_aliases": clean_name,
            "source_intents": f"{clean_name} workflow | {clean_name} flowchart | {clean_name} process",
            "source_section_titles": "Workflow",
            "source_summary": description,
        },
    )
