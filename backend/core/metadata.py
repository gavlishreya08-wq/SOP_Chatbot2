from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from langchain_core.documents import Document


def _display_page(doc: Document) -> str | None:
    page_label = doc.metadata.get("page_label")
    if page_label:
        return str(page_label)

    page = doc.metadata.get("page")
    if isinstance(page, int):
        return str(page + 1)
    if isinstance(page, str) and page and page != "unknown":
        return page
    return None


def _display_title(filename: str) -> str:
    title = Path(filename).stem
    title = title.replace("&", " and ")
    title = title.replace("_", " ").replace("-", " ")
    title = title.replace("(", " ").replace(")", " ")
    title = " ".join(title.split())
    return title


def _display_section(doc: Document) -> str | None:
    section = str(doc.metadata.get("section_title", "")).strip()
    if not section:
        return None
    if section in {"Document Profile", "Document Guide"}:
        return None
    return section


def _sort_pages(pages: list[str]) -> list[str]:
    numeric = [page for page in pages if page.isdigit()]
    ranges = [page for page in pages if "-" in page and page.split("-", 1)[0].isdigit()]
    non_numeric = [page for page in pages if page not in numeric and page not in ranges]
    return (
        sorted(numeric, key=int)
        + sorted(ranges, key=lambda page: int(page.split("-", 1)[0]))
        + non_numeric
    )


def format_sources(docs: list[Document]) -> dict | None:
    if not docs:
        return None

    sources = [d.metadata.get("source") for d in docs if d.metadata.get("source")]
    if not sources:
        return None

    main_source = Counter(sources).most_common(1)[0][0]

    for d in docs:
        if d.metadata.get("source") == main_source:
            link = (d.metadata.get("link") or d.metadata.get("pdf_link") or "").strip()
            version = str(d.metadata.get("version", "NA"))
            created = str(d.metadata.get("created_date", "NA"))
            title = _display_title(main_source)
            pages = []
            citations = []
            seen_citations = set()
            for doc in docs:
                if doc.metadata.get("source") != main_source:
                    continue
                page = _display_page(doc)
                if page and page not in pages:
                    pages.append(page)
                citation_key = (page, _display_section(doc))
                if citation_key in seen_citations:
                    continue
                seen_citations.add(citation_key)
                if page or citation_key[1]:
                    citations.append(
                        {
                            "page": page,
                            "section": citation_key[1],
                        }
                    )
            pages = _sort_pages(pages)
            citations = sorted(
                citations,
                key=lambda item: (
                    int(item["page"].split("-", 1)[0])
                    if item["page"] and item["page"].split("-", 1)[0].isdigit()
                    else 10**6,
                    item["section"] or "",
                ),
            )

            parsed = urlparse(link)
            valid_link = parsed.scheme in ("http", "https") and bool(parsed.netloc)

            return {
                "title": title,
                "filename": main_source,
                "link": link if valid_link else None,
                "version": version,
                "created_date": created,
                "pages": pages,
                "citations": citations[:8],
            }

    return None
