from __future__ import annotations

from types import SimpleNamespace

from langchain_core.documents import Document


def make_doc(
    content: str,
    source: str,
    *,
    page: int | str = 0,
    page_label: str | None = None,
    **metadata,
) -> Document:
    doc_metadata = {
        "source": source,
        "page": page,
        "type": "text",
        "chunk_id": metadata.pop("chunk_id", f"{source}-{page}-{abs(hash(content))}"),
    }
    if page_label is not None:
        doc_metadata["page_label"] = page_label
    doc_metadata.update(metadata)
    return Document(page_content=content, metadata=doc_metadata)


class FakeVectorStore:
    def __init__(
        self,
        *,
        global_results: list[tuple[Document, float]] | None = None,
        filtered_results: dict[str, list[tuple[Document, float]]] | None = None,
        metadatas: list[dict] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        self.global_results = global_results or []
        self.filtered_results = filtered_results or {}
        self.metadatas = metadatas or []
        self.documents = documents or []

    def get(self, include=None):
        return {"metadatas": self.metadatas, "documents": self.documents}

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
        where_document=None,
        **kwargs,
    ):
        if filter and "source" in filter:
            results = self.filtered_results.get(filter["source"], [])
        else:
            results = self.global_results
        return results[:k]


class FakeLLM:
    def __init__(
        self,
        *,
        responses: list[str] | None = None,
        stream_chunks: list[str] | None = None,
    ) -> None:
        self.responses = responses or []
        self.stream_chunks = stream_chunks or []
        self.calls: list[list] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        content = self.responses.pop(0) if self.responses else ""
        return SimpleNamespace(content=content)

    async def astream(self, messages):
        self.calls.append(messages)
        for chunk in self.stream_chunks:
            yield SimpleNamespace(content=chunk)
