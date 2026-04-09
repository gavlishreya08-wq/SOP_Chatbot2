from __future__ import annotations

from collections import Counter

from backend.rag.loader import load_pdfs
from backend.rag.splitter import split_docs
from backend.rag.vectorstore import create_vectorstore


def main() -> int:
    docs = load_pdfs()
    chunks = split_docs(docs)
    create_vectorstore(chunks)

    source_counts = Counter(doc.metadata.get("source", "unknown") for doc in chunks)
    print(f"Loaded documents: {len(docs)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Indexed sources: {len(source_counts)}")
    print("Top sources by chunk count:")
    for source, count in source_counts.most_common(15):
        print(f"  {source}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
