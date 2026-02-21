from langchain_community.document_loaders import PyPDFLoader, TextLoader
from pathlib import Path
from typing import List


def load_pdfs(directory: str = "sop_documents") -> List:
    """
    Load ALL SOP documents (PDF + TXT) from directory
    """

    docs = []
    directory = Path(directory)

    if not directory.exists():
        return docs

    # Load PDFs
    for pdf in directory.rglob("*.pdf"):
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())

    # Load TXT files (downloaded from auto-sync)
    for txt in directory.rglob("*.txt"):
        loader = TextLoader(str(txt), encoding="utf-8")
        docs.extend(loader.load())

    return docs
