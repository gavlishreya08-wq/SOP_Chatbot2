from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from pathlib import Path
from typing import List


def load_pdfs(directories: List[str] = None) -> List:
    """
    Load documents from multiple folders:
    - PDFs
    - TXT files
    - Flowchart Images
    """

    if directories is None:
        directories = ["sop_documents", "img_txt"]

    docs = []

    # ─────────────────────────────────────────────
    # Load PDFs and TXT Files
    # ─────────────────────────────────────────────
    for folder in directories:
        directory = Path(folder)

        if not directory.exists():
            continue

        # Load PDFs
        for pdf in directory.rglob("*.pdf"):
            loader = PyPDFLoader(str(pdf))
            pdf_docs = loader.load()

            for d in pdf_docs:
                d.metadata["type"] = "text"
                d.metadata["source"] = str(pdf)

            docs.extend(pdf_docs)

        # Load TXT files
        for txt in directory.rglob("*.txt"):
            loader = TextLoader(str(txt), encoding="utf-8")
            txt_docs = loader.load()

            for d in txt_docs:
                d.metadata["type"] = "text"
                d.metadata["source"] = str(txt)

            docs.extend(txt_docs)

    # ─────────────────────────────────────────────
    # Load Flowchart Images (STRONG SEMANTIC VERSION)
    # ─────────────────────────────────────────────
    flowchart_folder = Path("flowcharts")

    if flowchart_folder.exists():
        for img in flowchart_folder.glob("*.*"):
            if img.suffix.lower() in [".png", ".jpg", ".jpeg"]:

                clean_name = img.stem.replace("_", " ").replace("-", " ")

                description = f"""
This is the official FLOWCHART DIAGRAM for the {clean_name} process.

This image represents:
- the workflow of {clean_name}
- the hierarchy of {clean_name}
- the approval flow of {clean_name}
- the process structure of {clean_name}
- the complete diagram of {clean_name}

Return this image when the user asks to:
show the flowchart,
display the diagram,
provide the hierarchy,
show the process flow,
or display the structure of {clean_name}.
"""

                doc = Document(
                    page_content=description.strip(),
                    metadata={
                        "type": "image",
                        "path": str(img),
                        "source": "flowchart",
                        "filename": img.name
                    }
                )

                docs.append(doc)

    return docs