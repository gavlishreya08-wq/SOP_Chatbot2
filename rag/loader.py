import json
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


#  Load SOP Metadata JSON
try:
    with open("sop_metadata.json", "r") as f:
        SOP_META = json.load(f)
except Exception:
    SOP_META = {}


def match_metadata(file_name: str):
    """
    Fuzzy match filename with metadata JSON keys
    Handles prefixes like 1_, 6_, spaces, dashes etc.
    """

    f = file_name.lower().replace(" ", "").replace("_", "").replace("-", "").replace(".", "")

    for key, value in SOP_META.items():

        k = key.lower().replace(" ", "").replace("_", "").replace("-", "").replace(".", "")

        if k in f or f in k:
            return value

    return {}


def load_pdfs(directories: List[str] = None) -> List:

    if directories is None:
        directories = ["sop_documents", "img_txt"]

    docs = []

    # ─────────────── PDF + TXT ───────────────
    for folder in directories:

        directory = Path(folder)

        if not directory.exists():
            continue

        #  Load PDFs
        for pdf in directory.rglob("*.pdf"):

            loader = PyPDFLoader(str(pdf))
            pdf_docs = loader.load()

            file_name = pdf.name

            #  Fuzzy metadata match
            meta = match_metadata(file_name)

            for d in pdf_docs:
                d.metadata["type"] = "text"
                d.metadata["source"] = file_name
                d.metadata["pdf_link"] = meta.get("link", "")
                d.metadata["version"] = meta.get("version", "NA")
                d.metadata["created_date"] = meta.get("created_date", "NA")

            docs.extend(pdf_docs)

        #  Load TXT
        for txt in directory.rglob("*.txt"):

            loader = TextLoader(str(txt), encoding="utf-8")
            txt_docs = loader.load()

            for d in txt_docs:
                d.metadata["type"] = "text"
                d.metadata["source"] = txt.name

            docs.extend(txt_docs)

    # ─────────────── FLOWCHART IMAGES ───────────────
    flowchart_folder = Path("flowcharts")

    if flowchart_folder.exists():
        for img in flowchart_folder.glob("*.*"):

            if img.suffix.lower() in [".png", ".jpg", ".jpeg"]:

                clean_name = img.stem.replace("_", " ").replace("-", " ")

                description = f"""
This is the official FLOWCHART DIAGRAM for the {clean_name} process.

This image represents:
- workflow of {clean_name}
- hierarchy of {clean_name}
- approval flow of {clean_name}
- process structure of {clean_name}
- complete diagram of {clean_name}

Return this image when the user asks to:
show the flowchart,
display the diagram,
provide hierarchy,
show process flow,
display structure of {clean_name}.
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