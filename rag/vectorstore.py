from langchain_community.vectorstores import Chroma
from rag.embeddings import get_embeddings
import os

PERSIST_DIRECTORY = "chroma_db"

def create_vectorstore(chunks):
    embeddings = get_embeddings()

    # Create or load persistent Chroma DB
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    # Add new documents
    vectorstore.add_documents(chunks)

    # Persist to disk
    vectorstore.persist()

    return vectorstore


def load_existing_vectorstore():
    embeddings = get_embeddings()

    if os.path.exists(PERSIST_DIRECTORY):
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    return None