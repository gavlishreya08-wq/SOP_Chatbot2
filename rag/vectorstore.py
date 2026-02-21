from langchain_community.vectorstores import FAISS
from rag.embeddings import get_embeddings
import os

INDEX_PATH = "faiss_index"

def create_vectorstore(chunks):
    embeddings = get_embeddings()

    # If index already exists, load it
    if os.path.exists(INDEX_PATH):
        vectorstore = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    # Always save after update
    vectorstore.save_local(INDEX_PATH)

    return vectorstore


def load_existing_vectorstore():
    embeddings = get_embeddings()

    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None