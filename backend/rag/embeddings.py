from langchain_huggingface import HuggingFaceEmbeddings

_instance = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _instance
    if _instance is None:
        _instance = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _instance
