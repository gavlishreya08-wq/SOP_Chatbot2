from langchain_huggingface import HuggingFaceEmbeddings

_instance = None
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    global _instance
    if _instance is None:
        _instance = HuggingFaceEmbeddings(
            model_name=DEFAULT_EMBEDDING_MODEL,
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )
    return _instance
