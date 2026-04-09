import logging
import os
import stat
import shutil
import time
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from backend.config import settings
from backend.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)


def _remove_readonly(func, path, _exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _clear_persist_dir(persist_dir: str) -> None:
    path = Path(persist_dir)
    if not path.exists():
        return

    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            shutil.rmtree(path, onerror=_remove_readonly)
            logger.info("Cleared existing vectorstore at %s", persist_dir)
            return
        except PermissionError as exc:
            last_error = exc
            logger.warning(
                "Vectorstore directory is locked on attempt %d/5: %s",
                attempt,
                persist_dir,
            )
            time.sleep(attempt)

    raise RuntimeError(
        f"Could not clear {persist_dir}. Stop any running backend or rebuild process using Chroma and try again."
    ) from last_error


def create_vectorstore(chunks: list[Document]) -> Chroma:
    persist_dir = settings.chroma_db_dir

    # Remove existing DB to prevent duplicates on full rebuild
    _clear_persist_dir(persist_dir)

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    logger.info("Created vectorstore with %d chunks", len(chunks))
    return vectorstore


def load_existing_vectorstore() -> Chroma | None:
    persist_dir = settings.chroma_db_dir
    if not Path(persist_dir).exists():
        return None

    embeddings = get_embeddings()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )


def add_documents_to_vectorstore(chunks: list[Document]) -> Chroma:
    vectorstore = load_existing_vectorstore()
    if vectorstore is None:
        return create_vectorstore(chunks)
    vectorstore.add_documents(chunks)
    logger.info("Added %d chunks to existing vectorstore", len(chunks))
    return vectorstore
