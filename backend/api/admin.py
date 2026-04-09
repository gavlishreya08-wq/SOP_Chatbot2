import logging
import shutil
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from backend.config import settings
from backend.core.feedback import (
    clear_failed_query,
    get_analytics_summary,
    get_failed_queries,
    get_feedback,
)
from backend.core.sync import SOPSync
from backend.core.llm import get_llm
from backend.core.rag_chain import RAGChain
from backend.rag.loader import load_pdfs
from backend.rag.splitter import split_docs
from backend.rag.vectorstore import add_documents_to_vectorstore, create_vectorstore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])
security = HTTPBearer()


class LoginRequest(BaseModel):
    password: str


class LoginResponse(BaseModel):
    token: str


def create_token() -> str:
    payload = {
        "admin": True,
        "exp": datetime.now(timezone.utc) + timedelta(hours=24),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def verify_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(
            credentials.credentials, settings.jwt_secret, algorithms=["HS256"]
        )
        if not payload.get("admin"):
            raise HTTPException(status_code=403, detail="Not authorized")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def refresh_rag_runtime(req: Request, vectorstore) -> None:
    default_provider = settings.llm_provider.lower()
    default_chain = RAGChain(
        llm=get_llm(default_provider),
        vectorstore=vectorstore,
    )
    req.app.state.vectorstore = vectorstore
    req.app.state.rag_chain = default_chain
    req.app.state.rag_chains = {default_provider: default_chain}


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    if request.password != settings.admin_password:
        raise HTTPException(status_code=401, detail="Incorrect password")
    return LoginResponse(token=create_token())


@router.post("/sync", dependencies=[Depends(verify_admin)])
async def sync_sops(req: Request):
    syncer = SOPSync()
    result = syncer.sync()

    if result["changed_files"]:
        logger.info("Updating vectorstore with %d changed files", len(result["changed_files"]))
        docs = load_pdfs(filepaths=result["changed_files"])
        chunks = split_docs(docs)
        vectorstore = add_documents_to_vectorstore(chunks)
        refresh_rag_runtime(req, vectorstore)

    return {
        "success": True,
        "new": result["new"],
        "updated": result["updated"],
        "unchanged": result["unchanged"],
    }


@router.post("/rebuild", dependencies=[Depends(verify_admin)])
async def rebuild_index(req: Request):
    logger.info("Full vectorstore rebuild requested")
    docs = load_pdfs()
    chunks = split_docs(docs)
    vectorstore = create_vectorstore(chunks)
    refresh_rag_runtime(req, vectorstore)

    return {"success": True, "chunks": len(chunks)}


@router.get("/status", dependencies=[Depends(verify_admin)])
async def get_status():
    syncer = SOPSync()
    return syncer.get_status()


@router.get("/config", dependencies=[Depends(verify_admin)])
async def get_config():
    return {
        "llm_provider": settings.llm_provider,
        "gemini_model": settings.gemini_model,
        "groq_model": settings.groq_model,
    }


# ── Analytics ───────────────────────────────────────────────────────────

@router.get("/analytics", dependencies=[Depends(verify_admin)])
async def analytics():
    return get_analytics_summary()


@router.get("/feedback", dependencies=[Depends(verify_admin)])
async def feedback_list(limit: int = 200):
    return get_feedback(limit)


@router.get("/failed-queries", dependencies=[Depends(verify_admin)])
async def failed_queries(limit: int = 200):
    return get_failed_queries(limit)


@router.delete("/failed-queries/{index}", dependencies=[Depends(verify_admin)])
async def dismiss_failed_query(index: int):
    if clear_failed_query(index):
        return {"success": True}
    raise HTTPException(status_code=404, detail="Query not found")
