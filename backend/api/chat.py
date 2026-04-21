import json
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.core.feedback import (
    clear_failed_query,
    delete_conversation,
    get_analytics_summary,
    get_conversation,
    get_conversations,
    get_failed_queries,
    get_feedback,
    save_conversation,
    save_feedback,
    search_conversations,
)
from backend.core.llm import (
    get_fallback_provider,
    get_llm,
    get_provider_status,
    is_provider_configured,
    mark_provider_error,
    mark_provider_healthy,
)
from backend.core.rag_chain import RAGChain

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

ANSWER_MODES = ("brief", "detailed", "checklist", "step-by-step", "only-responsibilities", "only-objective")


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    active_sop: str | None = None
    stream: bool = True
    llm_provider: Literal["gemini", "groq"] | None = None
    answer_mode: str = "detailed"
    source_locked: bool = False
    cursor_offset: int = 0
    page_limit: int = 15


class ChatResponse(BaseModel):
    answer: str
    sources: dict | None = None
    followup: str | None = None
    active_sop: str | None = None
    image: str | None = None
    confidence: str | None = None
    suggestions: list[str] | None = None
    has_more: bool = False
    next_offset: int | None = None


class CompareRequest(BaseModel):
    question: str
    sop_a: str
    sop_b: str


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: Literal["up", "down"]
    active_sop: str | None = None
    comment: str = ""


class SaveConversationRequest(BaseModel):
    conversation_id: str
    messages: list[dict]
    title: str = ""


def select_rag_chain(req: Request, llm_provider: str | None):
    if not llm_provider:
        rag_chain = getattr(req.app.state, "rag_chain", None)
        if rag_chain is None:
            raise HTTPException(status_code=503, detail="RAG runtime is not ready.")
        return rag_chain, None

    provider = llm_provider.lower()
    rag_chains = dict(getattr(req.app.state, "rag_chains", {}))
    if provider in rag_chains:
        return rag_chains[provider], None

    if not is_provider_configured(provider):
        raise HTTPException(
            status_code=400,
            detail=f"{provider.title()} is not configured on the server.",
        )

    vectorstore = getattr(req.app.state, "vectorstore", None)
    base_chain = getattr(req.app.state, "rag_chain", None)
    if vectorstore is None and base_chain is not None:
        vectorstore = base_chain.vectorstore
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Vectorstore is not ready.")

    try:
        rag_chain = RAGChain(
            llm=get_llm(provider),
            vectorstore=vectorstore,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    rag_chains[provider] = rag_chain
    req.app.state.rag_chains = rag_chains
    return rag_chain, None


def _get_fallback_chain(req: Request, primary_provider: str):
    """Try to get a fallback RAG chain if primary provider fails."""
    fallback = get_fallback_provider(primary_provider or "gemini")
    if not fallback:
        return None, None

    vectorstore = getattr(req.app.state, "vectorstore", None)
    if vectorstore is None:
        base_chain = getattr(req.app.state, "rag_chain", None)
        if base_chain is not None:
            vectorstore = getattr(base_chain, "vectorstore", None)
    if vectorstore is None:
        return None, None

    rag_chains = dict(getattr(req.app.state, "rag_chains", {}))
    if fallback in rag_chains:
        return rag_chains[fallback], fallback

    try:
        chain = RAGChain(llm=get_llm(fallback), vectorstore=vectorstore)
        rag_chains[fallback] = chain
        req.app.state.rag_chains = rag_chains
        return chain, fallback
    except Exception:
        return None, None


@router.post("/chat")
async def chat(request: ChatRequest, req: Request):
    rag_chain, _ = select_rag_chain(req, request.llm_provider)
    provider_name = request.llm_provider or "gemini"
    answer_mode = request.answer_mode if request.answer_mode in ANSWER_MODES else "detailed"

    if request.stream:
        async def event_stream():
            nonlocal rag_chain, provider_name
            used_fallback = False
            try:
                async for event in rag_chain.stream_query(
                    request.message, request.history, request.active_sop,
                    answer_mode=answer_mode,
                    source_locked=request.source_locked,
                    llm_provider=provider_name,
                    cursor_offset=request.cursor_offset,
                    page_limit=request.page_limit,
                ):
                    yield f"data: {json.dumps(event)}\n\n"
                mark_provider_healthy(provider_name)
            except Exception as e:
                logger.exception("Stream error with %s", provider_name)
                mark_provider_error(provider_name, str(e))
                # Try fallback
                try:
                    fallback_chain, fallback_name = _get_fallback_chain(req, provider_name)
                except Exception:
                    fallback_chain, fallback_name = None, None
                if fallback_chain and not used_fallback:
                    used_fallback = True
                    logger.info("Falling back to %s", fallback_name)
                    yield f"data: {json.dumps({'type': 'fallback', 'content': f'Switched to {fallback_name} (primary unavailable)'})}\n\n"
                    try:
                        async for event in fallback_chain.stream_query(
                            request.message, request.history, request.active_sop,
                            answer_mode=answer_mode,
                            source_locked=request.source_locked,
                            llm_provider=fallback_name,
                            cursor_offset=request.cursor_offset,
                            page_limit=request.page_limit,
                        ):
                            yield f"data: {json.dumps(event)}\n\n"
                        mark_provider_healthy(fallback_name)
                    except Exception as e2:
                        logger.exception("Fallback stream error")
                        yield f"data: {json.dumps({'type': 'error', 'content': str(e2)})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        result = await rag_chain.query(
            request.message, request.history, request.active_sop,
            answer_mode=answer_mode,
            source_locked=request.source_locked,
            llm_provider=provider_name,
            cursor_offset=request.cursor_offset,
            page_limit=request.page_limit,
        )
        mark_provider_healthy(provider_name)
        return ChatResponse(**result)
    except Exception as e:
        logger.exception("Query error with %s", provider_name)
        mark_provider_error(provider_name, str(e))
        fallback_chain, fallback_name = _get_fallback_chain(req, provider_name)
        if fallback_chain:
            try:
                result = await fallback_chain.query(
                    request.message, request.history, request.active_sop,
                    answer_mode=answer_mode,
                    source_locked=request.source_locked,
                    llm_provider=fallback_name,
                    cursor_offset=request.cursor_offset,
                    page_limit=request.page_limit,
                )
                mark_provider_healthy(fallback_name)
                return ChatResponse(**result)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=str(e))


# ── Compare mode ────────────────────────────────────────────────────────

@router.post("/compare")
async def compare_sops(request: CompareRequest, req: Request):
    rag_chain = getattr(req.app.state, "rag_chain", None)
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG runtime is not ready.")
    result = await rag_chain.compare_sops(request.question, request.sop_a, request.sop_b)
    return result


# ── SOP list ────────────────────────────────────────────────────────────

@router.get("/sops")
async def list_sops(req: Request):
    rag_chain = getattr(req.app.state, "rag_chain", None)
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG runtime is not ready.")
    return rag_chain.get_all_sop_titles()


# ── Feedback ────────────────────────────────────────────────────────────

@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    record = save_feedback(
        question=request.question,
        answer=request.answer,
        rating=request.rating,
        active_sop=request.active_sop,
        comment=request.comment,
    )
    return {"success": True, "record": record}


# ── Conversations ───────────────────────────────────────────────────────

@router.post("/conversations")
async def save_conv(request: SaveConversationRequest):
    result = save_conversation(request.conversation_id, request.messages, request.title)
    return result


@router.get("/conversations")
async def list_convs(limit: int = 50):
    return get_conversations(limit)


@router.get("/conversations/search")
async def search_convs(q: str, limit: int = 20):
    return search_conversations(q, limit)


@router.get("/conversations/{conversation_id}")
async def get_conv(conversation_id: str):
    conv = get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.delete("/conversations/{conversation_id}")
async def delete_conv(conversation_id: str):
    if delete_conversation(conversation_id):
        return {"success": True}
    raise HTTPException(status_code=404, detail="Conversation not found")


# ── Provider status ─────────────────────────────────────────────────────

@router.get("/status")
async def provider_status():
    return get_provider_status()
