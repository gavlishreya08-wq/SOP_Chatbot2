import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.admin import router as admin_router
from backend.api.chat import router as chat_router
from backend.config import PROJECT_ROOT, settings
from backend.core.llm import get_available_llm_options, get_llm, get_provider_model, get_provider_status
from backend.core.rag_chain import RAGChain
from backend.rag.loader import load_pdfs
from backend.rag.splitter import split_docs
from backend.rag.vectorstore import create_vectorstore, load_existing_vectorstore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: build RAG chain
    logger.info("Initializing RAG pipeline...")
    logger.info("LLM provider: %s", settings.llm_provider)

    vectorstore = load_existing_vectorstore()
    if vectorstore is None:
        logger.info("No existing vectorstore found, building from scratch...")
        docs = load_pdfs()
        chunks = split_docs(docs)
        vectorstore = create_vectorstore(chunks)
        logger.info("Vectorstore built with %d chunks", len(chunks))
    else:
        logger.info("Loaded existing vectorstore")

    default_provider = settings.llm_provider.lower()
    llm = get_llm(default_provider)
    default_chain = RAGChain(llm=llm, vectorstore=vectorstore)
    app.state.vectorstore = vectorstore
    app.state.rag_chain = default_chain
    app.state.rag_chains = {default_provider: default_chain}
    logger.info("RAG chain ready")

    yield

    # Shutdown
    logger.info("Shutting down")


app = FastAPI(
    title="Prakriya AI - SOP Chatbot",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(chat_router)
app.include_router(admin_router)


# Serve static files (flowcharts, etc.)
flowcharts_path = Path(settings.flowcharts_dir)
if flowcharts_path.exists():
    app.mount(
        "/static/flowcharts",
        StaticFiles(directory=str(flowcharts_path)),
        name="flowcharts",
    )


# Health check
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "model": get_provider_model(settings.llm_provider),
        "available_models": get_available_llm_options(),
        "provider_status": get_provider_status(),
    }


# Serve React build in production
frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
