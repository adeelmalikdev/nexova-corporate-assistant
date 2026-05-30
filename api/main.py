from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes.auth import router as auth_router
from api.routes.employees import router as employees_router
from api.routes.ingest import router as ingest_router
from api.routes.query import router as query_router
from agents.orchestrator import Orchestrator
from database.db import init_db
from rag.ingestor import DocumentIngestor
from rag.embedder import Embedder
from rag.query_processor import QueryProcessor
from rag.reranker import Reranker
from rag.retriever import HybridRetriever
from vectordb.chroma_client import ChromaClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    await init_db()
    app.state.chroma_client = ChromaClient()
    app.state.embedder = Embedder()
    app.state.reranker = Reranker()
    app.state.hybrid_retriever = HybridRetriever()
    app.state.document_ingestor = DocumentIngestor()
    app.state.query_processor = QueryProcessor()
    app.state.orchestrator = Orchestrator()
    logger.info("Vector store, retriever, and agent singletons initialized")
    logger.info("✓ Nexova Corporate Assistant API ready | timestamp=%s", datetime.utcnow().isoformat())
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Nexova Corporate Assistant API",
    version="1.0.0",
    lifespan=lifespan
)

# TODO: restrict origins, methods, and headers in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth")
app.include_router(employees_router, prefix="/employees")
app.include_router(query_router, prefix="/query")
app.include_router(ingest_router, prefix="/ingest")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = str(uuid4())
    logger.exception("Unhandled exception | request_id=%s | path=%s", request_id, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "request_id": request_id},
    )


@app.get("/health", tags=["Health"])
async def health():
    chroma_client = getattr(app.state, "chroma_client", None)
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "chroma_status": "ok" if chroma_client and chroma_client.ping() else "unhealthy",
        "models_loaded": Embedder.is_loaded() and Reranker.is_loaded(),
    }