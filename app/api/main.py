"""
FastAPI application factory for UK LawAssistant.

This module is responsible for:
    1. Creating the FastAPI app instance
    2. Managing the application lifespan (startup/shutdown)
    3. Configuring CORS, middleware, and error handlers
    4. Mounting the API routes

Why a lifespan context manager instead of @app.on_event?
    - @app.on_event("startup") is deprecated in FastAPI
    - The lifespan pattern is cleaner — resources created in startup
      are guaranteed to be cleaned up in shutdown
    - It's also easier to test — you can pass a mock lifespan

Why store db/rag/search on app.state?
    - These are expensive to create (DB connections, HTTP clients)
    - Creating them per-request would be slow and wasteful
    - app.state is the standard FastAPI pattern for shared resources
    - Every route handler accesses them via request.app.state

Run with:
    uvicorn app.api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import router
from app.models.database import LawDatabase
from app.search.legal_search import LegalSearchEngine
from app.rag.pipeline import RAGPipeline
from app.llm.relax_client import RelaxAIClient

logger = logging.getLogger(__name__)

# Path to static files (HTML/CSS/JS)
STATIC_DIR = Path(__file__).parent.parent.parent / "static"


# ────────────────────────────────────────────────────────────────────
# Lifespan — startup and shutdown
# ────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.

    Startup:
        - Open database connection
        - Create search engine
        - Create RAG pipeline (with LLM client)

    Shutdown:
        - Close LLM HTTP client
        - Close database connection

    Everything is stored on app.state so route handlers can access it.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    logger.info("Starting UK LawAssistant API...")

    # 1. Database
    db = LawDatabase()
    await db.initialize()
    stats = await db.get_stats()
    logger.info(
        "Database ready: %d documents, %d nodes",
        stats.get("documents", 0),
        stats.get("total_nodes", 0),
    )

    # 2. Search engine
    search = LegalSearchEngine(db)

    # 3. LLM client
    llm = RelaxAIClient()
    await llm.__aenter__()

    # 4. RAG pipeline (reuses shared db, search, and llm — no duplication)
    rag = RAGPipeline.from_components(db=db, search=search, llm=llm)

    # Store on app.state for route handlers
    app.state.db = db
    app.state.search = search
    app.state.llm = llm
    app.state.rag = rag

    logger.info("UK LawAssistant API is ready!")

    yield  # ← App runs here

    # Shutdown
    logger.info("Shutting down UK LawAssistant API...")
    await llm.__aexit__(None, None, None)
    await db.close()
    logger.info("Shutdown complete.")


# ────────────────────────────────────────────────────────────────────
# App factory
# ────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns a fully configured app ready to be served by uvicorn.
    """
    app = FastAPI(
        title="UK LawAssistant API",
        description=(
            "Sovereign AI Legal Intelligence System for UK legislation. "
            "Ask questions about UK Acts of Parliament and get grounded, "
            "cited answers powered by Retrieval-Augmented Generation."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────
    # Allow frontend (React/Next.js) to call the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",   # React dev server
            "http://localhost:5173",   # Vite dev server
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────
    app.include_router(router)

    # ── Static files (CSS, JS) ────────────────────────────────────
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ── Serve index.html at root ──────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(STATIC_DIR / "index.html"))

    return app


# ── Module-level app instance (for uvicorn) ──────────────────────
app = create_app()
