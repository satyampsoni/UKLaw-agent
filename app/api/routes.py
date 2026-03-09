"""
FastAPI route handlers for UK LawAssistant.

Each endpoint follows the same pattern:
    1. Validate request (Pydantic does this automatically)
    2. Call the appropriate pipeline method
    3. Transform internal types into response schemas
    4. Return JSON

Why separate routes from the app factory (main.py)?
    - Routes define WHAT the API does
    - main.py defines HOW the app starts (lifespan, middleware, CORS)
    - This separation makes testing easier — you can test routes
      without starting the full server

Error handling:
    - Pydantic validation errors → 422 (automatic)
    - Pipeline errors → 500 with a message
    - Not found → 404
    - All errors return JSON, never HTML
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import (
    AskRequest,
    AskResponse,
    LegislationSource,
    ExplainRequest,
    ExplainResponse,
    SearchRequest,
    SearchResponse,
    SearchSectionResult,
    HealthResponse,
    StatsResponse,
)

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# Router
# ────────────────────────────────────────────────────────────────────

router = APIRouter()


# ────────────────────────────────────────────────────────────────────
# Health check
# ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request):
    """
    Health check endpoint.

    Returns database status, document count, and node count.
    Use this for monitoring, load balancers, and uptime checks.
    """
    db = request.app.state.db
    try:
        stats = await db.get_stats()
        return HealthResponse(
            status="ok",
            database="connected",
            documents=stats.get("documents", 0),
            nodes=stats.get("total_nodes", 0),
        )
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return HealthResponse(
            status="error",
            database=f"error: {e}",
            documents=0,
            nodes=0,
        )


# ────────────────────────────────────────────────────────────────────
# Ask — main RAG endpoint
# ────────────────────────────────────────────────────────────────────

@router.post("/api/ask", response_model=AskResponse, tags=["Legal Q&A"])
async def ask_question(body: AskRequest, request: Request):
    """
    Ask a legal question and get a grounded answer.

    The system searches UK legislation, retrieves relevant sections,
    and uses an LLM to generate an answer citing specific provisions.

    - **question**: Your legal question (3–1000 chars)
    - **document_id**: Optional — restrict to one Act (e.g. "dpa-2018")
    - **strict_mode**: If true, only quotes statutory text
    """
    rag = request.app.state.rag

    # Override strict mode if requested
    original_strict = rag._strict_mode
    if body.strict_mode:
        rag._strict_mode = True

    try:
        response = await rag.ask(
            question=body.question,
            document_id=body.document_id,
        )
    except Exception as e:
        logger.error("RAG pipeline error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
    finally:
        rag._strict_mode = original_strict

    return AskResponse(
        answer=response.answer,
        sources=[
            LegislationSource(**src)
            for src in response.sources_summary
        ],
        has_sources=response.has_sources,
        acts_consulted=sorted(response.acts_consulted),
        search_latency_ms=round(response.search_latency_ms, 1),
        llm_latency_ms=round(response.llm_response.latency_ms, 1),
        total_latency_ms=round(response.total_latency_ms, 1),
        tokens_used=response.tokens_used,
    )


# ────────────────────────────────────────────────────────────────────
# Explain — plain English section explanation
# ────────────────────────────────────────────────────────────────────

@router.post("/api/explain", response_model=ExplainResponse, tags=["Legal Q&A"])
async def explain_section(body: ExplainRequest, request: Request):
    """
    Explain a specific section of UK legislation in plain English.

    Provide the section's node ID (e.g. "dpa-2018:section-47") and
    get a clear, structured explanation covering:
    - What it requires/permits
    - Who it applies to
    - Conditions and exceptions
    - Practical implications
    """
    rag = request.app.state.rag

    try:
        response = await rag.explain_section(body.section_id)
    except Exception as e:
        logger.error("Explain error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explain error: {e}")

    if "not found" in response.answer.lower():
        raise HTTPException(
            status_code=404,
            detail=f"Section '{body.section_id}' not found in the database.",
        )

    return ExplainResponse(
        explanation=response.answer,
        section_id=body.section_id,
        tokens_used=response.tokens_used,
        latency_ms=round(response.total_latency_ms, 1),
    )


# ────────────────────────────────────────────────────────────────────
# Search — legislation search without LLM
# ────────────────────────────────────────────────────────────────────

@router.post("/api/search", response_model=SearchResponse, tags=["Search"])
async def search_legislation(body: SearchRequest, request: Request):
    """
    Search UK legislation without calling the LLM.

    Returns matching sections with full text, ranked by BM25 relevance.
    Useful for browsing legislation or building custom prompts.
    """
    search_engine = request.app.state.search

    try:
        result = await search_engine.search(
            query=body.query,
            limit=body.limit,
            document_id=body.document_id,
        )
    except Exception as e:
        logger.error("Search error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

    sections = [
        SearchSectionResult(
            citation=ctx.citation,
            title=ctx.section_node.title,
            hierarchy=ctx.hierarchy_path,
            text=ctx.full_text or ctx.section_node.text or "",
            score=round(ctx.match_score, 2),
        )
        for ctx in result.sections
    ]

    return SearchResponse(
        query=body.query,
        sections=sections,
        total_fts_hits=result.total_fts_hits,
        acts_found=sorted(result.documents_covered),
    )


# ────────────────────────────────────────────────────────────────────
# Stats
# ────────────────────────────────────────────────────────────────────

@router.get("/api/stats", response_model=StatsResponse, tags=["System"])
async def get_stats(request: Request):
    """
    Get database statistics.

    Returns total documents, nodes, text size, and per-Act breakdown.
    """
    db = request.app.state.db

    try:
        stats = await db.get_stats()
    except Exception as e:
        logger.error("Stats error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Stats error: {e}")

    # Transform by_document dict into a list of dicts for the response
    by_document = stats.get("by_document", {})
    acts_list = [
        {"id": doc_id, "nodes": count}
        for doc_id, count in by_document.items()
    ]

    return StatsResponse(
        documents=stats.get("documents", 0),
        total_nodes=stats.get("total_nodes", 0),
        total_text_chars=stats.get("total_text_chars", 0),
        acts=acts_list,
    )
