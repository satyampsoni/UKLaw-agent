"""
API request/response schemas for UK LawAssistant.

These Pydantic models define the exact shape of every JSON request
and response. They serve three purposes:

    1. Validation — FastAPI automatically validates incoming JSON
       against these schemas, returning 422 errors with clear
       messages for malformed requests.

    2. Documentation — FastAPI generates OpenAPI/Swagger docs
       directly from these models. Every field description shows
       up in the interactive API docs.

    3. Serialisation — response models ensure we never accidentally
       leak internal fields (like raw database rows or LLM API keys).

Naming convention:
    - *Request  → incoming JSON body
    - *Response → outgoing JSON body
    - *Source   → nested object within a response
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ────────────────────────────────────────────────────────────────────
# Ask endpoint
# ────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    """Request body for POST /api/ask."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The legal question to ask.",
        json_schema_extra={"examples": [
            "What are the lawful bases for processing personal data?",
        ]},
    )
    document_id: str | None = Field(
        default=None,
        description=(
            "Optional: restrict search to a specific Act. "
            'e.g. "dpa-2018", "osa-2023", "cra-2015".'
        ),
    )
    strict_mode: bool = Field(
        default=False,
        description=(
            "If true, the LLM will ONLY quote statutory text — "
            "no explanations or external knowledge."
        ),
    )


class LegislationSource(BaseModel):
    """A single legislation source that was consulted."""

    citation: str = Field(description="e.g. 'DPA 2018, s.6'")
    title: str = Field(description="Section title")
    hierarchy: str = Field(
        description="Full path in the Act: 'Act > Part > Chapter > Section'"
    )
    score: float = Field(description="BM25 relevance score (more negative = better)")


class AskResponse(BaseModel):
    """Response body for POST /api/ask."""

    answer: str = Field(description="The LLM's grounded answer")
    sources: list[LegislationSource] = Field(
        description="Legislation sections that were consulted"
    )
    has_sources: bool = Field(
        description="Whether the answer is grounded in retrieved legislation"
    )
    acts_consulted: list[str] = Field(
        description='List of Act IDs consulted, e.g. ["dpa-2018"]'
    )

    # Performance metrics
    search_latency_ms: float = Field(description="Time spent searching (ms)")
    llm_latency_ms: float = Field(description="Time spent waiting for LLM (ms)")
    total_latency_ms: float = Field(description="Total request time (ms)")
    tokens_used: int = Field(description="Total LLM tokens consumed")


# ────────────────────────────────────────────────────────────────────
# Explain endpoint
# ────────────────────────────────────────────────────────────────────

class ExplainRequest(BaseModel):
    """Request body for POST /api/explain."""

    section_id: str = Field(
        ...,
        min_length=3,
        description=(
            "The node ID of the section to explain. "
            'e.g. "dpa-2018:section-47"'
        ),
    )


class ExplainResponse(BaseModel):
    """Response body for POST /api/explain."""

    explanation: str = Field(description="Plain English explanation")
    section_id: str = Field(description="The section that was explained")
    tokens_used: int = Field(description="Total LLM tokens consumed")
    latency_ms: float = Field(description="Total request time (ms)")


# ────────────────────────────────────────────────────────────────────
# Search endpoint (search without LLM — just legislation retrieval)
# ────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    """Request body for POST /api/search."""

    query: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="Search query for legislation.",
    )
    document_id: str | None = Field(
        default=None,
        description="Optional: restrict to a specific Act.",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of sections to return.",
    )


class SearchSectionResult(BaseModel):
    """A single search result with full section text."""

    citation: str
    title: str
    hierarchy: str
    text: str = Field(description="Full section text including subsections")
    score: float


class SearchResponse(BaseModel):
    """Response body for POST /api/search."""

    query: str
    sections: list[SearchSectionResult]
    total_fts_hits: int = Field(
        description="Raw number of FTS matches before deduplication"
    )
    acts_found: list[str]


# ────────────────────────────────────────────────────────────────────
# Stats / health
# ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = Field(description="'ok' or 'error'")
    database: str = Field(description="Database connection status")
    documents: int = Field(description="Number of Acts in database")
    nodes: int = Field(description="Total number of legal nodes")
    version: str = Field(default="0.1.0")


class StatsResponse(BaseModel):
    """Response body for GET /api/stats."""

    documents: int
    total_nodes: int
    total_text_chars: int
    acts: list[dict] = Field(
        description="Per-Act breakdown: id, title, node count"
    )
