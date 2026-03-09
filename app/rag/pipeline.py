"""
RAG (Retrieval-Augmented Generation) pipeline for UK LawAssistant.

This is the orchestrator that connects three systems:

    Search (Phase 3) → Prompts (this phase) → LLM (Phase 1)

Flow for every user question:
    1. User asks: "What are the lawful bases for processing personal data?"
    2. Search engine finds relevant sections (FTS5 → section expansion)
    3. format_for_prompt() turns those sections into a context block
    4. build_rag_prompt() wraps the context + question into a user message
    5. RelaxAIClient sends system_prompt + user_message to the LLM
    6. LLM returns an answer grounded in the retrieved legislation
    7. We package the answer + metadata into a RAGResponse

Why RAG instead of just asking the LLM?
  - LLMs hallucinate legal provisions. GPT-4 will confidently cite
    "Section 12(3) of the Data Protection Act" even if it doesn't exist.
  - RAG ensures the LLM only sees REAL statutory text.
  - The LLM's job becomes explanation, not recall — it's much better at that.

Why not fine-tune instead?
  - UK legislation changes frequently (new Acts, amendments, repeals).
  - Fine-tuning bakes knowledge into weights — stale in months.
  - RAG lets us update the database and immediately get correct answers.
  - Fine-tuning is also expensive and requires GPU infrastructure.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from app.llm.relax_client import RelaxAIClient, LLMResponse, TokenUsage, Message
from app.models.database import LawDatabase
from app.search.legal_search import LegalSearchEngine, SearchResult
from app.rag.prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_STRICT,
    build_rag_prompt,
    build_followup_prompt,
    build_explain_section_prompt,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Response types
# ────────────────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """
    Complete response from the RAG pipeline.

    Contains both the LLM's answer and the search metadata so the
    frontend can show:
      - The answer text
      - Which sections were consulted (for transparency)
      - Performance metrics (latency, tokens used)
      - Whether the answer is grounded (has_sources)
    """
    answer: str
    search_result: SearchResult
    llm_response: LLMResponse

    # Timing
    search_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    @property
    def has_sources(self) -> bool:
        """Whether the answer is grounded in retrieved legislation."""
        return self.search_result.has_results

    @property
    def sources_summary(self) -> list[dict]:
        """
        Summary of legislation sources consulted.

        Returns a list of dicts suitable for frontend display:
        [
            {"citation": "DPA 2018, s.6", "title": "Lawfulness...", "score": -12.3},
            ...
        ]
        """
        return [
            {
                "citation": ctx.citation,
                "title": ctx.section_node.title,
                "hierarchy": ctx.hierarchy_path,
                "score": round(ctx.match_score, 2),
            }
            for ctx in self.search_result.sections
        ]

    @property
    def tokens_used(self) -> int:
        return self.llm_response.usage.total_tokens

    @property
    def acts_consulted(self) -> set[str]:
        return self.search_result.documents_covered


# ────────────────────────────────────────────────────────────────────
# Conversation turn (for multi-turn support)
# ────────────────────────────────────────────────────────────────────

@dataclass
class ConversationTurn:
    """A single Q&A exchange in a conversation."""
    question: str
    answer: str
    citations: list[str] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────
# RAG Pipeline
# ────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Orchestrates: search → prompt construction → LLM call → response packaging.

    Usage:
        async with RAGPipeline() as rag:
            response = await rag.ask("What is the right to erasure?")
            print(response.answer)
            print(response.sources_summary)
    """

    def __init__(
        self,
        db_path: str | None = None,
        max_context_chars: int = 12000,
        search_limit: int = 5,
        strict_mode: bool = False,
    ):
        """
        Args:
            db_path: Path to SQLite database. None = default.
            max_context_chars: Max chars of legislation in prompt.
                12,000 chars ≈ 3,000 tokens — leaves room for the
                question and a 5,000-token response within most
                model context windows.
            search_limit: Max sections to retrieve per query.
            strict_mode: If True, uses SYSTEM_PROMPT_STRICT (no external
                knowledge, quotes only). Default: balanced mode.
        """
        self._db_path = db_path
        self._max_context_chars = max_context_chars
        self._search_limit = search_limit
        self._strict_mode = strict_mode

        # Initialised in __aenter__ or from_components()
        self._db: LawDatabase | None = None
        self._search: LegalSearchEngine | None = None
        self._llm: RelaxAIClient | None = None
        self._owns_resources = True  # Whether we created db/llm ourselves

        # Conversation history for multi-turn
        self._history: list[ConversationTurn] = []

    @classmethod
    def from_components(
        cls,
        db: LawDatabase,
        search: LegalSearchEngine,
        llm: RelaxAIClient,
        max_context_chars: int = 12000,
        search_limit: int = 5,
        strict_mode: bool = False,
    ) -> "RAGPipeline":
        """
        Create a RAGPipeline from pre-initialised components.

        Used by the FastAPI lifespan to share a single DB connection
        and LLM client across the search engine and RAG pipeline,
        instead of duplicating them.
        """
        instance = cls(
            max_context_chars=max_context_chars,
            search_limit=search_limit,
            strict_mode=strict_mode,
        )
        instance._db = db
        instance._search = search
        instance._llm = llm
        instance._owns_resources = False  # Don't close — caller owns them
        return instance

    async def __aenter__(self) -> "RAGPipeline":
        """Open database and LLM connections."""
        self._db = LawDatabase(self._db_path)
        await self._db.initialize()
        self._search = LegalSearchEngine(self._db)
        self._llm = RelaxAIClient()
        await self._llm.__aenter__()
        self._owns_resources = True
        return self

    async def __aexit__(self, *args) -> None:
        """Close connections only if we created them."""
        if self._owns_resources:
            if self._llm:
                await self._llm.__aexit__(*args)
            if self._db:
                await self._db.close()

    # ──────────────────────────────────────────────────────────────
    # Main API: ask a question
    # ──────────────────────────────────────────────────────────────

    async def ask(
        self,
        question: str,
        document_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> RAGResponse:
        """
        Ask a legal question and get a grounded answer.

        This is the primary entry point. It:
          1. Searches for relevant legislation
          2. Builds a RAG prompt with the legislation context
          3. Sends it to the LLM
          4. Returns the answer with source metadata

        Args:
            question: The user's legal question.
            document_id: Optional — restrict search to one Act
                (e.g. "dpa-2018" to only search the DPA).
            temperature: Override LLM temperature for this query.
            max_tokens: Override max response tokens.

        Returns:
            RAGResponse with answer, sources, and metrics.
        """
        total_start = time.perf_counter()

        # 1. Search for relevant legislation
        search_start = time.perf_counter()
        search_result = await self._search.search(
            query=question,
            limit=self._search_limit,
            document_id=document_id,
        )
        search_ms = (time.perf_counter() - search_start) * 1000

        logger.info(
            "Search complete: %d FTS hits → %d sections (%.0fms)",
            search_result.total_fts_hits,
            len(search_result.sections),
            search_ms,
        )

        # 2. Format legislation context for the prompt
        context = search_result.format_for_prompt(
            max_chars=self._max_context_chars,
        )

        # 3. Build the user message
        if self._history:
            # Multi-turn: include conversation summary
            summary = self._build_conversation_summary()
            user_message = build_followup_prompt(
                question=question,
                context=context,
                conversation_summary=summary,
            )
        else:
            user_message = build_rag_prompt(
                question=question,
                context=context,
            )

        # 4. Choose system prompt
        system_prompt = (
            SYSTEM_PROMPT_STRICT if self._strict_mode else SYSTEM_PROMPT
        )

        # 5. Call the LLM
        llm_response = await self._llm.chat(
            user_message=user_message,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        total_ms = (time.perf_counter() - total_start) * 1000

        logger.info(
            "RAG complete: %d tokens, %.0fms total (%.0fms search + %.0fms LLM)",
            llm_response.usage.total_tokens,
            total_ms,
            search_ms,
            llm_response.latency_ms,
        )

        # 6. Record in conversation history
        citations = [ctx.citation for ctx in search_result.sections]
        self._history.append(ConversationTurn(
            question=question,
            answer=llm_response.content,
            citations=citations,
        ))

        return RAGResponse(
            answer=llm_response.content,
            search_result=search_result,
            llm_response=llm_response,
            search_latency_ms=round(search_ms, 2),
            total_latency_ms=round(total_ms, 2),
        )

    # ──────────────────────────────────────────────────────────────
    # Explain a section
    # ──────────────────────────────────────────────────────────────

    async def explain_section(
        self,
        section_id: str,
    ) -> RAGResponse:
        """
        Explain a specific section in plain English.

        Fetches the section's full text from the database and asks
        the LLM to explain it. No search needed — we already know
        which section the user wants.

        Args:
            section_id: The node ID of the section (e.g. "dpa-2018:section-6").

        Returns:
            RAGResponse with the explanation.
        """
        total_start = time.perf_counter()

        # Get the section's full text
        section_text = await self._db.get_subtree_text(section_id)
        if not section_text:
            # Try to get the node directly
            node = await self._db.get_node(section_id)
            if node and node.text:
                section_text = node.text
            else:
                # Return an empty response
                return RAGResponse(
                    answer=f"Section '{section_id}' not found in the database.",
                    search_result=SearchResult(
                        query=section_id, sections=[], total_fts_hits=0
                    ),
                    llm_response=LLMResponse(
                        content=f"Section '{section_id}' not found.",
                        model="",
                        usage=TokenUsage(),
                        latency_ms=0,
                    ),
                )

        # Build prompt and call LLM
        user_message = build_explain_section_prompt(section_text)
        system_prompt = SYSTEM_PROMPT

        llm_response = await self._llm.chat(
            user_message=user_message,
            system_prompt=system_prompt,
        )

        total_ms = (time.perf_counter() - total_start) * 1000

        # Build a minimal SearchResult for consistency
        node = await self._db.get_node(section_id)
        search_result = SearchResult(
            query=f"Explain {section_id}",
            sections=[],
            total_fts_hits=1 if node else 0,
        )

        return RAGResponse(
            answer=llm_response.content,
            search_result=search_result,
            llm_response=llm_response,
            total_latency_ms=round(total_ms, 2),
        )

    # ──────────────────────────────────────────────────────────────
    # Streaming
    # ──────────────────────────────────────────────────────────────

    async def ask_stream(
        self,
        question: str,
        document_id: str | None = None,
    ) -> tuple[SearchResult, AsyncIterator[str]]:
        """
        Ask a question and stream the LLM's response.

        Returns both the search result (available immediately) and
        an async iterator of response chunks. This lets the frontend
        show sources immediately while the answer streams in.

        Args:
            question: The user's legal question.
            document_id: Optional Act filter.

        Returns:
            Tuple of (SearchResult, async iterator of text chunks).

        Usage:
            async with RAGPipeline() as rag:
                search_result, stream = await rag.ask_stream("What is GDPR?")
                # Show sources immediately
                print(search_result.format_for_prompt())
                # Stream the answer
                async for chunk in stream:
                    print(chunk, end="", flush=True)
        """
        # 1. Search (blocking — we need results before calling LLM)
        search_result = await self._search.search(
            query=question,
            limit=self._search_limit,
            document_id=document_id,
        )

        # 2. Build prompt
        context = search_result.format_for_prompt(
            max_chars=self._max_context_chars,
        )
        user_message = build_rag_prompt(question=question, context=context)
        system_prompt = (
            SYSTEM_PROMPT_STRICT if self._strict_mode else SYSTEM_PROMPT
        )

        # 3. Return search result + stream iterator
        stream = self._llm.stream(
            user_message=user_message,
            system_prompt=system_prompt,
        )

        return search_result, stream

    # ──────────────────────────────────────────────────────────────
    # Conversation management
    # ──────────────────────────────────────────────────────────────

    def clear_history(self) -> None:
        """Reset conversation history for a new session."""
        self._history.clear()

    @property
    def conversation_length(self) -> int:
        """Number of Q&A turns in the current conversation."""
        return len(self._history)

    def _build_conversation_summary(self) -> str:
        """
        Build a brief summary of prior conversation turns.

        We don't send the full history — that would blow the token
        budget. Instead, we summarise each turn in one line so the
        LLM has continuity without the cost.

        We keep only the last 3 turns to stay within ~300 tokens.
        """
        recent = self._history[-3:]  # Last 3 turns max
        lines = []
        for i, turn in enumerate(recent, 1):
            # Truncate answer to first 150 chars
            answer_preview = turn.answer[:150]
            if len(turn.answer) > 150:
                answer_preview += "..."
            lines.append(
                f"Q{i}: {turn.question}\n"
                f"A{i}: {answer_preview}\n"
                f"Sources: {', '.join(turn.citations[:3])}"
            )
        return "\n\n".join(lines)
