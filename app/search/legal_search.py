"""
Legal search engine for UK LawAssistant.

This is the intelligence layer between raw FTS5 and the RAG pipeline.
Raw database search returns individual nodes — but that's not what
an LLM needs. For a good answer about "lawful processing", the LLM
needs:

    1. The matching sections (FTS5 results)
    2. Where they sit in the Act (ancestor context)
    3. Their full text including subsections (subtree expansion)
    4. Deduplication (multiple subsections of s.6 shouldn't repeat s.6)

This module builds that complete "search context" by:
    FTS5 match → expand to section level → gather ancestors → assemble

The output is a SearchResult containing everything the RAG prompt
needs: ranked sections with hierarchy paths, citation, and full text.

Why not just dump raw search results into the LLM prompt?
  - A subsection match gives you "DPA 2018, s.6(1)" with 20 words
  - But the LLM needs the FULL section 6 (all subsections) + knowledge
    that it's in "Part 2 > Chapter 2 > Lawfulness of processing"
  - Without this context, the LLM hallucinates the surrounding text
  - With it, the LLM can cite accurately and explain in context
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.models.database import LawDatabase
from app.models.legal_document import LegalNode, NodeType


# ────────────────────────────────────────────────────────────────────
# Search result types
# ────────────────────────────────────────────────────────────────────

@dataclass
class SectionContext:
    """
    A single search hit expanded to section level with full context.

    This is what the RAG pipeline consumes — one self-contained
    chunk of legal text with all the context an LLM needs to
    answer accurately.
    """
    # The matched node (could be section, subsection, or clause)
    matched_node: LegalNode
    match_score: float

    # The section-level ancestor (or the node itself if it IS a section)
    section_node: LegalNode

    # Full hierarchy: section → chapter → part → act
    ancestors: list[LegalNode] = field(default_factory=list)

    # Complete text of the section including all children
    full_text: str = ""

    @property
    def citation(self) -> str:
        return self.section_node.citation

    @property
    def hierarchy_path(self) -> str:
        return self.section_node.hierarchy_path

    @property
    def document_id(self) -> str:
        return self.section_node.document_id


@dataclass
class SearchResult:
    """
    Complete search result with all context needed for RAG.
    """
    query: str
    sections: list[SectionContext]
    total_fts_hits: int

    @property
    def has_results(self) -> bool:
        return len(self.sections) > 0

    @property
    def documents_covered(self) -> set[str]:
        return {s.document_id for s in self.sections}

    def format_for_prompt(self, max_chars: int = 12000) -> str:
        """
        Format search results into a text block for the LLM prompt.

        This is the critical bridge between search and RAG.
        The format is designed to be:
          - Unambiguous (clear section boundaries)
          - Citeable (every chunk has a citation)
          - Hierarchical (ancestry shows where it fits)
          - Truncated (respects token limits)
        """
        if not self.sections:
            return "No relevant legislation found for this query."

        parts: list[str] = []
        char_count = 0

        for i, ctx in enumerate(self.sections):
            # Build the context block for this section
            block_parts: list[str] = []

            # Header with citation and hierarchy
            block_parts.append(f"═══ [{i+1}] {ctx.citation} ═══")
            block_parts.append(f"Location: {ctx.hierarchy_path}")
            block_parts.append(f"Title: {ctx.section_node.title}")
            block_parts.append("")

            # Full section text
            if ctx.full_text:
                block_parts.append(ctx.full_text)
            elif ctx.section_node.text:
                block_parts.append(ctx.section_node.text)

            block_parts.append("")

            block = "\n".join(block_parts)

            # Check token budget
            if char_count + len(block) > max_chars and i > 0:
                parts.append(
                    f"\n[{len(self.sections) - i} more sections omitted — "
                    f"token limit reached]"
                )
                break

            parts.append(block)
            char_count += len(block)

        return "\n".join(parts)


# ────────────────────────────────────────────────────────────────────
# Search engine
# ────────────────────────────────────────────────────────────────────

# Node types that represent "content-bearing" levels we want to search
SEARCHABLE_TYPES = [
    NodeType.SECTION,
    NodeType.SUBSECTION,
    NodeType.CLAUSE,
    NodeType.PARAGRAPH,
]

# Node types that are at "section level" — the unit of context
SECTION_LEVEL_TYPES = {NodeType.SECTION, NodeType.CROSSHEADING}


class LegalSearchEngine:
    """
    Search engine that queries legislation and builds LLM-ready context.

    Usage:
        engine = LegalSearchEngine(db)
        result = await engine.search("lawful processing personal data")
        prompt_text = result.format_for_prompt()
    """

    def __init__(self, db: LawDatabase):
        self._db = db

    async def search(
        self,
        query: str,
        limit: int = 10,
        document_id: str | None = None,
        expand_context: bool = True,
    ) -> SearchResult:
        """
        Search for legislation matching a query.

        Steps:
          1. FTS5 search for matching nodes
          2. Expand each hit up to section level
          3. Deduplicate (multiple subsection hits → one section)
          4. Gather ancestors for hierarchy context
          5. Get full section text (all children)

        Args:
            query: Natural language search query
            limit: Max sections to return
            document_id: Optional filter to one Act
            expand_context: Whether to expand to section level (default: True)

        Returns:
            SearchResult with expanded, deduplicated, ranked sections.
        """
        # 1. Raw FTS5 search
        raw_hits = await self._db.search(
            query=query,
            limit=limit * 3,  # Over-fetch because dedup will reduce
            document_id=document_id,
            node_types=SEARCHABLE_TYPES,
        )

        if not raw_hits:
            return SearchResult(query=query, sections=[], total_fts_hits=0)

        total_fts_hits = len(raw_hits)

        if not expand_context:
            # Return raw hits without expansion
            sections = []
            for node, score in raw_hits[:limit]:
                sections.append(SectionContext(
                    matched_node=node,
                    match_score=score,
                    section_node=node,
                    full_text=node.text,
                ))
            return SearchResult(
                query=query,
                sections=sections,
                total_fts_hits=total_fts_hits,
            )

        # 2. Expand each hit to section level and deduplicate
        seen_sections: dict[str, SectionContext] = {}

        for node, score in raw_hits:
            # Find the section-level ancestor
            section_node = await self._find_section_ancestor(node)

            # Deduplicate: keep the best score for each section
            if section_node.id in seen_sections:
                existing = seen_sections[section_node.id]
                # BM25 scores are negative — more negative = better match
                if score < existing.match_score:
                    seen_sections[section_node.id].match_score = score
                    seen_sections[section_node.id].matched_node = node
                continue

            seen_sections[section_node.id] = SectionContext(
                matched_node=node,
                match_score=score,
                section_node=section_node,
            )

        # 3. Enrich each section with ancestors and full text
        for ctx in seen_sections.values():
            # Get hierarchy (section → chapter → part → act)
            ctx.ancestors = await self._db.get_ancestors(ctx.section_node.id)

            # Get complete section text including all children
            ctx.full_text = await self._db.get_subtree_text(ctx.section_node.id)

        # 4. Sort by relevance and limit
        sections = sorted(
            seen_sections.values(),
            key=lambda s: s.match_score,  # BM25: more negative = better
        )[:limit]

        return SearchResult(
            query=query,
            sections=sections,
            total_fts_hits=total_fts_hits,
        )

    async def _find_section_ancestor(self, node: LegalNode) -> LegalNode:
        """
        Walk up the tree to find the section-level ancestor.

        If the node IS a section, return it directly.
        If it's a subsection/clause, walk up until we hit a section.
        If no section ancestor exists (orphan), return the node itself.
        """
        if node.node_type in SECTION_LEVEL_TYPES:
            return node

        current = node
        # Walk up at most 5 levels (safety limit)
        for _ in range(5):
            if not current.parent_id:
                return current

            parent = await self._db.get_node(current.parent_id)
            if not parent:
                return current

            if parent.node_type in SECTION_LEVEL_TYPES:
                return parent

            current = parent

        return current  # Safety: return whatever we have

    async def search_by_citation(
        self,
        citation: str,
        limit: int = 5,
    ) -> SearchResult:
        """
        Search specifically by legal citation.

        Searches the citation field directly, which is useful when
        the user mentions "section 6" or "s.6(1)".
        """
        # Search the citation column specifically
        raw_hits = await self._db.search(
            query=citation,
            limit=limit,
        )

        sections = []
        for node, score in raw_hits:
            section_node = await self._find_section_ancestor(node)
            full_text = await self._db.get_subtree_text(section_node.id)
            ancestors = await self._db.get_ancestors(section_node.id)

            sections.append(SectionContext(
                matched_node=node,
                match_score=score,
                section_node=section_node,
                ancestors=ancestors,
                full_text=full_text,
            ))

        return SearchResult(
            query=citation,
            sections=sections,
            total_fts_hits=len(raw_hits),
        )
