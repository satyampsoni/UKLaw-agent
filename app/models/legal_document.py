"""
PageIndex Legal Document Model.

This is the core data model for UK LawAssistant. Instead of naive text
chunking (which destroys legal document hierarchy), we model legislation
as a tree of typed nodes:

    Act → Part → Chapter → CrossHeading → Section → Subsection → Clause

Each node carries its full ancestry path, enabling:
  - Precise citation generation ("Data Protection Act 2018, s.6(1)")
  - Hierarchical context retrieval (walk up to get parent context)
  - Cross-reference linking between sections
  - Structured search over titles, text, and hierarchy

This model maps directly to the XML structure returned by
legislation.gov.uk's free API, so parsing is a natural tree walk
rather than heuristic text splitting.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class NodeType(str, Enum):
    """
    Legal document hierarchy levels.

    Maps to legislation.gov.uk XML elements:
      ACT          → <Legislation> root
      PART         → <Part>
      CHAPTER      → <Chapter>
      CROSSHEADING → <Pblock> (grouping of sections under a heading)
      SECTION      → <P1> (numbered section)
      SUBSECTION   → <P2> (numbered subsection within a section)
      CLAUSE       → <P3> / <P4> (lettered/numbered sub-clauses)
      SCHEDULE     → <Schedule>
      PARAGRAPH    → <P> within schedules
    """
    ACT = "act"
    PART = "part"
    CHAPTER = "chapter"
    CROSSHEADING = "crossheading"
    SECTION = "section"
    SUBSECTION = "subsection"
    CLAUSE = "clause"
    SCHEDULE = "schedule"
    PARAGRAPH = "paragraph"


# Depth mapping for each node type in the hierarchy
NODE_DEPTH: dict[NodeType, int] = {
    NodeType.ACT: 0,
    NodeType.PART: 1,
    NodeType.CHAPTER: 2,
    NodeType.CROSSHEADING: 3,
    NodeType.SECTION: 3,
    NodeType.SUBSECTION: 4,
    NodeType.CLAUSE: 5,
    NodeType.SCHEDULE: 1,
    NodeType.PARAGRAPH: 2,
}


class DocumentSource(str, Enum):
    """Where the document was ingested from."""
    LEGISLATION_GOV_UK = "legislation.gov.uk"
    HANSARD = "hansard.parliament.uk"
    GOV_UK = "gov.uk"


class DocumentStatus(str, Enum):
    """Current status of the legislation."""
    IN_FORCE = "in_force"
    PARTIALLY_IN_FORCE = "partially_in_force"
    NOT_YET_IN_FORCE = "not_yet_in_force"
    REPEALED = "repealed"
    AMENDED = "amended"


class LegalNode(BaseModel):
    """
    A single node in the PageIndex legal document tree.

    This is the atomic unit of the system. Every piece of legal text —
    from a full Act down to a single clause — is represented as a
    LegalNode with its full hierarchical context.
    """

    # Identity
    id: str = Field(
        ...,
        description="Unique node ID. Format: '{act_slug}-{node_type}-{number}'.",
    )
    document_id: str = Field(
        ...,
        description="Root document identifier. Example: 'dpa-2018'",
    )

    # Hierarchy
    node_type: NodeType = Field(
        ...,
        description="Type of this legal node in the hierarchy",
    )
    depth: int = Field(
        ...,
        ge=0,
        le=10,
        description="Depth in the tree. ACT=0, PART=1, etc.",
    )
    parent_id: Optional[str] = Field(
        default=None,
        description="ID of parent node. None for root (ACT) nodes.",
    )
    order_index: int = Field(
        default=0,
        ge=0,
        description="Position among siblings. Preserves document order.",
    )
    hierarchy_path: str = Field(
        ...,
        description="Full breadcrumb path. "
                    "Example: 'Data Protection Act 2018 > Part 2 > Chapter 2 > Section 6'",
    )

    # Content
    title: str = Field(
        ...,
        description="Title or heading of this node.",
    )
    text: str = Field(
        default="",
        description="Direct text content of this node (not including children).",
    )
    citation: str = Field(
        ...,
        description="Human-readable legal citation. "
                    "Example: 'Data Protection Act 2018, s.6(1)'",
    )

    # Source
    source: DocumentSource = Field(
        default=DocumentSource.LEGISLATION_GOV_UK,
    )
    source_url: str = Field(
        default="",
        description="Direct URL to this section on the source website.",
    )

    # Metadata
    year: Optional[int] = Field(default=None)
    status: DocumentStatus = Field(default=DocumentStatus.IN_FORCE)
    keywords: list[str] = Field(default_factory=list)
    cross_references: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    # Timestamps
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    @computed_field
    @property
    def text_hash(self) -> str:
        """SHA-256 hash of text for change detection during re-ingestion."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

    @computed_field
    @property
    def search_text(self) -> str:
        """
        Combined text for full-text search indexing.

        Concatenates title + hierarchy + text + keywords so that a search
        for "data protection lawful processing" matches across all fields.
        """
        parts = [self.title, self.hierarchy_path, self.text, " ".join(self.keywords)]
        return " ".join(part for part in parts if part)

    @computed_field
    @property
    def is_leaf(self) -> bool:
        """Whether this is a content-bearing leaf node."""
        return self.node_type in {
            NodeType.SECTION,
            NodeType.SUBSECTION,
            NodeType.CLAUSE,
            NodeType.PARAGRAPH,
        }


class LegalDocument(BaseModel):
    """
    A complete legal document as a flat list of tree nodes.

    Nodes form a tree via parent_id, but are stored flat for
    efficient database insertion, indexing, and querying.
    """

    document_id: str
    title: str
    source: DocumentSource = DocumentSource.LEGISLATION_GOV_UK
    source_url: str = ""
    year: Optional[int] = None
    status: DocumentStatus = DocumentStatus.IN_FORCE
    nodes: list[LegalNode] = Field(default_factory=list)
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    @computed_field
    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @computed_field
    @property
    def total_text_length(self) -> int:
        return sum(len(n.text) for n in self.nodes)

    def get_node(self, node_id: str) -> Optional[LegalNode]:
        """Find a node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_children(self, parent_id: str) -> list[LegalNode]:
        """Get direct children of a node, in document order."""
        return sorted(
            [n for n in self.nodes if n.parent_id == parent_id],
            key=lambda n: n.order_index,
        )

    def get_ancestors(self, node_id: str) -> list[LegalNode]:
        """
        Walk up from a node to the root (Act).

        Returns ancestors from immediate parent → root.
        This is the key operation for RAG context building:
        given a matched section, get all ancestor context so
        the LLM knows WHERE in the Act this section lives.
        """
        ancestors: list[LegalNode] = []
        current = self.get_node(node_id)
        if not current:
            return ancestors
        while current and current.parent_id:
            parent = self.get_node(current.parent_id)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        return ancestors

    def get_subtree_text(self, node_id: str) -> str:
        """
        Get text of a node and ALL its descendants.

        Used to get the complete text of a Section including
        all subsections and clauses.
        """
        parts: list[str] = []
        self._collect_text(node_id, parts)
        return "\n".join(parts)

    def _collect_text(self, node_id: str, parts: list[str]) -> None:
        """Recursively collect text from a node and its children."""
        node = self.get_node(node_id)
        if not node:
            return
        if node.text:
            parts.append(f"{node.citation}: {node.text}")
        for child in self.get_children(node_id):
            self._collect_text(child.id, parts)
