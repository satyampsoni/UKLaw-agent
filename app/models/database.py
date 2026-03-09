"""
Database layer for UK LawAssistant.

Uses SQLite with FTS5 (Full-Text Search) for the MVP.
SQLite is the right choice here because:
  - Zero setup — no Docker, no external service
  - FTS5 gives us BM25 ranking out of the box
  - At our scale (3 Acts, ~1000 sections), SQLite is plenty fast
  - Single-file database that's easy to version and backup
  - We migrate to PostgreSQL when we scale to thousands of Acts

The schema has two tables:
  1. `legal_nodes` — stores every PageIndex node
  2. `legal_nodes_fts` — FTS5 virtual table for full-text search

aiosqlite gives us async access so the database doesn't block
the event loop during API requests.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import aiosqlite

from app.models.legal_document import (
    LegalNode,
    NodeType,
    DocumentSource,
    DocumentStatus,
)

logger = logging.getLogger(__name__)

# Default database path — in project root
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "uklaw.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CREATE_NODES_TABLE = """
CREATE TABLE IF NOT EXISTS legal_nodes (
    id              TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL,
    node_type       TEXT NOT NULL,
    depth           INTEGER NOT NULL,
    parent_id       TEXT,
    order_index     INTEGER NOT NULL DEFAULT 0,
    hierarchy_path  TEXT NOT NULL,
    title           TEXT NOT NULL,
    text            TEXT NOT NULL DEFAULT '',
    citation        TEXT NOT NULL,
    source          TEXT NOT NULL DEFAULT 'legislation.gov.uk',
    source_url      TEXT NOT NULL DEFAULT '',
    year            INTEGER,
    status          TEXT NOT NULL DEFAULT 'in_force',
    keywords        TEXT NOT NULL DEFAULT '[]',
    cross_references TEXT NOT NULL DEFAULT '[]',
    metadata        TEXT NOT NULL DEFAULT '{}',
    text_hash       TEXT NOT NULL DEFAULT '',
    search_text     TEXT NOT NULL DEFAULT '',
    ingested_at     TEXT NOT NULL,

    FOREIGN KEY (parent_id) REFERENCES legal_nodes(id)
);
"""

CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_nodes_document_id ON legal_nodes(document_id);
CREATE INDEX IF NOT EXISTS idx_nodes_parent_id ON legal_nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_node_type ON legal_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_depth ON legal_nodes(depth);
CREATE INDEX IF NOT EXISTS idx_nodes_year ON legal_nodes(year);
"""

# FTS5 virtual table for full-text search with BM25 ranking.
# We index: title, hierarchy_path, text, keywords (as combined search_text)
# plus the citation for exact citation searches.
CREATE_FTS_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS legal_nodes_fts USING fts5(
    id UNINDEXED,
    title,
    text,
    citation,
    hierarchy_path,
    search_text,
    content='legal_nodes',
    content_rowid='rowid',
    tokenize='porter unicode61'
);
"""

# Triggers to keep FTS index in sync with the main table.
# When a row is inserted/updated/deleted in legal_nodes,
# the FTS index is automatically updated.
CREATE_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS legal_nodes_ai AFTER INSERT ON legal_nodes BEGIN
    INSERT INTO legal_nodes_fts(rowid, id, title, text, citation, hierarchy_path, search_text)
    VALUES (new.rowid, new.id, new.title, new.text, new.citation, new.hierarchy_path, new.search_text);
END;

CREATE TRIGGER IF NOT EXISTS legal_nodes_ad AFTER DELETE ON legal_nodes BEGIN
    INSERT INTO legal_nodes_fts(legal_nodes_fts, rowid, id, title, text, citation, hierarchy_path, search_text)
    VALUES ('delete', old.rowid, old.id, old.title, old.text, old.citation, old.hierarchy_path, old.search_text);
END;

CREATE TRIGGER IF NOT EXISTS legal_nodes_au AFTER UPDATE ON legal_nodes BEGIN
    INSERT INTO legal_nodes_fts(legal_nodes_fts, rowid, id, title, text, citation, hierarchy_path, search_text)
    VALUES ('delete', old.rowid, old.id, old.title, old.text, old.citation, old.hierarchy_path, old.search_text);
    INSERT INTO legal_nodes_fts(rowid, id, title, text, citation, hierarchy_path, search_text)
    VALUES (new.rowid, new.id, new.title, new.text, new.citation, new.hierarchy_path, new.search_text);
END;
"""

# Documents table — one row per Act/document
CREATE_DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    document_id     TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    source          TEXT NOT NULL DEFAULT 'legislation.gov.uk',
    source_url      TEXT NOT NULL DEFAULT '',
    year            INTEGER,
    status          TEXT NOT NULL DEFAULT 'in_force',
    node_count      INTEGER NOT NULL DEFAULT 0,
    total_text_length INTEGER NOT NULL DEFAULT 0,
    ingested_at     TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------


class LawDatabase:
    """
    Async SQLite database for UK LawAssistant.

    Usage:
        db = LawDatabase()
        await db.initialize()

        # Insert nodes
        await db.insert_nodes(nodes)

        # Search
        results = await db.search("lawful processing")

        await db.close()
    """

    def __init__(self, db_path: Path | str | None = None):
        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Create database file, tables, indexes, and FTS."""
        # Ensure data directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row

        # Enable WAL mode for better concurrent read performance
        await self._db.execute("PRAGMA journal_mode=WAL")
        # Enable foreign keys
        await self._db.execute("PRAGMA foreign_keys=ON")

        # Create schema
        await self._db.executescript(CREATE_DOCUMENTS_TABLE)
        await self._db.executescript(CREATE_NODES_TABLE)
        await self._db.executescript(CREATE_INDEXES)
        await self._db.executescript(CREATE_FTS_TABLE)
        await self._db.executescript(CREATE_FTS_TRIGGERS)
        await self._db.commit()

        logger.info(f"Database initialized at {self._db_path}")

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    def _ensure_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._db

    # ------------------------------------------------------------------
    # Insert operations
    # ------------------------------------------------------------------

    async def insert_document(
        self,
        document_id: str,
        title: str,
        source: str,
        source_url: str,
        year: int | None,
        status: str,
        node_count: int,
        total_text_length: int,
    ) -> None:
        """Insert or replace a document record."""
        db = self._ensure_db()
        await db.execute(
            """
            INSERT OR REPLACE INTO documents
                (document_id, title, source, source_url, year, status,
                 node_count, total_text_length, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id, title, source, source_url, year, status,
                node_count, total_text_length,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await db.commit()

    async def insert_nodes(self, nodes: list[LegalNode]) -> int:
        """
        Insert a batch of LegalNodes into the database.

        Uses INSERT OR REPLACE so re-ingestion updates existing nodes.
        Returns the number of nodes inserted.
        """
        db = self._ensure_db()

        rows = []
        for node in nodes:
            rows.append((
                node.id,
                node.document_id,
                node.node_type.value,
                node.depth,
                node.parent_id,
                node.order_index,
                node.hierarchy_path,
                node.title,
                node.text,
                node.citation,
                node.source.value,
                node.source_url,
                node.year,
                node.status.value,
                json.dumps(node.keywords),
                json.dumps(node.cross_references),
                json.dumps(node.metadata),
                node.text_hash,
                node.search_text,
                node.ingested_at.isoformat(),
            ))

        await db.executemany(
            """
            INSERT OR REPLACE INTO legal_nodes
                (id, document_id, node_type, depth, parent_id, order_index,
                 hierarchy_path, title, text, citation, source, source_url,
                 year, status, keywords, cross_references, metadata,
                 text_hash, search_text, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        await db.commit()

        logger.info(f"Inserted {len(rows)} nodes")
        return len(rows)

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    async def get_node(self, node_id: str) -> Optional[LegalNode]:
        """Get a single node by ID."""
        db = self._ensure_db()
        async with db.execute(
            "SELECT * FROM legal_nodes WHERE id = ?", (node_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return self._row_to_node(row)
        return None

    async def get_children(self, parent_id: str) -> list[LegalNode]:
        """Get all direct children of a node, in document order."""
        db = self._ensure_db()
        async with db.execute(
            "SELECT * FROM legal_nodes WHERE parent_id = ? ORDER BY order_index",
            (parent_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_node(row) for row in rows]

    async def get_ancestors(self, node_id: str) -> list[LegalNode]:
        """
        Walk up the tree from a node to the root.

        Returns ancestors from immediate parent → root.
        Critical for RAG context building.
        """
        ancestors: list[LegalNode] = []
        current_id: str | None = node_id

        while current_id:
            node = await self.get_node(current_id)
            if not node or not node.parent_id:
                break
            parent = await self.get_node(node.parent_id)
            if parent:
                ancestors.append(parent)
                current_id = parent.id
            else:
                break

        return ancestors

    async def get_document_nodes(self, document_id: str) -> list[LegalNode]:
        """Get all nodes for a document, in order."""
        db = self._ensure_db()
        async with db.execute(
            "SELECT * FROM legal_nodes WHERE document_id = ? ORDER BY depth, order_index",
            (document_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_node(row) for row in rows]

    async def get_subtree_text(self, node_id: str) -> str:
        """Get combined text of a node and all its descendants."""
        node = await self.get_node(node_id)
        if not node:
            return ""

        parts: list[str] = []
        await self._collect_text(node_id, parts)
        return "\n".join(parts)

    async def _collect_text(self, node_id: str, parts: list[str]) -> None:
        """Recursively collect text from a node and its children."""
        node = await self.get_node(node_id)
        if not node:
            return
        if node.text:
            parts.append(f"{node.citation}: {node.text}")
        children = await self.get_children(node_id)
        for child in children:
            await self._collect_text(child.id, parts)

    # ------------------------------------------------------------------
    # Full-text search (FTS5 with BM25 ranking)
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 10,
        document_id: str | None = None,
        node_types: list[NodeType] | None = None,
    ) -> list[tuple[LegalNode, float]]:
        """
        Full-text search across all legal nodes.

        Uses SQLite FTS5 with BM25 ranking. Returns nodes
        ordered by relevance score (lower = more relevant in BM25).

        Args:
            query: Natural language search query
            limit: Max results to return
            document_id: Optional filter by document
            node_types: Optional filter by node types

        Returns:
            List of (node, score) tuples, most relevant first.
        """
        db = self._ensure_db()

        # Build FTS5 query — wrap in quotes for phrase matching,
        # or use raw for multi-term boolean matching
        fts_query = self._build_fts_query(query)

        sql = """
            SELECT legal_nodes.*, bm25(legal_nodes_fts) as score
            FROM legal_nodes_fts
            JOIN legal_nodes ON legal_nodes.id = legal_nodes_fts.id
            WHERE legal_nodes_fts MATCH ?
        """
        params: list = [fts_query]

        if document_id:
            sql += " AND legal_nodes.document_id = ?"
            params.append(document_id)

        if node_types:
            placeholders = ",".join("?" * len(node_types))
            sql += f" AND legal_nodes.node_type IN ({placeholders})"
            params.extend(nt.value for nt in node_types)

        sql += " ORDER BY score LIMIT ?"
        params.append(limit)

        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
            results = []
            for row in rows:
                node = self._row_to_node(row)
                score = row["score"]
                results.append((node, score))
            return results

    def _build_fts_query(self, query: str) -> str:
        """
        Convert a natural language query into an FTS5 query.

        Strategy:
        - Strip punctuation (FTS5 special chars: * ? " ^ { } ( ) : )
        - Remove stopwords that add noise
        - Join with OR so BM25 can rank by relevance
        - BM25 naturally scores docs with MORE matching terms higher
        - This avoids the AND problem where "lawful bases processing
          personal data" returns 0 hits because no single node
          contains ALL 5 words

        Why OR + BM25 is the right approach:
        - AND is too strict for natural language questions
        - OR returns more results but BM25 ranks them correctly
        - A node matching 4/5 terms scores much higher than 1/5
        - The search engine's section expansion + limit handles noise
        """
        import re
        # Strip everything except letters, digits, and spaces
        cleaned = re.sub(r"[^\w\s]", " ", query)
        words = cleaned.strip().split()
        # Only remove the most basic stopwords — keep legal terms
        stopwords = {
            "a", "an", "the", "is", "of", "in", "to", "and", "or",
            "for", "on", "by", "it", "at", "be", "as", "do", "no",
            "if", "so", "up", "we", "he", "me", "my", "am", "are",
            "was", "has", "had", "not", "but", "its", "can", "did",
            "how", "all", "what", "when", "who", "will", "with",
            "from", "have", "this", "that", "they", "been", "does",
            "each", "were", "than", "them", "then", "into", "some",
            "her", "him", "his", "our", "you", "she", "may", "any",
        }
        filtered = [w for w in words if len(w) >= 2 and w.lower() not in stopwords]
        if not filtered:
            # If everything was a stopword, use the longest original word
            words = cleaned.strip().split()
            if words:
                return max(words, key=len)
            return query
        # Use OR so BM25 ranking does the heavy lifting
        return " OR ".join(filtered)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_stats(self) -> dict:
        """Get database statistics."""
        db = self._ensure_db()

        stats = {}

        async with db.execute("SELECT COUNT(*) as cnt FROM documents") as cur:
            row = await cur.fetchone()
            stats["documents"] = row["cnt"]

        async with db.execute("SELECT COUNT(*) as cnt FROM legal_nodes") as cur:
            row = await cur.fetchone()
            stats["total_nodes"] = row["cnt"]

        async with db.execute(
            "SELECT node_type, COUNT(*) as cnt FROM legal_nodes GROUP BY node_type"
        ) as cur:
            rows = await cur.fetchall()
            stats["by_type"] = {row["node_type"]: row["cnt"] for row in rows}

        async with db.execute(
            "SELECT document_id, COUNT(*) as cnt FROM legal_nodes GROUP BY document_id"
        ) as cur:
            rows = await cur.fetchall()
            stats["by_document"] = {row["document_id"]: row["cnt"] for row in rows}

        async with db.execute(
            "SELECT SUM(LENGTH(text)) as total FROM legal_nodes"
        ) as cur:
            row = await cur.fetchone()
            stats["total_text_chars"] = row["total"] or 0

        return stats

    async def rebuild_fts(self) -> None:
        """
        Rebuild the FTS5 index from scratch.

        Call this after bulk delete + insert operations where the
        trigger-based sync may leave stale rowid references.
        FTS5's 'rebuild' command re-reads all rows from the content
        table and rebuilds the index.
        """
        db = self._ensure_db()
        await db.execute(
            "INSERT INTO legal_nodes_fts(legal_nodes_fts) VALUES('rebuild')"
        )
        await db.commit()
        logger.info("FTS index rebuilt")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_node(row) -> LegalNode:
        """Convert a database row to a LegalNode."""
        return LegalNode(
            id=row["id"],
            document_id=row["document_id"],
            node_type=NodeType(row["node_type"]),
            depth=row["depth"],
            parent_id=row["parent_id"],
            order_index=row["order_index"],
            hierarchy_path=row["hierarchy_path"],
            title=row["title"],
            text=row["text"],
            citation=row["citation"],
            source=DocumentSource(row["source"]),
            source_url=row["source_url"],
            year=row["year"],
            status=DocumentStatus(row["status"]),
            keywords=json.loads(row["keywords"]),
            cross_references=json.loads(row["cross_references"]),
            metadata=json.loads(row["metadata"]),
            ingested_at=datetime.fromisoformat(row["ingested_at"]),
        )
