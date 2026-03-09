"""
Seed orchestrator — the full ingestion pipeline.

This module ties together the three Phase 2 components:

    fetch_acts.py   →   parse_xml.py   →   database.py
    (download XML)      (build tree)       (store + index)

It's designed to be idempotent: running it twice doesn't duplicate
data because we use INSERT OR REPLACE in the database and track
each document's ingestion status.

The pipeline for each Act:
  1. Download the full XML from legislation.gov.uk
  2. Parse it into a LegalDocument (flat list of LegalNode objects)
  3. Delete any existing nodes for that Act (clean re-ingestion)
  4. Insert all new nodes
  5. Update the documents table with metadata

Why a separate orchestrator instead of putting this in a script?
  - Testability: we can call seed_act() from tests
  - Reusability: the API layer can trigger re-ingestion
  - Separation: the script just calls this, doesn't contain logic
"""

from __future__ import annotations

import asyncio
import time

from rich.console import Console
from rich.table import Table

from app.ingestion.fetch_acts import ActInfo, LegislationFetcher, MVP_ACTS
from app.ingestion.parse_xml import parse_act_xml
from app.models.database import LawDatabase

console = Console()


async def seed_act(
    db: LawDatabase,
    fetcher: LegislationFetcher,
    act: ActInfo,
) -> int:
    """
    Ingest a single Act: fetch → parse → store.

    Returns the number of nodes inserted.
    """
    start = time.time()

    # 1. Fetch XML
    xml_bytes = await fetcher.fetch_act(act)

    # 2. Parse into PageIndex tree
    console.print(f"  [dim]Parsing {act.short_title}...[/dim]")
    doc = parse_act_xml(
        xml_bytes=xml_bytes,
        document_id=act.document_id,
        short_title=act.short_title,
        full_title=act.full_title,
        year=act.year,
        base_url=act.base_url,
    )
    console.print(
        f"  [green]✓[/green] Parsed: {doc.node_count} nodes, "
        f"{doc.total_text_length:,} chars"
    )

    # 3. Clear any existing data for this Act (idempotent re-ingestion)
    db_conn = db._ensure_db()
    await db_conn.execute(
        "DELETE FROM legal_nodes WHERE document_id = ?",
        (act.document_id,),
    )
    await db_conn.execute(
        "DELETE FROM documents WHERE document_id = ?",
        (act.document_id,),
    )
    await db_conn.commit()

    # 4. Insert all nodes
    console.print(f"  [dim]Storing {doc.node_count} nodes...[/dim]")
    count = await db.insert_nodes(doc.nodes)

    # 5. Record document metadata
    await db.insert_document(
        document_id=act.document_id,
        title=act.full_title,
        source="legislation.gov.uk",
        source_url=act.base_url,
        year=act.year,
        status="in_force",
        node_count=doc.node_count,
        total_text_length=doc.total_text_length,
    )

    elapsed = time.time() - start
    console.print(
        f"  [bold green]✓ {act.short_title} seeded: "
        f"{count} nodes in {elapsed:.1f}s[/bold green]"
    )
    return count


async def seed_all(
    acts: list[ActInfo] | None = None,
    db_path: str | None = None,
) -> dict:
    """
    Run the full seed pipeline for all MVP Acts.

    Returns stats dict with counts and timing.
    """
    acts = acts or MVP_ACTS
    total_start = time.time()

    console.print("\n[bold]═══ UK LawAssistant — Database Seeder ═══[/bold]\n")
    console.print(f"Acts to ingest: {len(acts)}")
    for act in acts:
        console.print(f"  • {act.full_title} ({act.year})")
    console.print()

    # Initialize database
    db = LawDatabase(db_path) if db_path else LawDatabase()
    await db.initialize()
    console.print(f"[green]✓[/green] Database ready at {db._db_path}\n")

    # Fetch and ingest each Act
    total_nodes = 0
    async with LegislationFetcher() as fetcher:
        for i, act in enumerate(acts):
            console.print(f"[bold]── [{i+1}/{len(acts)}] {act.full_title} ──[/bold]")
            count = await seed_act(db, fetcher, act)
            total_nodes += count
            console.print()

    # Rebuild FTS index after all bulk inserts
    console.print("[dim]Rebuilding search index...[/dim]")
    await db.rebuild_fts()
    console.print("[green]✓[/green] Search index rebuilt\n")

    # Final stats
    stats = await db.get_stats()
    elapsed = time.time() - total_start

    # Print summary table
    table = Table(title="Seed Summary", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total documents", str(stats["documents"]))
    table.add_row("Total nodes", f"{stats['total_nodes']:,}")
    table.add_row("Total text", f"{stats['total_text_chars']:,} chars")
    table.add_row("Time", f"{elapsed:.1f}s")

    # Node type breakdown
    if stats.get("by_type"):
        table.add_section()
        for node_type, count in sorted(stats["by_type"].items()):
            table.add_row(f"  {node_type}", str(count))

    # Per-document breakdown
    if stats.get("by_document"):
        table.add_section()
        for doc_id, count in sorted(stats["by_document"].items()):
            table.add_row(f"  {doc_id}", str(count))

    console.print(table)

    # Quick search test to verify FTS works
    console.print("\n[bold]Quick search test:[/bold]")
    test_queries = ["data protection", "online safety", "consumer rights"]
    for q in test_queries:
        results = await db.search(q, limit=3)
        if results:
            top = results[0]
            console.print(
                f"  \"{q}\" → {top[0].citation} "
                f"(score: {top[1]:.2f})"
            )
        else:
            console.print(f"  \"{q}\" → [yellow]no results[/yellow]")

    await db.close()

    console.print(f"\n[bold green]✓ Seeding complete![/bold green]\n")

    return {
        "total_nodes": total_nodes,
        "elapsed": elapsed,
        "stats": stats,
    }
