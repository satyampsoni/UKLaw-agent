"""
Test the legal search engine against the seeded database.

Usage:
    python -m scripts.test_search

Prerequisites:
    python -m scripts.seed_database  (must be run first)
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel

from app.models.database import LawDatabase
from app.search.legal_search import LegalSearchEngine

console = Console()


async def run_tests():
    # Connect to the seeded database
    db = LawDatabase()
    await db.initialize()

    engine = LegalSearchEngine(db)

    stats = await db.get_stats()
    console.print(f"\n[bold]Database: {stats['total_nodes']:,} nodes across {stats['documents']} Acts[/bold]\n")

    # ── Test 1: Basic legal concept search ──────────────────────────
    console.print("[bold]Test 1: Search for 'lawful processing personal data'[/bold]")
    result = await engine.search("lawful processing personal data", limit=5)
    _print_result(result)

    # ── Test 2: Specific Act search ─────────────────────────────────
    console.print("[bold]Test 2: Search within DPA 2018 for 'data protection officer'[/bold]")
    result = await engine.search("data protection officer", limit=5, document_id="dpa-2018")
    _print_result(result)

    # ── Test 3: Online Safety search ────────────────────────────────
    console.print("[bold]Test 3: Search for 'illegal content duties'[/bold]")
    result = await engine.search("illegal content duties", limit=5)
    _print_result(result)

    # ── Test 4: Consumer law search ─────────────────────────────────
    console.print("[bold]Test 4: Search for 'unfair contract terms'[/bold]")
    result = await engine.search("unfair contract terms", limit=5)
    _print_result(result)

    # ── Test 5: Show formatted RAG prompt ───────────────────────────
    console.print("[bold]Test 5: RAG prompt format for 'right to erasure'[/bold]")
    result = await engine.search("right to erasure", limit=3)
    prompt_text = result.format_for_prompt(max_chars=3000)
    console.print(Panel(
        prompt_text[:2000] + ("..." if len(prompt_text) > 2000 else ""),
        title="What the LLM sees",
        border_style="green",
    ))

    # ── Test 6: Citation search ─────────────────────────────────────
    console.print("\n[bold]Test 6: Citation search for 'section 6'[/bold]")
    result = await engine.search_by_citation("section 6", limit=3)
    _print_result(result)

    await db.close()
    console.print("\n[bold green]✓ All search tests complete![/bold green]\n")


def _print_result(result):
    console.print(f"  Query: \"{result.query}\"")
    console.print(f"  FTS hits: {result.total_fts_hits}, Sections: {len(result.sections)}")
    console.print(f"  Acts covered: {result.documents_covered}")

    for i, ctx in enumerate(result.sections[:3]):
        text_preview = ctx.full_text[:100].replace("\n", " ") if ctx.full_text else "(no text)"
        console.print(f"  [{i+1}] {ctx.citation} (score: {ctx.match_score:.2f})")
        console.print(f"      Path: {ctx.hierarchy_path}")
        console.print(f"      Text: {text_preview}...")

    console.print()


def main():
    # Check database exists
    db_path = Path("data/uklaw.db")
    if not db_path.exists():
        console.print("[red]Database not found. Run 'python -m scripts.seed_database' first.[/red]")
        sys.exit(1)

    asyncio.run(run_tests())


if __name__ == "__main__":
    main()
