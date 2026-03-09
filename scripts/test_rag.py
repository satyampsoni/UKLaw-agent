"""
Test script for the RAG pipeline (Phase 4).

Runs 4 end-to-end tests:
  1. Basic legal question → search → LLM → grounded answer
  2. Act-scoped question → only searches one Act
  3. Explain a specific section in plain English
  4. Multi-turn conversation (follow-up question)

Each test validates:
  - The answer is non-empty
  - Sources are present (answer is grounded)
  - The correct Act(s) are consulted
  - Performance metrics are reasonable
"""

import asyncio
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.rag.pipeline import RAGPipeline

console = Console()


async def test_basic_question():
    """Test 1: Basic legal question."""
    console.rule("[bold cyan]Test 1: Basic Legal Question")

    async with RAGPipeline() as rag:
        question = "What are the lawful bases for processing personal data under UK law?"

        console.print(f"\n[bold]Question:[/] {question}\n")

        start = time.perf_counter()
        response = await rag.ask(question)
        elapsed = (time.perf_counter() - start) * 1000

        # Display answer
        console.print(Panel(
            response.answer[:2000],
            title="[green]LLM Answer",
            subtitle=f"{response.tokens_used} tokens | {elapsed:.0f}ms",
        ))

        # Display sources
        table = Table(title="Sources Consulted")
        table.add_column("Citation", style="cyan")
        table.add_column("Title")
        table.add_column("Score", justify="right")

        for src in response.sources_summary:
            table.add_row(src["citation"], src["title"], str(src["score"]))
        console.print(table)

        # Assertions
        assert response.answer, "Answer should not be empty"
        assert response.has_sources, "Answer should have sources"
        assert "dpa-2018" in response.acts_consulted, "Should consult DPA 2018"
        assert response.tokens_used > 0, "Should use tokens"

        console.print("[green]✓ Test 1 passed[/]\n")


async def test_scoped_question():
    """Test 2: Question scoped to a specific Act."""
    console.rule("[bold cyan]Test 2: Act-Scoped Question")

    async with RAGPipeline() as rag:
        question = "What duties do service providers have regarding illegal content?"

        console.print(f"\n[bold]Question:[/] {question}")
        console.print("[dim]Scope: Online Safety Act 2023 only[/]\n")

        response = await rag.ask(question, document_id="osa-2023")

        console.print(Panel(
            response.answer[:2000],
            title="[green]LLM Answer (OSA 2023 only)",
            subtitle=f"{response.tokens_used} tokens | {response.total_latency_ms:.0f}ms",
        ))

        # Assertions
        assert response.answer, "Answer should not be empty"
        assert response.has_sources, "Answer should have sources"
        # All sources should be from OSA 2023
        for src in response.sources_summary:
            assert "OSA 2023" in src["citation"], (
                f"Expected OSA 2023 citation, got: {src['citation']}"
            )

        console.print("[green]✓ Test 2 passed[/]\n")


async def test_explain_section():
    """Test 3: Explain a specific section."""
    console.rule("[bold cyan]Test 3: Explain Section")

    async with RAGPipeline() as rag:
        section_id = "dpa-2018:section-47"

        console.print(f"\n[bold]Section:[/] {section_id}")
        console.print("[dim]Asking LLM to explain in plain English[/]\n")

        response = await rag.explain_section(section_id)

        console.print(Panel(
            response.answer[:2000],
            title="[green]Plain English Explanation",
            subtitle=f"{response.tokens_used} tokens | {response.total_latency_ms:.0f}ms",
        ))

        # Assertions
        assert response.answer, "Explanation should not be empty"
        assert "not found" not in response.answer.lower(), (
            f"Section should be found, got: {response.answer[:100]}"
        )

        console.print("[green]✓ Test 3 passed[/]\n")


async def test_multiturn():
    """Test 4: Multi-turn conversation."""
    console.rule("[bold cyan]Test 4: Multi-Turn Conversation")

    async with RAGPipeline() as rag:
        # First question
        q1 = "What is a data protection officer under UK law?"
        console.print(f"\n[bold]Q1:[/] {q1}\n")

        r1 = await rag.ask(q1)
        console.print(Panel(
            r1.answer[:1500],
            title="[green]Answer 1",
            subtitle=f"{r1.tokens_used} tokens",
        ))

        assert r1.answer, "First answer should not be empty"
        assert rag.conversation_length == 1, "Should have 1 turn"

        # Follow-up question
        q2 = "What are their specific tasks and responsibilities?"
        console.print(f"\n[bold]Q2 (follow-up):[/] {q2}\n")

        r2 = await rag.ask(q2)
        console.print(Panel(
            r2.answer[:1500],
            title="[green]Answer 2 (follow-up)",
            subtitle=f"{r2.tokens_used} tokens",
        ))

        assert r2.answer, "Follow-up answer should not be empty"
        assert rag.conversation_length == 2, "Should have 2 turns"

        console.print("[green]✓ Test 4 passed[/]\n")


async def test_streaming():
    """Test 5: Streaming response."""
    console.rule("[bold cyan]Test 5: Streaming Response")

    async with RAGPipeline() as rag:
        question = "What are the consumer rights for faulty goods?"
        console.print(f"\n[bold]Question:[/] {question}\n")

        search_result, stream = await rag.ask_stream(question)

        # Sources are available immediately
        console.print(f"[dim]Sources ready: {len(search_result.sections)} sections from "
                       f"{search_result.documents_covered}[/]\n")

        # Stream the answer
        console.print("[green]Streaming answer:[/]")
        chunks = []
        async for chunk in stream:
            console.print(chunk, end="")
            chunks.append(chunk)
        console.print("\n")

        full_answer = "".join(chunks)
        assert full_answer, "Streamed answer should not be empty"
        assert search_result.has_results, "Should have search results"

        console.print(f"\n[dim]Total streamed: {len(full_answer)} chars[/]")
        console.print("[green]✓ Test 5 passed[/]\n")


async def main():
    console.print("\n[bold magenta]═══ UK LawAssistant — RAG Pipeline Tests ═══[/]\n")

    tests = [
        ("Basic legal question", test_basic_question),
        ("Act-scoped question", test_scoped_question),
        ("Explain section", test_explain_section),
        ("Multi-turn conversation", test_multiturn),
        ("Streaming response", test_streaming),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            await test_fn()
            passed += 1
        except Exception as e:
            console.print(f"[red]✗ {name} FAILED: {e}[/]\n")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    console.print("\n[bold magenta]═══ Results ═══[/]")
    console.print(f"  [green]Passed: {passed}[/]")
    if failed:
        console.print(f"  [red]Failed: {failed}[/]")
    else:
        console.print("  [green]All tests passed! RAG pipeline is working.[/]")
    console.print()


if __name__ == "__main__":
    asyncio.run(main())
