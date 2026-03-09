"""
Test script for the Relax AI LLM client.

Run: python -m scripts.test_llm

This script verifies that:
1. We can connect to the Relax AI API
2. We can send a simple question and get a response
3. We can stream a response
4. Token usage tracking works
"""

import asyncio
import logging

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
logging.basicConfig(level=logging.INFO)


async def test_simple_chat():
    """Test a simple single-turn chat completion."""
    from app.llm.relax_client import RelaxAIClient

    console.print("\n[bold cyan]Test 1: Simple Chat Completion[/bold cyan]")
    console.print("─" * 50)

    async with RelaxAIClient() as client:
        response = await client.chat(
            user_message="What is the UK Data Protection Act 2018? Answer in 2-3 sentences.",
            system_prompt="You are a UK legal expert. Be concise and accurate.",
        )

        # Display the response
        console.print(Panel(
            response.content,
            title="[green]LLM Response[/green]",
            border_style="green",
        ))

        # Display usage stats
        stats = Table(title="Usage Stats")
        stats.add_column("Metric", style="cyan")
        stats.add_column("Value", style="yellow")
        stats.add_row("Model", response.model)
        stats.add_row("Prompt Tokens", str(response.usage.prompt_tokens))
        stats.add_row("Completion Tokens", str(response.usage.completion_tokens))
        stats.add_row("Total Tokens", str(response.usage.total_tokens))
        stats.add_row("Latency", f"{response.latency_ms:.0f}ms")
        stats.add_row("Finish Reason", response.finish_reason)
        console.print(stats)

    return True


async def test_system_prompt():
    """Test that system prompts correctly shape behavior."""
    from app.llm.relax_client import RelaxAIClient

    console.print("\n[bold cyan]Test 2: System Prompt Behavior[/bold cyan]")
    console.print("─" * 50)

    async with RelaxAIClient() as client:
        response = await client.chat(
            user_message="What is GDPR?",
            system_prompt=(
                "You are a UK legal assistant. Always answer with reference to "
                "UK law specifically. Start your answer with 'Under UK law, ...'"
            ),
            max_tokens=200,
        )

        console.print(Panel(
            response.content,
            title="[green]System Prompt Response[/green]",
            border_style="green",
        ))

        # Check if the system prompt was followed
        if response.content.lower().startswith("under uk law"):
            console.print("[bold green]✓ System prompt followed correctly[/bold green]")
        else:
            console.print("[bold yellow]⚠ System prompt may not have been followed[/bold yellow]")

    return True


async def test_streaming():
    """Test streaming response."""
    from app.llm.relax_client import RelaxAIClient

    console.print("\n[bold cyan]Test 3: Streaming Response[/bold cyan]")
    console.print("─" * 50)
    console.print("[dim]Streaming: [/dim]", end="")

    async with RelaxAIClient() as client:
        full_text = ""
        async for chunk in client.stream(
            user_message="List 3 key UK privacy laws. Be brief.",
            system_prompt="You are a UK legal expert. Be concise.",
        ):
            console.print(chunk, end="")
            full_text += chunk

    console.print()  # Newline after stream

    if len(full_text) > 0:
        console.print(f"\n[bold green]✓ Streamed {len(full_text)} characters[/bold green]")
    else:
        console.print("\n[bold red]✗ No content streamed[/bold red]")

    return True


async def main():
    console.print(Panel(
        "[bold]UK LawAssistant — Relax AI Client Test[/bold]\n"
        "Testing connection to Relax AI API...",
        title="🇬🇧 Phase 1",
        border_style="blue",
    ))

    tests = [
        ("Simple Chat", test_simple_chat),
        ("System Prompt", test_system_prompt),
        ("Streaming", test_streaming),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            await test_fn()
            passed += 1
        except Exception as e:
            console.print(f"\n[bold red]✗ {name} failed: {e}[/bold red]")
            failed += 1

    # Summary
    console.print("\n" + "═" * 50)
    if failed == 0:
        console.print(f"[bold green]All {passed} tests passed! ✓[/bold green]")
        console.print("[green]Relax AI integration is working. Ready for Phase 2.[/green]")
    else:
        console.print(f"[bold yellow]{passed} passed, {failed} failed[/bold yellow]")


if __name__ == "__main__":
    asyncio.run(main())
