"""
Async fetcher for UK legislation XML from legislation.gov.uk.

This module handles downloading Act XML from the official UK government
API. It's intentionally simple — a thin async wrapper around httpx that:

  1. Knows the URL pattern for Acts: /ukpga/{year}/{chapter}/data.xml
  2. Handles retries and rate-limiting (the API has no auth but we
     should be polite — it's a public service)
  3. Returns raw bytes for the XML parser to consume

We define our MVP act catalogue here too — the 3 Acts we seed the
database with. This is a deliberate MVP choice: start with 3 Acts
that cover AI/data/consumer law, prove the pipeline works end-to-end,
then expand later.

Why async?
  - We can fetch multiple Acts in parallel during seeding
  - The same httpx client style as our LLM client
  - FastAPI (Phase 5) is async, so everything stays consistent
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
from rich.console import Console

console = Console()


# ────────────────────────────────────────────────────────────────────
# Act catalogue — the Acts we know how to fetch
# ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ActInfo:
    """Metadata for an Act we want to ingest."""
    document_id: str       # Our internal slug
    short_title: str       # For citations: "DPA 2018"
    full_title: str        # Full name: "Data Protection Act 2018"
    year: int
    chapter: int           # Statute book chapter number
    act_type: str = "ukpga"  # UK Public General Act

    @property
    def base_url(self) -> str:
        return f"https://www.legislation.gov.uk/{self.act_type}/{self.year}/{self.chapter}"

    @property
    def xml_url(self) -> str:
        return f"{self.base_url}/data.xml"


# ── MVP Acts ──────────────────────────────────────────────────────
# Three Acts that cover the core of AI/data/consumer law in the UK.
# Each one tests different XML structures:
#   - DPA 2018: Large act with deep nesting, many schedules
#   - OSA 2023: Recent act, complex cross-referencing
#   - CRA 2015: Consumer-focused, different Part structure

MVP_ACTS: list[ActInfo] = [
    ActInfo(
        document_id="dpa-2018",
        short_title="DPA 2018",
        full_title="Data Protection Act 2018",
        year=2018,
        chapter=12,
    ),
    ActInfo(
        document_id="osa-2023",
        short_title="OSA 2023",
        full_title="Online Safety Act 2023",
        year=2023,
        chapter=50,
    ),
    ActInfo(
        document_id="cra-2015",
        short_title="CRA 2015",
        full_title="Consumer Rights Act 2015",
        year=2015,
        chapter=15,
    ),
]


# ────────────────────────────────────────────────────────────────────
# Fetcher
# ────────────────────────────────────────────────────────────────────

class LegislationFetcher:
    """
    Downloads Act XML from legislation.gov.uk.

    Usage:
        async with LegislationFetcher() as fetcher:
            xml_bytes = await fetcher.fetch_act(MVP_ACTS[0])
    """

    def __init__(
        self,
        timeout: float = 60.0,
        max_retries: int = 3,
        delay_between_requests: float = 1.0,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.delay = delay_between_requests
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> LegislationFetcher:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=True,
            headers={
                "Accept": "application/xml",
                "User-Agent": "UK-LawAssistant/1.0 (legal-research-tool)",
            },
        )
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def fetch_act(self, act: ActInfo) -> bytes:
        """
        Download the full XML for an Act.

        Retries on transient errors (5xx, timeouts).
        Raises on permanent failures (404, etc).

        Returns raw XML bytes.
        """
        if not self._client:
            raise RuntimeError("Use 'async with LegislationFetcher()' context manager")

        url = act.xml_url
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                console.print(
                    f"  [dim]Fetching {act.short_title} "
                    f"(attempt {attempt}/{self.max_retries})...[/dim]"
                )
                response = await self._client.get(url)
                response.raise_for_status()

                size_kb = len(response.content) / 1024
                console.print(
                    f"  [green]✓[/green] {act.short_title}: "
                    f"{size_kb:.0f} KB downloaded"
                )
                return response.content

            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500:
                    # Client error (404, 403, etc) — don't retry
                    console.print(
                        f"  [red]✗[/red] {act.short_title}: "
                        f"HTTP {e.response.status_code} — not retrying"
                    )
                    raise
                last_error = e
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e

            # Wait before retry (exponential backoff)
            if attempt < self.max_retries:
                wait = self.delay * (2 ** (attempt - 1))
                console.print(f"  [yellow]⟳[/yellow] Retrying in {wait:.1f}s...")
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"Failed to fetch {act.short_title} after {self.max_retries} attempts: "
            f"{last_error}"
        )

    async def fetch_all(self, acts: list[ActInfo] | None = None) -> dict[str, bytes]:
        """
        Fetch all Acts sequentially (to be polite to the API).

        Returns a dict mapping document_id → XML bytes.
        """
        acts = acts or MVP_ACTS
        results: dict[str, bytes] = {}

        console.print(f"\n[bold]Fetching {len(acts)} Acts from legislation.gov.uk[/bold]\n")

        for act in acts:
            xml_bytes = await self.fetch_act(act)
            results[act.document_id] = xml_bytes

            # Be polite — wait between requests
            if act != acts[-1]:
                await asyncio.sleep(self.delay)

        total_kb = sum(len(v) for v in results.values()) / 1024
        console.print(
            f"\n[bold green]✓ All {len(results)} Acts fetched "
            f"({total_kb:.0f} KB total)[/bold green]\n"
        )
        return results
