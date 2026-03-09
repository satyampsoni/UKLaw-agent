"""
Seed the UK LawAssistant database with MVP Acts.

Usage:
    python -m scripts.seed_database

This downloads 3 UK Acts from legislation.gov.uk, parses them
into our PageIndex tree model, and stores them in the local
SQLite database with FTS5 full-text search indexing.

Takes about 30-60 seconds depending on network speed.
"""

import asyncio
import sys
from pathlib import Path

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.seed import seed_all


def main():
    result = asyncio.run(seed_all())

    if result["total_nodes"] == 0:
        print("\n⚠ No nodes were ingested. Check network connectivity.")
        sys.exit(1)

    print(f"Database location: data/uklaw.db")
    print(f"Total nodes: {result['total_nodes']:,}")
    print(f"Time: {result['elapsed']:.1f}s")


if __name__ == "__main__":
    main()
