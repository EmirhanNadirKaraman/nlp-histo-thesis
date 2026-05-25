#!/usr/bin/env python3
"""Import a JSON embedding cache into the SQLite backend (B-064).

Non-destructive + idempotent: reads the JSON cache, writes a sibling ``.sqlite``;
the JSON file is never modified or deleted, and re-running skips existing keys.
No API calls. See ``eval/silver/matcher.import_json_cache_to_sqlite``.

Usage (from repo root):
  python scripts/import_embedding_cache_sqlite.py
  python scripts/import_embedding_cache_sqlite.py --json eval/data/embedding_cache_gemini.json
  python scripts/import_embedding_cache_sqlite.py --json X.json --sqlite Y.sqlite
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.silver.matcher import import_json_cache_to_sqlite

DEFAULT_JSON = "eval/data/embedding_cache_gemini.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="Import a JSON embedding cache into SQLite")
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--sqlite", default=None, help="default: <json> with a .sqlite suffix")
    ap.add_argument("--batch", type=int, default=2000)
    args = ap.parse_args()

    if not Path(args.json).exists():
        print(f"JSON cache not found: {args.json}", file=sys.stderr)
        return 2

    stats = import_json_cache_to_sqlite(args.json, args.sqlite, batch=args.batch)
    print(f"source JSON     : {stats['source']}")
    print(f"dest SQLite     : {stats['dest']}")
    print(f"json entries    : {stats['json_entries']}")
    print(f"rows imported   : {stats['imported']}")
    print(f"rows skipped    : {stats['skipped']}  (already present)")
    print(f"dim distribution: {stats['dims']}")
    print(f"SQLite total    : {stats['total']} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
