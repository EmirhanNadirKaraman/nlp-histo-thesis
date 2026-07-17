#!/usr/bin/env python3
"""Verify a voter_cache.json has L1/L2/L3 populated before sweeping.

The MAP cascade sweep needs L3 outputs for escalated chunks; an empty-L3 cache
silently degrades every escalation-sensitive metric (this was the failure mode
behind the stale primer — a datestamped Sonnet ID that 404'd, leaving L3 empty).
Run this after `collect`, before `sweep`. Exits non-zero if L3 is empty.

Pure file read — no API keys, no DB.

Usage (from repo root):
  python scripts/verify_voter_cache_l3.py [path/to/voter_cache.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT = Path("eval/data/map_primer/voter_cache.json")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"voter_cache not found: {path}", file=sys.stderr)
        return 2

    cache = json.loads(path.read_text(encoding="utf-8"))
    chunks = l1 = l2 = l3 = 0
    for entry in cache.values():
        for cid in entry.get("chunk_map", {}):
            chunks += 1
            if any(entry.get("l1", {}).get(cid) or []):
                l1 += 1
            if any(entry.get("l2", {}).get(cid) or []):
                l2 += 1
            if entry.get("l3", {}).get(cid):
                l3 += 1

    print(f"{path}")
    print(f"  cases={len(cache)}  chunks={chunks}")
    print(f"  L1 chunks with >=1 output: {l1}")
    print(f"  L2 chunks with >=1 output: {l2}")
    print(f"  L3 chunks with output:     {l3}")
    if l3 == 0:
        print("FAIL: L3 is empty — re-prime/collect before sweeping "
              "(every escalated chunk would contribute no findings).")
        return 1
    print("OK: L3 populated — safe to sweep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
