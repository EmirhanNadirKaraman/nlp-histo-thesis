#!/usr/bin/env python3
"""E13 Part 2 — inspect the merged relation claim-pair dataset (no API).

Prints totals, label / difficulty / relation-subtype counts, topic & entity
diversity, the first few examples per label, and any near-duplicate warnings.
Read-only; operates on the committed ``relation_claim_pairs_300.jsonl``.

  python -m eval.silver.relation_pairs.inspect
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from eval.paths import REPO_ROOT as _REPO_ROOT  # noqa: E402

from eval.silver.relation_pairs.validate_and_merge import (  # noqa: E402
    LABELS,
    OUT_JSONL,
    OUT_META,
    validate_full,
)


def load_dataset(path: Path = OUT_JSONL) -> list[dict]:
    if not path.exists():
        print(f"Dataset not found: {path}\n"
              "Run validate_and_merge first (after pasting all 10 batches).")
        raise SystemExit(1)
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _bar(count: int, total: int, width: int = 30) -> str:
    filled = int(round(width * count / total)) if total else 0
    return "█" * filled + "·" * (width - filled)


def main() -> None:
    records = load_dataset()
    errs, warnings, stats = validate_full(records)

    print(f"relation_claim_pairs — {stats['total']} pairs   ({OUT_JSONL.relative_to(_REPO_ROOT)})")
    if OUT_META.exists():
        meta = json.loads(OUT_META.read_text())
        print(f"  generator={meta.get('generator_model')} @ T={meta.get('temperature')}  "
              f"validated_at={meta.get('validated_at')}  content_hash={meta.get('content_hash')}")
    print()

    print("gold_label:")
    for lab in LABELS:
        c = stats["label_counts"].get(lab, 0)
        print(f"  {lab:14} {c:4d}  {_bar(c, stats['total'])}")
    print("\ndifficulty:")
    for diff, c in sorted(stats["difficulty_counts"].items()):
        print(f"  {str(diff):14} {c:4d}  {_bar(c, stats['total'])}")

    print("\nrelation_subtype (top):")
    for sub, c in Counter(stats["relation_subtype_counts"]).most_common(12):
        print(f"  {str(sub):22} {c:4d}")

    print(f"\ndiversity: {stats['n_unique_entities']} unique entities, "
          f"{stats['n_unique_topics']} unique topics")

    print("\nexamples (first 3 per label):")
    by_label: dict[str, list[dict]] = {lab: [] for lab in LABELS}
    for r in records:
        by_label.get(r.get("gold_label"), []).append(r)
    for lab in LABELS:
        print(f"  ── {lab} ──")
        for r in by_label[lab][:3]:
            print(f"    [{r.get('id')} | {r.get('difficulty')} | {r.get('relation_subtype')}]")
            print(f"      A: {r.get('claim_a')}")
            print(f"      B: {r.get('claim_b')}")

    if warnings:
        print(f"\nwarnings ({len(warnings)}):")
        for w in warnings[:15]:
            print(f"  ! {w}")
    if errs:
        print(f"\nVALIDATION ERRORS ({len(errs)}) — dataset is not clean:")
        for e in errs:
            print(f"  ✗ {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
