#!/usr/bin/env python3
"""
compare_corpus_relations.py — A/B diff two corpus_relations.json runs.

Built for the predicate-vs-verbatim NLI input A/B (RelateConfig
``use_verbatim_for_nli`` flag), but works for any pair of corpus relation
outputs that share the same canonical_rule universe.

Reports:
  1. Per-file totals + label distribution (SUPPORT / CONTRADICT / etc.)
  2. Per-(rule_id_a, rule_id_b) label change matrix
  3. NLI-score delta distribution (mean, p50, p95 absolute change)
  4. Same-paper vs cross-paper breakdown
  5. Up to 5 spot-check examples for each kind of label flip

Usage::

    python scripts/eval/compare_corpus_relations.py \\
        --label-a predicate \\
        --file-a out/summaries/corpus_relations_predicate.json \\
        --label-b verbatim  \\
        --file-b out/summaries/corpus_relations_verbatim.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def _relations(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return data.get("relations", []) or []


def _key(rel: dict) -> tuple[str, str]:
    """(rule_id_a, rule_id_b) — order-canonicalised so swapped-A/B doesn't double-count."""
    a, b = rel.get("rule_id_a") or "", rel.get("rule_id_b") or ""
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--label-a", default="A", help="Short label for file A")
    p.add_argument("--file-a", required=True, type=Path)
    p.add_argument("--label-b", default="B", help="Short label for file B")
    p.add_argument("--file-b", required=True, type=Path)
    p.add_argument(
        "--examples", type=int, default=5,
        help="Max spot-check examples per flip kind (default 5).",
    )
    args = p.parse_args(argv)

    if not args.file_a.exists():
        print(f"ERROR: missing {args.file_a}", file=sys.stderr)
        return 2
    if not args.file_b.exists():
        print(f"ERROR: missing {args.file_b}", file=sys.stderr)
        return 2

    rels_a = _relations(args.file_a)
    rels_b = _relations(args.file_b)
    by_key_a = {_key(r): r for r in rels_a}
    by_key_b = {_key(r): r for r in rels_b}
    keys_all = set(by_key_a) | set(by_key_b)

    # ── 1. Totals ──────────────────────────────────────────────────────────
    print(f"\n=== Totals ===")
    print(f"  {args.label_a:>12}: {len(rels_a):>4} relations  ({args.file_a.name})")
    print(f"  {args.label_b:>12}: {len(rels_b):>4} relations  ({args.file_b.name})")
    print(f"  shared pairs : {len(set(by_key_a) & set(by_key_b)):>4}")
    print(f"  {args.label_a}-only    : {len(set(by_key_a) - set(by_key_b)):>4}")
    print(f"  {args.label_b}-only    : {len(set(by_key_b) - set(by_key_a)):>4}")

    # ── 2. Label distribution ──────────────────────────────────────────────
    la = Counter(r.get("relation_type") for r in rels_a)
    lb = Counter(r.get("relation_type") for r in rels_b)
    labels = sorted(set(la) | set(lb))
    print(f"\n=== Label distribution ===")
    print(f"  {'label':<18} {args.label_a:>10} {args.label_b:>10}  Δ")
    for lbl in labels:
        print(f"  {lbl:<18} {la.get(lbl, 0):>10} {lb.get(lbl, 0):>10}  "
              f"{lb.get(lbl, 0) - la.get(lbl, 0):+d}")

    # ── 3. Same- vs cross-paper breakdown ──────────────────────────────────
    sa = Counter(r.get("comparison_scope") for r in rels_a)
    sb = Counter(r.get("comparison_scope") for r in rels_b)
    print(f"\n=== Scope distribution ===")
    for s in sorted(set(sa) | set(sb)):
        print(f"  {s:<14} {args.label_a}={sa.get(s,0):>3}  {args.label_b}={sb.get(s,0):>3}  "
              f"Δ={sb.get(s,0) - sa.get(s,0):+d}")

    # ── 4. Per-pair label change matrix ────────────────────────────────────
    flip_counts: Counter = Counter()
    flip_examples: dict[tuple[str, str], list[tuple[dict, dict]]] = {}
    for k in sorted(by_key_a.keys() & by_key_b.keys()):
        ra, rb = by_key_a[k], by_key_b[k]
        ka, kb = ra.get("relation_type") or "—", rb.get("relation_type") or "—"
        if ka != kb:
            flip_counts[(ka, kb)] += 1
            flip_examples.setdefault((ka, kb), []).append((ra, rb))

    if flip_counts:
        print(f"\n=== Label flips on shared pairs ===")
        print(f"  ({args.label_a:>10}) → ({args.label_b:>10})   count")
        for (ka, kb), n in flip_counts.most_common():
            print(f"  {ka:>12} → {kb:>12}    {n}")
    else:
        print(f"\n=== Label flips on shared pairs: none ===")

    # ── 5. NLI score deltas on shared pairs ───────────────────────────────
    deltas_ab: list[float] = []
    deltas_ba: list[float] = []
    for k in by_key_a.keys() & by_key_b.keys():
        ra, rb = by_key_a[k], by_key_b[k]
        try:
            deltas_ab.append(float(rb["nli_score_a_to_b"]) - float(ra["nli_score_a_to_b"]))
            deltas_ba.append(float(rb["nli_score_b_to_a"]) - float(ra["nli_score_b_to_a"]))
        except (KeyError, TypeError, ValueError):
            continue
    if deltas_ab:
        all_abs = [abs(x) for x in deltas_ab + deltas_ba]
        all_abs.sort()
        n = len(all_abs)
        p50 = all_abs[n // 2]
        p95 = all_abs[min(n - 1, int(0.95 * n))]
        print(f"\n=== NLI-score deltas (B - A) over shared pairs ===")
        print(f"  n samples       : {len(deltas_ab) + len(deltas_ba)}")
        print(f"  mean |Δ|        : {statistics.mean(all_abs):.4f}")
        print(f"  median |Δ|      : {p50:.4f}")
        print(f"  p95 |Δ|         : {p95:.4f}")
        print(f"  max |Δ|         : {all_abs[-1]:.4f}")

    # ── 6. Spot-check examples per flip kind ──────────────────────────────
    if flip_counts:
        print(f"\n=== Spot-check examples (up to {args.examples} per flip kind) ===")
        for (ka, kb), pairs in flip_examples.items():
            print(f"\n--- {ka} → {kb}  ({len(pairs)} pairs) ---")
            for ra, rb in pairs[: args.examples]:
                print(f"  {ra.get('rule_id_a')} ↔ {ra.get('rule_id_b')}")
                print(f"    pmcid_a: {ra.get('pmcid_a')}")
                print(f"    pmcid_b: {ra.get('pmcid_b')}")
                print(f"    subject={ra.get('subject_entity')}  outcome={ra.get('outcome_entity')}")
                print(f"    predicate_a: {(ra.get('predicate_a') or '')[:120]!r}")
                print(f"    predicate_b: {(ra.get('predicate_b') or '')[:120]!r}")
                print(f"    NLI a→b: {args.label_a}={ra.get('nli_score_a_to_b'):.3f} "
                      f"{args.label_b}={rb.get('nli_score_a_to_b'):.3f}")
                print(f"    NLI b→a: {args.label_a}={ra.get('nli_score_b_to_a'):.3f} "
                      f"{args.label_b}={rb.get('nli_score_b_to_a'):.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
