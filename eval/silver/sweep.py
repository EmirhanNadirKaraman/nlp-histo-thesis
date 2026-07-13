"""
Sweep similarity threshold from 0.40 to 0.80 and report precision/recall/F1.

Embeddings are computed once and cached; all subsequent threshold steps run
without any API calls.

Usage:
  python -m eval.silver.sweep --silver eval/data/silver_findings_related15.jsonl \\
                               --pipeline eval/data/pipeline_findings_related15.jsonl
  python -m eval.silver.sweep --silver ... --pipeline ... --embedder gemini
  python -m eval.silver.sweep --silver ... --pipeline ... --embedder both
  python -m eval.silver.sweep --silver ... --pipeline ... --split dev --min-threshold 0.40 --max-threshold 0.80

NOTE: Use only the dev split for threshold selection. Never use --split test
for calibration — that contaminates the held-out evaluation set.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

from eval.silver.matching.embedders import GeminiEmbedder, OpenAIEmbedder
from eval.silver.data.jsonl_utils import read_jsonl
from eval.silver.matching.matcher import (
    DEFAULT_CACHE_PATH,
    DEFAULT_GEMINI_CACHE_PATH,
    STRICT_FIELDS,
    EmbeddingCache,
    compute_sim_matrix,
    make_embedding_cache,
    match_from_matrix,
)
from eval.silver.data.schemas import PipelineCaseOutput, SilverCaseResult
from eval.silver.data.split import filter_by_split

SILVER_PATH   = Path("eval/data/silver_findings_related15.jsonl")
PIPELINE_PATH = Path("eval/data/pipeline_findings_related15.jsonl")
REPORTS_DIR   = Path("eval/reports")


def _thresholds(min_t: float, max_t: float, step: float) -> list[float]:
    result = []
    t = min_t
    while t <= max_t + 1e-9:
        result.append(round(t, 4))
        t += step
    return result


def _strict_tp_for_results(match_results) -> float:
    total = 0.0
    for r in match_results:
        for pair in r.matched:
            has_important_mismatch = any(m.field in STRICT_FIELDS for m in pair.field_mismatches)
            total += 0.5 if has_important_mismatch else 1.0
    return total


def _metrics_at_threshold(
    cases: list[tuple],  # [(silver, pipeline, sim_matrix)]
    threshold: float,
) -> dict:
    all_results = []
    n_silver = 0
    n_pipeline = 0

    for silver, pipeline, sim in cases:
        result = match_from_matrix(silver, pipeline, sim, threshold)
        all_results.append(result)
        n_silver += len(silver.findings)
        n_pipeline += len(pipeline.findings)

    n_matched = sum(len(r.matched) for r in all_results)
    n_field_mismatches = sum(
        len(pair.field_mismatches) for r in all_results for pair in r.matched
    )

    precision = n_matched / n_pipeline if n_pipeline else 0.0
    recall = n_matched / n_silver if n_silver else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    stp = _strict_tp_for_results(all_results)
    sp = stp / n_pipeline if n_pipeline else 0.0
    sr = stp / n_silver if n_silver else 0.0
    strict_f1 = (2 * sp * sr / (sp + sr)) if (sp + sr) else 0.0

    return {
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "strict_f1": round(strict_f1, 4),
        "n_matched": n_matched,
        "n_silver": n_silver,
        "n_pipeline": n_pipeline,
        "n_field_mismatches": n_field_mismatches,
    }


def _run_sweep(
    label: str,
    embedder: object,
    cache: EmbeddingCache,
    common: list[str],
    silver_by_case: dict,
    pipeline_by_case: dict,
    thresholds: list[float],
    args,
) -> tuple[list[dict], dict]:
    """Compute sim matrices once, then sweep all thresholds. Returns (rows, best)."""
    logger.info("[%s] Computing similarity matrices…", label)
    cases_with_matrices: list[tuple] = []
    for case_id in common:
        silver = silver_by_case[case_id]
        pipeline = pipeline_by_case[case_id]
        sim, _, _ = compute_sim_matrix(silver, pipeline, embedder, cache)
        cases_with_matrices.append((silver, pipeline, sim))

    logger.info("[%s] All embeddings ready. Starting threshold sweep…", label)
    rows = []
    for t in thresholds:
        row = _metrics_at_threshold(cases_with_matrices, t)
        row["split"] = args.split
        row["seed"] = args.seed
        row["dev_fraction"] = args.dev_fraction
        row["embedder"] = label
        rows.append(row)

    best = max(rows, key=lambda r: r["f1"])
    return rows, best


def _print_table(label: str, rows: list[dict], best: dict, n_cases: int, args) -> None:
    print(f"\n{'─' * 80}")
    print(f"MAP Silver Eval — Threshold Sweep  [{label}]")
    print(f"Split: {args.split}  |  Cases: {n_cases}  |  "
          f"Seed: {args.seed}  |  Dev fraction: {args.dev_fraction}")
    print("NOTE: These are dev-set results only — do not report as final performance.")
    print(f"{'─' * 80}")
    print(f"{'Threshold':>10}  {'Precision':>9}  {'Recall':>7}  {'F1':>7}  "
          f"{'Strict F1':>9}  {'Matched':>7}  {'Mismatches':>10}")
    print(f"{'─' * 80}")
    for row in rows:
        marker = " ← best F1" if row["threshold"] == best["threshold"] else ""
        print(
            f"{row['threshold']:>10.2f}  {row['precision']:>9.3f}  "
            f"{row['recall']:>7.3f}  {row['f1']:>7.3f}  "
            f"{row['strict_f1']:>9.3f}  {row['n_matched']:>7}  "
            f"{row['n_field_mismatches']:>10}{marker}"
        )
    print(f"{'─' * 80}")
    print(f"\nBest F1 threshold: {best['threshold']:.2f}  "
          f"(F1={best['f1']:.3f}, Strict F1={best['strict_f1']:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep similarity threshold for MAP silver eval")
    parser.add_argument("--silver",   required=True,
                        help="Silver findings JSONL (REQUIRED — pick the set explicitly).")
    parser.add_argument("--pipeline", required=True,
                        help="Pipeline findings JSONL (REQUIRED — must match the silver set).")
    parser.add_argument("--reports",  default=str(REPORTS_DIR))
    parser.add_argument("--embed-cache",        default=str(DEFAULT_CACHE_PATH),
                        help="Cache path for OpenAI embeddings")
    parser.add_argument("--embed-cache-gemini", default=str(DEFAULT_GEMINI_CACHE_PATH),
                        help="Cache path for Gemini embeddings")
    parser.add_argument("--embedder", default="openai", choices=["openai", "gemini", "both"],
                        help="Embedding provider to use (default: openai)")
    parser.add_argument("--split",    default="dev", choices=["dev", "test", "all"],
                        help="Split to evaluate on. Default: dev.")
    parser.add_argument("--dev-fraction", type=float, default=0.8)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--min-threshold", type=float, default=0.40)
    parser.add_argument("--max-threshold", type=float, default=0.80)
    parser.add_argument("--step",     type=float, default=0.05)
    args = parser.parse_args()

    silver_path   = Path(args.silver)
    pipeline_path = Path(args.pipeline)
    reports_dir   = Path(args.reports)

    for p in (silver_path, pipeline_path):
        if not p.exists():
            print(f"File not found: {p}", file=sys.stderr)
            sys.exit(1)

    use_openai = args.embedder in ("openai", "both")
    use_gemini = args.embedder in ("gemini", "both")

    openai_key = os.environ.get("OPENAI_API_KEY") if use_openai else None
    gemini_key = os.environ.get("GOOGLE_API_KEY") if use_gemini else None

    if use_openai and not openai_key:
        print("OPENAI_API_KEY not set — required for OpenAI embeddings", file=sys.stderr)
        sys.exit(1)
    if use_gemini and not gemini_key:
        print("GOOGLE_API_KEY not set — required for Gemini embeddings", file=sys.stderr)
        sys.exit(1)

    if args.split == "test":
        print(
            "WARNING: Running sweep on test split contaminates held-out evaluation.\n"
            "Use --split dev for threshold calibration.",
            file=sys.stderr,
        )

    # Load and filter by split
    silver_by_case: dict[str, SilverCaseResult] = {}
    for rec in read_jsonl(silver_path, SilverCaseResult):
        silver_by_case[rec.case_id] = rec

    pipeline_by_case: dict[str, PipelineCaseOutput] = {}
    for rec in read_jsonl(pipeline_path, PipelineCaseOutput):
        pipeline_by_case[rec.case_id] = rec

    common = sorted(set(silver_by_case) & set(pipeline_by_case))

    class _Stub:
        def __init__(self, case_id): self.case_id = case_id
    filtered = filter_by_split(
        [_Stub(c) for c in common],
        args.split,
        dev_fraction=args.dev_fraction,
        seed=args.seed,
    )
    common = [s.case_id for s in filtered]

    if not common:
        print("No cases in selected split.", file=sys.stderr)
        sys.exit(1)

    evaluated_case_ids = sorted(common)
    logger.info("Split=%s  Cases=%d  (seed=%d, dev_fraction=%.2f)",
                args.split, len(common), args.seed, args.dev_fraction)

    thresholds = _thresholds(args.min_threshold, args.max_threshold, args.step)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    sweep_results: list[tuple[str, list[dict], dict]] = []  # (label, rows, best)

    if use_openai:
        from eval.silver.matching.embedders import OPENAI_MODEL
        cache = make_embedding_cache(Path(args.embed_cache), OPENAI_MODEL)
        embedder = OpenAIEmbedder(openai_key)  # type: ignore[arg-type]
        rows, best = _run_sweep(
            "openai", embedder, cache, common,
            silver_by_case, pipeline_by_case, thresholds, args,
        )
        sweep_results.append(("openai", rows, best))
        all_rows.extend(rows)

    if use_gemini:
        from eval.silver.matching.embedders import GEMINI_MODEL
        cache = make_embedding_cache(Path(args.embed_cache_gemini), GEMINI_MODEL)
        embedder = GeminiEmbedder(gemini_key)  # type: ignore[arg-type]
        rows, best = _run_sweep(
            "gemini", embedder, cache, common,
            silver_by_case, pipeline_by_case, thresholds, args,
        )
        sweep_results.append(("gemini", rows, best))
        all_rows.extend(rows)

    # Write CSV (all embedders combined)
    csv_path = reports_dir / f"sweep_{timestamp}.csv"
    fieldnames = [
        "embedder", "threshold", "precision", "recall", "f1", "strict_f1",
        "n_matched", "n_silver", "n_pipeline", "n_field_mismatches",
        "split", "seed", "dev_fraction",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    # Write provenance JSON
    import json
    prov_path = reports_dir / f"sweep_{timestamp}_provenance.json"
    prov_path.write_text(json.dumps({
        "timestamp": timestamp,
        "embedder": args.embedder,
        "split": args.split,
        "seed": args.seed,
        "dev_fraction": args.dev_fraction,
        "evaluated_case_ids": evaluated_case_ids,
        "silver_path": str(silver_path),
        "pipeline_path": str(pipeline_path),
        "threshold_range": [args.min_threshold, args.max_threshold, args.step],
    }, indent=2), encoding="utf-8")

    # Print tables
    for label, rows, best in sweep_results:
        _print_table(label, rows, best, len(common), args)

    print(f"\nSweep CSV    → {csv_path}")
    print(f"Provenance   → {prov_path}")


if __name__ == "__main__":
    main()
