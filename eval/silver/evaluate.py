"""
Match pipeline findings against silver findings, compute metrics, write report.

Usage:
  python -m eval.silver.evaluate --silver eval/data/silver_findings_related15.jsonl \\
                                  --pipeline eval/data/pipeline_findings_related15.jsonl
  python -m eval.silver.evaluate --silver ... --pipeline ... --split test --inspect
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

from eval.silver.embedders import GeminiEmbedder, OpenAIEmbedder
from eval.silver.inspect import write_inspection_report
from eval.silver.jsonl_utils import read_jsonl, write_jsonl
from eval.silver.matcher import (
    DEFAULT_CACHE_PATH,
    DEFAULT_GEMINI_CACHE_PATH,
    EMBEDDING_MODEL,
    GEMINI_EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD,
    EmbeddingCache,
    compute_metrics,
    make_embedding_cache,
    compute_sim_matrix,
    match_from_matrix,
    MATCHERS,
)
from eval.silver.schemas import (
    EvalMetrics,
    MatchResult,
    PipelineCaseOutput,
    SilverCaseResult,
    SourceCase,
)
from eval.silver.split import filter_by_split

SILVER_PATH   = Path("eval/data/silver_findings_related15.jsonl")
PIPELINE_PATH = Path("eval/data/pipeline_findings_related15.jsonl")
SOURCE_PATH   = Path("eval/data/source_cases_related15.jsonl")
REPORTS_DIR   = Path("eval/reports")


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline vs silver findings")
    parser.add_argument("--silver",   required=True,
                        help="Silver findings JSONL (REQUIRED — pick the set explicitly, "
                             "e.g. silver_findings_heldout15.jsonl; no default so an eval "
                             "can't silently run on the calibration set).")
    parser.add_argument("--pipeline", required=True,
                        help="Pipeline findings JSONL (REQUIRED — must match the silver set, "
                             "e.g. pipeline_findings_heldout15.jsonl).")
    parser.add_argument("--source",   default=str(SOURCE_PATH),
                        help="Source cases JSONL (required with --inspect)")
    parser.add_argument("--reports",  default=str(REPORTS_DIR))
    parser.add_argument("--embed-cache", default=None,
                        help="Embedding cache path (default depends on --embedder)")
    parser.add_argument("--embedder", default="openai", choices=["openai", "gemini"],
                        help="Embedding provider (default: openai)")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD,
                        help=f"Similarity threshold (default: {SIMILARITY_THRESHOLD})")
    parser.add_argument("--matcher", default="optimal", choices=sorted(MATCHERS),
                        help="One-to-one matching strategy. Default: optimal (Hungarian) — the "
                             "PRIMARY matcher the calibration sweep selects on, so the held-out "
                             "score stays consistent with calibration. Pass --matcher greedy for "
                             "the secondary diagnostic.")
    parser.add_argument("--split", default="all", choices=["dev", "test", "all"],
                        help="Case split to evaluate. Default: all.")
    parser.add_argument("--dev-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect", action="store_true",
                        help="Write detailed inspection Markdown + CSVs")
    args = parser.parse_args()

    silver_path   = Path(args.silver)
    pipeline_path = Path(args.pipeline)
    reports_dir   = Path(args.reports)

    for p in (silver_path, pipeline_path):
        if not p.exists():
            print(f"File not found: {p}", file=sys.stderr)
            sys.exit(1)

    if args.embedder == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("GOOGLE_API_KEY not set — required for Gemini embeddings", file=sys.stderr)
            sys.exit(1)
        embedder = GeminiEmbedder(api_key)
        embed_model = GEMINI_EMBEDDING_MODEL
        default_cache = DEFAULT_GEMINI_CACHE_PATH
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY not set — required for OpenAI embeddings", file=sys.stderr)
            sys.exit(1)
        embedder = OpenAIEmbedder(api_key)
        embed_model = EMBEDDING_MODEL
        default_cache = DEFAULT_CACHE_PATH

    embed_cache_path = Path(args.embed_cache) if args.embed_cache else default_cache

    # Load data
    silver_by_case: dict[str, SilverCaseResult] = {}
    for rec in read_jsonl(silver_path, SilverCaseResult):
        silver_by_case[rec.case_id] = rec  # latest per case_id wins

    pipeline_by_case: dict[str, PipelineCaseOutput] = {}
    for rec in read_jsonl(pipeline_path, PipelineCaseOutput):
        pipeline_by_case[rec.case_id] = rec

    common = set(silver_by_case) & set(pipeline_by_case)
    only_silver = set(silver_by_case) - set(pipeline_by_case)
    only_pipeline = set(pipeline_by_case) - set(silver_by_case)

    if only_silver:
        logger.warning("%d case(s) have silver but no pipeline output: %s",
                       len(only_silver), sorted(only_silver)[:5])
    if only_pipeline:
        logger.warning("%d case(s) have pipeline output but no silver: %s",
                       len(only_pipeline), sorted(only_pipeline)[:5])

    # Apply split filter
    class _Stub:
        def __init__(self, case_id): self.case_id = case_id
    filtered = filter_by_split(
        [_Stub(c) for c in common],
        args.split,
        dev_fraction=args.dev_fraction,
        seed=args.seed,
    )
    common = sorted(s.case_id for s in filtered)

    if not common:
        print("No cases in selected split.", file=sys.stderr)
        sys.exit(1)

    logger.info("Split=%s  Evaluating %d common case(s)…", args.split, len(common))

    # Embedding cache
    cache = make_embedding_cache(embed_cache_path, embed_model)

    # Match
    match_fn = MATCHERS[args.matcher]
    match_results: list[MatchResult] = []
    for case_id in common:
        silver = silver_by_case[case_id]
        pipeline = pipeline_by_case[case_id]
        sim, _, _ = compute_sim_matrix(silver, pipeline, embedder, cache)
        result = match_from_matrix(silver, pipeline, sim, args.threshold, match_fn=match_fn)
        match_results.append(result)
        logger.info("  %s — matched %d / %d silver (pipeline has %d)",
                    case_id, len(result.matched),
                    len(silver.findings), len(pipeline.findings))

    silver_list = [silver_by_case[c] for c in common]
    pipeline_list = [pipeline_by_case[c] for c in common]

    prompt_version = silver_list[0].prompt_version if silver_list else "unknown"
    model = silver_list[0].model if silver_list else "unknown"

    metrics = compute_metrics(
        match_results, silver_list, pipeline_list, prompt_version, model,
        split=args.split,
        split_seed=args.seed,
        dev_fraction=args.dev_fraction,
        threshold=args.threshold,
    )

    # Write outputs
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    reports_dir.mkdir(parents=True, exist_ok=True)

    match_path = reports_dir / f"match_results_{timestamp}.jsonl"
    write_jsonl(match_path, match_results)

    metrics_path = reports_dir / f"metrics_{timestamp}.json"
    metrics_path.write_text(metrics.model_dump_json(indent=2))

    # Inspection report
    if args.inspect:
        source_path = Path(args.source)
        source_by_case: dict[str, SourceCase] = {}
        if source_path.exists():
            for rec in read_jsonl(source_path, SourceCase):
                source_by_case[rec.case_id] = rec
        else:
            logger.warning("Source file not found: %s — inspection will omit source text", source_path)

        write_inspection_report(
            match_results=match_results,
            silver_by_case={c: silver_by_case[c] for c in common},
            pipeline_by_case={c: pipeline_by_case[c] for c in common},
            source_by_case=source_by_case,
            reports_dir=reports_dir,
            timestamp=timestamp,
        )

    # Print summary
    print(f"\n{'─' * 60}")
    print(f"Evaluation results  ({timestamp})")
    print(f"Split: {args.split}  |  Threshold: {args.threshold}  |  "
          f"Cases: {metrics.n_cases}")
    print(f"{'─' * 60}")
    print(f"Silver findings:    {metrics.n_silver_findings}")
    print(f"Pipeline findings:  {metrics.n_pipeline_findings}")
    print(f"Matched pairs:      {metrics.n_matched}")
    print(f"Field mismatches:   {metrics.n_field_mismatches}")
    print(f"{'─' * 60}")
    print(f"Precision:          {metrics.precision:.3f}")
    print(f"Recall:             {metrics.recall:.3f}")
    print(f"F1 (loose):         {metrics.f1:.3f}")
    print(f"F1 (strict):        {metrics.strict_f1:.3f}")
    print(f"Avg similarity:     {metrics.avg_similarity:.3f}")
    print(f"{'─' * 60}")
    print(f"\nMatch results → {match_path}")
    print(f"Metrics      → {metrics_path}")


if __name__ == "__main__":
    main()
