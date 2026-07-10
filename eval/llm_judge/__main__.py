"""
eval.llm_judge.__main__ — CLI entry point.

Usage:
  python -m eval.llm_judge --mode sync --tests q1,q2 --n 5
  python -m eval.llm_judge --mode batch --tests q1,q2,q3,q5 --n 15
  python -m eval.llm_judge --batch-id msgbatch_... --poll-interval-seconds 60
  python -m eval.llm_judge --mode batch --no-submit --tests q1 --n 10
"""
from __future__ import annotations

import argparse
import logging

from .runner import RunConfig, run

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-judge evaluation harness (Phase 1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    parser.add_argument(
        "--mode", choices=["sync", "batch"], default="sync",
        help="Execution mode: sync (one-at-a-time) or batch (Anthropic Batch API)",
    )

    # Paper selection
    parser.add_argument("--n", type=int, default=15, help="Papers to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-words", type=int, default=1500, help="Minimum paper word count")
    parser.add_argument("--max-words", type=int, default=15000, help="Maximum paper word count")

    # Test selection
    parser.add_argument(
        "--tests", default="q1,q2,q3,q5",
        help="Comma-separated list of tests to run (q1,q2,q3,q5)",
    )

    # Per-test sampling
    parser.add_argument("--q1-findings", type=int, default=20, help="MAP findings per paper (Q1)")
    parser.add_argument("--q2-relations", type=int, default=10, help="Relations per paper (Q2)")
    parser.add_argument(
        "--q3-with-extraction-paragraphs", type=int, default=3,
        help="Paragraphs with extractions per paper (Q3)",
    )
    parser.add_argument(
        "--q3-zero-extraction-paragraphs", type=int, default=2,
        help="Zero-extraction paragraphs per paper (Q3)",
    )
    parser.add_argument("--q5-paragraphs", type=int, default=5, help="Paragraphs per paper (Q5)")

    # Batch API
    parser.add_argument(
        "--batch-id", type=str, default=None,
        help="Resume/retrieve an existing batch by ID",
    )
    parser.add_argument(
        "--poll-interval-seconds", type=int, default=30,
        help="Seconds between batch status polls",
    )
    parser.add_argument(
        "--no-submit", action="store_true",
        help="Build pending request file but do not submit (batch mode only)",
    )
    parser.add_argument(
        "--max-requests", type=int, default=None,
        help="Cap the number of API requests",
    )

    # Cache
    parser.add_argument(
        "--force-refresh-cache", action="store_true",
        help="Clear cache for current prompt/schema version before running",
    )

    # Dry run
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build all requests and log prompts to dry_run_requests.jsonl without calling the API",
    )

    # Q2 specific
    parser.add_argument(
        "--show-pipeline-label", action="store_true",
        help="Show pipeline label to Opus in Q2 (default: blind judging)",
    )

    args = parser.parse_args()

    cfg = RunConfig(
        mode=args.mode,
        tests={t.strip() for t in args.tests.split(",")},
        n=args.n,
        seed=args.seed,
        min_words=args.min_words,
        max_words=args.max_words,
        q1_findings=args.q1_findings,
        q2_relations=args.q2_relations,
        q3_with_extraction_paragraphs=args.q3_with_extraction_paragraphs,
        q3_zero_extraction_paragraphs=args.q3_zero_extraction_paragraphs,
        q5_paragraphs=args.q5_paragraphs,
        show_pipeline_label=args.show_pipeline_label,
        force_refresh_cache=args.force_refresh_cache,
        max_requests=args.max_requests,
        no_submit=args.no_submit,
        dry_run=args.dry_run,
        batch_id=args.batch_id,
        poll_interval=args.poll_interval_seconds,
    )

    run(cfg)


if __name__ == "__main__":
    main()
