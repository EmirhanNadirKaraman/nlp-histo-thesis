"""Layer A frozen-artifact sweep over ``grounding_threshold``.

Reads existing summarisation artifacts under ``--input`` (default
``out/summaries``), gathers every persisted finding (surviving +
grounding-rejected) with its NLI ``grounding_score``, and simulates
retention at each candidate threshold. No API calls, no NLI re-inference,
no full-pipeline replay.

Usage::

    python eval/sweeps/grounding.py \\
        [--input out/summaries] \\
        [--thresholds 0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95] \\
        [--out-csv eval/results/grounding_sweep.csv] \\
        [--out-md  eval/results/grounding_sweep.md] \\
        [--strict-single-config]

See ``eval/sweeps/README.md`` for the Layer A vs Layer B distinction.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

# Support both ``python eval/sweeps/grounding.py`` (script form) and
# ``from eval.sweeps.grounding import ...`` (test form).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _lib  # type: ignore  # noqa: E402

SWEEP_NAME = "grounding_threshold"
DEFAULT_INPUT = Path("out/summaries")
# Outputs default into the generated tree, not into eval/results/ (B-108).
# `eval/results/grounding_sweep.md` is a tracked thesis snapshot: 5 papers, run
# grounding_compare_calv1_runB_20260516T163007, config hash 149023b87374cbc2.
# This sweep has no frozen input pin -- it reads whatever is in out/summaries --
# so a default run tracks the corpus, not that snapshot, and writing there
# rewrote a published table with numbers matching no published result. Pass
# --out-md explicitly to regenerate the snapshot deliberately.
DEFAULT_OUT_CSV = Path("out/eval_sweeps/grounding_sweep.csv")
DEFAULT_OUT_MD = Path("out/eval_sweeps/grounding_sweep.md")
DEFAULT_THRESHOLDS = "0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95"

logger = logging.getLogger("eval.sweeps.grounding")


def parse_thresholds(arg: str) -> list[float]:
    """Parse ``"0.3,0.5,0.7"`` → ``[0.3, 0.5, 0.7]``. Raise on any bad token."""
    raw_tokens = [t.strip() for t in arg.split(",") if t.strip()]
    if not raw_tokens:
        raise argparse.ArgumentTypeError("--thresholds must list at least one value")
    out: list[float] = []
    for tok in raw_tokens:
        try:
            out.append(float(tok))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--thresholds: invalid float {tok!r}"
            ) from exc
    return sorted(out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--thresholds", type=parse_thresholds, default=parse_thresholds(DEFAULT_THRESHOLDS),
        help=f"Comma-separated thresholds (default: {DEFAULT_THRESHOLDS})",
    )
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument(
        "--strict-single-config", action="store_true",
        help="Exit rc=2 if the input contains multiple pipeline_config_hash "
             "values or multiple run_ids. By default the sweep runs and "
             "appends a 'mixed configurations' warning to the report.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    summaries_dir = args.input / "summaries"
    if not summaries_dir.exists():
        logger.error("Summaries directory not found: %s", summaries_dir)
        return 1

    records, gather_warnings = _lib.gather_findings_with_grounding(summaries_dir)
    if not records:
        logger.error("No findings gathered from %s — nothing to sweep.", summaries_dir)
        return 1

    metadata = _lib.build_sweep_metadata(
        sweep_name=SWEEP_NAME,
        summaries_dir=summaries_dir,
        input_dir=args.input,
        out_csv=args.out_csv,
        out_md=args.out_md,
        thresholds=args.thresholds,
    )

    warnings = list(gather_warnings)
    mixed_hashes = len(metadata.pipeline_config_hashes) > 1
    mixed_runs = len(metadata.run_ids) > 1
    if mixed_hashes:
        warnings.append(
            "Multiple pipeline_config_hashes detected in the input: "
            + ", ".join(metadata.pipeline_config_hashes)
            + ". The retention table mixes producer configurations — "
            "interpret deltas with care or re-run with --strict-single-config."
        )
    if mixed_runs:
        warnings.append(
            "Multiple run_ids detected in the input: "
            + ", ".join(metadata.run_ids)
            + ". This usually means artifacts from more than one run are "
            "co-mingled in --input."
        )
    if (mixed_hashes or mixed_runs) and args.strict_single_config:
        for w in warnings:
            logger.error(w)
        logger.error(
            "Exiting with rc=2 because --strict-single-config was set."
        )
        return 2

    # Sanity: all-missing-score corpus.
    n_scored = sum(1 for r in records if r.grounding_score is not None)
    if n_scored == 0:
        warnings.insert(
            0,
            "All findings missing grounding_score — was the producer run "
            "executed with grounding enabled or with NLI scores persisted? "
            "The retention table will report 0/0/0 rows.",
        )

    sweep_rows = _lib.simulate_threshold_sweep(records, args.thresholds)

    per_paper_detail = _build_per_paper_detail(records)
    producer_thresholds = [r.producer_threshold for r in records]

    _lib.write_sweep_csv(sweep_rows, args.out_csv, metadata)
    _lib.write_sweep_markdown(
        sweep_rows, args.out_md, metadata,
        warnings=warnings,
        per_paper_detail=per_paper_detail,
        producer_thresholds=producer_thresholds,
    )

    n_missing = len(records) - n_scored
    pct_missing = (n_missing / len(records) * 100.0) if records else 0.0
    logger.info(
        "Sweep over %d thresholds, %d findings (%d scored, %d missing, "
        "%.2f%% missing) across %d papers, %d pipeline_config_hashes — "
        "wrote %s and %s",
        len(args.thresholds), len(records), n_scored, n_missing, pct_missing,
        len(metadata.pmcids), len(metadata.pipeline_config_hashes),
        args.out_csv, args.out_md,
    )
    return 0


def _build_per_paper_detail(records: list[_lib.FindingRecord]) -> list[dict]:
    by_pmcid_total: Counter = Counter()
    by_pmcid_scored: Counter = Counter()
    by_pmcid_rejected: Counter = Counter()
    by_pmcid_producer: dict[str, float | None] = {}
    for r in records:
        by_pmcid_total[r.pmcid] += 1
        if r.grounding_score is not None:
            by_pmcid_scored[r.pmcid] += 1
        if not r.kept_at_producer_threshold:
            by_pmcid_rejected[r.pmcid] += 1
        by_pmcid_producer.setdefault(r.pmcid, r.producer_threshold)
    return [
        {
            "pmcid": pmcid,
            "total": by_pmcid_total[pmcid],
            "scored": by_pmcid_scored[pmcid],
            "grounding_rejected": by_pmcid_rejected[pmcid],
            "producer_threshold": by_pmcid_producer.get(pmcid),
        }
        for pmcid in sorted(by_pmcid_total)
    ]


if __name__ == "__main__":
    raise SystemExit(main())
