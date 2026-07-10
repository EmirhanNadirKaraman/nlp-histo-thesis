"""CLI entry point for the offline calibration-set selector.

Example::

    python -m eval.paper_selection.run_select \\
        --output-version calibration_set_v1

Reads from the ingestion DB by default; pass ``--jsonl path/to/papers.jsonl``
to use the JSONL fallback. No LLM/API calls are made anywhere in this script.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .export import write_calibration_set
from .fingerprints import FingerprintConfig, build_fingerprints
from .ilp_selectors import (
    ILPConfig,
    pulp_available,
    select_calibration_set_ilp,
)
from .loaders import DBLoader, JSONLLoader
from .models import PaperFingerprint, SelectionResult
from .selectors import SelectionConfig, select_calibration_set

logger = logging.getLogger("eval.paper_selection.run_select")


# ── Validation ───────────────────────────────────────────────────────────────

def _validate_selection(
    result: SelectionResult,
    fingerprints: list[PaperFingerprint],
    *,
    cfg: SelectionConfig,
    allow_overlap: bool,
) -> list[str]:
    """Return a list of human-readable warnings/errors. Empty = clean."""
    issues: list[str] = []
    by_pmcid = {fp.pmcid: fp for fp in fingerprints}

    pmcids = result.all_pmcids()
    expected = cfg.k_related + cfg.k_diverse + cfg.k_hard
    if not allow_overlap and len(set(pmcids)) != expected:
        issues.append(
            f"expected {expected} unique PMCIDs, got {len(set(pmcids))} unique "
            f"out of {len(pmcids)} (allow_overlap={allow_overlap})"
        )

    for pmcid in pmcids:
        fp = by_pmcid.get(pmcid)
        if fp is None:
            issues.append(f"selected PMCID {pmcid!r} has no fingerprint")
            continue
        if fp.is_empty():
            issues.append(f"selected PMCID {pmcid} has zero text — skip or fix loader")
        bucket = result.rationale.get(pmcid, {}).get("bucket")
        if not fp.has_useful_entities() and bucket != "hard":
            issues.append(
                f"selected PMCID {pmcid} has zero useful entities "
                f"(bucket={bucket}); only hard-bucket may keep these intentionally"
            )

    # Mean relatedness sanity: related > diverse, hard > {related, diverse}
    from .metrics import Relatedness
    rel_metric = cfg.relatedness if isinstance(cfg.relatedness, Relatedness) else Relatedness()
    hardness_metric = cfg.hardness

    def _mean_pairwise(pmcids: list[str]) -> float:
        fps = [by_pmcid[p] for p in pmcids if p in by_pmcid]
        if len(fps) < 2:
            return 0.0
        scores = [
            rel_metric.score(fps[i], fps[j])
            for i in range(len(fps)) for j in range(i + 1, len(fps))
        ]
        return sum(scores) / len(scores)

    def _mean_hardness(pmcids: list[str]) -> float:
        fps = [by_pmcid[p] for p in pmcids if p in by_pmcid]
        if not fps:
            return 0.0
        return sum(hardness_metric.breakdown(fp).normalized_hardness for fp in fps) / len(fps)

    rel_mean = _mean_pairwise(result.related)
    div_mean = _mean_pairwise(result.diverse)
    # Pairwise relatedness only makes sense with ≥2 papers per bucket.
    if (len(result.related) >= 2 and len(result.diverse) >= 2
            and rel_mean <= div_mean):
        issues.append(
            f"related mean pairwise relatedness ({rel_mean:.3f}) "
            f"is not greater than diverse ({div_mean:.3f})"
        )

    hard_h = _mean_hardness(result.hard)
    rel_h  = _mean_hardness(result.related)
    div_h  = _mean_hardness(result.diverse)
    if result.hard and (hard_h <= rel_h or hard_h <= div_h):
        issues.append(
            f"hard mean hardness ({hard_h:.3f}) is not greater than "
            f"related ({rel_h:.3f}) / diverse ({div_h:.3f})"
        )

    return issues


# ── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m eval.paper_selection.run_select",
        description="Offline calibration-set selector (no LLM calls).",
    )
    p.add_argument("--output-version", default="calibration_set_v1",
                   help="Identifier used in output filenames (default: calibration_set_v1)")
    p.add_argument("--export-dir", default="configs/paper_selection",
                   help="Directory where YAML/JSON/CSV are written")
    p.add_argument("--jsonl", default=None,
                   help="Path to a JSONL papers export. If unset, uses the DB.")
    p.add_argument("--pmcid", action="append", default=None,
                   help="Restrict the candidate pool to these PMCIDs (repeatable).")

    p.add_argument("--k-related", type=int, default=5)
    p.add_argument("--k-diverse", type=int, default=5)
    p.add_argument("--k-hard",    type=int, default=5)

    p.add_argument("--max-sentences-related", type=int, default=350)
    p.add_argument("--max-sentences-diverse", type=int, default=350)
    p.add_argument("--min-sentences",         type=int, default=20)
    p.add_argument("--max-text-elements", type=int, default=None,
                   help="Drop any paper with more than N text elements before "
                        "fingerprinting. Applies to all buckets (including hard). "
                        "Useful for excluding textbook-chapter / thesis outliers.")
    p.add_argument("--allow-overlap", default="false",
                   help="Allow a paper to appear in more than one bucket (true|false).")

    p.add_argument("--strategy", choices=("greedy", "ilp"), default="greedy",
                   help="Selection strategy. 'ilp' requires PuLP (pip install pulp).")
    p.add_argument("--time-limit-seconds", type=float, default=None,
                   help="Per-bucket ILP solver time budget (default: ILPConfig "
                        "default — 30s). Pass 0 to disable. Ignored under --strategy greedy.")
    p.add_argument("--candidate-limit", type=int, default=None,
                   help="Override per-bucket candidate caps with a single value. "
                        "When omitted, bucket-specific defaults are used "
                        "(related=80, diverse=100, hard=120). Pass 0 to disable "
                        "the cap entirely (slower but exhaustive).")
    p.add_argument("--solver-verbose", action="store_true",
                   help="Stream CBC progress (LP root, cuts, node count, gap) to stdout.")
    p.add_argument("--ilp-fallback-greedy", action="store_true",
                   help="If PuLP is missing, silently fall back to greedy instead of erroring.")

    p.add_argument("--dry-run", action="store_true",
                   help="Compute and print the selection but do not write files.")
    p.add_argument("--log-level", default="INFO")
    return p


def _yes(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # ── Load papers ─────────────────────────────────────────────────────────
    if args.jsonl:
        loader = JSONLLoader(args.jsonl)
        logger.info("Loading from JSONL: %s", args.jsonl)
    else:
        loader = DBLoader()
        logger.info("Loading from database")

    pmcids = args.pmcid
    t_load = time.perf_counter()
    raw_papers = list(loader.iter_papers(pmcids))
    logger.info("Loaded %d papers in %.1fs", len(raw_papers),
                time.perf_counter() - t_load)
    if not raw_papers:
        logger.error("No papers loaded — aborting")
        return 2

    # ── Outlier filter ──────────────────────────────────────────────────────
    if args.max_text_elements is not None:
        kept = [p for p in raw_papers if len(p.text_elements) <= args.max_text_elements]
        dropped = len(raw_papers) - len(kept)
        logger.info("--max-text-elements %d: kept %d, dropped %d",
                    args.max_text_elements, len(kept), dropped)
        if not kept:
            logger.error("All papers filtered out by --max-text-elements — aborting")
            return 2
        raw_papers = kept

    # ── Fingerprint ─────────────────────────────────────────────────────────
    fp_cfg = FingerprintConfig()
    t_fp = time.perf_counter()
    fingerprints = build_fingerprints(raw_papers, fp_cfg)
    logger.info("Fingerprints built in %.1fs", time.perf_counter() - t_fp)

    # ── Select ──────────────────────────────────────────────────────────────
    sel_cfg = SelectionConfig(
        k_related=args.k_related,
        k_diverse=args.k_diverse,
        k_hard=args.k_hard,
        max_sentences_related=args.max_sentences_related,
        max_sentences_diverse=args.max_sentences_diverse,
        min_sentences=args.min_sentences,
    )
    allow_overlap = _yes(args.allow_overlap)

    t_sel = time.perf_counter()
    if args.strategy == "ilp":
        if not pulp_available() and not args.ilp_fallback_greedy:
            logger.error(
                "PuLP not installed. Install with `pip install pulp` "
                "or rerun with --strategy greedy / --ilp-fallback-greedy."
            )
            return 3
        ilp_cfg = ILPConfig(solver_verbose=args.solver_verbose)
        if args.candidate_limit is not None:
            ilp_cfg.candidate_limit = args.candidate_limit
        if args.time_limit_seconds is not None:
            ilp_cfg.time_limit_seconds = (
                args.time_limit_seconds if args.time_limit_seconds > 0 else None
            )
        if ilp_cfg.candidate_limit is None:
            limit_repr = (f"per-bucket(related={ilp_cfg.related_candidate_limit}, "
                          f"diverse={ilp_cfg.diverse_candidate_limit}, "
                          f"hard={ilp_cfg.hard_candidate_limit})")
        elif ilp_cfg.candidate_limit <= 0:
            limit_repr = "disabled"
        else:
            limit_repr = f"override={ilp_cfg.candidate_limit}"
        logger.info("Using ILP strategy (candidate_limit=%s, time_limit=%s, "
                    "allow_overlap=%s, solver_verbose=%s)",
                    limit_repr, ilp_cfg.time_limit_seconds, allow_overlap,
                    ilp_cfg.solver_verbose)
        result = select_calibration_set_ilp(
            fingerprints, config=sel_cfg, ilp_config=ilp_cfg,
            allow_overlap=allow_overlap,
            fallback_to_greedy=args.ilp_fallback_greedy,
        )
    else:
        logger.info("Using greedy strategy (allow_overlap=%s)", allow_overlap)
        result = select_calibration_set(fingerprints, config=sel_cfg, allow_overlap=allow_overlap)
    logger.info("Selection complete in %.1fs", time.perf_counter() - t_sel)

    # ── Print summary ───────────────────────────────────────────────────────
    print(f"\n=== Calibration set: {args.output_version} ===")
    for bucket in ("related", "diverse", "hard"):
        ids = getattr(result, bucket)
        print(f"  {bucket} ({len(ids)}): {ids}")

    issues = _validate_selection(result, fingerprints, cfg=sel_cfg, allow_overlap=allow_overlap)
    if issues:
        print("\nWarnings:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nAll sanity checks passed.")

    # ── Write outputs ───────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[dry-run] Skipping file writes.")
        return 0

    paths = write_calibration_set(
        result, fingerprints,
        version=args.output_version,
        export_dir=Path(args.export_dir),
    )
    print(f"\nWrote: {paths.yaml_path}")
    print(f"Wrote: {paths.json_path}")
    print(f"Wrote: {paths.csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
