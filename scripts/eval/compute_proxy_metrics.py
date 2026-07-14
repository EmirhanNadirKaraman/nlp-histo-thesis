"""Phase 1 — proxy metrics over frozen knowledge_extraction artifacts.

Hard scope: this script never imports NLI/LLM/embedding libraries and never
re-executes any pipeline stage. It reads the JSON, JSONL, and CSV files the
knowledge_extraction pipeline already writes and emits one row per pmcid plus one
``__aggregate__`` row.

Usage::

    python scripts/eval/compute_proxy_metrics.py \\
        --input out/summaries \\
        --out eval/results/proxy_metrics.csv

Outputs:
- ``<--out>`` CSV (one row per pmcid + one aggregate row).
- ``<--out parent>/proxy_metrics_aggregate.json`` with the aggregate row and
  raw ``status_counts``.
- ``<--out parent>/proxy_metrics.meta.json`` with ``git_commit``, ``created_at``,
  ``schema_version``, ``input_dir``, ``n_papers``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

# Support both ``python scripts/eval/compute_proxy_metrics.py`` (script form)
# and ``from compute_proxy_metrics import ...`` (test form). We don't add a
# ``scripts/__init__.py`` because other code in ``scripts/`` is loaded via
# importlib by name (e.g. tests/scripts/test_poll_interval_defaults.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _lib  # type: ignore  # noqa: E402

SCHEMA_VERSION = "v1"

SUCCESS_LIKE_STATUSES = frozenset({"success", "skipped", "ok", "completed"})

CSV_COLUMNS: list[str] = [
    # Identity
    "pmcid",
    "run_id",
    "status",
    "is_success_like",
    "pipeline_config_hash",
    # MAP-level counts
    "raw_map_finding_count",
    "grounded_finding_count",
    "grounded_finding_count_source",
    "grounding_rejection_rate",
    "grounding_rejection_rate_source",
    "grounding_threshold_used",
    "mean_grounding_score",
    "mean_grounding_score_source",
    # Downstream stage counts
    "normal_finding_count",
    "non_groupable_total",
    "canonical_rule_count",
    "final_rule_count",
    # Relations
    "relations_total",
    "support_edge_count",
    "contradiction_edge_count",
    "scope_qualify_edge_count",
    "unclear_no_direction_rate",
    # Dedup
    "duplicate_rate",
    "duplicate_rate_source",
    # Final-rule slices
    "top_k_final_rule_count_score_ge_0_5",
    "top_k_final_rule_count_top10",
    # Cascade
    "l1_chunks",
    "l2_chunks",
    "l3_chunks",
    "polarity_conflict_chunks",
    "cache_hits",
    "cache_hits_source",
    "cache_misses",
    "cache_misses_source",
    # Cost
    "total_input_tokens",
    "total_output_tokens",
    "token_count_source",
    "estimated_total_cost_usd",
    "estimated_total_cost_usd_source",
    # Timing
    "run_duration_s",
]

SOURCE_COLUMNS: frozenset[str] = frozenset(
    c for c in CSV_COLUMNS if c.endswith("_source")
)

NUMERIC_SUM_COLUMNS: frozenset[str] = frozenset(
    {
        "raw_map_finding_count",
        "grounded_finding_count",
        "normal_finding_count",
        "non_groupable_total",
        "canonical_rule_count",
        "final_rule_count",
        "relations_total",
        "support_edge_count",
        "contradiction_edge_count",
        "scope_qualify_edge_count",
        "top_k_final_rule_count_score_ge_0_5",
        "top_k_final_rule_count_top10",
        "l1_chunks",
        "l2_chunks",
        "l3_chunks",
        "polarity_conflict_chunks",
        "cache_hits",
        "cache_misses",
        "total_input_tokens",
        "total_output_tokens",
        "estimated_total_cost_usd",
        "run_duration_s",
    }
)

NUMERIC_MEAN_COLUMNS: frozenset[str] = frozenset(
    {
        "grounding_rejection_rate",
        "mean_grounding_score",
        "unclear_no_direction_rate",
        "duplicate_rate",
        "grounding_threshold_used",
    }
)

logger = logging.getLogger("compute_proxy_metrics")


# ---------------------------------------------------------------------------
# Per-paper computation
# ---------------------------------------------------------------------------


def compute_per_paper_row(
    summary_path: Path,
    chunk_summary_by_pmcid: dict[str, list[dict[str, str]]],
    cascade_decisions_dir: Path,
    cost_by_paper: dict[str, dict[str, float]],
    runs_by_pmcid: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary = _lib.load_summary(summary_path)
    pmcid = summary.get("pmcid") or _lib.coerce_pmcid_from_path(summary_path)
    rejection_summary = summary.get("rejection_summary") or {}
    map_chunks = (summary.get("audit_trail") or {}).get("map_chunks") or []
    canonical_rules = summary.get("canonical_rules") or []
    final_rules = summary.get("final_rules") or []
    relations = summary.get("relations") or []

    row: dict[str, Any] = {col: float("nan") for col in CSV_COLUMNS}
    for col in SOURCE_COLUMNS:
        row[col] = "missing"

    # Identity ---------------------------------------------------------------
    row["pmcid"] = pmcid
    row["run_id"] = summary.get("run_id", "")
    raw_status = summary.get("status", "") or ""
    row["status"] = raw_status
    row["is_success_like"] = int(raw_status in SUCCESS_LIKE_STATUSES)
    row["pipeline_config_hash"] = summary.get("pipeline_config_hash", "")

    # MAP-level counts -------------------------------------------------------
    raw_count = rejection_summary.get("map_findings_total")
    if raw_count is None:
        raw_count = sum(len(c.get("findings") or []) for c in map_chunks)
    row["raw_map_finding_count"] = int(raw_count or 0)

    rejected = rejection_summary.get("map_grounding_rejected")
    if raw_count is not None and rejected is not None:
        row["grounded_finding_count"] = int(raw_count) - int(rejected)
        row["grounded_finding_count_source"] = "rejection_summary"
    else:
        row["grounded_finding_count"] = float("nan")
        row["grounded_finding_count_source"] = "missing"

    grr = rejection_summary.get("grounding_rejection_rate")
    if grr is not None:
        row["grounding_rejection_rate"] = float(grr)
        row["grounding_rejection_rate_source"] = "rejection_summary"

    row["grounding_threshold_used"] = (
        rejection_summary.get("grounding_threshold")
        if rejection_summary.get("grounding_threshold") is not None
        else float("nan")
    )

    scores: list[float] = []
    for chunk in map_chunks:
        for finding in chunk.get("findings") or []:
            gs = finding.get("grounding_score")
            if isinstance(gs, (int, float)):
                scores.append(float(gs))
    if scores:
        row["mean_grounding_score"] = sum(scores) / len(scores)
        row["mean_grounding_score_source"] = "audit_trail"

    # Downstream counts ------------------------------------------------------
    row["normal_finding_count"] = int(rejection_summary.get("normal_findings_total") or 0)
    row["non_groupable_total"] = int(rejection_summary.get("non_groupable_total") or 0)
    row["canonical_rule_count"] = len(canonical_rules)
    row["final_rule_count"] = len(final_rules)

    # Relations --------------------------------------------------------------
    rel_types = Counter((r.get("relation_type") or "").upper() for r in relations)
    row["relations_total"] = len(relations)
    row["support_edge_count"] = int(rel_types.get("SUPPORT", 0))
    row["contradiction_edge_count"] = int(rel_types.get("CONTRADICT", 0))
    row["scope_qualify_edge_count"] = int(rel_types.get("SCOPE_QUALIFY", 0))

    if canonical_rules:
        n_unclear = sum(
            1
            for r in canonical_rules
            if (r.get("direction") or "").lower() in {"unclear", "no_direction"}
        )
        row["unclear_no_direction_rate"] = n_unclear / len(canonical_rules)
    else:
        row["unclear_no_direction_rate"] = float("nan")

    # Duplicates: not present in current rejection_summary schema. Mark missing.
    row["duplicate_rate"] = float("nan")
    row["duplicate_rate_source"] = "missing"

    # Final-rule slices ------------------------------------------------------
    row["top_k_final_rule_count_score_ge_0_5"] = sum(
        1 for r in final_rules if isinstance(r.get("final_score"), (int, float)) and r["final_score"] >= 0.5
    )
    row["top_k_final_rule_count_top10"] = min(10, len(final_rules))

    # Cascade decisions ------------------------------------------------------
    l1 = l2 = l3 = 0
    polarity_conflicts = 0
    cascade_seen = False
    for rec in _lib.iter_cascade_decisions(cascade_decisions_dir, pmcid):
        cascade_seen = True
        level = (rec.get("level") or "").lower()
        if level == "l1":
            l1 += 1
        elif level == "l2":
            l2 += 1
        elif level == "l3":
            l3 += 1
        for code in rec.get("reason_codes") or []:
            if "polarity" in str(code).lower():
                polarity_conflicts += 1
                break
    if cascade_seen:
        row["l1_chunks"] = l1
        row["l2_chunks"] = l2
        row["l3_chunks"] = l3
        row["polarity_conflict_chunks"] = polarity_conflicts
    else:
        row["l1_chunks"] = float("nan")
        row["l2_chunks"] = float("nan")
        row["l3_chunks"] = float("nan")
        row["polarity_conflict_chunks"] = float("nan")

    # Chunk-summary cache stats ----------------------------------------------
    chunk_rows = chunk_summary_by_pmcid.get(pmcid, [])
    if chunk_rows:
        hits = sum(1 for r in chunk_rows if _to_bool(r.get("cache_hit")))
        misses = sum(1 for r in chunk_rows if _to_bool(r.get("cache_hit")) is False)
        row["cache_hits"] = hits
        row["cache_hits_source"] = "chunk_summary"
        row["cache_misses"] = misses
        row["cache_misses_source"] = "chunk_summary"
    else:
        row["cache_hits"] = float("nan")
        row["cache_hits_source"] = "missing"
        row["cache_misses"] = float("nan")
        row["cache_misses_source"] = "missing"

    # Cost -------------------------------------------------------------------
    cost_bucket = cost_by_paper.get(pmcid)
    if cost_bucket is not None:
        row["total_input_tokens"] = int(cost_bucket.get("input_tokens", 0))
        row["total_output_tokens"] = int(cost_bucket.get("output_tokens", 0))
        row["token_count_source"] = "cost_jsonl"
        row["estimated_total_cost_usd"] = float(cost_bucket.get("cost_usd", 0.0))
        row["estimated_total_cost_usd_source"] = "cost_jsonl"
    else:
        row["total_input_tokens"] = float("nan")
        row["total_output_tokens"] = float("nan")
        row["token_count_source"] = "missing"
        row["estimated_total_cost_usd"] = float("nan")
        row["estimated_total_cost_usd_source"] = "missing"

    # Timing -----------------------------------------------------------------
    run_rec = runs_by_pmcid.get(pmcid) or runs_by_pmcid.get(_strip_run_suffix(pmcid))
    if run_rec is not None and isinstance(run_rec.get("duration_s"), (int, float)):
        row["run_duration_s"] = float(run_rec["duration_s"])
    else:
        row["run_duration_s"] = float("nan")

    _log_missing(pmcid, row)
    return row


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def _strip_run_suffix(pmcid: str) -> str:
    """Best-effort fallback: ``PMC123_main`` → ``PMC123`` if exact match fails."""
    if "_" in pmcid:
        return pmcid.split("_", 1)[0]
    return pmcid


def _log_missing(pmcid: str, row: dict[str, Any]) -> None:
    missing = [c.removesuffix("_source") for c in SOURCE_COLUMNS if row.get(c) == "missing"]
    if missing:
        logger.info("[%s] metrics missing: %s", pmcid, ", ".join(sorted(missing)))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def build_aggregate_row(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (csv_row, nested_json)."""
    csv_row: dict[str, Any] = {col: "" for col in CSV_COLUMNS}
    csv_row["pmcid"] = "__aggregate__"

    pipeline_hashes = {r.get("pipeline_config_hash", "") for r in rows if r.get("pipeline_config_hash")}
    if len(pipeline_hashes) == 1:
        csv_row["pipeline_config_hash"] = next(iter(pipeline_hashes))
    elif len(pipeline_hashes) > 1:
        csv_row["pipeline_config_hash"] = "MIXED"

    statuses = [r.get("status", "") for r in rows]
    status_counts = dict(Counter(s for s in statuses if s))
    csv_row["status"] = ""
    n_success_like = sum(1 for r in rows if r.get("is_success_like") == 1)
    csv_row["is_success_like"] = n_success_like

    sum_stats: dict[str, float] = {}
    for col in NUMERIC_SUM_COLUMNS:
        values = [_to_float(r.get(col)) for r in rows]
        finite = [v for v in values if v is not None]
        if finite:
            total = sum(finite)
            sum_stats[col] = total
            csv_row[col] = _csv_render(total, col)
        else:
            sum_stats[col] = float("nan")
            csv_row[col] = ""

    mean_stats: dict[str, dict[str, float]] = {}
    for col in NUMERIC_MEAN_COLUMNS:
        values = [_to_float(r.get(col)) for r in rows]
        finite = [v for v in values if v is not None]
        if finite:
            mean_stats[col] = {
                "mean": mean(finite),
                "median": median(finite),
                "p90": _percentile(finite, 0.90),
                "n": len(finite),
            }
            csv_row[col] = _csv_render(mean(finite), col)
        else:
            mean_stats[col] = {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "n": 0}
            csv_row[col] = ""

    for col in SOURCE_COLUMNS:
        seen = sorted({str(r.get(col, "")) for r in rows if r.get(col)})
        csv_row[col] = "|".join(seen)

    aggregate_json: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "n_papers": len(rows),
        "n_success_like": n_success_like,
        "status_counts": status_counts,
        "pipeline_config_hashes": sorted(pipeline_hashes),
        "sums": sum_stats,
        "rate_stats": mean_stats,
        "source_distribution": {
            col: dict(Counter(str(r.get(col, "")) for r in rows))
            for col in SOURCE_COLUMNS
        },
    }
    return csv_row, aggregate_json


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def _csv_render(value: Any, col: str) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if col in NUMERIC_SUM_COLUMNS and col not in {"estimated_total_cost_usd", "run_duration_s"}:
            return int(round(value))
    return value


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_csv(rows: list[dict[str, Any]], aggregate_row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _render_csv_cell(row.get(col)) for col in CSV_COLUMNS})
        writer.writerow({col: aggregate_row.get(col, "") for col in CSV_COLUMNS})


def _render_csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
    return value


def write_aggregate_json(aggregate_json: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(aggregate_json), f, indent=2, allow_nan=False)
        f.write("\n")


def write_meta_json(meta: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(meta), f, indent=2, allow_nan=False)
        f.write("\n")


def _sanitize_for_json(value: Any) -> Any:
    """Recursively replace NaN/Inf floats with ``None`` so JSON stays valid."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("out/summaries"),
        help="Path to the summaries root (default: out/summaries).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval/results/proxy_metrics.csv"),
        help="CSV output path (default: eval/results/proxy_metrics.csv).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
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
    traces_dir = args.input / "traces"
    cascade_decisions_dir = args.input / "cascade_decisions"
    cost_root = args.input / "cost"

    if not summaries_dir.exists():
        logger.error("Summaries directory not found: %s", summaries_dir)
        return 1

    chunk_summary_by_pmcid = _lib.load_chunk_summary_csv(traces_dir / "chunk_summary.csv")
    cost_by_paper = _lib.aggregate_cost_by_paper(cost_root)
    runs_by_pmcid = _lib.load_runs_by_pmcid(traces_dir / "runs.jsonl")

    rows: list[dict[str, Any]] = []
    for summary_path in _lib.iter_summary_paths(summaries_dir):
        try:
            row = compute_per_paper_row(
                summary_path,
                chunk_summary_by_pmcid,
                cascade_decisions_dir,
                cost_by_paper,
                runs_by_pmcid,
            )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping %s: %s", summary_path.name, exc)
            continue
        rows.append(row)

    if not rows:
        logger.error("No summary files found in %s", summaries_dir)
        return 1

    aggregate_row, aggregate_json = build_aggregate_row(rows)

    out_csv = args.out
    out_dir = out_csv.parent
    out_aggregate = out_dir / "proxy_metrics_aggregate.json"
    out_meta = out_dir / "proxy_metrics.meta.json"

    write_csv(rows, aggregate_row, out_csv)
    write_aggregate_json(aggregate_json, out_aggregate)
    write_meta_json(
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": _lib.now_iso_utc(),
            "git_commit": _lib.get_git_commit(),
            "input_dir": str(args.input),
            "n_papers": len(rows),
            "csv_path": str(out_csv),
            "aggregate_json_path": str(out_aggregate),
        },
        out_meta,
    )

    logger.info(
        "Wrote %d per-paper rows + aggregate to %s", len(rows), out_csv
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
