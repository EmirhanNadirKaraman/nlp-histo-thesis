"""
JSONL serialization and optional CSV export helpers.

JSONL is the canonical trace format; CSV is a derived convenience export.

Primary outputs
---------------
``{trace_dir}/runs.jsonl``    — one record per paper run
``{trace_dir}/chunks.jsonl``  — one record per MAP chunk

Derived CSV outputs (not the source of truth)
----------------------------------------------
``{trace_dir}/run_summary.csv``     — key run-level metrics, one row per run
``{trace_dir}/chunk_summary.csv``   — key chunk-level metrics, one row per chunk
``{trace_dir}/pairwise_scores.csv`` — one row per voter pair per chunk
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .collector import TraceCollector


# ── Primary write helpers ──────────────────────────────────────────────────────


def append_jsonl(record: dict[str, Any], path: Path) -> None:
    """Append one JSON record as a newline-delimited line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def flush_collector(
    collector: TraceCollector,
    trace_dir: Path,
    status: str = "success",
    error: str | None = None,
) -> None:
    """
    Finalize the collector and append its records to the canonical JSONL files.

    Parameters
    ----------
    collector:
        A populated TraceCollector for one run.
    trace_dir:
        Directory that will contain ``runs.jsonl`` and ``chunks.jsonl``.
    status:
        "success", "error", or "skipped".
    error:
        Error string, populated only when status=="error".
    """
    run_trace = collector.finalize(status, error)
    append_jsonl(run_trace.to_dict(), trace_dir / "runs.jsonl")
    for ct in collector.chunk_traces:
        append_jsonl(ct.to_dict(), trace_dir / "chunks.jsonl")


# ── CSV flatten helpers ────────────────────────────────────────────────────────


def export_run_summary_csv(runs_jsonl: Path, out_csv: Path) -> int:
    """
    Flatten ``runs.jsonl`` → ``run_summary.csv``.

    Returns the number of rows written (one per run).
    """
    rows = _load_jsonl(runs_jsonl)
    if not rows:
        return 0

    fields = [
        "run_id", "pmcid", "started_at", "duration_s", "status", "error",
        # Ingestion
        "sentence_count", "te_count",
        # Chunking
        "total_chunks", "chunk_size", "chunk_overlap",
        # MAP stage
        "cache_hits", "cache_misses", "escalations", "keeps", "rejects",
        "map_findings_out",
        # Grounding (MAP)
        "grounding_map_before", "grounding_map_after", "grounding_map_dropped",
        # Reduce stage
        "reduce_iterations", "reduce_batches",
        "reduce_cache_hits", "reduce_cache_misses", "reduce_findings_out",
        # Rules
        "rules_extracted",
        # Grounding (rules)
        "grounding_rules_before", "grounding_rules_after", "grounding_rules_dropped",
        # Meta
        "warning_count",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(_flatten_run(r))
    return len(rows)


def export_chunk_summary_csv(chunks_jsonl: Path, out_csv: Path) -> int:
    """
    Flatten ``chunks.jsonl`` → ``chunk_summary.csv``.

    Returns the number of rows written (one per chunk).
    """
    rows = _load_jsonl(chunks_jsonl)
    if not rows:
        return 0

    fields = [
        "run_id", "pmcid", "chunk_id",
        "sentence_count", "te_ids", "cache_hit", "escalated",
        "selected_voter_index",
        "voter_count", "eligible_voter_count",
        "deferral_score", "theta", "reject_theta",
        "decision", "reason",
        "text_preview",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(_flatten_chunk(r))
    return len(rows)


def export_pairwise_scores_csv(chunks_jsonl: Path, out_csv: Path) -> int:
    """
    Explode ``chunks.jsonl`` → ``pairwise_scores.csv``: one row per voter pair per chunk.

    Returns the number of rows written.
    Chunks with no agreement data (cache hits) produce no rows.
    """
    rows = _load_jsonl(chunks_jsonl)
    if not rows:
        return 0

    fields = [
        "run_id", "pmcid", "chunk_id",
        "voter_i", "voter_j", "score",
        "decision", "deferral_score",
        # Alignment component breakdown (present when EmbeddingSimilarityStrategy is used)
        "claim_count_a", "claim_count_b",
        "coverage_a_to_b", "coverage_b_to_a", "base",
        "count_factor", "reuse_factor",
        "polarity_contradiction_ratio", "numeric_contradiction_ratio",
        "contradiction_ratio", "contradiction_factor",
        "pre_grounding_score", "grounding_factor",
    ]
    written = 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            agreement = r.get("agreement") or {}
            decision = agreement.get("decision")
            deferral = agreement.get("deferral_score")
            for ps in agreement.get("pairwise_scores", []):
                row: dict = {
                    "run_id": r.get("run_id"),
                    "pmcid": r.get("pmcid"),
                    "chunk_id": r.get("chunk_id"),
                    "voter_i": ps.get("voter_i"),
                    "voter_j": ps.get("voter_j"),
                    "score": ps.get("score"),
                    "decision": decision,
                    "deferral_score": deferral,
                }
                # Merge breakdown fields if present in the pairwise_score record.
                for k in (
                    "claim_count_a", "claim_count_b",
                    "coverage_a_to_b", "coverage_b_to_a", "base",
                    "count_factor", "reuse_factor",
                    "polarity_contradiction_ratio", "numeric_contradiction_ratio",
                    "contradiction_ratio", "contradiction_factor",
                    "pre_grounding_score", "grounding_factor",
                ):
                    row[k] = ps.get(k)
                writer.writerow(row)
                written += 1
    return written


def export_all_csv(trace_dir: Path) -> dict[str, int]:
    """
    Derive all CSV exports from the JSONL files already in ``trace_dir``.

    Returns a dict mapping export name → row count.
    Call this after a batch run to produce analysis-ready CSVs.
    """
    runs_path = trace_dir / "runs.jsonl"
    chunks_path = trace_dir / "chunks.jsonl"
    counts: dict[str, int] = {}

    if runs_path.exists():
        counts["run_summary"] = export_run_summary_csv(
            runs_path, trace_dir / "run_summary.csv"
        )
    if chunks_path.exists():
        counts["chunk_summary"] = export_chunk_summary_csv(
            chunks_path, trace_dir / "chunk_summary.csv"
        )
        counts["pairwise_scores"] = export_pairwise_scores_csv(
            chunks_path, trace_dir / "pairwise_scores.csv"
        )
    return counts


# ── Internal helpers ───────────────────────────────────────────────────────────


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _g(d: Any, *keys: str, default: Any = None) -> Any:
    """Safe nested get over dicts."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
        if d is None:
            return default
    return d


def _flatten_run(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": r.get("run_id"),
        "pmcid": r.get("pmcid"),
        "started_at": r.get("started_at"),
        "duration_s": r.get("duration_s"),
        "status": r.get("status"),
        "error": r.get("error"),
        "sentence_count": _g(r, "ingestion", "sentence_count"),
        "te_count": _g(r, "ingestion", "te_count"),
        "total_chunks": _g(r, "chunking", "total_chunks"),
        "chunk_size": _g(r, "chunking", "chunk_size"),
        "chunk_overlap": _g(r, "chunking", "chunk_overlap"),
        "cache_hits": _g(r, "map_stage", "cache_hits"),
        "cache_misses": _g(r, "map_stage", "cache_misses"),
        "escalations": _g(r, "map_stage", "escalations"),
        "keeps": _g(r, "map_stage", "keeps"),
        "rejects": _g(r, "map_stage", "rejects"),
        "map_findings_out": _g(r, "map_stage", "total_findings_out"),
        "grounding_map_before": _g(r, "grounding_map", "items_before"),
        "grounding_map_after": _g(r, "grounding_map", "items_after"),
        "grounding_map_dropped": _g(r, "grounding_map", "dropped"),
        "reduce_iterations": _g(r, "reduce_stage", "iterations"),
        "reduce_batches": _g(r, "reduce_stage", "total_batches"),
        "reduce_cache_hits": _g(r, "reduce_stage", "cache_hits"),
        "reduce_cache_misses": _g(r, "reduce_stage", "cache_misses"),
        "reduce_findings_out": _g(r, "reduce_stage", "output_finding_count"),
        "rules_extracted": _g(r, "rule_stage", "rules_extracted"),
        "grounding_rules_before": _g(r, "grounding_rules", "items_before"),
        "grounding_rules_after": _g(r, "grounding_rules", "items_after"),
        "grounding_rules_dropped": _g(r, "grounding_rules", "dropped"),
        "warning_count": len(r.get("warnings", [])),
    }


def _flatten_chunk(r: dict[str, Any]) -> dict[str, Any]:
    agreement = r.get("agreement") or {}
    return {
        "run_id": r.get("run_id"),
        "pmcid": r.get("pmcid"),
        "chunk_id": r.get("chunk_id"),
        "sentence_count": r.get("sentence_count"),
        "te_ids": json.dumps(r.get("te_ids", [])),
        "cache_hit": r.get("cache_hit"),
        "escalated": r.get("escalated"),
        "selected_voter_index": r.get("selected_voter_index"),
        "voter_count": len(r.get("voters", [])),
        "eligible_voter_count": len(agreement.get("eligible_voter_indices", [])),
        "deferral_score": agreement.get("deferral_score"),
        "theta": agreement.get("theta"),
        "reject_theta": agreement.get("reject_theta"),
        "decision": agreement.get("decision"),
        "reason": agreement.get("reason"),
        "text_preview": (r.get("text_preview") or "")[:200],
    }
