#!/usr/bin/env python3
"""
Compare two PDF-extraction run manifests.

Each manifest is the JSON file produced by ``RunManifestWriter`` (look under
``<out_root>/run_metadata/run_*.json``).  This script reads two of them, the
A side (baseline) and the B side (variant), and emits a human-readable
Markdown summary plus an optional JSON diff so two pipeline configurations
can be compared knob-by-knob without re-running the pipeline.

Usage::

    python scripts/eval/compare_pdf_runs.py \\
        out/sweeps/baseline/run_metadata/run_*.json \\
        out/sweeps/tatr095/run_metadata/run_*.json

    python scripts/eval/compare_pdf_runs.py \\
        baseline.json variant.json \\
        --md-out reports/baseline_vs_tatr095.md \\
        --json-out reports/baseline_vs_tatr095.json

The script is observability-only — it never reads source PDFs, never runs
the pipeline, and never modifies the input manifests.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Fields we surface in the per-document table.  Add new ones here when the
# manifest schema grows; downstream callers tolerate missing keys.
_PER_DOC_COUNT_FIELDS: Tuple[str, ...] = (
    "n_text_rows", "n_figures", "n_tables",
)


# ── IO ────────────────────────────────────────────────────────────────────────


def load_manifest(path: Path) -> Dict[str, Any]:
    """Read a manifest file or raise with a useful message."""
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest {path} is not valid JSON: {exc}") from exc


# ── Diff helpers ──────────────────────────────────────────────────────────────


def _docs_by_pmcid(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {d.get("pmcid"): d for d in manifest.get("documents", []) if d.get("pmcid")}


def _summary(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return manifest.get("summary") or {}


def _all_keys(*dicts: Dict[str, Any]) -> List[str]:
    keys: set = set()
    for d in dicts:
        keys.update(d.keys())
    return sorted(keys)


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return round(b - a, 4)


def _delta_pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or a == 0:
        return None
    return round((b - a) / a * 100.0, 1)


def compute_summary_diff(a: Dict[str, Any], b: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Top-level batch-summary deltas."""
    sa, sb = _summary(a), _summary(b)
    keys = (
        "n_attempted", "n_with_stats", "n_ok", "n_failed",
        "total_wall_seconds", "mean_wall_seconds",
        "reason_in_header_zone_sum",
    )
    rows: List[Dict[str, Any]] = []
    for k in keys:
        av = sa.get(k)
        bv = sb.get(k)
        rows.append({"key": k, "a": av, "b": bv, "delta": _delta(av, bv)})
    return rows


def compute_counts_diff(a: Dict[str, Any], b: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Union of `summary.counts_sum` keys, A vs B with delta + delta%."""
    ca = _summary(a).get("counts_sum") or {}
    cb = _summary(b).get("counts_sum") or {}
    rows: List[Dict[str, Any]] = []
    for k in _all_keys(ca, cb):
        av = ca.get(k, 0)
        bv = cb.get(k, 0)
        rows.append({
            "key":    k,
            "a":      av,
            "b":      bv,
            "delta":  _delta(av, bv),
            "delta_pct": _delta_pct(av, bv),
        })
    return rows


def compute_reason_diff(a: Dict[str, Any], b: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Union of `summary.reason_histogram_sum` keys."""
    ra = _summary(a).get("reason_histogram_sum") or {}
    rb = _summary(b).get("reason_histogram_sum") or {}
    rows: List[Dict[str, Any]] = []
    for k in _all_keys(ra, rb):
        av = ra.get(k, 0)
        bv = rb.get(k, 0)
        rows.append({
            "key":    k,
            "a":      av,
            "b":      bv,
            "delta":  _delta(av, bv),
            "delta_pct": _delta_pct(av, bv),
        })
    return rows


def compute_status_changes(
    a: Dict[str, Any], b: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """PMCIDs whose status differs between A and B (incl. only-in-A / only-in-B)."""
    da = _docs_by_pmcid(a)
    db = _docs_by_pmcid(b)
    changes: List[Dict[str, Any]] = []
    for pmcid in sorted(set(da.keys()) | set(db.keys())):
        sa = (da.get(pmcid) or {}).get("status")
        sb = (db.get(pmcid) or {}).get("status")
        if sa != sb:
            changes.append({"pmcid": pmcid, "status_a": sa, "status_b": sb})
    return changes


def compute_per_doc_count_diffs(
    a: Dict[str, Any], b: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Per-document field-level deltas for the count fields we surface."""
    da = _docs_by_pmcid(a)
    db = _docs_by_pmcid(b)
    diffs: List[Dict[str, Any]] = []
    for pmcid in sorted(set(da.keys()) | set(db.keys())):
        doc_a = da.get(pmcid) or {}
        doc_b = db.get(pmcid) or {}
        for field in _PER_DOC_COUNT_FIELDS:
            av = doc_a.get(field)
            bv = doc_b.get(field)
            if av is None and bv is None:
                continue
            if av != bv:
                diffs.append({
                    "pmcid": pmcid,
                    "field": field,
                    "a":     av,
                    "b":     bv,
                    "delta": _delta(av, bv),
                })
    return diffs


# ── Rendering ─────────────────────────────────────────────────────────────────


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 1000 else f"{value:.1f}"
    return str(value)


def _md_table(headers: List[str], rows: Iterable[List[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(_fmt(c) for c in r) + " |")
    return "\n".join(lines)


def render_markdown(
    a: Dict[str, Any],
    b: Dict[str, Any],
    *,
    path_a: Optional[Path] = None,
    path_b: Optional[Path] = None,
) -> str:
    out: List[str] = []
    out.append("# PDF-extraction run comparison")
    out.append("")
    if path_a or path_b:
        out.append(f"* **A (baseline):** `{path_a}`")
        out.append(f"* **B (variant):**  `{path_b}`")
        out.append("")

    # ── Overview ────────────────────────────────────────────────────────────
    out.append("## Overview")
    out.append("")
    overview_rows = [
        ["run_id",        a.get("run_id"),        b.get("run_id")],
        ["config_digest", a.get("config_digest"), b.get("config_digest")],
        ["started_at",    a.get("started_at"),    b.get("started_at")],
        ["finished_at",   a.get("finished_at"),   b.get("finished_at")],
        ["host",          a.get("host"),          b.get("host")],
        ["python",        a.get("python"),        b.get("python")],
        ["git.sha",
         (a.get("git") or {}).get("sha"),
         (b.get("git") or {}).get("sha")],
        ["git.branch",
         (a.get("git") or {}).get("branch"),
         (b.get("git") or {}).get("branch")],
        ["git.dirty",
         (a.get("git") or {}).get("dirty"),
         (b.get("git") or {}).get("dirty")],
    ]
    out.append(_md_table(["field", "A", "B"], overview_rows))
    out.append("")

    same = a.get("config_digest") == b.get("config_digest")
    out.append("✅ Config digests match — observed differences are NOT due to "
               "configuration."
               if same else
               "⚠️  Config digests differ — A and B were run with different knobs.")
    out.append("")

    # ── Summary ────────────────────────────────────────────────────────────
    out.append("## Batch summary")
    out.append("")
    out.append(_md_table(
        ["metric", "A", "B", "Δ (B − A)"],
        [[r["key"], r["a"], r["b"], r["delta"]]
         for r in compute_summary_diff(a, b)],
    ))
    out.append("")

    # ── Counts ─────────────────────────────────────────────────────────────
    cd = compute_counts_diff(a, b)
    out.append("## Counts (summed across documents)")
    out.append("")
    out.append(_md_table(
        ["key", "A", "B", "Δ (B − A)", "Δ %"],
        [[r["key"], r["a"], r["b"], r["delta"], r["delta_pct"]] for r in cd],
    ))
    out.append("")

    # ── Reason histogram ───────────────────────────────────────────────────
    rd = compute_reason_diff(a, b)
    out.append("## Reason histogram (NodeScorer R0/R1/R2/R3/R-color)")
    out.append("")
    out.append(_md_table(
        ["code", "A", "B", "Δ (B − A)", "Δ %"],
        [[r["key"], r["a"], r["b"], r["delta"], r["delta_pct"]] for r in rd],
    ))
    out.append("")

    # ── Status changes ─────────────────────────────────────────────────────
    sc = compute_status_changes(a, b)
    out.append("## Per-document status changes")
    out.append("")
    if sc:
        out.append(_md_table(
            ["pmcid", "status A", "status B"],
            [[r["pmcid"], r["status_a"], r["status_b"]] for r in sc],
        ))
    else:
        out.append("_No status changes._")
    out.append("")

    # ── Per-doc count deltas ───────────────────────────────────────────────
    cdd = compute_per_doc_count_diffs(a, b)
    out.append("## Per-document count differences")
    out.append("")
    if cdd:
        out.append(_md_table(
            ["pmcid", "field", "A", "B", "Δ (B − A)"],
            [[r["pmcid"], r["field"], r["a"], r["b"], r["delta"]] for r in cdd],
        ))
    else:
        out.append("_No per-document count differences._")
    out.append("")

    return "\n".join(out)


def render_json_diff(
    a: Dict[str, Any],
    b: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "run_a": {
            "run_id":        a.get("run_id"),
            "config_digest": a.get("config_digest"),
        },
        "run_b": {
            "run_id":        b.get("run_id"),
            "config_digest": b.get("config_digest"),
        },
        "config_digest_equal":
            a.get("config_digest") == b.get("config_digest"),
        "summary_diff":     compute_summary_diff(a, b),
        "counts_diff":      compute_counts_diff(a, b),
        "reason_diff":      compute_reason_diff(a, b),
        "status_changes":   compute_status_changes(a, b),
        "per_doc_count_diffs": compute_per_doc_count_diffs(a, b),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare two PDF-extraction run manifests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("manifest_a", type=Path,
                   help="Baseline manifest (out/.../run_metadata/run_*.json).")
    p.add_argument("manifest_b", type=Path,
                   help="Variant manifest.")
    p.add_argument("--md-out",   type=Path, default=None,
                   help="Write Markdown report here.  Defaults to stdout.")
    p.add_argument("--json-out", type=Path, default=None,
                   help="Also write a machine-readable JSON diff here.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    try:
        a = load_manifest(args.manifest_a)
        b = load_manifest(args.manifest_b)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    md = render_markdown(a, b, path_a=args.manifest_a, path_b=args.manifest_b)

    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(md)
        print(f"wrote markdown report → {args.md_out}", file=sys.stderr)
    else:
        sys.stdout.write(md)
        sys.stdout.write("\n")

    if args.json_out is not None:
        payload = render_json_diff(a, b)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"wrote json diff → {args.json_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
