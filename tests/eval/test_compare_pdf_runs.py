"""
Tests for ``scripts/eval/compare_pdf_runs.py``.

The script is observability-only — it reads two run manifests and renders
a Markdown summary plus an optional JSON diff.  These tests check the pure
diff helpers directly and round-trip the CLI via ``cmp_mod.main()``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest


# Load ``scripts/eval/compare_pdf_runs.py`` by file path so the test does not
# require ``scripts`` to be an importable package (mirrors
# ``tests/eval/test_compute_proxy_metrics.py``).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "eval" / "compare_pdf_runs.py"


@pytest.fixture(scope="module")
def cmp_mod():
    spec = importlib.util.spec_from_file_location(
        "compare_pdf_runs_under_test", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    try:
        yield mod
    finally:
        sys.modules.pop(spec.name, None)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _manifest(
    *,
    run_id: str,
    config_digest: str,
    documents: list,
    counts_sum: Dict[str, int] | None = None,
    reasons_sum: Dict[str, int] | None = None,
    total_wall: float = 0.0,
    n_ok: int = 0,
    n_failed: int = 0,
    n_attempted: int | None = None,
    header_zone_sum: int = 0,
) -> Dict[str, Any]:
    n_attempted = len(documents) if n_attempted is None else n_attempted
    n_with_stats = sum(1 for d in documents if d.get("status") != "no_stats")
    mean_wall = round(total_wall / n_with_stats, 4) if n_with_stats else 0.0
    return {
        "run_id":        run_id,
        "started_at":    "2026-05-17T10:00:00Z",
        "finished_at":   "2026-05-17T10:05:00Z",
        "git":           {"sha": "deadbeef", "branch": "test", "dirty": False},
        "host":          "ci",
        "python":        "3.12.0",
        "input":         {"runner": "ParallelBatchRunner", "n_paths": n_attempted},
        "config_digest": config_digest,
        "config":        {},
        "batch_stats":   {"processed": n_ok, "failed": n_failed, "skipped": 0},
        "summary": {
            "n_attempted":          n_attempted,
            "n_with_stats":         n_with_stats,
            "n_ok":                 n_ok,
            "n_failed":             n_failed,
            "total_wall_seconds":   round(total_wall, 4),
            "mean_wall_seconds":    mean_wall,
            "counts_sum":           counts_sum or {},
            "reason_histogram_sum": reasons_sum or {},
            "reason_in_header_zone_sum": header_zone_sum,
        },
        "documents":     documents,
    }


def _doc(pmcid: str, status: str = "ok", wall: float = 1.0,
         text_rows: int | None = None, figures: int | None = None,
         tables: int | None = None,
         failed_stage: str | None = None) -> Dict[str, Any]:
    return {
        "pmcid":         pmcid,
        "status":        status,
        "wall_seconds":  wall,
        "failed_stage":  failed_stage,
        "n_text_rows":   text_rows,
        "n_figures":     figures,
        "n_tables":      tables,
    }


def _baseline() -> Dict[str, Any]:
    return _manifest(
        run_id="rid-A", config_digest="aaaaaaaaaaaa",
        documents=[
            _doc("PMC1", "ok", wall=10.0, text_rows=14, figures=10, tables=0),
            _doc("PMC2", "ok", wall=6.0,  text_rows=20, figures=3,  tables=1),
        ],
        counts_sum={"text_rows": 34, "figures_cropped": 13, "tables_cropped": 1},
        reasons_sum={"R1_blank_pixels": 59, "R3_dense_text": 7},
        total_wall=16.0, n_ok=2, header_zone_sum=18,
    )


def _variant() -> Dict[str, Any]:
    """B has one more rejected node, an extra failed doc, and one extra figure."""
    return _manifest(
        run_id="rid-B", config_digest="bbbbbbbbbbbb",
        documents=[
            _doc("PMC1", "ok", wall=11.0, text_rows=14, figures=11, tables=0),
            _doc("PMC2", "failed", wall=2.5, failed_stage="STEP1_LAYOUT"),
        ],
        counts_sum={"text_rows": 14, "figures_cropped": 11, "tables_cropped": 0},
        reasons_sum={"R1_blank_pixels": 60, "R3_dense_text": 9, "R0_empty": 1},
        total_wall=13.5, n_ok=1, n_failed=1, header_zone_sum=20,
    )


# ── Pure diff helpers ─────────────────────────────────────────────────────────


def test_summary_diff_shape_and_values(cmp_mod) -> None:
    rows = cmp_mod.compute_summary_diff(_baseline(), _variant())
    by_key = {r["key"]: r for r in rows}
    assert by_key["n_ok"] == {"key": "n_ok", "a": 2, "b": 1, "delta": -1.0}
    assert by_key["n_failed"]["delta"] == 1.0
    assert by_key["mean_wall_seconds"]["delta"] == pytest.approx(-1.25)
    assert by_key["reason_in_header_zone_sum"]["delta"] == 2.0


def test_counts_diff_union_keys_and_deltas(cmp_mod) -> None:
    rows = cmp_mod.compute_counts_diff(_baseline(), _variant())
    by_key = {r["key"]: r for r in rows}
    assert by_key["text_rows"]["a"] == 34
    assert by_key["text_rows"]["b"] == 14
    assert by_key["text_rows"]["delta"] == -20
    assert by_key["text_rows"]["delta_pct"] == pytest.approx(-58.8, abs=0.1)
    assert by_key["figures_cropped"]["delta"] == -2  # 13 → 11
    assert by_key["tables_cropped"]["delta"] == -1   # 1 → 0


def test_reason_diff_handles_new_codes(cmp_mod) -> None:
    rows = cmp_mod.compute_reason_diff(_baseline(), _variant())
    by_key = {r["key"]: r for r in rows}
    # R0_empty only present in B
    assert by_key["R0_empty"]["a"] == 0
    assert by_key["R0_empty"]["b"] == 1
    assert by_key["R0_empty"]["delta_pct"] is None  # cannot %-of-zero
    assert by_key["R1_blank_pixels"]["delta"] == 1
    assert by_key["R3_dense_text"]["delta"] == 2


def test_status_changes_lists_only_diffs(cmp_mod) -> None:
    changes = cmp_mod.compute_status_changes(_baseline(), _variant())
    assert changes == [
        {"pmcid": "PMC2", "status_a": "ok", "status_b": "failed"},
    ]


def test_status_changes_handles_missing_docs(cmp_mod) -> None:
    a = _baseline()
    b = _variant()
    # Drop PMC2 from B entirely; should still appear as a status change.
    b["documents"] = [d for d in b["documents"] if d["pmcid"] != "PMC2"]
    changes = cmp_mod.compute_status_changes(a, b)
    assert len(changes) == 1
    assert changes[0]["pmcid"] == "PMC2"
    assert changes[0]["status_a"] == "ok"
    assert changes[0]["status_b"] is None  # missing in B


def test_per_doc_count_diffs(cmp_mod) -> None:
    diffs = cmp_mod.compute_per_doc_count_diffs(_baseline(), _variant())
    # PMC1 changed n_figures (10 → 11); n_text_rows + n_tables identical
    pmc1 = [r for r in diffs if r["pmcid"] == "PMC1"]
    assert pmc1 == [{"pmcid": "PMC1", "field": "n_figures",
                     "a": 10, "b": 11, "delta": 1}]
    # PMC2 changed all three (B has nones since it failed)
    pmc2 = {r["field"]: r for r in diffs if r["pmcid"] == "PMC2"}
    assert pmc2["n_text_rows"] == {"pmcid": "PMC2", "field": "n_text_rows",
                                    "a": 20, "b": None, "delta": None}


def test_identical_inputs_produce_zero_diffs(cmp_mod) -> None:
    base = _baseline()
    rows = cmp_mod.compute_counts_diff(base, base)
    assert all(r["delta"] == 0 for r in rows)
    rows = cmp_mod.compute_reason_diff(base, base)
    assert all(r["delta"] == 0 for r in rows)
    assert cmp_mod.compute_status_changes(base, base) == []
    assert cmp_mod.compute_per_doc_count_diffs(base, base) == []


# ── Rendering ─────────────────────────────────────────────────────────────────


def test_render_markdown_contains_key_sections(cmp_mod) -> None:
    md = cmp_mod.render_markdown(_baseline(), _variant())
    assert "# PDF-extraction run comparison" in md
    assert "## Overview" in md
    assert "## Batch summary" in md
    assert "## Counts (summed across documents)" in md
    assert "## Reason histogram" in md
    assert "## Per-document status changes" in md
    assert "## Per-document count differences" in md
    # Spot-checks
    assert "rid-A" in md and "rid-B" in md
    assert "aaaaaaaaaaaa" in md and "bbbbbbbbbbbb" in md
    assert "PMC2" in md
    assert "R1_blank_pixels" in md


def test_render_markdown_warns_when_digests_differ(cmp_mod) -> None:
    md = cmp_mod.render_markdown(_baseline(), _variant())
    assert "Config digests differ" in md


def test_render_markdown_confirms_match_when_digests_equal(cmp_mod) -> None:
    a = _baseline()
    b = _baseline()  # same digest
    md = cmp_mod.render_markdown(a, b)
    assert "Config digests match" in md
    assert "No status changes." in md
    assert "No per-document count differences." in md


def test_render_json_diff_structure(cmp_mod) -> None:
    out = cmp_mod.render_json_diff(_baseline(), _variant())
    expected_top = {
        "run_a", "run_b", "config_digest_equal",
        "summary_diff", "counts_diff", "reason_diff",
        "status_changes", "per_doc_count_diffs",
    }
    assert expected_top.issubset(out.keys())
    assert out["config_digest_equal"] is False
    assert out["run_a"]["run_id"] == "rid-A"
    assert out["run_b"]["run_id"] == "rid-B"
    # The JSON must be serialisable as-is
    json.dumps(out)


# ── load_manifest ─────────────────────────────────────────────────────────────


def test_load_manifest_missing_file_raises(cmp_mod, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cmp_mod.load_manifest(tmp_path / "nonexistent.json")


def test_load_manifest_invalid_json_raises(cmp_mod, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not valid {{")
    with pytest.raises(ValueError, match="not valid JSON"):
        cmp_mod.load_manifest(bad)


# ── CLI entry-point ───────────────────────────────────────────────────────────


def _write(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload))


def test_main_writes_md_and_json(cmp_mod, tmp_path: Path, capsys) -> None:
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    _write(a_path, _baseline())
    _write(b_path, _variant())

    md_out = tmp_path / "report.md"
    json_out = tmp_path / "diff.json"

    rc = cmp_mod.main([
        str(a_path), str(b_path),
        "--md-out", str(md_out),
        "--json-out", str(json_out),
    ])
    assert rc == 0

    md_text = md_out.read_text()
    assert "# PDF-extraction run comparison" in md_text

    diff = json.loads(json_out.read_text())
    assert diff["run_a"]["run_id"] == "rid-A"
    assert diff["run_b"]["run_id"] == "rid-B"
    assert diff["config_digest_equal"] is False

    # When --md-out is set, stdout stays clean of the report body
    captured = capsys.readouterr()
    assert "# PDF-extraction run comparison" not in captured.out


def test_main_prints_markdown_to_stdout_without_md_out(cmp_mod, tmp_path: Path, capsys) -> None:
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    _write(a_path, _baseline())
    _write(b_path, _variant())

    rc = cmp_mod.main([str(a_path), str(b_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "# PDF-extraction run comparison" in out
    assert "Config digests differ" in out


def test_main_returns_2_on_missing_file(cmp_mod, tmp_path: Path, capsys) -> None:
    a_path = tmp_path / "a.json"
    _write(a_path, _baseline())
    rc = cmp_mod.main([str(a_path), str(tmp_path / "missing.json")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "manifest not found" in err


def test_main_identical_inputs_produce_no_status_or_count_diffs(cmp_mod, tmp_path: Path, capsys) -> None:
    """Same manifest used twice — no changes flagged in the report."""
    a_path = tmp_path / "a.json"
    _write(a_path, _baseline())

    rc = cmp_mod.main([str(a_path), str(a_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "No status changes." in out
    assert "No per-document count differences." in out
    assert "Config digests match" in out
