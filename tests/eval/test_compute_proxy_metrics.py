"""Phase 1 unit tests for `scripts/eval/compute_proxy_metrics.py`.

Covers:
- golden-file behaviour on a two-paper fixture (one rich, one sparse);
- graceful NaN handling when ``rejection_summary`` is missing;
- import-safety guarantee (no NLI/LLM/embedding libs pulled in).
"""

from __future__ import annotations

import csv
import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "eval"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "summaries"


@pytest.fixture()
def proxy_module():
    """Import compute_proxy_metrics by file path, then clean up sys.path/modules."""
    spec = importlib.util.spec_from_file_location(
        "compute_proxy_metrics_under_test",
        SCRIPT_DIR / "compute_proxy_metrics.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)
    sys.modules.pop("_lib", None)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_phase1_golden(tmp_path: Path, proxy_module) -> None:
    out_csv = tmp_path / "proxy_metrics.csv"
    rc = proxy_module.main([
        "--input", str(FIXTURE_DIR),
        "--out", str(out_csv),
        "--log-level", "WARNING",
    ])
    assert rc == 0

    rows = _read_csv(out_csv)
    assert len(rows) == 3, "two papers + one aggregate row"
    by_pmcid = {r["pmcid"]: r for r in rows}
    assert set(by_pmcid) == {"PAPER_OK", "PAPER_SPARSE", "__aggregate__"}

    ok = by_pmcid["PAPER_OK"]
    sparse = by_pmcid["PAPER_SPARSE"]
    agg = by_pmcid["__aggregate__"]

    # Identity ------------------------------------------------------------
    assert ok["status"] == "success"
    assert ok["is_success_like"] == "1"
    assert sparse["status"] == "skipped"
    assert sparse["is_success_like"] == "1"

    # MAP-level -----------------------------------------------------------
    assert ok["raw_map_finding_count"] == "10"
    assert ok["grounded_finding_count"] == "7"
    assert ok["grounded_finding_count_source"] == "rejection_summary"
    assert ok["grounding_rejection_rate"] == "0.3"
    assert ok["grounding_threshold_used"] == "0.6"
    # Mean of [0.9, 0.7, 0.5] = 0.7
    assert math.isclose(float(ok["mean_grounding_score"]), 0.7, rel_tol=1e-9)
    assert ok["mean_grounding_score_source"] == "audit_trail"

    # Sparse paper: rejection_summary missing → NaN/empty + missing sources.
    assert sparse["raw_map_finding_count"] == "0"
    assert sparse["grounded_finding_count"] == ""
    assert sparse["grounded_finding_count_source"] == "missing"
    assert sparse["grounding_rejection_rate"] == ""
    assert sparse["grounding_rejection_rate_source"] == "missing"
    assert sparse["mean_grounding_score"] == ""
    assert sparse["mean_grounding_score_source"] == "missing"

    # Downstream counts ---------------------------------------------------
    assert ok["canonical_rule_count"] == "4"
    assert ok["final_rule_count"] == "3"
    assert ok["relations_total"] == "2"
    assert ok["support_edge_count"] == "1"
    assert ok["contradiction_edge_count"] == "1"
    assert ok["scope_qualify_edge_count"] == "0"
    # 2 of 4 directions are unclear/no_direction = 0.5
    assert math.isclose(float(ok["unclear_no_direction_rate"]), 0.5, rel_tol=1e-9)
    # Sparse paper has 0 canonicals → unclear rate is NaN/empty.
    assert sparse["unclear_no_direction_rate"] == ""

    # Final-rule slices ---------------------------------------------------
    # final_score >= 0.5 in [0.82, 0.40, 0.55] → 2.
    assert ok["top_k_final_rule_count_score_ge_0_5"] == "2"
    assert ok["top_k_final_rule_count_top10"] == "3"
    assert sparse["top_k_final_rule_count_top10"] == "0"

    # Cascade -------------------------------------------------------------
    assert ok["l1_chunks"] == "1"
    assert ok["l2_chunks"] == "1"
    assert ok["l3_chunks"] == "1"
    assert ok["polarity_conflict_chunks"] == "1"
    # No cascade file for PAPER_SPARSE → all four columns NaN/empty.
    assert sparse["l1_chunks"] == ""
    assert sparse["polarity_conflict_chunks"] == ""

    # Cache stats ---------------------------------------------------------
    assert ok["cache_hits"] == "1"
    assert ok["cache_misses"] == "1"
    assert ok["cache_hits_source"] == "chunk_summary"

    # Cost ----------------------------------------------------------------
    assert ok["total_input_tokens"] == "1200"
    assert ok["total_output_tokens"] == "400"
    assert ok["token_count_source"] == "cost_jsonl"
    assert math.isclose(float(ok["estimated_total_cost_usd"]), 0.0123, rel_tol=1e-9)
    # Sparse paper not in any cost_report → missing.
    assert sparse["total_input_tokens"] == ""
    assert sparse["token_count_source"] == "missing"

    # Timing --------------------------------------------------------------
    assert math.isclose(float(ok["run_duration_s"]), 12.5, rel_tol=1e-9)
    assert math.isclose(float(sparse["run_duration_s"]), 0.3, rel_tol=1e-9)

    # Aggregate -----------------------------------------------------------
    assert agg["pipeline_config_hash"] == "MIXED"
    assert agg["is_success_like"] == "2", "both papers are success-like"
    # Sum across papers.
    assert agg["raw_map_finding_count"] == "10"
    assert agg["canonical_rule_count"] == "4"
    assert agg["final_rule_count"] == "3"
    assert agg["total_input_tokens"] == "1200"
    # Source distribution stringifies to pipe-joined list.
    assert set(agg["grounded_finding_count_source"].split("|")) == {
        "missing",
        "rejection_summary",
    }


def test_aggregate_json_status_counts(tmp_path: Path, proxy_module) -> None:
    out_csv = tmp_path / "proxy_metrics.csv"
    proxy_module.main([
        "--input", str(FIXTURE_DIR),
        "--out", str(out_csv),
        "--log-level", "WARNING",
    ])
    agg_path = tmp_path / "proxy_metrics_aggregate.json"
    with agg_path.open("r", encoding="utf-8") as f:
        agg = json.load(f)
    assert agg["schema_version"] == "v1"
    assert agg["n_papers"] == 2
    assert agg["n_success_like"] == 2
    assert agg["status_counts"] == {"success": 1, "skipped": 1}
    # JSON must not contain NaN (allow_nan=False at write time).
    raw = agg_path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert agg["source_distribution"]["mean_grounding_score_source"] == {
        "audit_trail": 1,
        "missing": 1,
    }


def test_meta_json_has_required_fields(tmp_path: Path, proxy_module) -> None:
    out_csv = tmp_path / "proxy_metrics.csv"
    proxy_module.main([
        "--input", str(FIXTURE_DIR),
        "--out", str(out_csv),
        "--log-level", "WARNING",
    ])
    meta_path = tmp_path / "proxy_metrics.meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["schema_version"] == "v1"
    assert "created_at" in meta
    assert meta["n_papers"] == 2
    assert meta["input_dir"].endswith("summaries")


def test_missing_input_dir(tmp_path: Path, proxy_module) -> None:
    rc = proxy_module.main([
        "--input", str(tmp_path / "does-not-exist"),
        "--out", str(tmp_path / "metrics.csv"),
        "--log-level", "ERROR",
    ])
    assert rc == 1


BANNED_MODULE_PREFIXES: tuple[str, ...] = (
    "transformers",
    "torch",
    "openai",
    "anthropic",
    "google.genai",
    "google.generativeai",
    "vertexai",
    "sentence_transformers",
    "pipeline.stages.knowledge_extraction.llm_providers",
)


def test_import_safety_no_nli_or_llm_modules(tmp_path: Path) -> None:
    """Importing the Phase 1 entry point must not pull in NLI/LLM/embedding modules."""
    # Use a subprocess so a previously-imported module in this pytest session
    # cannot pollute the check.
    import subprocess

    script = (
        "import sys, importlib, importlib.util;"
        f"spec = importlib.util.spec_from_file_location('m', r'{SCRIPT_DIR / 'compute_proxy_metrics.py'}');"
        "mod = importlib.util.module_from_spec(spec);"
        "sys.modules['m'] = mod;"
        "spec.loader.exec_module(mod);"
        "banned = "
        f"{BANNED_MODULE_PREFIXES!r};"
        "loaded = sorted({n for n in sys.modules if any(n == b or n.startswith(b + '.') for b in banned)});"
        "print('LOADED:' + (','.join(loaded) if loaded else 'none'));"
        "sys.exit(1 if loaded else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"banned imports detected: {result.stdout!r}"
