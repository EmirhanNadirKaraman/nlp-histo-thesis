"""Unit tests for ``eval/sweeps/_lib.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
LIB_PATH = REPO_ROOT / "eval" / "sweeps" / "_lib.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture()
def lib():
    spec = importlib.util.spec_from_file_location("eval_sweeps_lib_under_test", LIB_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


def test_gather_pulls_surviving_and_rejected(lib) -> None:
    summaries_dir = FIXTURES / "summaries" / "summaries"
    records, warnings = lib.gather_findings_with_grounding(summaries_dir)

    # PAPER_A: 4 surviving + 2 grounding-rejected = 6.
    # PAPER_B: 3 surviving + 0 rejected = 3.
    assert len(records) == 9
    paper_a = [r for r in records if r.pmcid == "PAPER_A"]
    paper_b = [r for r in records if r.pmcid == "PAPER_B"]
    assert len(paper_a) == 6
    assert len(paper_b) == 3
    a_surviving = [r for r in paper_a if r.kept_at_producer_threshold]
    a_rejected = [r for r in paper_a if not r.kept_at_producer_threshold]
    assert len(a_surviving) == 4
    assert len(a_rejected) == 2
    # Rejected scores are parsed from strings ("0.20", "0.10").
    assert sorted(r.grounding_score for r in a_rejected) == pytest.approx([0.10, 0.20])
    # Producer threshold round-trips as float.
    assert paper_a[0].producer_threshold == 0.5
    # No warnings — invariant holds on both papers.
    assert warnings == []


def test_simulate_threshold_sweep_denominators(lib) -> None:
    summaries_dir = FIXTURES / "summaries" / "summaries"
    records, _ = lib.gather_findings_with_grounding(summaries_dir)

    rows = lib.simulate_threshold_sweep(records, [0.30, 0.50, 1.00])
    rows_by_t = {r["threshold"]: r for r in rows}

    # At 0.30: PAPER_A keeps {0.95, 0.80, 0.55, 0.45}=4, rejects {0.20, 0.10}=2;
    # PAPER_B keeps {0.90, 0.60, 0.30}=3, rejects {}=0. Total kept=7, rej=2.
    r30 = rows_by_t[0.30]
    assert r30["total_findings"] == 9
    assert r30["scored_findings"] == 9
    assert r30["missing_score_count"] == 0
    assert r30["missing_score_pct"] == pytest.approx(0.0)
    assert r30["kept_findings"] == 7
    assert r30["rejected_findings"] == 2
    assert r30["kept_pct_of_scored"] == pytest.approx(7 / 9 * 100)
    assert r30["rejected_pct_of_scored"] == pytest.approx(2 / 9 * 100)
    assert r30["papers_covered"] == 2

    # At 0.50: kept=5 ({0.95,0.80,0.55,0.90,0.60}), rejected=4 ({0.45,0.20,0.10,0.30}).
    r50 = rows_by_t[0.50]
    assert r50["kept_findings"] == 5
    assert r50["rejected_findings"] == 4
    assert r50["mean_grounding_score_kept"] == pytest.approx(
        (0.95 + 0.80 + 0.55 + 0.90 + 0.60) / 5
    )
    assert r50["mean_grounding_score_rejected"] == pytest.approx(
        (0.45 + 0.20 + 0.10 + 0.30) / 4
    )

    # At 1.00: nothing scores >= 1.0 → kept=0, rejected=9. mean_kept is None.
    r100 = rows_by_t[1.00]
    assert r100["kept_findings"] == 0
    assert r100["rejected_findings"] == 9
    assert r100["mean_grounding_score_kept"] is None
    assert r100["mean_grounding_score_rejected"] == pytest.approx(
        (0.95 + 0.80 + 0.55 + 0.45 + 0.20 + 0.10 + 0.90 + 0.60 + 0.30) / 9
    )


def test_missing_score_denominator(lib) -> None:
    Record = lib.FindingRecord
    records = [
        Record(pmcid="P", chunk_id="C1", finding_index=0, grounding_score=0.9,
               kept_at_producer_threshold=True, producer_threshold=None, category=None),
        Record(pmcid="P", chunk_id="C1", finding_index=1, grounding_score=0.4,
               kept_at_producer_threshold=True, producer_threshold=None, category=None),
        Record(pmcid="P", chunk_id="C1", finding_index=2, grounding_score=None,
               kept_at_producer_threshold=True, producer_threshold=None, category=None),
    ]
    rows = lib.simulate_threshold_sweep(records, [0.5])
    row = rows[0]
    assert row["total_findings"] == 3
    assert row["scored_findings"] == 2
    assert row["missing_score_count"] == 1
    assert row["missing_score_pct"] == pytest.approx(1 / 3 * 100)
    assert row["kept_findings"] == 1
    assert row["rejected_findings"] == 1
    # Denominator is scored, not total.
    assert row["kept_pct_of_scored"] == pytest.approx(50.0)
    assert row["rejected_pct_of_scored"] == pytest.approx(50.0)


def test_simulate_handles_empty_input(lib) -> None:
    rows = lib.simulate_threshold_sweep([], [0.5])
    assert len(rows) == 1
    r = rows[0]
    assert r["total_findings"] == 0
    assert r["scored_findings"] == 0
    assert r["kept_findings"] == 0
    assert r["rejected_findings"] == 0
    assert r["kept_pct_of_scored"] == pytest.approx(0.0)
    assert r["mean_grounding_score_kept"] is None


def test_build_sweep_metadata_collects_unique_values(tmp_path: Path, lib) -> None:
    summaries_dir = FIXTURES / "summaries" / "summaries"
    metadata = lib.build_sweep_metadata(
        sweep_name="grounding_threshold",
        summaries_dir=summaries_dir,
        input_dir=summaries_dir.parent,
        out_csv=tmp_path / "x.csv",
        out_md=tmp_path / "x.md",
        thresholds=[0.3, 0.5, 0.7],
    )
    assert metadata.pmcids == ("PAPER_A", "PAPER_B")
    assert metadata.run_ids == ("shared_run",)
    assert metadata.pipeline_config_hashes == ("hashA",)
    assert metadata.thresholds == (0.3, 0.5, 0.7)
    assert metadata.schema_version == lib.SCHEMA_VERSION


def test_csv_writer_emits_metadata_comments_and_columns(tmp_path: Path, lib) -> None:
    summaries_dir = FIXTURES / "summaries" / "summaries"
    records, _ = lib.gather_findings_with_grounding(summaries_dir)
    metadata = lib.build_sweep_metadata(
        sweep_name="grounding_threshold",
        summaries_dir=summaries_dir,
        input_dir=summaries_dir.parent,
        out_csv=tmp_path / "x.csv",
        out_md=tmp_path / "x.md",
        thresholds=[0.3, 0.5, 0.7],
    )
    rows = lib.simulate_threshold_sweep(records, [0.3, 0.5, 0.7])
    lib.write_sweep_csv(rows, tmp_path / "x.csv", metadata)

    text = (tmp_path / "x.csv").read_text(encoding="utf-8")
    assert text.startswith("# sweep_name:")
    assert "# pipeline_config_hashes: " in text
    data_lines = [line for line in text.splitlines() if not line.startswith("#") and line.strip()]
    # 1 header + 3 thresholds.
    assert len(data_lines) == 4
    header = data_lines[0].split(",")
    for col in [
        "threshold", "total_findings", "scored_findings",
        "missing_score_count", "missing_score_pct",
        "kept_findings", "rejected_findings",
        "kept_pct_of_scored", "rejected_pct_of_scored",
        "papers_covered",
        "mean_grounding_score_kept", "mean_grounding_score_rejected",
    ]:
        assert col in header, f"missing {col} from CSV header"


def test_markdown_writer_contains_disclaimer_and_no_forbidden_phrasings(
    tmp_path: Path, lib
) -> None:
    summaries_dir = FIXTURES / "summaries" / "summaries"
    records, gather_warnings = lib.gather_findings_with_grounding(summaries_dir)
    metadata = lib.build_sweep_metadata(
        sweep_name="grounding_threshold",
        summaries_dir=summaries_dir,
        input_dir=summaries_dir.parent,
        out_csv=tmp_path / "x.csv",
        out_md=tmp_path / "x.md",
        thresholds=[0.3, 0.5, 0.7],
    )
    rows = lib.simulate_threshold_sweep(records, [0.3, 0.5, 0.7])
    lib.write_sweep_markdown(
        rows, tmp_path / "x.md", metadata,
        warnings=gather_warnings,
        per_paper_detail=[
            {"pmcid": "PAPER_A", "total": 6, "scored": 6, "grounding_rejected": 2, "producer_threshold": 0.5},
            {"pmcid": "PAPER_B", "total": 3, "scored": 3, "grounding_rejected": 0, "producer_threshold": 0.5},
        ],
        producer_thresholds=[0.5, 0.5],
    )
    text = (tmp_path / "x.md").read_text(encoding="utf-8")

    # Positive assertion: the disclaimer must appear.
    assert "does NOT measure accuracy, precision, or recall" in text, (
        "Markdown report must contain the explicit disclaimer."
    )

    # Negative assertions: must not claim to measure these metrics.
    for forbidden in [
        "accuracy score", "precision score", "recall score",
        "F1 score", "F-1 score", "precision/recall",
        "accuracy=", "precision=", "recall=",
    ]:
        assert forbidden not in text, (
            f"Markdown report unexpectedly contains the phrase {forbidden!r}, "
            "which implies the report measures that metric."
        )

    # Tables.
    assert "## Retention by threshold" in text
    assert "kept_pct_of_scored" in text
    assert "## Per-paper detail (producer threshold = 0.5)" in text


def test_get_git_commit_is_optional(lib, tmp_path: Path) -> None:
    """Outside a git repo, returns None instead of crashing."""
    commit = lib.get_git_commit(repo_dir=tmp_path)
    assert commit is None or isinstance(commit, str)


def test_sanitize_for_json_handles_nan(lib) -> None:
    assert lib.sanitize_for_json(float("nan")) is None
    assert lib.sanitize_for_json(float("inf")) is None
    assert lib.sanitize_for_json({"a": float("nan"), "b": 1.0}) == {"a": None, "b": 1.0}
    assert lib.sanitize_for_json([float("nan"), 2]) == [None, 2]


def test_invariant_warning_when_total_mismatches(tmp_path: Path, lib) -> None:
    """If rejection_summary.map_findings_total disagrees, warning is emitted."""
    summaries_root = tmp_path / "summaries"
    summaries_root.mkdir()
    (summaries_root / "P.json").write_text(
        '{"pmcid":"P","run_id":"r","pipeline_config_hash":"h",'
        '"audit_trail":{"map_chunks":[{"chunk_id":"C1","findings":['
        '{"claim":"x","grounding_score":0.9}'
        ']}]},'
        '"rejection_summary":{"map_findings_total":3,"rejected":[]}}',
        encoding="utf-8",
    )
    _records, warnings = lib.gather_findings_with_grounding(summaries_root)
    assert any("possible missing or duplicated records" in w for w in warnings)


def test_invariant_warning_when_total_absent(tmp_path: Path, lib) -> None:
    summaries_root = tmp_path / "summaries"
    summaries_root.mkdir()
    (summaries_root / "P.json").write_text(
        '{"pmcid":"P","audit_trail":{"map_chunks":[{"chunk_id":"C1","findings":['
        '{"claim":"x","grounding_score":0.9}'
        ']}]},'
        '"rejection_summary":{}}',
        encoding="utf-8",
    )
    _records, warnings = lib.gather_findings_with_grounding(summaries_root)
    assert any("map_findings_total is absent" in w for w in warnings)
