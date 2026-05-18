"""
Tests for ``scripts/eval/score_pdf_variants.py``.

Covers:
* ``classify_figure`` / ``classify_table`` — full label vocabulary.
* ``metrics`` — corner cases (zero division, all-None).
* ``score_kind`` — TP / FP / FN / unlabelled / skipped accounting.
* ``load_emitted_crops`` — figures + tables sets from sweep media.json.
* ``load_labels`` — per-variant first, legacy fallback, no-fallback mode.
* ``detector_modes`` — reads detector from manifest; routes DOCLING → docling labels.
* ``main`` — full CLI round-trip with synthetic sweeps + labels.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "eval" / "score_pdf_variants.py"


@pytest.fixture(scope="module")
def scr():
    spec = importlib.util.spec_from_file_location("score_pdf_variants_uut", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    try:
        yield mod
    finally:
        sys.modules.pop(spec.name, None)


# ── Classifiers ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("label,expected", [
    ("correct", "tp"),
    ("correct figure", "tp"),
    ("correct-ish", "tp"),  # starts with "correct"
    ("icon", "fp"),
    ("incorrect", "fp"),
    ("CORRECT", "tp"),  # case-insensitive
    ("  correct  ", "tp"),  # trimmed
    ("other", "skip"),
    ("skipped", "skip"),
    ("", "skip"),
])
def test_classify_figure(scr, label, expected) -> None:
    assert scr.classify_figure(label) == expected


@pytest.mark.parametrize("label,expected", [
    ("correct", "tp"),
    ("missing footnotes — text below cut", "tp"),
    ("incorrect", "fp"),
    ("wrong caption (footer matched)", "fp"),
    ("crop is too big", "fp"),
    ("weird crop? check later", "fp"),
    ("other", "skip"),
    ("", "skip"),
])
def test_classify_table(scr, label, expected) -> None:
    assert scr.classify_table(label) == expected


# ── metrics ───────────────────────────────────────────────────────────────────


def test_metrics_normal(scr) -> None:
    m = scr.metrics(tp=8, fp=2, fn=4)
    assert m["precision"] == pytest.approx(0.8)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 * 0.8 * (2 / 3) / (0.8 + 2 / 3))


def test_metrics_zero_tp(scr) -> None:
    m = scr.metrics(tp=0, fp=5, fn=3)
    assert m["precision"] == 0
    assert m["recall"] == 0
    assert m["f1"] is None  # no harmonic mean with zeros


def test_metrics_no_predictions(scr) -> None:
    m = scr.metrics(tp=0, fp=0, fn=5)
    assert m["precision"] is None
    assert m["recall"] == 0
    assert m["f1"] is None


# ── score_kind ────────────────────────────────────────────────────────────────


def test_score_kind_counts_tp_fp_unlabelled(scr) -> None:
    emitted = {"a.png", "b.png", "c.png", "d.png"}
    labels = {"a.png": "correct", "b.png": "incorrect", "c.png": "other"}
    m = scr.score_kind(emitted, labels, scr.classify_table, total_actual=5)
    assert m["tp"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 4   # total_actual=5 - tp=1
    assert m["skipped"] == 1
    assert m["unlabelled"] == 1
    assert m["emitted"] == 4
    assert m["labelled"] == 3


def test_score_kind_no_ground_truth_for_recall(scr) -> None:
    """When neither total_actual nor fn_count is given, recall + F1 are None."""
    emitted = {"x.png"}
    labels = {"x.png": "correct"}
    m = scr.score_kind(emitted, labels, scr.classify_table)
    assert m["tp"] == 1
    assert m["fn"] == 0
    assert m["precision"] == 1.0
    assert m["recall"] is None
    assert m["f1"] is None


def test_score_kind_with_fn_count_path(scr) -> None:
    """``fn_count`` provides FN directly (used for figures via missed_figures)."""
    emitted = {"a.png", "b.png"}
    labels = {"a.png": "correct", "b.png": "correct"}
    m = scr.score_kind(emitted, labels, scr.classify_figure, fn_count=3)
    assert m["tp"] == 2
    assert m["fn"] == 3
    assert m["precision"] == 1.0
    assert m["recall"] == pytest.approx(0.4)  # 2 / (2 + 3)


def test_score_kind_with_fn_count_zero_yields_100_percent_recall(scr) -> None:
    """No missed figures + at least one TP → R=100%."""
    emitted = {"a.png"}
    labels = {"a.png": "correct"}
    m = scr.score_kind(emitted, labels, scr.classify_figure, fn_count=0)
    assert m["tp"] == 1
    assert m["fn"] == 0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_score_kind_fn_count_takes_priority_over_total_actual(scr) -> None:
    """If both are given (shouldn't happen), fn_count wins."""
    emitted = {"a.png"}
    labels = {"a.png": "correct"}
    m = scr.score_kind(emitted, labels, scr.classify_figure,
                       total_actual=10, fn_count=2)
    assert m["fn"] == 2  # not 10 - 1


def test_score_kind_total_actual_clipped_at_zero(scr) -> None:
    """If we mistakenly label more TPs than the ground truth claims exist,
    FN is clamped to 0 (not negative)."""
    emitted = {"a.png", "b.png", "c.png"}
    labels = {k: "correct" for k in emitted}
    m = scr.score_kind(emitted, labels, scr.classify_table, total_actual=2)
    assert m["tp"] == 3
    assert m["fn"] == 0  # max(0, 2 - 3) = 0
    assert m["recall"] == 1.0  # 3 / (3 + 0)


# ── load_emitted_crops ────────────────────────────────────────────────────────


def _write_sweep_with_crops(sweep_dir: Path, pmcid: str, *,
                             figures: list[str] | None = None,
                             tables: list[str] | None = None) -> None:
    json_dir = sweep_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"pmcid": pmcid, "figures": [], "tables": []}
    for n in figures or []:
        payload["figures"].append({"image_path": f"out/figs/{n}"})
    for n in tables or []:
        payload["tables"].append({"image_path": f"out/tabs/{n}"})
    (json_dir / f"{pmcid}_media.json").write_text(json.dumps(payload))


def test_load_emitted_crops_splits_figures_and_tables(tmp_path, scr) -> None:
    sweep = tmp_path / "sweep"
    _write_sweep_with_crops(sweep, "PMC_A",
                             figures=["f1.png", "f2.png"], tables=["t1.png"])
    _write_sweep_with_crops(sweep, "PMC_B", tables=["t2.png"])
    figs, tabs = scr.load_emitted_crops(sweep)
    assert figs == {"f1.png", "f2.png"}
    assert tabs == {"t1.png", "t2.png"}


def test_load_emitted_crops_empty(tmp_path, scr) -> None:
    sweep = tmp_path / "empty_sweep"
    (sweep / "json").mkdir(parents=True)
    figs, tabs = scr.load_emitted_crops(sweep)
    assert figs == set() and tabs == set()


# ── load_labels ───────────────────────────────────────────────────────────────


def test_load_labels_prefers_per_variant(tmp_path, scr, monkeypatch) -> None:
    monkeypatch.setattr(scr, "_ANN_ROOT", tmp_path)
    (tmp_path / "E1").mkdir()
    (tmp_path / "E1" / "json_tables_full.json").write_text(json.dumps(
        {"x.png": "correct"}
    ))
    # Legacy file with a different value — must NOT be used
    (tmp_path / "annotations_json_tables_full.json").write_text(json.dumps(
        {"x.png": "incorrect", "y.png": "correct"}
    ))
    labels = scr.load_labels("json_tables_full", "E1")
    assert labels == {"x.png": "correct"}


def test_load_labels_falls_back_to_legacy(tmp_path, scr, monkeypatch) -> None:
    monkeypatch.setattr(scr, "_ANN_ROOT", tmp_path)
    (tmp_path / "annotations_json_tables_full.json").write_text(json.dumps(
        {"y.png": "correct"}
    ))
    # No per-variant file → fallback to legacy
    labels = scr.load_labels("json_tables_full", "E1")
    assert labels == {"y.png": "correct"}


def test_load_labels_no_fallback_returns_empty(tmp_path, scr, monkeypatch) -> None:
    monkeypatch.setattr(scr, "_ANN_ROOT", tmp_path)
    (tmp_path / "annotations_json_tables_full.json").write_text(json.dumps(
        {"y.png": "correct"}
    ))
    labels = scr.load_labels("json_tables_full", "E1", fallback_legacy=False)
    assert labels == {}


def test_load_labels_malformed_per_variant_falls_back(tmp_path, scr, monkeypatch) -> None:
    monkeypatch.setattr(scr, "_ANN_ROOT", tmp_path)
    (tmp_path / "E1").mkdir()
    (tmp_path / "E1" / "json_tables_full.json").write_text("not json {")
    (tmp_path / "annotations_json_tables_full.json").write_text(json.dumps(
        {"y.png": "correct"}
    ))
    labels = scr.load_labels("json_tables_full", "E1")
    assert labels == {"y.png": "correct"}


# ── detector_modes ───────────────────────────────────────────────────────────


def _write_manifest(sweep: Path, detector: str) -> None:
    rm = sweep / "run_metadata"
    rm.mkdir(parents=True, exist_ok=True)
    (rm / "run_TEST.json").write_text(json.dumps({
        "config": {"table_detector": detector},
    }))


def test_detector_modes_hybrid(tmp_path, scr) -> None:
    sweep = tmp_path / "sweep"
    _write_manifest(sweep, "HYBRID")
    tab, fig = scr.detector_modes(sweep)
    assert tab == "json_tables_full"
    assert fig == "json_figures"


def test_detector_modes_docling(tmp_path, scr) -> None:
    sweep = tmp_path / "sweep"
    _write_manifest(sweep, "DOCLING")
    tab, fig = scr.detector_modes(sweep)
    assert tab == "json_tables_docling"
    assert fig == "json_figures"


def test_detector_modes_default_hybrid_when_no_manifest(tmp_path, scr) -> None:
    sweep = tmp_path / "no_manifest"
    sweep.mkdir()
    tab, fig = scr.detector_modes(sweep)
    assert tab == "json_tables_full"  # default
    assert fig == "json_figures"


# ── main() CLI ────────────────────────────────────────────────────────────────


def _build_eval_scenario(tmp_path, scr, monkeypatch) -> dict:
    """Construct a self-contained sweeps + labels + ground_truth scenario.

    Returns the paths so the test can read the resulting reports.
    """
    sweeps_root = tmp_path / "sweeps"
    ann_root = tmp_path / "annotations"
    gt_csv = tmp_path / "ground_truth.csv"

    monkeypatch.setattr(scr, "_ANN_ROOT", ann_root)
    monkeypatch.setattr(scr, "_GT_CSV", gt_csv)

    # Two variants: baseline (HYBRID), and detector_docling.
    base = sweeps_root / "baseline"
    _write_sweep_with_crops(base, "PMC_A",
                             figures=["PMC_A_Figure_1.png"],
                             tables=["PMC_A_Table_1.png", "PMC_A_Table_2.png"])
    _write_manifest(base, "HYBRID")

    docl = sweeps_root / "detector_docling"
    _write_sweep_with_crops(docl, "PMC_A",
                             figures=["PMC_A_Figure_1.png"],
                             tables=["PMC_A_Table_1.png"])
    _write_manifest(docl, "DOCLING")

    # Labels (legacy shared files — no per-variant yet)
    ann_root.mkdir()
    (ann_root / "annotations_json_tables_full.json").write_text(json.dumps({
        "PMC_A_Table_1.png": "correct",
        "PMC_A_Table_2.png": "incorrect",
    }))
    (ann_root / "annotations_json_tables_docling.json").write_text(json.dumps({
        "PMC_A_Table_1.png": "correct",
    }))
    (ann_root / "annotations_json_figures.json").write_text(json.dumps({
        "PMC_A_Figure_1.png": "correct",
    }))

    gt_csv.write_text(
        "pmcid,missed_figures,missed_tables,total_tables\n"
        "PMC_A,0,0,2\n"
    )
    return {"sweeps_root": sweeps_root, "ann_root": ann_root, "gt_csv": gt_csv}


def test_main_full_round_trip(tmp_path, scr, monkeypatch, capsys) -> None:
    s = _build_eval_scenario(tmp_path, scr, monkeypatch)
    md_out = tmp_path / "out.md"
    json_out = tmp_path / "out.json"
    rc = scr.main([
        "--sweeps-root", str(s["sweeps_root"]),
        "--md-out", str(md_out),
        "--json-out", str(json_out),
    ])
    assert rc == 0

    rows = json.loads(json_out.read_text())
    by_key = {(r["variant"], r["kind"]): r for r in rows}

    # Baseline HYBRID: 2 emitted tables (T_1 correct, T_2 incorrect) → TP=1, FP=1
    bt = by_key[("baseline", "tables")]
    assert bt["tp"] == 1
    assert bt["fp"] == 1
    assert bt["fn"] == 1  # total_tables=2 - tp=1
    assert bt["precision"] == 0.5
    assert bt["recall"] == 0.5
    assert bt["f1"] == 0.5

    # Docling: 1 emitted table (T_1 correct) → TP=1, FP=0
    dt = by_key[("detector_docling", "tables")]
    assert dt["tp"] == 1
    assert dt["fp"] == 0
    assert dt["fn"] == 1
    assert dt["precision"] == 1.0
    assert dt["recall"] == 0.5

    # Figures: 1 emitted, 1 correct → P=1.0
    assert by_key[("baseline", "figures")]["precision"] == 1.0

    # Markdown report has the expected header
    assert "# Per-variant PDF-extraction P/R/F1" in md_out.read_text()


def test_main_to_stdout_when_no_md_out(tmp_path, scr, monkeypatch, capsys) -> None:
    s = _build_eval_scenario(tmp_path, scr, monkeypatch)
    rc = scr.main(["--sweeps-root", str(s["sweeps_root"])])
    assert rc == 0
    out = capsys.readouterr().out
    assert "# Per-variant PDF-extraction P/R/F1" in out
    assert "baseline" in out
    assert "detector_docling" in out


def test_main_no_legacy_fallback_drops_to_zero_when_no_per_variant(
    tmp_path, scr, monkeypatch, capsys
) -> None:
    s = _build_eval_scenario(tmp_path, scr, monkeypatch)
    rc = scr.main([
        "--sweeps-root", str(s["sweeps_root"]),
        "--no-legacy-fallback",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    # Without legacy fallback, no per-variant files exist → TP=0 everywhere
    assert "| baseline | tables | — | 0.0% | — | 0 |" in out or \
           "| baseline | tables | 0.0% | 0.0% | — | 0 |" in out or \
           "| baseline | tables | — | — | — | 0 |" in out
