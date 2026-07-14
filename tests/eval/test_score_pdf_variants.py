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


from tests.paths import REPO_ROOT as _REPO_ROOT
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
    # Exact match only (2026-05-19) — no prefix matching.
    ("correct", "tp"),
    ("correct figure", "skip"),                # no longer prefix-matches "correct"
    ("correct-ish", "skip"),                   # same
    ("icon", "fp"),
    ("incorrect", "fp"),
    ("CORRECT", "tp"),                          # case-insensitive normalize OK
    ("  correct  ", "tp"),                      # trimmed OK
    ("other", "skip"),
    ("skipped", "skip"),
    ("", "skip"),
])
def test_classify_figure(scr, label, expected) -> None:
    assert scr.classify_figure(label) == expected


@pytest.mark.parametrize("label,expected", [
    # Exact match only (2026-05-19) — no prefix matching.
    ("correct", "tp"),
    ("missing footnotes", "tp"),                # exact match for the bare label
    ("missing footnotes — text below cut", "skip"),  # no longer prefix-matches
    ("incorrect", "fp"),
    ("wrong caption", "fp"),                    # exact match
    ("wrong caption (footer matched)", "skip"),  # no longer prefix-matches
    ("crop is too big", "skip"),                # legacy label not in exact set
    ("weird crop? check later", "skip"),        # same
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
        "--legacy",
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
    rc = scr.main([
        "--legacy","--sweeps-root", str(s["sweeps_root"])])
    assert rc == 0
    out = capsys.readouterr().out
    # Tab-aligned terminal format (2026-05-19): header row + data rows
    # separated by tabs.  No markdown title.
    assert "variant" in out
    assert "baseline" in out
    assert "detector_docling" in out
    assert "\t" in out


# ── Rubric: load_rubric ───────────────────────────────────────────────────────


def test_load_rubric_falls_back_when_path_missing(tmp_path, scr) -> None:
    rubric = scr.load_rubric(tmp_path / "does_not_exist.yaml")
    # built-in defaults present
    assert "correct" in rubric
    assert rubric["correct"]["crop"] == 1.0


def test_load_rubric_reads_yaml(tmp_path, scr) -> None:
    p = tmp_path / "rubric.yaml"
    p.write_text(
        "labels:\n"
        "  correct: {crop: 1.0, caption: 1.0, footnote: 1.0}\n"
        "  custom_label: {crop: 0.5, caption: n/a, footnote: n/a}\n"
    )
    rubric = scr.load_rubric(p)
    assert rubric["custom_label"]["crop"] == 0.5
    assert rubric["custom_label"]["caption"] == "n/a"


def test_load_rubric_handles_malformed_yaml(tmp_path, scr) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("not: valid: yaml: :::")
    # Should fall back without raising
    rubric = scr.load_rubric(p)
    assert "correct" in rubric


# ── Rubric: parse_label_to_rubric ────────────────────────────────────────────


def test_parse_label_direct_match(scr) -> None:
    rubric = scr.load_rubric(None)
    out = scr.parse_label_to_rubric("correct", rubric)
    assert out == {"crop": 1.0, "caption": 1.0, "footnote": 1.0, "mask": 1.0}


def test_parse_label_no_prefix_match(scr) -> None:
    """Prefix matching is disabled — a descriptive suffix on a known label
    is treated as an unknown label and returns None.  User must add an
    explicit rubric entry (via --review-unknown or by editing YAML)."""
    rubric = scr.load_rubric(None)
    out = scr.parse_label_to_rubric("wrong caption (footer matched as caption)", rubric)
    assert out is None


def test_parse_label_compound_not_auto_combined(scr) -> None:
    """Compound semantics are NOT auto-derived AND no longer prefix-match.
    A '+'-joined string with no explicit rubric entry returns None."""
    rubric = scr.load_rubric(None)
    out = scr.parse_label_to_rubric("wrong caption + missing footnotes", rubric)
    assert out is None


def test_parse_label_explicit_compound_in_rubric(scr, tmp_path) -> None:
    """The user can still get compound semantics by adding the explicit
    label to the rubric YAML."""
    p = tmp_path / "rubric.yaml"
    p.write_text(
        "labels:\n"
        "  wrong caption and missing footnotes:\n"
        "    crop: 1.0\n"
        "    caption: 0.0\n"
        "    footnote: 0.0\n"
    )
    rubric = scr.load_rubric(p)
    out = scr.parse_label_to_rubric("wrong caption and missing footnotes", rubric)
    assert out == {"crop": 1.0, "caption": 0.0, "footnote": 0.0}


def test_parse_label_unknown_returns_none(scr) -> None:
    rubric = scr.load_rubric(None)
    assert scr.parse_label_to_rubric("totally unknown label", rubric) is None


def test_parse_label_empty_string_returns_none(scr) -> None:
    rubric = scr.load_rubric(None)
    assert scr.parse_label_to_rubric("", rubric) is None
    assert scr.parse_label_to_rubric("   ", rubric) is None


def test_parse_label_case_insensitive(scr) -> None:
    rubric = scr.load_rubric(None)
    out = scr.parse_label_to_rubric("CORRECT", rubric)
    assert out["crop"] == 1.0


# ── score_kind_rubric ─────────────────────────────────────────────────────────


def test_score_kind_rubric_all_correct(scr) -> None:
    rubric = scr.load_rubric(None)
    emitted = {"a.png", "b.png"}
    labels = {"a.png": "correct", "b.png": "correct"}
    out = scr.score_kind_rubric(emitted, labels, rubric, total_actual=2)
    # Crop: 2 TP, 0 FP, FN=0
    assert out["by_dim"]["crop"]["tp"] == 2
    assert out["by_dim"]["crop"]["fp"] == 0
    assert out["by_dim"]["crop"]["f1"] == 1.0
    # Caption/footnote: 2 TP, 0 FP
    assert out["by_dim"]["caption"]["tp"] == 2
    assert out["by_dim"]["footnote"]["tp"] == 2
    # Strict: both items perfect → 2 TP
    assert out["strict"]["tp"] == 2
    assert out["strict"]["fp"] == 0


def test_score_kind_rubric_wrong_caption_breaks_strict_only(scr) -> None:
    # Use an in-memory rubric so the test doesn't depend on which labels
    # happen to exist in the live YAML.
    rubric = {
        "wrong caption": {"crop": 1.0, "caption": 0.0, "footnote": 1.0, "mask": 1.0},
    }
    emitted = {"a.png"}
    labels = {"a.png": "wrong caption"}
    out = scr.score_kind_rubric(emitted, labels, rubric, total_actual=1)
    # Crop dim: TP (crop score 1.0)
    assert out["by_dim"]["crop"]["tp"] == 1
    # Caption dim: FP (0.0)
    assert out["by_dim"]["caption"]["fp"] == 1
    # Footnote dim: TP (1.0)
    assert out["by_dim"]["footnote"]["tp"] == 1
    # Strict: FP because caption < 1.0
    assert out["strict"]["tp"] == 0
    assert out["strict"]["fp"] == 1


def test_score_kind_rubric_incorrect_excludes_caption_footnote(scr) -> None:
    """'incorrect' has crop=0, caption=n/a, footnote=n/a.  Counts as crop FP,
    excluded from caption/footnote dims entirely."""
    rubric = scr.load_rubric(None)
    emitted = {"a.png"}
    labels = {"a.png": "incorrect"}
    out = scr.score_kind_rubric(emitted, labels, rubric, total_actual=1)
    assert out["by_dim"]["crop"]["fp"] == 1
    # Caption/footnote dims: nothing scored → None
    assert out["by_dim"]["caption"] is None
    assert out["by_dim"]["footnote"] is None
    # Strict: every dim is n/a or 0 → no dim scored ≥ 1.0 → skipped from strict
    assert out["strict"]["tp"] == 0


def test_score_kind_rubric_crop_too_big_minor_partial(scr) -> None:
    """Severity-split rubric (2026-05-19): crop too big minor → crop=0.75.
    At threshold 0.5 → crop TP.  Strict (threshold 1.0) → FP."""
    rubric = scr.load_rubric(None)
    emitted = {"a.png"}
    labels = {"a.png": "crop too big minor"}
    out = scr.score_kind_rubric(emitted, labels, rubric, total_actual=1)
    assert out["by_dim"]["crop"]["tp"] == 1   # 0.75 >= 0.5
    assert out["strict"]["fp"] == 1            # 0.75 < 1.0


def test_score_kind_rubric_crop_too_small_major_below_dim_threshold(scr) -> None:
    """Severity-split rubric: crop score 0.25, below 0.5 threshold → crop FP.

    Uses an in-memory rubric so the test doesn't depend on whether the
    `crop too small major` label is in the live YAML at any given moment.
    """
    rubric = {
        "crop too small major": {"crop": 0.25, "caption": 1.0, "footnote": 1.0, "mask": 0.0},
    }
    emitted = {"a.png"}
    labels = {"a.png": "crop too small major"}
    out = scr.score_kind_rubric(emitted, labels, rubric, total_actual=1)
    assert out["by_dim"]["crop"]["fp"] == 1


def test_score_kind_rubric_figures_kind_overrides_footnote_to_na(scr) -> None:
    """Fix A (2026-05-19): kind="figures" forces footnote=n/a regardless
    of rubric value.  `correct` has footnote=1.0 in the rubric — without
    the override it would count as a footnote TP for a figure item.
    With every figure routed to footnote=n/a, the footnote dim has
    tp+fp=0 and is reported as None (scorer collapses empty dims)."""
    rubric = scr.load_rubric(None)
    emitted = {"fig.png"}
    labels = {"fig.png": "correct"}
    out_fig = scr.score_kind_rubric(
        emitted, labels, rubric, fn_count=0, kind="figures",
    )
    # footnote dim should be None (every item routed to n/a, no TP/FP).
    assert out_fig["by_dim"]["footnote"] is None

    # Same label on a table still counts as footnote TP.
    out_tab = scr.score_kind_rubric(
        emitted, labels, rubric, total_actual=1, kind="tables",
    )
    assert out_tab["by_dim"]["footnote"]["tp"] == 1


def test_score_kind_rubric_unknown_label_unrecognised(scr) -> None:
    rubric = scr.load_rubric(None)
    emitted = {"a.png"}
    labels = {"a.png": "totally novel free-text label"}
    out = scr.score_kind_rubric(emitted, labels, rubric, total_actual=1)
    assert out["counts"]["unrecognised"] == 1


def test_score_kind_rubric_unlabelled_counted(scr) -> None:
    rubric = scr.load_rubric(None)
    emitted = {"a.png", "b.png"}
    labels = {"a.png": "correct"}
    out = scr.score_kind_rubric(emitted, labels, rubric, total_actual=2)
    assert out["counts"]["unlabelled"] == 1


# ── Mask dimension ────────────────────────────────────────────────────────────


def test_mask_dim_correct_label_scores_tp(scr) -> None:
    """correct: crop=1, mask=1 → both dims TP."""
    rubric = scr.load_rubric(None)
    emitted = {"a.png"}
    labels = {"a.png": "correct"}
    out = scr.score_kind_rubric(emitted, labels, rubric, total_actual=1)
    assert out["by_dim"]["crop"]["tp"] == 1
    assert out["by_dim"]["mask"]["tp"] == 1
    assert out["by_dim"]["mask"]["fp"] == 0


def test_mask_dim_icon_crop_fp_mask_tp(scr) -> None:
    """icon: crop=0 (figure-output FP) but mask=1 (masking-side OK)."""
    rubric = scr.load_rubric(None)
    emitted = {"a.png"}
    labels = {"a.png": "icon"}
    out = scr.score_kind_rubric(emitted, labels, rubric, total_actual=1)
    # crop: this is a figure-output FP
    assert out["by_dim"]["crop"]["fp"] == 1
    assert out["by_dim"]["crop"]["tp"] == 0
    # mask: icons are correctly masked
    assert out["by_dim"]["mask"]["tp"] == 1
    assert out["by_dim"]["mask"]["fp"] == 0
    # caption / footnote: n/a → not scored
    assert out["by_dim"]["caption"] is None
    assert out["by_dim"]["footnote"] is None
    # Icon counter
    assert out["counts"]["icon"] == 1


def test_mask_dim_incorrect_both_dims_fp(scr) -> None:
    """incorrect (random paragraph, not an image at all): crop=0 AND mask=0."""
    rubric = scr.load_rubric(None)
    emitted = {"a.png"}
    labels = {"a.png": "incorrect"}
    out = scr.score_kind_rubric(emitted, labels, rubric, total_actual=1)
    assert out["by_dim"]["crop"]["fp"] == 1
    assert out["by_dim"]["mask"]["fp"] == 1


def test_mask_dim_missed_figures_contribute_to_mask_fn(scr) -> None:
    """When the corpus has missed_figures > 0, mask recall is dragged down."""
    rubric = scr.load_rubric(None)
    emitted = {"a.png", "b.png"}
    labels = {"a.png": "correct", "b.png": "correct"}
    # 3 figures missed entirely (not detected → not masked).
    out = scr.score_kind_rubric(emitted, labels, rubric, fn_count=3)
    mask = out["by_dim"]["mask"]
    assert mask["tp"] == 2
    assert mask["fn"] == 3                     # uses fn_count
    assert mask["recall"] == pytest.approx(2 / 5)


def test_mask_dim_crop_metrics_unchanged_by_mask_addition(scr) -> None:
    """The crop dim still computes exactly as before — adding mask must not
    affect crop's TP/FP/FN math."""
    rubric = scr.load_rubric(None)
    emitted = {"a.png", "b.png", "c.png"}
    labels = {
        "a.png": "correct",
        "b.png": "incorrect",
        "c.png": "icon",
    }
    out = scr.score_kind_rubric(emitted, labels, rubric, total_actual=2)
    # Crop: TP=1 (correct), FP=2 (incorrect + icon)
    assert out["by_dim"]["crop"]["tp"] == 1
    assert out["by_dim"]["crop"]["fp"] == 2
    # FN: max(0, total_actual=2 - tp=1) = 1
    assert out["by_dim"]["crop"]["fn"] == 1


def test_mask_dim_fp_by_label_breakdown(scr) -> None:
    """Per-dim FP counts get broken down by label so we can say e.g.
    'of the crop FPs, N are icons, M are incorrect'."""
    rubric = scr.load_rubric(None)
    emitted = {"a.png", "b.png", "c.png", "d.png"}
    labels = {
        "a.png": "icon",
        "b.png": "icon",
        "c.png": "incorrect",
        "d.png": "correct",
    }
    out = scr.score_kind_rubric(emitted, labels, rubric, total_actual=2)
    crop_fps = out["by_dim"]["crop"]["fp_by_label"]
    assert crop_fps == {"icon": 2, "incorrect": 1}
    mask_fps = out["by_dim"]["mask"]["fp_by_label"]
    # Only incorrect contributes to mask FP (icons are mask=1)
    assert mask_fps == {"incorrect": 1}
    assert out["counts"]["icon"] == 2


def test_main_no_legacy_fallback_drops_to_zero_when_no_per_variant(
    tmp_path, scr, monkeypatch, capsys
) -> None:
    s = _build_eval_scenario(tmp_path, scr, monkeypatch)
    rc = scr.main([
        "--legacy",
        "--sweeps-root", str(s["sweeps_root"]),
        "--no-legacy-fallback",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    # Tab-aligned terminal format (2026-05-19): the "baseline" + "tables"
    # row exists, and its TP column is 0 (no per-variant labels exist and
    # legacy fallback is disabled).  Format-agnostic assertion: find the
    # baseline tables row and confirm TP=0 (column index after F1).
    lines = [ln for ln in out.splitlines() if "baseline" in ln and "tables" in ln]
    assert lines, "expected a 'baseline tables' row in stdout"
    cells = [c.strip() for c in lines[0].split("\t")]
    # Legacy columns: variant, kind, P, R, F1, TP, FP, FN, ...
    assert cells[5] == "0", f"expected TP=0, got {cells[5]!r} in row: {cells}"


# ── Review-unknown helpers ────────────────────────────────────────────────────


def test_collect_unknown_labels(scr) -> None:
    rubric = {"correct": {"crop": 1.0}}
    dicts = [
        {"a.png": "correct", "b.png": "mystery one"},
        {"c.png": "MYSTERY ONE", "d.png": "another"},  # case-insensitive dedupe
        {"e.png": ""},  # empty ignored
    ]
    out = scr.collect_unknown_labels(dicts, rubric)
    assert out == ["another", "mystery one"]


def test_append_rubric_entry_creates_file(scr, tmp_path) -> None:
    p = tmp_path / "rubric.yaml"
    scr.append_rubric_entry(p, "new label", {
        "crop": 1.0, "caption": 0.5, "footnote": "n/a", "mask": "skip"})
    txt = p.read_text()
    assert "labels:" in txt
    assert "'new label':" in txt
    assert "crop: 1.0" in txt
    assert "caption: 0.5" in txt
    assert "footnote: n/a" in txt
    assert "mask: skip" in txt


def test_append_rubric_entry_preserves_existing_content(scr, tmp_path) -> None:
    p = tmp_path / "rubric.yaml"
    p.write_text("# top comment\nlabels:\n  correct:\n    crop: 1.0\n")
    scr.append_rubric_entry(p, "added", {
        "crop": 0.0, "caption": "n/a", "footnote": "n/a", "mask": 0.0})
    txt = p.read_text()
    assert "# top comment" in txt
    assert "correct:" in txt
    assert "'added':" in txt
    # YAML parses cleanly
    import yaml
    data = yaml.safe_load(txt)
    assert "correct" in data["labels"]
    assert data["labels"]["added"] == {
        "crop": 0.0, "caption": "n/a", "footnote": "n/a", "mask": 0.0}


def test_append_rubric_entry_quotes_apostrophes(scr, tmp_path) -> None:
    p = tmp_path / "rubric.yaml"
    p.write_text("labels:\n")
    scr.append_rubric_entry(p, "label with 'quote'", {
        "crop": 1.0, "caption": 1.0, "footnote": 1.0, "mask": 1.0})
    import yaml
    data = yaml.safe_load(p.read_text())
    assert "label with 'quote'" in data["labels"]
