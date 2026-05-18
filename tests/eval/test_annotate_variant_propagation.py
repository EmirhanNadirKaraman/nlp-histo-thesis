"""
Tests for the ``eval/annotate.py`` extensions:

* ``ann_path`` with/without ``variant``
* ``load_annotations`` / ``save_annotations`` round-trip per variant
* ``load_share_map`` graceful when file missing
* ``propagate_label`` — single peer, multi peer, source variant excluded,
  per-peer write failure suppressed
* ``_loader_for_mode`` — legacy ``eval/out/`` layout vs sweep-dir layout
* ``_parse_cli`` — all new flag combinations
* ``resolve_pdf_dir`` / ``resolve_pdf_path`` — explicit, auto from manifest,
  None when neither
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# annotate.py is a regular module under eval/ — direct import works.
from eval import annotate as ann_mod


# ── ann_path ──────────────────────────────────────────────────────────────────


def test_ann_path_legacy(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ann_mod, "ANN_DIR", tmp_path / "annotations")
    p = ann_mod.ann_path("json_tables_full")
    assert p == tmp_path / "annotations" / "annotations_json_tables_full.json"


def test_ann_path_with_variant(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ann_mod, "ANN_DIR", tmp_path / "annotations")
    p = ann_mod.ann_path("json_tables_full", variant="baseline_evalcfg")
    assert p == tmp_path / "annotations" / "baseline_evalcfg" / "json_tables_full.json"


# ── load / save round-trip ────────────────────────────────────────────────────


def test_save_then_load_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ann_mod, "ANN_DIR", tmp_path)
    monkeypatch.setattr(ann_mod, "SHARE_MAP_PATH", tmp_path / "share_map.json")

    ann = {"PMC_X_Table_1_p2.png": "correct", "PMC_Y_Table_1_p1.png": "incorrect"}
    ann_mod.save_annotations("json_tables_full", ann, variant="E1_baseline")
    loaded = ann_mod.load_annotations("json_tables_full", variant="E1_baseline")
    assert loaded == ann
    # Path matches the per-variant layout
    assert (tmp_path / "E1_baseline" / "json_tables_full.json").exists()


def test_load_returns_empty_dict_when_file_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ann_mod, "ANN_DIR", tmp_path)
    assert ann_mod.load_annotations("json_tables_full") == {}
    assert ann_mod.load_annotations("json_tables_full", variant="ghost") == {}


# ── load_share_map ────────────────────────────────────────────────────────────


def test_load_share_map_missing_returns_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ann_mod, "SHARE_MAP_PATH", tmp_path / "missing.json")
    assert ann_mod.load_share_map() == {}


def test_load_share_map_malformed_returns_empty(tmp_path, monkeypatch) -> None:
    bad = tmp_path / "share_map.json"
    bad.write_text("not valid {{")
    monkeypatch.setattr(ann_mod, "SHARE_MAP_PATH", bad)
    assert ann_mod.load_share_map() == {}


def test_load_share_map_valid(tmp_path, monkeypatch) -> None:
    sm = {"PMC_X_Table_1_p2.png": ["baseline", "tatr_095"]}
    p = tmp_path / "share_map.json"
    p.write_text(json.dumps(sm))
    monkeypatch.setattr(ann_mod, "SHARE_MAP_PATH", p)
    assert ann_mod.load_share_map() == sm


# ── propagate_label ───────────────────────────────────────────────────────────


def test_propagate_label_writes_to_peers_but_not_source(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ann_mod, "ANN_DIR", tmp_path)
    share_map = {"k.png": ["E1", "E2", "E3"]}
    peers = ann_mod.propagate_label(
        key="k.png",
        label="correct",
        mode="json_tables_full",
        source_variant="E1",
        share_map=share_map,
    )
    assert peers == ["E2", "E3"]
    # Both peer files were written with the label
    for v in peers:
        data = json.loads((tmp_path / v / "json_tables_full.json").read_text())
        assert data["k.png"] == "correct"
    # Source variant must NOT have been written
    assert not (tmp_path / "E1" / "json_tables_full.json").exists()


def test_propagate_label_merges_with_existing_peer_labels(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ann_mod, "ANN_DIR", tmp_path)
    # Pre-seed E2 with an existing label for a different key
    (tmp_path / "E2").mkdir()
    (tmp_path / "E2" / "json_tables_full.json").write_text(json.dumps({
        "other.png": "incorrect",
    }))
    ann_mod.propagate_label("k.png", "correct", "json_tables_full",
                             source_variant="E1", share_map={"k.png": ["E1", "E2"]})
    data = json.loads((tmp_path / "E2" / "json_tables_full.json").read_text())
    assert data == {"other.png": "incorrect", "k.png": "correct"}


def test_propagate_label_no_peers_returns_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ann_mod, "ANN_DIR", tmp_path)
    # Only the source variant emitted this crop
    peers = ann_mod.propagate_label("solo.png", "correct", "json_tables_full",
                                     source_variant="E1",
                                     share_map={"solo.png": ["E1"]})
    assert peers == []
    assert not (tmp_path / "E1").exists()


def test_propagate_label_unknown_key_returns_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ann_mod, "ANN_DIR", tmp_path)
    peers = ann_mod.propagate_label("ghost.png", "correct", "json_tables_full",
                                     source_variant="E1", share_map={})
    assert peers == []


def test_propagate_label_per_peer_write_failure_suppressed(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(ann_mod, "ANN_DIR", tmp_path)
    # Make _atomic_write fail for E2 only by patching it to raise when the
    # target path contains "E2".
    orig_atomic = ann_mod._atomic_write

    def _broken_atomic(path, data):
        if "E2" in str(path):
            raise OSError("E2 disk full")
        return orig_atomic(path, data)

    monkeypatch.setattr(ann_mod, "_atomic_write", _broken_atomic)
    peers = ann_mod.propagate_label("k.png", "correct", "json_tables_full",
                                     source_variant="E1",
                                     share_map={"k.png": ["E1", "E2", "E3"]})
    # E3 succeeds, E2 fails but is suppressed (no exception out)
    assert peers == ["E3"]
    out = capsys.readouterr().out
    assert "propagation to E2 failed" in out


# ── _loader_for_mode ──────────────────────────────────────────────────────────


def test_loader_legacy_layout_uses_subdirs() -> None:
    folder, glob, parser = ann_mod._loader_for_mode("json_tables_full", ann_mod.OUT_DIR)
    assert folder == ann_mod.OUT_DIR / "json" / "full"
    assert glob == "*_media.json"


def test_loader_sweep_layout_uses_flat_json_dir(tmp_path) -> None:
    sweep = tmp_path / "sweep"
    folder, glob, parser = ann_mod._loader_for_mode("json_tables_full", sweep)
    assert folder == sweep / "json"  # flat — no /full /docling /docling_recon
    folder2, _, _ = ann_mod._loader_for_mode("json_tables_docling", sweep)
    assert folder2 == sweep / "json"
    folder3, _, _ = ann_mod._loader_for_mode("json_tables_docling_recon", sweep)
    assert folder3 == sweep / "json"


def test_loader_text_modes_route_correctly(tmp_path) -> None:
    sweep = tmp_path / "sweep"
    assert ann_mod._loader_for_mode("text", sweep)[0] == sweep / "text"
    assert ann_mod._loader_for_mode("text_raw", sweep)[0] == sweep / "text_raw"
    assert ann_mod._loader_for_mode("docling_full", sweep)[0] == sweep / "docling_full"


# ── _parse_cli ────────────────────────────────────────────────────────────────


def test_parse_cli_minimal() -> None:
    args = ann_mod._parse_cli(["json_tables_full"])
    assert args.mode == "json_tables_full"
    assert args.sweep is None
    assert args.variant is None
    assert args.pdf_dir is None


def test_parse_cli_full_flags() -> None:
    args = ann_mod._parse_cli([
        "json_tables_full",
        "--sweep", "out/sweeps/baseline",
        "--variant", "E1_baseline",
        "--pdf-dir", "/tmp/pdfs",
    ])
    assert args.mode == "json_tables_full"
    assert args.sweep == Path("out/sweeps/baseline")
    assert args.variant == "E1_baseline"
    assert args.pdf_dir == Path("/tmp/pdfs")


def test_parse_cli_invalid_mode_exits() -> None:
    with pytest.raises(SystemExit):
        ann_mod._parse_cli(["bogus_mode"])


# ── resolve_pdf_dir / resolve_pdf_path ────────────────────────────────────────


def _make_manifest(sweep_dir: Path, pdf_dir: str) -> None:
    (sweep_dir / "run_metadata").mkdir(parents=True, exist_ok=True)
    (sweep_dir / "run_metadata" / "run_TEST.json").write_text(json.dumps({
        "input": {"pdf_dir": pdf_dir},
    }))


def test_resolve_pdf_dir_explicit_wins(tmp_path) -> None:
    sweep = tmp_path / "sweep"
    _make_manifest(sweep, "/manifest/path")
    out = ann_mod.resolve_pdf_dir(sweep, Path("/explicit/path"))
    assert out == Path("/explicit/path")


def test_resolve_pdf_dir_from_manifest(tmp_path) -> None:
    sweep = tmp_path / "sweep"
    _make_manifest(sweep, "/manifest/path")
    out = ann_mod.resolve_pdf_dir(sweep, None)
    assert out == Path("/manifest/path")


def test_resolve_pdf_dir_no_sweep_no_explicit() -> None:
    assert ann_mod.resolve_pdf_dir(None, None) is None


def test_resolve_pdf_dir_sweep_without_manifest_returns_none(tmp_path) -> None:
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    assert ann_mod.resolve_pdf_dir(sweep, None) is None


def test_resolve_pdf_dir_malformed_manifest_returns_none(tmp_path) -> None:
    sweep = tmp_path / "sweep"
    (sweep / "run_metadata").mkdir(parents=True)
    (sweep / "run_metadata" / "run_BAD.json").write_text("not json {")
    assert ann_mod.resolve_pdf_dir(sweep, None) is None


def test_resolve_pdf_path(tmp_path) -> None:
    pdf_dir = tmp_path / "pdfs"
    p = ann_mod.resolve_pdf_path("PMC_ABC", pdf_dir)
    assert p == pdf_dir / "PMC_ABC.pdf"
    # No pdf_dir → None
    assert ann_mod.resolve_pdf_path("PMC_X", None) is None
