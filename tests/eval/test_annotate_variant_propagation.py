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


# ── resolve_visualization_path ────────────────────────────────────────────────


def test_resolve_visualization_path_returns_existing_file(tmp_path) -> None:
    sweep = tmp_path / "sweep"
    viz_dir = sweep / "visualization"
    viz_dir.mkdir(parents=True)
    target = viz_dir / "PMC_ABC_layout_vis.pdf"
    target.write_bytes(b"%PDF-1.4\n")
    assert ann_mod.resolve_visualization_path(sweep, "PMC_ABC") == target


def test_resolve_visualization_path_none_when_file_absent(tmp_path) -> None:
    sweep = tmp_path / "sweep"
    (sweep / "visualization").mkdir(parents=True)
    # File doesn't exist
    assert ann_mod.resolve_visualization_path(sweep, "PMC_MISSING") is None


def test_resolve_visualization_path_none_when_sweep_none() -> None:
    assert ann_mod.resolve_visualization_path(None, "PMC_X") is None


def test_resolve_visualization_path_none_when_dir_missing(tmp_path) -> None:
    """Sweep dir exists but visualization/ subdir does not (e.g. viz disabled)."""
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    assert ann_mod.resolve_visualization_path(sweep, "PMC_X") is None


# ── open_pdf ──────────────────────────────────────────────────────────────────


def test_open_pdf_no_op_on_missing_path(tmp_path, monkeypatch) -> None:
    """``open`` must not run if the path doesn't exist."""
    calls = []
    monkeypatch.setattr(ann_mod.os, "system", lambda cmd: calls.append(cmd) or 0)
    ann_mod.open_pdf(tmp_path / "does_not_exist.pdf")
    assert calls == []


def test_open_pdf_no_op_on_none() -> None:
    ann_mod.open_pdf(None)  # no exception, no os.system call


def test_open_pdf_invokes_open_then_osascript_with_page(tmp_path, monkeypatch) -> None:
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    calls = []
    monkeypatch.setattr(ann_mod.os, "system",
                        lambda cmd: calls.append(cmd) or 0)
    monkeypatch.setattr(ann_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(ann_mod.sys, "platform", "darwin")
    ann_mod.open_pdf(pdf, page=3)
    # First call: open the PDF
    assert calls[0].startswith("open ")
    assert str(pdf) in calls[0]
    # Second call: osascript navigating to page
    assert any("osascript" in c for c in calls[1:])
    nav = next(c for c in calls if "osascript" in c)
    assert "Preview" in nav
    assert "command down, option down" in nav
    assert '"3"' in nav  # the page keystroke


def test_open_pdf_skips_osascript_when_page_is_one(tmp_path, monkeypatch) -> None:
    """Page 1 is already shown on open — no navigation needed."""
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    calls = []
    monkeypatch.setattr(ann_mod.os, "system",
                        lambda cmd: calls.append(cmd) or 0)
    monkeypatch.setattr(ann_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(ann_mod.sys, "platform", "darwin")
    ann_mod.open_pdf(pdf, page=1)
    assert len(calls) == 1
    assert calls[0].startswith("open ")


def test_open_pdf_skips_osascript_when_page_is_none(tmp_path, monkeypatch) -> None:
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    calls = []
    monkeypatch.setattr(ann_mod.os, "system",
                        lambda cmd: calls.append(cmd) or 0)
    monkeypatch.setattr(ann_mod.sys, "platform", "darwin")
    ann_mod.open_pdf(pdf, page=None)
    assert len(calls) == 1


def test_open_pdf_skips_osascript_on_non_darwin(tmp_path, monkeypatch) -> None:
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    calls = []
    monkeypatch.setattr(ann_mod.os, "system",
                        lambda cmd: calls.append(cmd) or 0)
    monkeypatch.setattr(ann_mod.sys, "platform", "linux")
    ann_mod.open_pdf(pdf, page=5)
    # Only the open call; no osascript invocation
    assert len(calls) == 1
    assert calls[0].startswith("open ")
    assert "osascript -e" not in calls[0]


# ── UX helpers: recent labels + next-unlabelled cursor + cancel ──────────────


def test_collect_recent_labels_returns_only_custom() -> None:
    ann = {
        "a.png": "correct", "b.png": "incorrect",
        "d.png": "skipped", "e.png": "table_in_figure", "f.png": "wrong caption",
        "g.png": "",  # empty value — ignored
    }
    recent = ann_mod._collect_recent_labels(ann)
    assert recent == ["table_in_figure", "wrong caption"]


def test_collect_recent_labels_handles_empty() -> None:
    assert ann_mod._collect_recent_labels({}) == []


def test_collect_recent_labels_filters_standard_only() -> None:
    ann = {"a.png": "correct", "b.png": "skipped"}
    assert ann_mod._collect_recent_labels(ann) == []


def _mk_item(doc_id: str = "PMC_X", label: str = "tables", index: int = 0,
             image_path: str = ""):
    """Build a minimal Item with an ann_key derivable from image_path."""
    return ann_mod.Item(
        doc_id=doc_id, label=label, index=index, text="caption",
        meta={"image_path": image_path or f"out/x/{doc_id}_t{index}.png"},
    )


def test_next_unlabelled_index_finds_first_gap() -> None:
    items = [_mk_item(index=i, image_path=f"a_{i}.png") for i in range(5)]
    # Label items 0, 1, 3
    ann = {ann_mod.ann_key(items[0]): "correct",
           ann_mod.ann_key(items[1]): "correct",
           ann_mod.ann_key(items[3]): "incorrect"}
    # start=0 → first unlabelled is item 2
    assert ann_mod._next_unlabelled_index(items, ann, 0) == 2
    # start=3 → next unlabelled is item 4
    assert ann_mod._next_unlabelled_index(items, ann, 3) == 4


def test_next_unlabelled_index_returns_total_when_done() -> None:
    items = [_mk_item(index=i, image_path=f"a_{i}.png") for i in range(3)]
    ann = {ann_mod.ann_key(item): "correct" for item in items}
    # Everything labelled → returns past-end
    assert ann_mod._next_unlabelled_index(items, ann, 0) == 3


def test_next_unlabelled_index_skips_propagated_labels() -> None:
    """Simulates propagation: items 1 and 2 got labels mid-session (from
    a peer variant); after labelling item 0, cursor should jump to item 3."""
    items = [_mk_item(index=i, image_path=f"a_{i}.png") for i in range(5)]
    ann = {
        ann_mod.ann_key(items[0]): "correct",  # just-labelled
        ann_mod.ann_key(items[1]): "correct",  # propagated
        ann_mod.ann_key(items[2]): "incorrect",  # propagated
    }
    # Cursor was on 0; after labelling, search starts at 1.
    assert ann_mod._next_unlabelled_index(items, ann, 1) == 3


def test_read_label_empty_input_returns_none(monkeypatch, capsys) -> None:
    """Empty enter cancels — caller treats None as "stay on this item"."""
    monkeypatch.setattr("builtins.input", lambda: "")
    # Bypass tty setup — termios calls would fail in pytest
    monkeypatch.setattr(ann_mod.termios, "tcgetattr", lambda fd: None)
    monkeypatch.setattr(ann_mod.termios, "tcsetattr", lambda *a, **kw: None)
    monkeypatch.setattr(ann_mod.sys.stdin, "fileno", lambda: 0)
    assert ann_mod.read_label() is None


def test_read_label_whitespace_input_returns_none(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda: "   ")
    monkeypatch.setattr(ann_mod.termios, "tcgetattr", lambda fd: None)
    monkeypatch.setattr(ann_mod.termios, "tcsetattr", lambda *a, **kw: None)
    monkeypatch.setattr(ann_mod.sys.stdin, "fileno", lambda: 0)
    assert ann_mod.read_label() is None


def test_read_label_keyboard_interrupt_returns_none(monkeypatch) -> None:
    def _boom():
        raise KeyboardInterrupt
    monkeypatch.setattr("builtins.input", _boom)
    monkeypatch.setattr(ann_mod.termios, "tcgetattr", lambda fd: None)
    monkeypatch.setattr(ann_mod.termios, "tcsetattr", lambda *a, **kw: None)
    monkeypatch.setattr(ann_mod.sys.stdin, "fileno", lambda: 0)
    assert ann_mod.read_label() is None


def test_read_label_eof_returns_none(monkeypatch) -> None:
    def _eof():
        raise EOFError
    monkeypatch.setattr("builtins.input", _eof)
    monkeypatch.setattr(ann_mod.termios, "tcgetattr", lambda fd: None)
    monkeypatch.setattr(ann_mod.termios, "tcsetattr", lambda *a, **kw: None)
    monkeypatch.setattr(ann_mod.sys.stdin, "fileno", lambda: 0)
    assert ann_mod.read_label() is None


def test_read_label_text_returns_stripped(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda: "  weird crop  ")
    monkeypatch.setattr(ann_mod.termios, "tcgetattr", lambda fd: None)
    monkeypatch.setattr(ann_mod.termios, "tcsetattr", lambda *a, **kw: None)
    monkeypatch.setattr(ann_mod.sys.stdin, "fileno", lambda: 0)
    assert ann_mod.read_label() == "weird crop"


def test_read_label_numeric_picks_from_recent(monkeypatch) -> None:
    """Typing '2' should resolve to the 2nd recent label."""
    monkeypatch.setattr("builtins.input", lambda: "2")
    monkeypatch.setattr(ann_mod.termios, "tcgetattr", lambda fd: None)
    monkeypatch.setattr(ann_mod.termios, "tcsetattr", lambda *a, **kw: None)
    monkeypatch.setattr(ann_mod.sys.stdin, "fileno", lambda: 0)
    result = ann_mod.read_label(recent=["table_in_figure", "wrong caption", "icon"])
    assert result == "wrong caption"


def test_read_label_numeric_out_of_range_treated_as_literal(monkeypatch) -> None:
    """If the digit is outside the menu range, treat it as a literal label."""
    monkeypatch.setattr("builtins.input", lambda: "9")
    monkeypatch.setattr(ann_mod.termios, "tcgetattr", lambda fd: None)
    monkeypatch.setattr(ann_mod.termios, "tcsetattr", lambda *a, **kw: None)
    monkeypatch.setattr(ann_mod.sys.stdin, "fileno", lambda: 0)
    assert ann_mod.read_label(recent=["a", "b"]) == "9"


def test_read_label_no_recent_menu_pass_through(monkeypatch) -> None:
    """recent=None → no menu, raw text returned."""
    monkeypatch.setattr("builtins.input", lambda: "2")
    monkeypatch.setattr(ann_mod.termios, "tcgetattr", lambda fd: None)
    monkeypatch.setattr(ann_mod.termios, "tcsetattr", lambda *a, **kw: None)
    monkeypatch.setattr(ann_mod.sys.stdin, "fileno", lambda: 0)
    assert ann_mod.read_label() == "2"  # literal — no menu to look up


# ── Rubric-backed label menu ─────────────────────────────────────────────────


def test_load_rubric_labels_reads_file(tmp_path, monkeypatch) -> None:
    p = tmp_path / "rubric.yaml"
    p.write_text(
        "labels:\n"
        "  correct:\n    crop: 1.0\n"
        "  table_in_figure:\n    crop: 0.0\n"
        "  wrong caption:\n    crop: 1.0\n"
    )
    monkeypatch.setattr(ann_mod, "_RUBRIC_PATH", p)
    out = ann_mod._load_rubric_labels()
    # 'correct' is a standard label and filtered out
    assert out == ["table_in_figure", "wrong caption"]


def test_load_rubric_labels_missing_file_returns_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ann_mod, "_RUBRIC_PATH", tmp_path / "absent.yaml")
    assert ann_mod._load_rubric_labels() == []


def test_collect_label_menu_merges_rubric_and_ann(tmp_path, monkeypatch) -> None:
    p = tmp_path / "rubric.yaml"
    p.write_text(
        "labels:\n"
        "  table_in_figure:\n    crop: 0.0\n"
        "  wrong caption:\n    crop: 1.0\n"
    )
    monkeypatch.setattr(ann_mod, "_RUBRIC_PATH", p)
    ann = {"a.png": "weird new label", "b.png": "table_in_figure"}
    menu = ann_mod._collect_label_menu(ann)
    # Rubric labels first (sorted), then non-rubric extras from ann
    assert menu == ["table_in_figure", "wrong caption", "weird new label"]


def test_collect_label_menu_no_rubric_falls_back_to_recent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ann_mod, "_RUBRIC_PATH", tmp_path / "absent.yaml")
    ann = {"a.png": "fresh label"}
    assert ann_mod._collect_label_menu(ann) == ["fresh label"]


# ── Kind-aware menu filtering (block-based, 2026-05-19) ──────────────────────


_NEW_FORMAT_RUBRIC = (
    "shared_labels:\n"
    "  crop too big minor:\n    crop: 0.75\n"
    "  wrong caption:\n    crop: 1.0\n"
    "figure_labels:\n"
    "  correct figure:\n    crop: 1.0\n"
    "  icon:\n    crop: 0.0\n"
    "table_labels:\n"
    "  missing footnotes:\n    crop: 1.0\n"
    "  table_in_figure:\n    crop: 0.0\n"
)


def test_load_rubric_labels_filters_figure_kind(tmp_path, monkeypatch) -> None:
    p = tmp_path / "rubric.yaml"
    p.write_text(_NEW_FORMAT_RUBRIC)
    monkeypatch.setattr(ann_mod, "_RUBRIC_PATH", p)
    out = ann_mod._load_rubric_labels(item_kind="figure")
    # shared + figure-only, no table-only
    assert "crop too big minor" in out
    assert "wrong caption" in out
    assert "correct figure" in out
    assert "icon" in out
    assert "missing footnotes" not in out
    assert "table_in_figure" not in out


def test_load_rubric_labels_filters_table_kind(tmp_path, monkeypatch) -> None:
    p = tmp_path / "rubric.yaml"
    p.write_text(_NEW_FORMAT_RUBRIC)
    monkeypatch.setattr(ann_mod, "_RUBRIC_PATH", p)
    out = ann_mod._load_rubric_labels(item_kind="table")
    assert "crop too big minor" in out
    assert "wrong caption" in out
    assert "missing footnotes" in out
    assert "table_in_figure" in out
    assert "correct figure" not in out
    assert "icon" not in out


def test_load_rubric_labels_no_kind_returns_everything(tmp_path, monkeypatch) -> None:
    p = tmp_path / "rubric.yaml"
    p.write_text(_NEW_FORMAT_RUBRIC)
    monkeypatch.setattr(ann_mod, "_RUBRIC_PATH", p)
    out = ann_mod._load_rubric_labels(item_kind=None)
    assert {"crop too big minor", "wrong caption",
            "correct figure", "icon",
            "missing footnotes", "table_in_figure"}.issubset(set(out))


def test_load_rubric_labels_legacy_format_still_works(tmp_path, monkeypatch) -> None:
    """Old `labels:` block (no kind blocks) returns everything regardless
    of item_kind — back-compat."""
    p = tmp_path / "rubric.yaml"
    p.write_text(
        "labels:\n"
        "  correct figure:\n    crop: 1.0\n"
        "  missing footnotes:\n    crop: 1.0\n"
        "  wrong caption:\n    crop: 1.0\n"
    )
    monkeypatch.setattr(ann_mod, "_RUBRIC_PATH", p)
    for kind in ("figure", "table", None):
        out = ann_mod._load_rubric_labels(item_kind=kind)
        assert "correct figure" in out
        assert "missing footnotes" in out
        assert "wrong caption" in out


def test_collect_label_menu_filters_for_figure(tmp_path, monkeypatch) -> None:
    p = tmp_path / "rubric.yaml"
    p.write_text(_NEW_FORMAT_RUBRIC)
    monkeypatch.setattr(ann_mod, "_RUBRIC_PATH", p)
    menu = ann_mod._collect_label_menu({}, item_kind="figure")
    assert "missing footnotes" not in menu
    assert "table_in_figure" not in menu
    assert "correct figure" in menu
    assert "crop too big minor" in menu


def test_collect_label_menu_filters_for_table(tmp_path, monkeypatch) -> None:
    p = tmp_path / "rubric.yaml"
    p.write_text(_NEW_FORMAT_RUBRIC)
    monkeypatch.setattr(ann_mod, "_RUBRIC_PATH", p)
    menu = ann_mod._collect_label_menu({}, item_kind="table")
    assert "correct figure" not in menu
    assert "icon" not in menu
    assert "missing footnotes" in menu
    assert "crop too big minor" in menu


# ── Bbox-aware propagation (new share-map format) ─────────────────────────────


def test_propagate_label_bbox_aware_only_writes_matching_group(tmp_path, monkeypatch) -> None:
    """Source bbox matches Group A → only A peers receive the label.
    Group B (different bbox) is not touched even though it emits the same filename."""
    monkeypatch.setattr(ann_mod, "ANN_DIR", tmp_path)
    share_map = {
        "k.png": [
            {"bbox": [50.0, 500.0, 520.0, 300.0], "variants": ["E1", "E2"]},
            {"bbox": [48.0, 540.0, 525.0, 200.0], "variants": ["E3_relaxed"]},
        ],
    }
    source_bbox = {"x1": 50.0, "y1": 500.0, "x2": 520.0, "y2": 300.0}
    peers = ann_mod.propagate_label(
        "k.png", "correct", "json_tables_full",
        source_variant="E1", share_map=share_map, source_bbox=source_bbox,
    )
    assert peers == ["E2"]
    assert (tmp_path / "E2" / "json_tables_full.json").exists()
    assert not (tmp_path / "E3_relaxed").exists()


def test_propagate_label_bbox_no_match_propagates_nowhere(tmp_path, monkeypatch) -> None:
    """Source bbox matches NO group → no propagation."""
    monkeypatch.setattr(ann_mod, "ANN_DIR", tmp_path)
    share_map = {
        "k.png": [
            {"bbox": [50.0, 500.0, 520.0, 300.0], "variants": ["E1", "E2"]},
        ],
    }
    # Source has a wildly different bbox
    src = {"x1": 999.0, "y1": 999.0, "x2": 1000.0, "y2": 1000.0}
    peers = ann_mod.propagate_label(
        "k.png", "correct", "json_tables_full",
        source_variant="E1", share_map=share_map, source_bbox=src,
    )
    assert peers == []


def test_propagate_label_bbox_quantized_to_tolerance(tmp_path, monkeypatch) -> None:
    """Source bbox within 1pt of a group's bbox → matches via quantization."""
    monkeypatch.setattr(ann_mod, "ANN_DIR", tmp_path)
    share_map = {
        "k.png": [
            {"bbox": [50.0, 500.0, 520.0, 300.0], "variants": ["E1", "E2"]},
        ],
    }
    # Source bbox jitter: .4 each direction — quantize_bbox rounds to .0
    src = {"x1": 50.4, "y1": 500.3, "x2": 519.7, "y2": 300.2}
    peers = ann_mod.propagate_label(
        "k.png", "correct", "json_tables_full",
        source_variant="E1", share_map=share_map, source_bbox=src,
    )
    assert peers == ["E2"]


def test_propagate_label_legacy_format_still_works(tmp_path, monkeypatch) -> None:
    """Old flat-list share_map still propagates (with no bbox check)."""
    monkeypatch.setattr(ann_mod, "ANN_DIR", tmp_path)
    share_map = {"k.png": ["E1", "E2", "E3"]}  # legacy flat list
    peers = ann_mod.propagate_label(
        "k.png", "correct", "json_tables_full",
        source_variant="E1", share_map=share_map, source_bbox=None,
    )
    assert sorted(peers) == ["E2", "E3"]


def test_quantize_bbox_handles_missing(tmp_path) -> None:
    assert ann_mod._quantize_bbox(None) == (0.0, 0.0, 0.0, 0.0)
    assert ann_mod._quantize_bbox({}) == (0.0, 0.0, 0.0, 0.0)
    assert ann_mod._quantize_bbox({"x1": 50.4, "y1": 500.6}) == (50.0, 501.0, 0.0, 0.0)


def test_find_peers_empty_entry_returns_empty() -> None:
    assert ann_mod._find_peers(None, "E1", None) == []
    assert ann_mod._find_peers([], "E1", None) == []
