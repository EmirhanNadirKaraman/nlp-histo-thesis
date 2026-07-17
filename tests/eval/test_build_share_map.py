"""
Tests for ``scripts/eval/build_share_map.py``.

Covers:
* ``quantize_bbox`` — bbox rounding to grouping tolerance.
* ``_collect_with_bbox`` — scans a sweep's media JSONs and returns
  (filename, quantized-bbox) pairs.
* ``build`` — groups variants by (filename, bbox) across sweeps.
* malformed media.json / missing json subdir are tolerated.
* ``write`` — atomic JSON write, parent dir created.
* CLI ``main()`` — happy path, missing sweeps root → exit 2, empty root → exit 2.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


from tests.paths import REPO_ROOT as _REPO_ROOT
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "eval" / "build_share_map.py"


@pytest.fixture(scope="module")
def bsm():
    spec = importlib.util.spec_from_file_location("build_share_map_uut", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    try:
        yield mod
    finally:
        sys.modules.pop(spec.name, None)


# ── Fixture helpers ───────────────────────────────────────────────────────────


def _write_media(
    json_dir: Path,
    pmcid: str,
    *,
    figures: list[tuple[str, dict]] | None = None,
    tables: list[tuple[str, dict]] | None = None,
) -> None:
    """``figures`` / ``tables`` are lists of (filename, bbox-dict) tuples."""
    json_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"pmcid": pmcid, "figures": [], "tables": []}
    for name, bbox in figures or []:
        payload["figures"].append({"image_path": f"out/sweeps/x/figures/{name}",
                                    "bbox": bbox})
    for name, bbox in tables or []:
        payload["tables"].append({"image_path": f"out/sweeps/x/tables/{name}",
                                    "bbox": bbox})
    (json_dir / f"{pmcid}_media.json").write_text(json.dumps(payload))


def _bbox(x1=50.0, y1=500.0, x2=520.0, y2=300.0) -> dict:
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _build_sweep(sweeps_root: Path, name: str, *, pmcid: str, **kwargs) -> Path:
    sweep = sweeps_root / name
    _write_media(sweep / "json", pmcid, **kwargs)
    return sweep


# ── quantize_bbox ─────────────────────────────────────────────────────────────


def test_quantize_bbox_rounds_to_tol(bsm) -> None:
    bb = {"x1": 50.4, "y1": 500.6, "x2": 520.1, "y2": 300.9}
    assert bsm.quantize_bbox(bb, 1.0) == (50.0, 501.0, 520.0, 301.0)


def test_quantize_bbox_handles_missing_keys(bsm) -> None:
    assert bsm.quantize_bbox({}, 1.0) == (0.0, 0.0, 0.0, 0.0)


def test_quantize_bbox_handles_none_values(bsm) -> None:
    assert bsm.quantize_bbox({"x1": None, "y1": 5.0, "x2": "junk", "y2": 10.4}, 1.0) \
        == (0.0, 5.0, 0.0, 10.0)


def test_quantize_bbox_larger_tolerance_collapses(bsm) -> None:
    bb_a = {"x1": 50.0, "y1": 500.0, "x2": 520.0, "y2": 300.0}
    bb_b = {"x1": 51.5, "y1": 502.0, "x2": 521.0, "y2": 301.5}
    # tol=5 → both collapse to same key
    assert bsm.quantize_bbox(bb_a, 5.0) == bsm.quantize_bbox(bb_b, 5.0)


# ── _collect_with_bbox ────────────────────────────────────────────────────────


def test_collect_with_bbox_returns_pairs(tmp_path, bsm) -> None:
    bb1 = _bbox()
    bb2 = _bbox(y1=200.0, y2=100.0)
    sweep = _build_sweep(
        tmp_path / "sweeps", "v1",
        pmcid="PMC_A",
        figures=[("PMC_A_Figure_1_p2.png", bb1)],
        tables=[("PMC_A_Table_1_p3.png", bb1),
                ("PMC_A_Table_2_p3.png", bb2)],
    )
    out = bsm._collect_with_bbox(sweep / "json", tol=1.0)
    names = {n for n, _ in out}
    assert names == {
        "PMC_A_Figure_1_p2.png",
        "PMC_A_Table_1_p3.png",
        "PMC_A_Table_2_p3.png",
    }
    # Spot-check one bbox tuple
    for n, key in out:
        if n == "PMC_A_Table_2_p3.png":
            assert key == (50.0, 200.0, 520.0, 100.0)


def test_collect_with_bbox_skips_missing_image_path(tmp_path, bsm) -> None:
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    (json_dir / "PMC_X_media.json").write_text(json.dumps({
        "tables": [
            {"image_path": "out/x/PMC_X_Table_1_p1.png", "bbox": _bbox()},
            {"image_path": None, "bbox": _bbox()},
            {"image_path": "", "bbox": _bbox()},
            {"bbox": _bbox()},  # no image_path key at all
        ],
    }))
    out = bsm._collect_with_bbox(json_dir, tol=1.0)
    assert [n for n, _ in out] == ["PMC_X_Table_1_p1.png"]


def test_collect_with_bbox_tolerates_malformed_media_json(tmp_path, bsm) -> None:
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    (json_dir / "good_media.json").write_text(json.dumps({
        "tables": [{"image_path": "x/PMC_G_Table_1_p1.png", "bbox": _bbox()}],
    }))
    (json_dir / "bad_media.json").write_text("not valid json {{")
    out = bsm._collect_with_bbox(json_dir, tol=1.0)
    assert [n for n, _ in out] == ["PMC_G_Table_1_p1.png"]


# ── build ─────────────────────────────────────────────────────────────────────


def test_build_groups_by_bbox(tmp_path, bsm) -> None:
    """Two variants emit the same filename but different bboxes → two groups."""
    root = tmp_path / "sweeps"
    bb_small = _bbox()
    bb_big   = _bbox(y2=200.0)  # extended downward
    _build_sweep(root, "baseline",
                 pmcid="PMC_A", tables=[("PMC_A_Table_1_p1.png", bb_small)])
    _build_sweep(root, "no_expand",
                 pmcid="PMC_A", tables=[("PMC_A_Table_1_p1.png", bb_small)])
    _build_sweep(root, "relaxed",
                 pmcid="PMC_A", tables=[("PMC_A_Table_1_p1.png", bb_big)])

    sm = bsm.build(root)
    groups = sm["PMC_A_Table_1_p1.png"]
    assert len(groups) == 2
    # Find each group by bbox content
    small_group = next(g for g in groups if g["bbox"] == list(bb_small.values())
                       or g["bbox"] == [bb_small["x1"], bb_small["y1"], bb_small["x2"], bb_small["y2"]])
    big_group = next(g for g in groups if g != small_group)
    assert sorted(small_group["variants"]) == ["baseline", "no_expand"]
    assert big_group["variants"] == ["relaxed"]


def test_build_single_group_when_bboxes_identical(tmp_path, bsm) -> None:
    root = tmp_path / "sweeps"
    bb = _bbox()
    _build_sweep(root, "v1", pmcid="PMC_X", tables=[("a.png", bb)])
    _build_sweep(root, "v2", pmcid="PMC_X", tables=[("a.png", bb)])

    sm = bsm.build(root)
    assert len(sm["a.png"]) == 1
    assert sm["a.png"][0]["variants"] == ["v1", "v2"]


def test_build_sub_pt_jitter_collapses_into_one_group(tmp_path, bsm) -> None:
    """Bboxes within tolerance round to the same key."""
    root = tmp_path / "sweeps"
    bb_a = _bbox(x1=50.1, y1=500.2)
    bb_b = _bbox(x1=49.9, y1=500.4)  # within 1 pt of bb_a
    _build_sweep(root, "v1", pmcid="PMC_X", tables=[("a.png", bb_a)])
    _build_sweep(root, "v2", pmcid="PMC_X", tables=[("a.png", bb_b)])

    sm = bsm.build(root, bbox_tolerance=1.0)
    assert len(sm["a.png"]) == 1
    assert sm["a.png"][0]["variants"] == ["v1", "v2"]


def test_build_returns_sorted_filenames(tmp_path, bsm) -> None:
    root = tmp_path / "sweeps"
    _build_sweep(root, "a", pmcid="PMC_Z", tables=[("z.png", _bbox())])
    _build_sweep(root, "b", pmcid="PMC_A", tables=[("a.png", _bbox())])
    sm = bsm.build(root)
    assert list(sm.keys()) == ["a.png", "z.png"]
    for groups in sm.values():
        for g in groups:
            assert g["variants"] == sorted(g["variants"])


def test_build_skips_variant_without_json_subdir(tmp_path, bsm) -> None:
    root = tmp_path / "sweeps"
    _build_sweep(root, "with_json", pmcid="PMC_A", tables=[("a.png", _bbox())])
    (root / "no_json").mkdir()
    sm = bsm.build(root)
    assert sm == {"a.png": [{"bbox": [50.0, 500.0, 520.0, 300.0], "variants": ["with_json"]}]}


def test_build_raises_when_root_missing(tmp_path, bsm) -> None:
    with pytest.raises(FileNotFoundError):
        bsm.build(tmp_path / "does_not_exist")


def test_build_raises_when_root_empty(tmp_path, bsm) -> None:
    empty = tmp_path / "empty_sweeps"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="no sweep variants"):
        bsm.build(empty)


# ── write ─────────────────────────────────────────────────────────────────────


def test_write_creates_parent_dir(tmp_path, bsm) -> None:
    out = tmp_path / "deep" / "nested" / "share_map.json"
    payload = {"a.png": [{"bbox": [0, 0, 10, 10], "variants": ["v1", "v2"]}]}
    bsm.write(payload, out)
    assert out.exists()
    assert json.loads(out.read_text()) == payload


def test_write_is_atomic_no_partial_on_failure(tmp_path, bsm, monkeypatch) -> None:
    out = tmp_path / "share_map.json"
    bsm.write({"existing.png": [{"bbox": [0, 0, 10, 10], "variants": ["v1"]}]}, out)
    original = json.loads(out.read_text())
    def boom(self, target):
        raise OSError("disk full")
    monkeypatch.setattr(Path, "replace", boom)
    with pytest.raises(OSError):
        bsm.write({"new.png": [{"bbox": [0, 0, 10, 10], "variants": ["v2"]}]}, out)
    assert json.loads(out.read_text()) == original


# ── CLI main() ────────────────────────────────────────────────────────────────


def test_main_writes_share_map(tmp_path, bsm, capsys) -> None:
    root = tmp_path / "sweeps"
    _build_sweep(root, "v1", pmcid="PMC_X", tables=[("t.png", _bbox())])
    out = tmp_path / "share_map.json"

    rc = bsm.main(["--sweeps-root", str(root), "--out", str(out), "--quiet"])
    assert rc == 0
    data = json.loads(out.read_text())
    assert "t.png" in data
    assert data["t.png"] == [{"bbox": [50.0, 500.0, 520.0, 300.0], "variants": ["v1"]}]
    captured = capsys.readouterr()
    assert "unique crop filenames: 1" in captured.out


def test_main_returns_2_on_missing_root(tmp_path, bsm, capsys) -> None:
    out = tmp_path / "share_map.json"
    rc = bsm.main([
        "--sweeps-root", str(tmp_path / "missing"),
        "--out", str(out), "--quiet",
    ])
    assert rc == 2
    assert not out.exists()
    assert "error" in capsys.readouterr().out.lower()


def test_main_returns_2_on_empty_root(tmp_path, bsm, capsys) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "share_map.json"
    rc = bsm.main(["--sweeps-root", str(empty), "--out", str(out), "--quiet"])
    assert rc == 2
    assert "no sweep variants" in capsys.readouterr().out
