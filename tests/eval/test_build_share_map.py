"""
Tests for ``scripts/eval/build_share_map.py``.

Covers:
* ``_collect`` — scans a sweep's media JSONs and returns crop filenames.
* ``build`` — inverts file→variant relation across all sweeps under a root.
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


_REPO_ROOT = Path(__file__).resolve().parents[2]
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
    figures: list[str] | None = None,
    tables: list[str] | None = None,
) -> None:
    json_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"pmcid": pmcid, "figures": [], "tables": []}
    for name in figures or []:
        payload["figures"].append({"image_path": f"out/sweeps/x/figures/{name}"})
    for name in tables or []:
        payload["tables"].append({"image_path": f"out/sweeps/x/tables/{name}"})
    (json_dir / f"{pmcid}_media.json").write_text(json.dumps(payload))


def _build_sweep(sweeps_root: Path, name: str, *, pmcid: str, **kwargs) -> Path:
    sweep = sweeps_root / name
    _write_media(sweep / "json", pmcid, **kwargs)
    return sweep


# ── _collect ──────────────────────────────────────────────────────────────────


def test_collect_returns_basenames(tmp_path, bsm) -> None:
    sweep = _build_sweep(
        tmp_path / "sweeps", "v1",
        pmcid="PMC_A",
        figures=["PMC_A_Figure_1_p2.png"],
        tables=["PMC_A_Table_1_p3.png", "PMC_A_Table_2_p3.png"],
    )
    crops = bsm._collect(sweep / "json")
    assert crops == {
        "PMC_A_Figure_1_p2.png",
        "PMC_A_Table_1_p3.png",
        "PMC_A_Table_2_p3.png",
    }


def test_collect_skips_missing_image_path(tmp_path, bsm) -> None:
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    (json_dir / "PMC_X_media.json").write_text(json.dumps({
        "tables": [
            {"image_path": "out/x/PMC_X_Table_1_p1.png"},
            {"image_path": None},
            {"image_path": ""},
            {},  # no image_path key at all
        ],
    }))
    assert bsm._collect(json_dir) == {"PMC_X_Table_1_p1.png"}


def test_collect_tolerates_malformed_media_json(tmp_path, bsm) -> None:
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    (json_dir / "good_media.json").write_text(json.dumps({
        "tables": [{"image_path": "x/PMC_G_Table_1_p1.png"}],
    }))
    (json_dir / "bad_media.json").write_text("not valid json {{")
    assert bsm._collect(json_dir) == {"PMC_G_Table_1_p1.png"}


# ── build ─────────────────────────────────────────────────────────────────────


def test_build_inverts_relation(tmp_path, bsm) -> None:
    root = tmp_path / "sweeps"
    _build_sweep(root, "baseline", pmcid="PMC_A",
                 tables=["PMC_A_Table_1_p1.png"], figures=["PMC_A_Figure_1_p2.png"])
    _build_sweep(root, "tatr_095", pmcid="PMC_A",
                 tables=["PMC_A_Table_1_p1.png", "PMC_A_Table_2_p1.png"])
    _build_sweep(root, "tatr_090", pmcid="PMC_A",
                 tables=["PMC_A_Table_1_p1.png", "PMC_A_Table_2_p1.png",
                         "PMC_A_Table_3_p2.png"])

    sm = bsm.build(root)
    assert sm["PMC_A_Table_1_p1.png"] == ["baseline", "tatr_090", "tatr_095"]
    assert sm["PMC_A_Table_2_p1.png"] == ["tatr_090", "tatr_095"]
    assert sm["PMC_A_Table_3_p2.png"] == ["tatr_090"]
    assert sm["PMC_A_Figure_1_p2.png"] == ["baseline"]


def test_build_returns_sorted_keys(tmp_path, bsm) -> None:
    root = tmp_path / "sweeps"
    _build_sweep(root, "a", pmcid="PMC_Z", tables=["z.png"])
    _build_sweep(root, "b", pmcid="PMC_A", tables=["a.png"])
    sm = bsm.build(root)
    assert list(sm.keys()) == ["a.png", "z.png"]
    for variants in sm.values():
        assert variants == sorted(variants)


def test_build_skips_variant_without_json_subdir(tmp_path, bsm, caplog) -> None:
    root = tmp_path / "sweeps"
    _build_sweep(root, "with_json", pmcid="PMC_A", tables=["a.png"])
    (root / "no_json").mkdir()  # variant dir but no json/
    sm = bsm.build(root)
    assert sm == {"a.png": ["with_json"]}


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
    payload = {"a.png": ["v1", "v2"]}
    bsm.write(payload, out)
    assert out.exists()
    assert json.loads(out.read_text()) == payload


def test_write_is_atomic_no_partial_on_failure(tmp_path, bsm, monkeypatch) -> None:
    out = tmp_path / "share_map.json"
    bsm.write({"existing.png": ["v1"]}, out)  # seed valid file
    original = json.loads(out.read_text())
    # Simulate Path.replace failing — the tmp file should never overwrite the real one.
    def boom(self, target):
        raise OSError("disk full")
    monkeypatch.setattr(Path, "replace", boom)
    with pytest.raises(OSError):
        bsm.write({"new.png": ["v2"]}, out)
    # Original file untouched
    assert json.loads(out.read_text()) == original


# ── CLI main() ────────────────────────────────────────────────────────────────


def test_main_writes_share_map(tmp_path, bsm, capsys) -> None:
    root = tmp_path / "sweeps"
    _build_sweep(root, "v1", pmcid="PMC_X", tables=["t.png"])
    out = tmp_path / "share_map.json"

    rc = bsm.main(["--sweeps-root", str(root), "--out", str(out), "--quiet"])
    assert rc == 0
    assert json.loads(out.read_text()) == {"t.png": ["v1"]}
    # Summary printed to stdout
    captured = capsys.readouterr()
    assert "total unique crop filenames: 1" in captured.out


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
