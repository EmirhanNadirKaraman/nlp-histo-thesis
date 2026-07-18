"""
Tests for ``pipeline/stages/pdf_text_extraction/outputs/manifest_writer.py``.

Covers:
* ``make_run_id`` — uniqueness across calls, expected shape.
* ``_git_info`` — best-effort, never raises.
* ``RunManifestWriter`` — input metadata, aggregation across per-doc stats
  files (counts sum, reason histogram sum, header-zone sum, mean wall,
  status breakdown), missing per-doc files, empty attempted list, JSON
  shape of the resulting manifest.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nlp_histo.pipeline.stages.pdf_text_extraction.config import PipelineConfig
from nlp_histo.pipeline.stages.pdf_text_extraction.outputs.manifest_writer import (
    RunManifestWriter,
    _git_info,
    make_run_id,
)


# make_run_id


def test_make_run_id_shape() -> None:
    rid = make_run_id()
    # YYYYMMDDTHHMMSSZ_xxxxxxxx
    assert re.fullmatch(r"\d{8}T\d{6}Z_[0-9a-f]{8}", rid), rid


def test_make_run_id_unique() -> None:
    ids = {make_run_id() for _ in range(50)}
    assert len(ids) == 50  # 8-hex uuid suffix collision-free in practice


# _git_info


def test_git_info_returns_dict_with_expected_keys() -> None:
    info = _git_info()
    assert set(info.keys()) == {"sha", "branch", "dirty"}
    # Either fully populated (running inside a git checkout) or fully None
    # (subprocess failures fall through).  Both shapes are acceptable.


# RunManifestWriter helpers


def _write_doc_stats(
    run_metadata_dir: Path,
    pmcid: str,
    *,
    status: str = "ok",
    wall: float = 1.0,
    counts: dict | None = None,
    reasons: dict | None = None,
    header_zone: int = 0,
    failed_stage: str | None = None,
    error: str | None = None,
) -> Path:
    """Synthesize a per-document stats JSON file."""
    run_metadata_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "pmcid": pmcid,
        "status": status,
        "wall_seconds": wall,
        "counts": counts or {},
        "reason_histogram": reasons or {},
        "reason_in_header_zone": header_zone,
        "failed_stage": failed_stage,
        "error": error,
    }
    path = run_metadata_dir / f"{pmcid}_stats.json"
    path.write_text(json.dumps(payload))
    return path


# RunManifestWriter


def test_manifest_writes_with_no_attempted(tmp_path: Path) -> None:
    """Empty batch must still produce a parseable manifest."""
    w = RunManifestWriter(cfg=PipelineConfig(), run_metadata_dir=tmp_path,
                          run_id="rid-empty")
    out = w.write(batch_stats={"processed": 0, "failed": 0, "skipped": 0})
    assert out is not None and out.exists()
    data = json.loads(out.read_text())
    assert data["run_id"] == "rid-empty"
    assert data["documents"] == []
    s = data["summary"]
    assert s["n_attempted"] == 0
    assert s["n_with_stats"] == 0
    assert s["n_ok"] == 0
    assert s["n_failed"] == 0
    assert s["total_wall_seconds"] == 0.0
    assert s["mean_wall_seconds"] == 0.0
    assert s["counts_sum"] == {}
    assert s["reason_histogram_sum"] == {}


def test_manifest_aggregates_counts_and_reasons(tmp_path: Path) -> None:
    _write_doc_stats(tmp_path, "PMC_A",
                     status="ok", wall=10.0,
                     counts={"text_rows": 14, "figures_cropped": 10},
                     reasons={"R1_blank_pixels": 40, "R3_dense_text": 7},
                     header_zone=7)
    _write_doc_stats(tmp_path, "PMC_B",
                     status="ok", wall=6.0,
                     counts={"text_rows": 20, "figures_cropped": 3},
                     reasons={"R1_blank_pixels": 19, "R0_empty": 2},
                     header_zone=18)

    w = RunManifestWriter(cfg=PipelineConfig(), run_metadata_dir=tmp_path,
                          run_id="rid-agg")
    w.set_attempted(["PMC_A", "PMC_B"])
    w.set_input(runner="test", pdf_dir=str(tmp_path), n_paths=2)
    out = w.write(batch_stats={"processed": 2, "failed": 0, "skipped": 0})
    data = json.loads(out.read_text())

    s = data["summary"]
    assert s["n_attempted"] == 2
    assert s["n_with_stats"] == 2
    assert s["n_ok"] == 2
    assert s["n_failed"] == 0
    assert s["total_wall_seconds"] == pytest.approx(16.0)
    assert s["mean_wall_seconds"] == pytest.approx(8.0)
    assert s["counts_sum"] == {"text_rows": 34, "figures_cropped": 13}
    assert s["reason_histogram_sum"] == {
        "R1_blank_pixels": 59, "R3_dense_text": 7, "R0_empty": 2,
    }
    assert s["reason_in_header_zone_sum"] == 25
    assert len(data["documents"]) == 2
    pmcids = {d["pmcid"] for d in data["documents"]}
    assert pmcids == {"PMC_A", "PMC_B"}


def test_manifest_records_failed_status(tmp_path: Path) -> None:
    _write_doc_stats(tmp_path, "PMC_OK", status="ok", wall=5.0,
                     counts={"text_rows": 4})
    _write_doc_stats(tmp_path, "PMC_FAIL", status="failed", wall=2.0,
                     failed_stage="STEP1_LAYOUT", error="RuntimeError: boom")

    w = RunManifestWriter(cfg=PipelineConfig(), run_metadata_dir=tmp_path,
                          run_id="rid-fail")
    w.set_attempted(["PMC_OK", "PMC_FAIL"])
    out = w.write(batch_stats={"processed": 1, "failed": 1, "skipped": 0})
    data = json.loads(out.read_text())

    s = data["summary"]
    assert s["n_ok"] == 1
    assert s["n_failed"] == 1
    failed_doc = next(d for d in data["documents"] if d["pmcid"] == "PMC_FAIL")
    assert failed_doc["status"] == "failed"
    assert failed_doc["failed_stage"] == "STEP1_LAYOUT"


def test_manifest_handles_missing_per_doc_stats(tmp_path: Path) -> None:
    """Documents the runner attempted but couldn't tag with stats."""
    _write_doc_stats(tmp_path, "PMC_OK", status="ok", wall=1.0)
    w = RunManifestWriter(cfg=PipelineConfig(), run_metadata_dir=tmp_path,
                          run_id="rid-missing")
    w.set_attempted(["PMC_OK", "PMC_MISSING"])
    out = w.write()
    data = json.loads(out.read_text())

    s = data["summary"]
    assert s["n_attempted"] == 2
    assert s["n_with_stats"] == 1
    missing_doc = next(d for d in data["documents"] if d["pmcid"] == "PMC_MISSING")
    assert missing_doc["status"] == "no_stats"


def test_manifest_payload_top_level_fields(tmp_path: Path) -> None:
    w = RunManifestWriter(cfg=PipelineConfig(), run_metadata_dir=tmp_path,
                          run_id="rid-shape")
    w.set_input(runner="test", pdf_dir=str(tmp_path), n_paths=0)
    out = w.write(batch_stats={"processed": 0, "failed": 0, "skipped": 0})
    data = json.loads(out.read_text())
    expected_top = {
        "run_id", "started_at", "finished_at", "git", "host", "python",
        "input", "config_digest", "config", "batch_stats", "summary",
        "documents",
    }
    assert expected_top.issubset(data.keys())
    assert data["python"]  # non-empty
    assert isinstance(data["config_digest"], str)


def test_manifest_input_meta_serialises_paths(tmp_path: Path) -> None:
    w = RunManifestWriter(cfg=PipelineConfig(), run_metadata_dir=tmp_path,
                          run_id="rid-paths")
    w.set_input(pdf_dir=Path("/tmp/x"), max_docs=None, n_paths=3)
    out = w.write()
    data = json.loads(out.read_text())
    assert data["input"]["pdf_dir"] == "/tmp/x"
    assert data["input"]["max_docs"] is None
    assert data["input"]["n_paths"] == 3


def test_manifest_runid_picked_when_omitted(tmp_path: Path) -> None:
    w = RunManifestWriter(cfg=PipelineConfig(), run_metadata_dir=tmp_path)
    assert re.fullmatch(r"\d{8}T\d{6}Z_[0-9a-f]{8}", w.run_id), w.run_id


def test_manifest_ignores_malformed_stats_file(tmp_path: Path) -> None:
    """A corrupt per-doc stats file must not crash aggregation."""
    (tmp_path / "PMC_BAD_stats.json").write_text("not json {{{")
    _write_doc_stats(tmp_path, "PMC_GOOD", status="ok", wall=2.5,
                     counts={"text_rows": 5})
    w = RunManifestWriter(cfg=PipelineConfig(), run_metadata_dir=tmp_path,
                          run_id="rid-bad")
    w.set_attempted(["PMC_BAD", "PMC_GOOD"])
    out = w.write()
    data = json.loads(out.read_text())
    s = data["summary"]
    # Bad file is logged + skipped; good file aggregates normally.
    assert s["n_with_stats"] == 1
    assert s["counts_sum"] == {"text_rows": 5}
