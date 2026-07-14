"""
Tests for ``pipeline/stages/pdf_text_extraction/outputs/stats_writer.py``.

Covers:
* ``classify_reason`` — full rejection-string vocabulary and the
  header-zone-marker suffix that ``NodeScorer`` appends.
* ``config_digest`` — stability, length, sensitivity to knob changes,
  insensitivity to ``PathConfig`` changes.
* ``config_snapshot`` — JSON-friendly shape covers all sub-configs.
* ``DocStatsCollector`` — recording, stage timing, status transitions,
  failure handling, write idempotency.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlp_histo.pipeline.stages.pdf_text_extraction.config import PipelineConfig
from nlp_histo.pipeline.stages.pdf_text_extraction.models.dto import BoundingBox, LayoutElement
from nlp_histo.pipeline.stages.pdf_text_extraction.models.scored_node import ScoredNode
from nlp_histo.pipeline.stages.pdf_text_extraction.outputs.stats_writer import (
    DocStatsCollector,
    classify_reason,
    config_digest,
    config_snapshot,
)


# ── Reason-code classifier ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Bare prefixes
        ("empty text", "R0_empty"),
        ("white-text ghost layer", "R_color_white"),
        ("hidden text layer", "R3_dense_text"),
        ("visually blank region", "R1_blank_pixels"),
        ("no fitz words", "R2_no_fitz_words"),
        # Real strings emitted by NodeScorer._score_text_node
        ("white-text ghost layer — 80% of characters are near-white",
         "R_color_white"),
        ("white-text ghost layer — 80% of characters are near-white in header zone",
         "R_color_white"),
        ("hidden text layer — 180 chars in 10.0pt bbox (18.0 chars/pt)",
         "R3_dense_text"),
        ("hidden text layer — 180 chars in 10.0pt bbox (18.0 chars/pt) in header zone",
         "R3_dense_text"),
        ("visually blank region — invisible text layer", "R1_blank_pixels"),
        ("visually blank region — invisible text layer in header zone",
         "R1_blank_pixels"),
        ("no fitz words in bbox (no pixel data available)", "R2_no_fitz_words"),
        ("no fitz words in bbox (no pixel data available) in header zone",
         "R2_no_fitz_words"),
        # OTHER bucket
        ("", "OTHER"),
        ("something unrecognised", "OTHER"),
        ("almost-empty text", "OTHER"),  # close but not a prefix
    ],
)
def test_classify_reason(raw: str, expected: str) -> None:
    assert classify_reason(raw) == expected


# ── config_digest ─────────────────────────────────────────────────────────────


def test_config_digest_stable_across_instances() -> None:
    """Two fresh PipelineConfig() instances must produce the same digest."""
    a = config_digest(PipelineConfig())
    b = config_digest(PipelineConfig())
    assert a == b
    assert len(a) == 12
    assert a != "unknown"


def test_config_digest_length_param() -> None:
    cfg = PipelineConfig()
    assert len(config_digest(cfg, length=8)) == 8
    assert len(config_digest(cfg, length=20)) == 20


def test_config_digest_changes_with_knob() -> None:
    base = PipelineConfig()
    tweak = PipelineConfig()
    tweak.tatr.threshold = 0.9  # default 0.99
    assert config_digest(base) != config_digest(tweak)


def test_config_digest_ignores_paths() -> None:
    """Two configs that differ ONLY in PathConfig values must hash identically."""
    a = PipelineConfig()
    b = PipelineConfig()
    b.paths.output_root = Path("out/sweeps/foo")
    b.paths.text_dir = Path("out/sweeps/foo/text")
    b.paths.figures_dir = Path("out/sweeps/foo/figures")
    b.paths.json_dir = Path("out/sweeps/foo/json")
    b.paths.run_metadata_dir = Path("out/sweeps/foo/run_metadata")
    assert config_digest(a) == config_digest(b)


def test_config_digest_changes_with_two_pass_knob() -> None:
    base = PipelineConfig()
    tweak = PipelineConfig()
    tweak.two_pass.render_dpi = 200  # default 150
    assert config_digest(base) != config_digest(tweak)


def test_config_digest_handles_non_dataclass_input() -> None:
    """Non-dataclass input must not raise — returns a stable digest string."""
    digest = config_digest(object())
    assert isinstance(digest, str)
    assert len(digest) == 12  # default length, even on the fallback path
    assert digest != "unknown"  # bare ``object()`` is JSON-able via default=str


def test_config_digest_returns_unknown_on_internal_failure(monkeypatch) -> None:
    """If config_payload() raises, the digest helper must swallow + return 'unknown'."""
    from nlp_histo.pipeline.stages.pdf_text_extraction.outputs import stats_writer

    def _blow_up(_):
        raise RuntimeError("inject failure")

    monkeypatch.setattr(stats_writer, "_config_payload", _blow_up)
    assert config_digest(PipelineConfig()) == "unknown"


# ── config_snapshot ───────────────────────────────────────────────────────────


def test_config_snapshot_includes_all_subconfigs() -> None:
    snap = config_snapshot(PipelineConfig())
    expected_keys = {
        "paths", "docling", "docling_text", "tatr", "masking", "filtering",
        "cropping", "text", "visualization", "database", "runtime", "two_pass",
        "table_detector",
    }
    assert expected_keys.issubset(snap.keys())


def test_config_snapshot_is_json_serializable() -> None:
    snap = config_snapshot(PipelineConfig())
    blob = json.dumps(snap)  # must not raise
    assert "two_pass" in blob


def test_config_snapshot_enum_serialised_as_value() -> None:
    snap = config_snapshot(PipelineConfig())
    # TableDetectorType.HYBRID → "HYBRID" (or "hybrid"); not "TableDetectorType.HYBRID"
    assert snap["table_detector"] in {"HYBRID", "hybrid", "TATR", "tatr",
                                       "DOCLING", "docling", "VLM", "vlm"}


# ── DocStatsCollector ─────────────────────────────────────────────────────────


def _make_collector(tmp_path: Path) -> DocStatsCollector:
    return DocStatsCollector(
        pmcid="PMC_TEST",
        pdf_path=Path("fake.pdf"),
        run_metadata_dir=tmp_path,
        run_id="run-abc",
        config_digest="abc123",
        config_used={"answer": 42},
    )


def _fake_element() -> LayoutElement:
    return LayoutElement(
        type="TEXT", page=1,
        bbox=BoundingBox(0.0, 100.0, 50.0, 0.0, page=1),
        text="x",
    )


def test_collector_writes_stats_file(tmp_path: Path) -> None:
    c = _make_collector(tmp_path)
    c.set_count("text_rows", 14)
    c.mark_ok()
    out = c.write()
    assert out is not None and out.exists()
    data = json.loads(out.read_text())
    assert data["pmcid"] == "PMC_TEST"
    assert data["status"] == "ok"
    assert data["counts"]["text_rows"] == 14
    assert data["run_id"] == "run-abc"
    assert data["config_digest"] == "abc123"
    assert data["config_used"] == {"answer": 42}


def test_collector_stage_records_timing(tmp_path: Path) -> None:
    c = _make_collector(tmp_path)
    with c.stage("STEP1"):
        pass
    c.mark_ok()
    data = json.loads(c.write().read_text())
    timings = data["stage_timings"]
    assert len(timings) == 1
    assert timings[0]["name"] == "STEP1"
    assert timings[0]["ok"] is True
    assert timings[0]["seconds"] >= 0.0


def test_collector_stage_captures_failure_and_reraises(tmp_path: Path) -> None:
    c = _make_collector(tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        with c.stage("STEP1"):
            raise RuntimeError("boom")
    c.mark_failed(RuntimeError("boom"))  # mimic runner's except branch
    data = json.loads(c.write().read_text())
    assert data["status"] == "failed"
    assert "RuntimeError" in data["error"]
    assert data["failed_stage"] == "STEP1"
    assert data["stage_timings"][0]["ok"] is False
    assert data["stage_timings"][0]["error"] == "RuntimeError"


def test_collector_record_scored_nodes_tallies_reasons(tmp_path: Path) -> None:
    c = _make_collector(tmp_path)
    nodes = [
        ScoredNode(element=_fake_element(), keep=True),
        ScoredNode(element=_fake_element(), keep=False,
                   rejection_reason="empty text"),
        ScoredNode(element=_fake_element(), keep=False,
                   rejection_reason="visually blank region — invisible text layer in header zone"),
        ScoredNode(element=_fake_element(), keep=False,
                   rejection_reason="hidden text layer — 200 chars in 10pt bbox (20 chars/pt)"),
        ScoredNode(element=_fake_element(), keep=False,
                   rejection_reason="completely unexpected reason"),
    ]
    c.record_scored_nodes(nodes)
    c.mark_ok()
    data = json.loads(c.write().read_text())
    assert data["counts"]["two_pass_kept"] == 1
    assert data["counts"]["two_pass_rejected"] == 4
    assert data["counts"]["two_pass_scored"] == 5
    assert data["reason_histogram"]["R0_empty"] == 1
    assert data["reason_histogram"]["R1_blank_pixels"] == 1
    assert data["reason_histogram"]["R3_dense_text"] == 1
    assert data["reason_histogram"]["OTHER"] == 1
    assert data["reason_in_header_zone"] == 1
    assert data["reason_unknown_examples"] == ["completely unexpected reason"]


def test_collector_unknown_examples_capped(tmp_path: Path) -> None:
    """At most 5 OTHER samples are retained — protects manifest size."""
    c = _make_collector(tmp_path)
    nodes = [
        ScoredNode(element=_fake_element(), keep=False,
                   rejection_reason=f"weird reason #{i}")
        for i in range(20)
    ]
    c.record_scored_nodes(nodes)
    c.mark_ok()
    data = json.loads(c.write().read_text())
    assert data["reason_histogram"]["OTHER"] == 20
    assert len(data["reason_unknown_examples"]) == 5


def test_collector_table_detection(tmp_path: Path) -> None:
    c = _make_collector(tmp_path)
    c.record_table_detection("HYBRID", 7, {"tatr": 5, "docling": 2})
    c.mark_ok()
    data = json.loads(c.write().read_text())
    assert data["table_detection"]["detector"] == "HYBRID"
    assert data["table_detection"]["n_regions"] == 7
    assert data["table_detection"]["per_source_counts"] == {"tatr": 5, "docling": 2}


def test_collector_mark_failed_overrides_running(tmp_path: Path) -> None:
    c = _make_collector(tmp_path)
    c.mark_failed(ValueError("oops"), stage="STEP3")
    data = json.loads(c.write().read_text())
    assert data["status"] == "failed"
    assert "ValueError" in data["error"]
    assert data["failed_stage"] == "STEP3"


def test_collector_mark_ok_after_failed_is_noop(tmp_path: Path) -> None:
    """mark_ok only transitions if status is still 'running'."""
    c = _make_collector(tmp_path)
    c.mark_failed(ValueError("oops"))
    c.mark_ok()
    data = json.loads(c.write().read_text())
    assert data["status"] == "failed"  # mark_ok must NOT override


def test_collector_write_is_idempotent(tmp_path: Path) -> None:
    c = _make_collector(tmp_path)
    c.mark_ok()
    out1 = c.write()
    assert out1 is not None
    # Manually clobber the file so we can detect a second write
    out1.write_text("CLOBBERED")
    out2 = c.write()
    assert out2 is None  # second call should be a no-op
    assert out1.read_text() == "CLOBBERED"


def test_collector_creates_run_metadata_dir(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "out" / "run_metadata"
    c = DocStatsCollector(
        pmcid="PMC_X", pdf_path=Path("fake.pdf"),
        run_metadata_dir=nested,
    )
    c.mark_ok()
    out = c.write()
    assert out is not None
    assert nested.is_dir()
    assert out.parent == nested


def test_collector_record_op_is_failure_safe(tmp_path: Path) -> None:
    """A faulty record call must not raise into the runner."""
    c = _make_collector(tmp_path)

    class Boom:
        @property
        def keep(self):
            raise RuntimeError("explode")
    # Should NOT raise:
    c.record_scored_nodes([Boom()])
    c.mark_ok()
    # write must still succeed:
    out = c.write()
    assert out is not None
