"""
PipelineRunner.run_document must produce a per-document stats file even
when ``_process`` raises — including failures that happen before any
``DocStatsCollector.stage(...)`` context is entered.

This is the user-imposed Stage-1 guardrail: collectors are built BEFORE the
first fail-prone step so failed documents are still diffable in a sweep.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


from nlp_histo.pipeline.stages.pdf_text_extraction.config import PipelineConfig
from nlp_histo.pipeline.stages.pdf_text_extraction.models.dto import BoundingBox, LayoutElement
from nlp_histo.pipeline.stages.pdf_text_extraction.models.scored_node import ScoredNode
from nlp_histo.pipeline.stages.pdf_text_extraction.runner import PipelineRunner


def _build_runner(tmp_path: Path) -> PipelineRunner:
    cfg = PipelineConfig()
    cfg.paths.output_root      = tmp_path / "out"
    cfg.paths.run_metadata_dir = tmp_path / "out" / "run_metadata"
    cfg.paths.blacklist_file   = tmp_path / "out" / "bl.json"
    cfg.paths.text_dir         = tmp_path / "out" / "text"
    cfg.paths.json_dir         = tmp_path / "out" / "json"
    cfg.paths.figures_dir      = tmp_path / "out" / "figures"
    cfg.paths.tables_dir       = tmp_path / "out" / "tables"
    cfg.paths.vis_dir          = tmp_path / "out" / "viz"
    cfg.paths.docling_full_dir = tmp_path / "out" / "docling_full"
    cfg.paths.docling_masked_dir = tmp_path / "out" / "docling_masked"
    cfg.paths.masked_pdf_dir   = tmp_path / "out" / "masked"
    cfg.paths.text_raw_dir     = tmp_path / "out" / "text_raw"
    cfg.database.enabled = False
    cfg.visualization.enabled = False
    cfg.runtime.skip_existing_in_db = False
    cfg.runtime.skip_existing_media_json = False
    cfg.runtime.update_blacklist_on_failure = False
    cfg.runtime.save_error_traces = False
    cfg.prepare()
    return PipelineRunner(cfg)


def _stats_for(runner: PipelineRunner, pmcid: str) -> dict[str, Any]:
    path = runner._cfg.paths.run_metadata_dir / f"{pmcid}_stats.json"
    assert path.exists(), f"stats file missing: {path}"
    return json.loads(path.read_text())


def test_success_path_writes_status_ok(tmp_path, monkeypatch) -> None:
    runner = _build_runner(tmp_path)

    def fake_process(self, pdf_path, pmcid):
        # Mimic the runner's own recording calls for a single fake stage
        with self._current_stats.stage("STEP1_LAYOUT"):
            pass
        self._current_stats.set_count("layout_elements_full", 10)
        return ["row1", "row2", "row3"]

    monkeypatch.setattr(PipelineRunner, "_process", fake_process)
    assert runner.run_document(Path("fake.pdf"), "PMC_OK") is True

    data = _stats_for(runner, "PMC_OK")
    assert data["status"] == "ok"
    assert data["counts"]["text_rows"] == 3
    assert data["counts"]["layout_elements_full"] == 10
    assert [t["name"] for t in data["stage_timings"]] == ["STEP1_LAYOUT"]
    assert data["error"] is None
    assert data["failed_stage"] is None


def test_mid_pipeline_failure_writes_status_failed(tmp_path, monkeypatch) -> None:
    runner = _build_runner(tmp_path)

    def fake_process(self, pdf_path, pmcid):
        with self._current_stats.stage("STEP1_LAYOUT"):
            raise RuntimeError("layout boom")

    monkeypatch.setattr(PipelineRunner, "_process", fake_process)
    assert runner.run_document(Path("fake.pdf"), "PMC_MID") is False

    data = _stats_for(runner, "PMC_MID")
    assert data["status"] == "failed"
    assert "RuntimeError" in data["error"]
    assert "layout boom" in data["error"]
    assert data["failed_stage"] == "STEP1_LAYOUT"
    assert len(data["stage_timings"]) == 1
    assert data["stage_timings"][0]["ok"] is False
    assert data["stage_timings"][0]["error"] == "RuntimeError"


def test_failure_before_any_stage_still_writes_stats(tmp_path, monkeypatch) -> None:
    """The user's "init BEFORE the first fail-prone step" requirement —
    a document that crashes before any stage context opens must still get
    a stats file with status=failed."""
    runner = _build_runner(tmp_path)

    def fake_process(self, pdf_path, pmcid):
        raise ValueError("died at the very start")

    monkeypatch.setattr(PipelineRunner, "_process", fake_process)
    assert runner.run_document(Path("fake.pdf"), "PMC_EARLY") is False

    data = _stats_for(runner, "PMC_EARLY")
    assert data["status"] == "failed"
    assert "ValueError" in data["error"]
    assert "died at the very start" in data["error"]
    assert data["failed_stage"] is None  # no stage context was ever entered
    assert data["stage_timings"] == []


def test_run_id_propagates_into_stats_file(tmp_path, monkeypatch) -> None:
    runner = _build_runner(tmp_path)

    def fake_process(self, pdf_path, pmcid):
        return []

    monkeypatch.setattr(PipelineRunner, "_process", fake_process)
    runner.run_document(Path("fake.pdf"), "PMC_RID", run_id="rid-from-test")

    data = _stats_for(runner, "PMC_RID")
    assert data["run_id"] == "rid-from-test"


def test_config_digest_stamped_in_stats(tmp_path, monkeypatch) -> None:
    runner = _build_runner(tmp_path)

    def fake_process(self, pdf_path, pmcid):
        return []

    monkeypatch.setattr(PipelineRunner, "_process", fake_process)
    runner.run_document(Path("fake.pdf"), "PMC_DIGEST")

    data = _stats_for(runner, "PMC_DIGEST")
    assert isinstance(data["config_digest"], str)
    assert len(data["config_digest"]) == 12


def test_failure_with_pre_failure_stages_captures_partial_progress(
    tmp_path, monkeypatch
) -> None:
    """A failure mid-batch must still record the stages that finished OK."""
    runner = _build_runner(tmp_path)

    def fake_process(self, pdf_path, pmcid):
        with self._current_stats.stage("STEP1_LAYOUT"):
            pass
        self._current_stats.set_count("layout_elements_full", 50)
        with self._current_stats.stage("STEP2_TABLE_DETECTION"):
            raise RuntimeError("tatr failed")

    monkeypatch.setattr(PipelineRunner, "_process", fake_process)
    assert runner.run_document(Path("fake.pdf"), "PMC_PARTIAL") is False

    data = _stats_for(runner, "PMC_PARTIAL")
    assert data["status"] == "failed"
    assert data["failed_stage"] == "STEP2_TABLE_DETECTION"
    assert data["counts"]["layout_elements_full"] == 50
    names = [t["name"] for t in data["stage_timings"]]
    assert names == ["STEP1_LAYOUT", "STEP2_TABLE_DETECTION"]
    assert data["stage_timings"][0]["ok"] is True
    assert data["stage_timings"][1]["ok"] is False


def test_skipped_blacklisted_does_not_write_stats(tmp_path, monkeypatch) -> None:
    """Blacklisted documents short-circuit before the collector is built — no
    stats file is produced.  (Sweep visibility for skips comes from the
    batch manifest's `batch_stats`, not per-doc files.)"""
    runner = _build_runner(tmp_path)
    runner._blacklist.add("PMC_SKIP", reason="prior failure")

    sentinel = {"called": False}

    def fake_process(self, pdf_path, pmcid):
        sentinel["called"] = True
        return []

    monkeypatch.setattr(PipelineRunner, "_process", fake_process)
    assert runner.run_document(Path("fake.pdf"), "PMC_SKIP") is False
    assert sentinel["called"] is False
    assert not (runner._cfg.paths.run_metadata_dir / "PMC_SKIP_stats.json").exists()


def test_scored_node_recording_via_runner(tmp_path, monkeypatch) -> None:
    """End-to-end check that scored-node tallies flow into the stats file
    when the runner threads them through ``self._current_stats``."""
    runner = _build_runner(tmp_path)

    def _el():
        return LayoutElement(
            type="TEXT", page=1,
            bbox=BoundingBox(0.0, 100.0, 50.0, 0.0, page=1),
            text="x",
        )

    def fake_process(self, pdf_path, pmcid):
        self._current_stats.record_scored_nodes([
            ScoredNode(element=_el(), keep=True),
            ScoredNode(element=_el(), keep=False, rejection_reason="empty text"),
            ScoredNode(element=_el(), keep=False,
                       rejection_reason="visually blank region in header zone"),
        ])
        return ["only_row"]

    monkeypatch.setattr(PipelineRunner, "_process", fake_process)
    runner.run_document(Path("fake.pdf"), "PMC_NODES")

    data = _stats_for(runner, "PMC_NODES")
    assert data["counts"]["two_pass_kept"] == 1
    assert data["counts"]["two_pass_rejected"] == 2
    assert data["reason_histogram"] == {"R0_empty": 1, "R1_blank_pixels": 1}
    assert data["reason_in_header_zone"] == 1
