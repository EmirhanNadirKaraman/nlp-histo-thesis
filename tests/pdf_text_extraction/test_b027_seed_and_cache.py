"""B-027 — pipeline-owned seeding + per-stage output cache.

Three groups of tests:

1. ``_StageCache`` primitive — behaviour around hits, misses, hash
   mismatches, sidecar / loader corruption, opt-out, write failure.
2. Per-stage round-trips — assert each (loader, dumper) preserves the
   downstream-relevant runtime types (Path, int dict keys, list[str]).
3. Runner wiring + seed — both standard and two-pass paths end up with
   three sidecars on first run and three CACHE HIT lines on second run.
   Second runs use a fresh ``PipelineRunner`` instance with
   raise-on-call mocks so any in-memory state leak would fail loudly.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nlp_histo.pipeline.stages.pdf_text_extraction.config import PipelineConfig
from nlp_histo.pipeline.stages.pdf_text_extraction.models.dto import (
    BoundingBox,
    DetectedRegion,
    HierarchicalRow,
    LayoutElement,
    LayoutResult,
    TableDetectionResult,
)
from nlp_histo.pipeline.stages.pdf_text_extraction.runner import PipelineRunner
from nlp_histo.pipeline.stages.pdf_text_extraction.stage_cache import (
    _StageCache,
    dump_hierarchical_rows,
    dump_layout_elements,
    dump_table_detection,
    load_hierarchical_rows,
    load_layout_elements,
    load_table_detection,
)


# ── _StageCache primitive ─────────────────────────────────────────────────────

def _trivial_dump(payload):
    return payload


def _trivial_load(path):
    return json.loads(Path(path).read_text())


def _build_cache(tmp_path: Path, enabled: bool = True) -> _StageCache:
    return _StageCache(root=tmp_path / "stage_cache", enabled=enabled)


def test_first_run_writes_artifact_and_sidecar(tmp_path):
    cache = _build_cache(tmp_path)
    calls = [0]

    def compute():
        calls[0] += 1
        return {"x": 1}

    result = cache.get_or_compute(
        stage_name="demo", pmcid="PMC1", config_hash="abc",
        compute_fn=compute, loader_fn=_trivial_load,
        dumper_fn=_trivial_dump, summarise_fn=lambda r: "1 item",
    )
    assert result == {"x": 1}
    assert calls == [1]
    artifact = tmp_path / "stage_cache" / "demo" / "PMC1.json"
    sidecar = artifact.with_suffix(".hash")
    assert artifact.exists() and sidecar.exists()
    assert sidecar.read_text() == "abc"


def test_second_run_skip_true_uses_cache(tmp_path):
    cache = _build_cache(tmp_path, enabled=True)
    calls = [0]

    def compute():
        calls[0] += 1
        return {"x": 1}

    cache.get_or_compute(
        stage_name="demo", pmcid="PMC1", config_hash="abc",
        compute_fn=compute, loader_fn=_trivial_load,
        dumper_fn=_trivial_dump, summarise_fn=lambda r: "1",
    )
    cache.get_or_compute(
        stage_name="demo", pmcid="PMC1", config_hash="abc",
        compute_fn=compute, loader_fn=_trivial_load,
        dumper_fn=_trivial_dump, summarise_fn=lambda r: "1",
    )
    assert calls == [1]


def test_skip_false_recomputes_even_if_artifact_exists(tmp_path):
    # First write with enabled=True
    cache_on = _build_cache(tmp_path, enabled=True)
    calls = [0]

    def compute():
        calls[0] += 1
        return {"x": 1}

    cache_on.get_or_compute(
        stage_name="demo", pmcid="PMC1", config_hash="abc",
        compute_fn=compute, loader_fn=_trivial_load,
        dumper_fn=_trivial_dump, summarise_fn=lambda r: "1",
    )
    # Second call with enabled=False must recompute
    cache_off = _build_cache(tmp_path, enabled=False)
    cache_off.get_or_compute(
        stage_name="demo", pmcid="PMC1", config_hash="abc",
        compute_fn=compute, loader_fn=_trivial_load,
        dumper_fn=_trivial_dump, summarise_fn=lambda r: "1",
    )
    assert calls == [2]


def test_corrupted_artifact_falls_through_and_overwrites(tmp_path, caplog):
    cache = _build_cache(tmp_path)
    calls = [0]

    def compute():
        calls[0] += 1
        return {"x": calls[0]}

    cache.get_or_compute(
        stage_name="demo", pmcid="PMC1", config_hash="abc",
        compute_fn=compute, loader_fn=_trivial_load,
        dumper_fn=_trivial_dump, summarise_fn=lambda r: "1",
    )
    artifact = tmp_path / "stage_cache" / "demo" / "PMC1.json"
    artifact.write_bytes(b"not json")

    with caplog.at_level(logging.WARNING,
                          logger="nlp_histo.pipeline.stages.pdf_text_extraction.stage_cache"):
        result = cache.get_or_compute(
            stage_name="demo", pmcid="PMC1", config_hash="abc",
            compute_fn=compute, loader_fn=_trivial_load,
            dumper_fn=_trivial_dump, summarise_fn=lambda r: "1",
        )
    assert calls == [2]
    assert result == {"x": 2}
    # Artifact got rewritten cleanly — we can parse it again
    assert json.loads(artifact.read_text()) == {"x": 2}
    msg = caplog.text
    assert "CACHE INVALID" in msg
    assert "JSONDecodeError" in msg or "ValueError" in msg


def test_hash_mismatch_invalidates_and_overwrites_sidecar_and_artifact(tmp_path, caplog):
    cache = _build_cache(tmp_path)
    calls = [0]

    def compute():
        calls[0] += 1
        return {"x": calls[0]}

    cache.get_or_compute(
        stage_name="demo", pmcid="PMC1", config_hash="hash_old",
        compute_fn=compute, loader_fn=_trivial_load,
        dumper_fn=_trivial_dump, summarise_fn=lambda r: "1",
    )
    artifact = tmp_path / "stage_cache" / "demo" / "PMC1.json"
    sidecar = artifact.with_suffix(".hash")
    old_artifact_bytes = artifact.read_bytes()

    with caplog.at_level(logging.INFO,
                          logger="nlp_histo.pipeline.stages.pdf_text_extraction.stage_cache"):
        cache.get_or_compute(
            stage_name="demo", pmcid="PMC1", config_hash="hash_new",
            compute_fn=compute, loader_fn=_trivial_load,
            dumper_fn=_trivial_dump, summarise_fn=lambda r: "1",
        )
    assert calls == [2]
    assert sidecar.read_text() == "hash_new"
    # Stale artifact bytes overwritten with the new compute payload
    assert artifact.read_bytes() != old_artifact_bytes
    assert json.loads(artifact.read_text()) == {"x": 2}
    assert "CACHE STALE" in caplog.text


def test_sidecar_read_failure_treated_as_cache_invalid(tmp_path, caplog):
    cache = _build_cache(tmp_path)
    calls = [0]

    def compute():
        calls[0] += 1
        return {"x": 1}

    cache.get_or_compute(
        stage_name="demo", pmcid="PMC1", config_hash="abc",
        compute_fn=compute, loader_fn=_trivial_load,
        dumper_fn=_trivial_dump, summarise_fn=lambda r: "1",
    )
    sidecar = tmp_path / "stage_cache" / "demo" / "PMC1.hash"
    sidecar.write_bytes(b"\xff\xfe\xfd")  # invalid UTF-8

    with caplog.at_level(logging.WARNING,
                          logger="nlp_histo.pipeline.stages.pdf_text_extraction.stage_cache"):
        cache.get_or_compute(
            stage_name="demo", pmcid="PMC1", config_hash="abc",
            compute_fn=compute, loader_fn=_trivial_load,
            dumper_fn=_trivial_dump, summarise_fn=lambda r: "1",
        )
    assert calls == [2]
    assert "CACHE INVALID" in caplog.text
    assert "sidecar read failed" in caplog.text
    assert "UnicodeDecodeError" in caplog.text
    # No double-log: STALE comes from a successful sidecar read with a
    # mismatching hash; we never reached that branch.
    assert "CACHE STALE" not in caplog.text


def test_unexpected_loader_exception_propagates(tmp_path):
    cache = _build_cache(tmp_path)

    def compute():
        return {"x": 1}

    cache.get_or_compute(
        stage_name="demo", pmcid="PMC1", config_hash="abc",
        compute_fn=compute, loader_fn=_trivial_load,
        dumper_fn=_trivial_dump, summarise_fn=lambda r: "1",
    )

    def bad_loader(_path):
        raise RuntimeError("loader bug")

    with pytest.raises(RuntimeError, match="loader bug"):
        cache.get_or_compute(
            stage_name="demo", pmcid="PMC1", config_hash="abc",
            compute_fn=compute, loader_fn=bad_loader,
            dumper_fn=_trivial_dump, summarise_fn=lambda r: "1",
        )


def test_write_failure_propagates(tmp_path, monkeypatch):
    cache = _build_cache(tmp_path)

    from nlp_histo.pipeline.stages.pdf_text_extraction import stage_cache as sc

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(sc, "_atomic_write_json", boom)

    with pytest.raises(OSError, match="disk full"):
        cache.get_or_compute(
            stage_name="demo", pmcid="PMC1", config_hash="abc",
            compute_fn=lambda: {"x": 1}, loader_fn=_trivial_load,
            dumper_fn=_trivial_dump, summarise_fn=lambda r: "1",
        )


def test_artifact_without_sidecar_logs_specific_message(tmp_path, caplog):
    cache = _build_cache(tmp_path)
    artifact = tmp_path / "stage_cache" / "demo" / "PMC1.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}")
    # No sidecar.

    with caplog.at_level(logging.INFO,
                          logger="nlp_histo.pipeline.stages.pdf_text_extraction.stage_cache"):
        cache.get_or_compute(
            stage_name="demo", pmcid="PMC1", config_hash="abc",
            compute_fn=lambda: {"x": 1}, loader_fn=_trivial_load,
            dumper_fn=_trivial_dump, summarise_fn=lambda r: "1",
        )
    assert "artifact present but sidecar missing" in caplog.text


# ── Per-stage round-trips with runtime-type assertions ────────────────────────

def test_table_detection_round_trip(tmp_path):
    original = TableDetectionResult(
        regions=[
            DetectedRegion(
                bbox=BoundingBox(x1=10, y1=700, x2=400, y2=600, page=1),
                score=0.97, source="tatr", label="table",
            ),
            DetectedRegion(
                bbox=BoundingBox(x1=20, y1=500, x2=380, y2=300, page=3),
                score=0.99, source="hybrid", label="table rotated",
            ),
        ],
        source="tatr",
        pdf_path=Path("/tmp/foo.pdf"),
        page_dims={1: {"width": 612.0, "height": 792.0},
                   3: {"width": 612.0, "height": 792.0}},
    )
    artifact = tmp_path / "td.json"
    artifact.write_text(json.dumps(dump_table_detection(original)))

    loaded = load_table_detection(artifact)

    # Stability: dump(load(dump(x))) byte-identical to dump(x)
    assert dump_table_detection(loaded) == dump_table_detection(original)
    # Runtime types preserved
    assert isinstance(loaded.pdf_path, Path)
    assert all(isinstance(k, int) for k in loaded.page_dims.keys())
    assert loaded.regions[0].bbox.page == 1
    assert loaded.regions[1].label == "table rotated"


def test_layout_element_list_round_trip(tmp_path):
    elements = [
        LayoutElement(
            type="TEXT", page=1, level=0,
            bbox=BoundingBox(x1=10, y1=700, x2=400, y2=680, page=1),
            text="Body paragraph",
        ),
        LayoutElement(
            type="SECTION_HEADER", page=1, level=2,
            bbox=BoundingBox(x1=10, y1=720, x2=200, y2=710, page=1),
            text="2.1 Methods",
        ),
        LayoutElement(  # text=None edge case
            type="CAPTION", page=2, level=0,
            bbox=BoundingBox(x1=15, y1=300, x2=380, y2=290, page=2),
            text=None,
        ),
    ]
    artifact = tmp_path / "el.json"
    artifact.write_text(json.dumps(dump_layout_elements(elements)))

    loaded = load_layout_elements(artifact)
    assert len(loaded) == 3
    for orig, got in zip(elements, loaded):
        assert isinstance(got, LayoutElement)
        assert got.type == orig.type
        assert got.page == orig.page
        assert got.level == orig.level
        assert got.text == orig.text
        assert got.bbox.x1 == orig.bbox.x1
        assert got.bbox.page == orig.bbox.page


def test_hierarchical_row_list_round_trip(tmp_path):
    rows = [
        HierarchicalRow(
            path_string="Root",
            path_list=[],
            depth=0,
            text="abstract paragraph",
            source_chunks=["abstract paragraph"],
        ),
        HierarchicalRow(
            path_string="Methods > 2.1 Staining",
            path_list=["Methods", "2.1 Staining"],
            depth=2,
            text="Slides were stained.",
            source_chunks=[],
        ),
        HierarchicalRow(
            path_string="Results > Survival",
            path_list=["Results", "Survival"],
            depth=2,
            text="Patients with high CD30 had worse OS.",
            source_chunks=["Patients with high CD30", "had worse OS."],
        ),
    ]
    artifact = tmp_path / "rows.json"
    artifact.write_text(json.dumps(dump_hierarchical_rows(rows)))

    loaded = load_hierarchical_rows(artifact)
    assert len(loaded) == 3
    for orig, got in zip(rows, loaded):
        assert isinstance(got, HierarchicalRow)
        assert got.path_string == orig.path_string
        assert got.path_list == orig.path_list
        assert isinstance(got.path_list, list)
        assert got.depth == orig.depth
        assert got.text == orig.text
        assert got.source_chunks == orig.source_chunks
        assert isinstance(got.source_chunks, list)
        assert all(isinstance(c, str) for c in got.source_chunks)


# ── Seed initialisation ──────────────────────────────────────────────────────

def _make_runner(tmp_path: Path, seed: int | None = 42, **runtime_kw) -> PipelineRunner:
    cfg = PipelineConfig()
    cfg.runtime.seed = seed
    cfg.runtime.skip_existing_outputs = False
    cfg.database.enabled = False
    cfg.paths.output_root = tmp_path
    cfg.paths.blacklist_file = tmp_path / "blacklist.json"
    for k, v in runtime_kw.items():
        setattr(cfg.runtime, k, v)
    return PipelineRunner(cfg)


def test_seed_pipeline_calls_random_seed_with_config_value(tmp_path, monkeypatch):
    seeds: list[int] = []
    import random
    monkeypatch.setattr(random, "seed", lambda s: seeds.append(s))
    _make_runner(tmp_path, seed=1234)
    assert seeds == [1234]


def test_seed_pipeline_none_skips_all_seeding(tmp_path, monkeypatch):
    import os
    import random
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    called: list[int] = []
    monkeypatch.setattr(random, "seed", lambda s: called.append(s))
    _make_runner(tmp_path, seed=None)
    assert called == []
    assert "PYTHONHASHSEED" not in os.environ


# Restore the nulled module INSIDE the test (try/finally) rather than via
# ``monkeypatch.setitem``: pytest-randomly's per-test teardown reseed calls the
# numpy/torch RNG seeders (``thinc.fix_random_seed`` → ``torch.manual_seed``,
# ``numpy.random.seed``) and runs BEFORE monkeypatch's undo. With the sentinel
# ``None`` still in ``sys.modules`` that reseed raises ``ModuleNotFoundError:
# import of <mod> halted; None in sys.modules``, which aborts teardown so the undo
# never runs — leaking ``None`` into every later test's reseed (a 350+ error
# cascade). Restoring before we return keeps ``sys.modules`` clean for the reseed.
def _null_module_during(name: str):
    _sentinel = object()
    real = sys.modules.get(name, _sentinel)
    sys.modules[name] = None
    def _restore():
        if real is _sentinel:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = real
    return _restore


def test_seed_pipeline_handles_missing_torch(tmp_path, monkeypatch):
    import random
    seeds: list[int] = []
    monkeypatch.setattr(random, "seed", lambda s: seeds.append(s))
    restore = _null_module_during("torch")
    try:
        _make_runner(tmp_path, seed=7)  # must not raise
    finally:
        restore()
    assert seeds == [7]


def test_seed_pipeline_handles_missing_numpy(tmp_path, monkeypatch):
    import random
    seeds: list[int] = []
    monkeypatch.setattr(random, "seed", lambda s: seeds.append(s))
    restore = _null_module_during("numpy")
    try:
        _make_runner(tmp_path, seed=11)  # must not raise
    finally:
        restore()
    assert seeds == [11]


# ── Runner wiring (mocks the three expensive stages) ──────────────────────────

def _fake_layout(pdf_path: Path) -> LayoutResult:
    return LayoutResult(
        elements=[
            LayoutElement(type="SECTION_HEADER", page=1, level=1,
                          bbox=BoundingBox(x1=10, y1=720, x2=200, y2=710, page=1),
                          text="Methods"),
            LayoutElement(type="TEXT", page=1, level=0,
                          bbox=BoundingBox(x1=10, y1=700, x2=400, y2=680, page=1),
                          text="Body."),
        ],
        page_dims={1: {"width": 612.0, "height": 792.0}},
        pdf_path=pdf_path,
        source="docling",
    )


def _fake_table_detection(pdf_path: Path) -> TableDetectionResult:
    return TableDetectionResult(
        regions=[
            DetectedRegion(
                bbox=BoundingBox(x1=20, y1=500, x2=380, y2=300, page=1),
                score=0.99, source="tatr", label="table",
            ),
        ],
        source="tatr",
        pdf_path=pdf_path,
        page_dims={1: {"width": 612.0, "height": 792.0}},
    )


def _fake_filtered_elements() -> list[LayoutElement]:
    return [
        LayoutElement(type="TEXT", page=1, level=0,
                      bbox=BoundingBox(x1=10, y1=700, x2=400, y2=680, page=1),
                      text="Filtered body."),
    ]


def _fake_rows() -> list[HierarchicalRow]:
    return [
        HierarchicalRow(
            path_string="Methods", path_list=["Methods"], depth=1,
            text="Filtered body.", source_chunks=["Filtered body."],
        ),
    ]


def _build_runner_with_mocks(
    tmp_path: Path,
    *,
    two_pass: bool,
    raise_stages: bool = False,
) -> tuple[PipelineRunner, dict]:
    """Build a runner whose three cached stages are mocked.

    raise_stages=True turns each mock into a raise-on-call stub so a cache
    miss on the second run trips the test.
    """
    cfg = PipelineConfig()
    cfg.runtime.skip_existing_outputs = True
    cfg.runtime.seed = 0
    cfg.database.enabled = False
    cfg.visualization.enabled = False  # avoid loading PyMuPDF visualizer
    cfg.two_pass.enabled = two_pass
    cfg.paths.output_root = tmp_path
    cfg.paths.blacklist_file = tmp_path / "blacklist.json"
    for d in (cfg.paths.docling_full_dir, cfg.paths.docling_masked_dir,
              cfg.paths.text_dir, cfg.paths.json_dir, cfg.paths.figures_dir,
              cfg.paths.tables_dir, cfg.paths.run_metadata_dir,
              cfg.paths.masked_pdf_dir, cfg.paths.text_raw_dir):
        d.mkdir(parents=True, exist_ok=True)

    runner = PipelineRunner(cfg)

    # Fake stage outputs (or raise-on-call if requested).
    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")

    layout = _fake_layout(pdf_path)
    masked_layout = _fake_layout(pdf_path)
    detection = _fake_table_detection(pdf_path)
    rows = _fake_rows()
    filtered = _fake_filtered_elements()

    counters = {"detect": 0, "filter": 0, "assemble": 0}

    def detect(*_a, **_k):
        if raise_stages:
            raise AssertionError("table detection mock invoked on second run")
        counters["detect"] += 1
        return detection

    def filter_elements(_els):
        if raise_stages:
            raise AssertionError("artifact filter mock invoked on second run")
        counters["filter"] += 1
        return filtered

    def assemble(_layout):
        if raise_stages:
            raise AssertionError("text assembler mock invoked on second run")
        counters["assemble"] += 1
        return rows

    # Replace _run_table_detection to skip detector instantiation entirely.
    runner._run_table_detection = detect  # type: ignore[assignment]

    # Stub the artifact-filter and text-assembler factories.
    af_stub = MagicMock()
    af_stub.filter_elements = filter_elements
    runner._get_artifact_filter = lambda: af_stub  # type: ignore[assignment]

    ta_stub = MagicMock()
    ta_stub.assemble = assemble
    runner._get_text_assembler = lambda: ta_stub  # type: ignore[assignment]

    # Skip Steps 1/3/4 entirely — return fake layouts directly. The
    # standard-mode shadow exercises the table_detection cache wrap that
    # the real `_steps_1_3_4_standard` runs (otherwise stubbing the whole
    # function would bypass that cache slot). Two-pass shadow returns
    # detection=None because `_process` wraps the deferred call itself.
    from nlp_histo.pipeline.stages.pdf_text_extraction.stage_cache import (
        StageName as _SN, load_table_detection as _ltd,
        dump_table_detection as _dtd,
    )

    def _shadow_standard(pdf, pmc, cache):
        det = cache.get_or_compute(
            stage_name=_SN.TABLE_DETECTION, pmcid=pmc,
            config_hash=runner._current_config_hash,
            compute_fn=lambda: detect(layout, pdf),
            loader_fn=_ltd, dumper_fn=_dtd,
            summarise_fn=lambda r: f"{len(r.regions)} regions",
        )
        return layout, layout, masked_layout, det

    runner._steps_1_3_4_standard = _shadow_standard  # type: ignore[assignment]
    runner._steps_1_3_4_two_pass = lambda pdf, pmc: (  # type: ignore[assignment]
        layout, layout, masked_layout, None,
    )

    # Stub outputs (Step 8) to a no-op so we don't need real writers.
    runner._get_outputs = lambda: []  # type: ignore[assignment]

    # Stub media cropper to no-op.
    cropper_stub = MagicMock()
    cropper_stub.crop = lambda *a, **k: ([], [])
    runner._get_media_cropper = lambda: cropper_stub  # type: ignore[assignment]

    return runner, counters


def _expected_artifacts(tmp_path: Path, pmcid: str) -> list[Path]:
    base = tmp_path / "stage_cache"
    return [
        base / "table_detection" / f"{pmcid}.json",
        base / "table_detection" / f"{pmcid}.hash",
        base / "filtering" / f"{pmcid}.json",
        base / "filtering" / f"{pmcid}.hash",
        base / "text_assembly" / f"{pmcid}.json",
        base / "text_assembly" / f"{pmcid}.hash",
    ]


def test_runner_first_run_writes_three_caches_standard_path(tmp_path):
    runner, counters = _build_runner_with_mocks(tmp_path, two_pass=False)
    assert runner.run_document(tmp_path / "fake.pdf", "PMC1") is True
    assert counters == {"detect": 1, "filter": 1, "assemble": 1}
    for p in _expected_artifacts(tmp_path, "PMC1"):
        assert p.exists(), p

    # Artifacts deserialise to the canned mock outputs — proves the cache
    # wraps the expensive call site, not just a factory.
    rows_loaded = load_hierarchical_rows(
        tmp_path / "stage_cache" / "text_assembly" / "PMC1.json"
    )
    assert rows_loaded == _fake_rows()
    detection_loaded = load_table_detection(
        tmp_path / "stage_cache" / "table_detection" / "PMC1.json"
    )
    assert detection_loaded.regions[0].score == 0.99
    filtered_loaded = load_layout_elements(
        tmp_path / "stage_cache" / "filtering" / "PMC1.json"
    )
    assert filtered_loaded[0].text == "Filtered body."


def test_runner_second_run_hits_three_caches_standard_path(tmp_path, caplog):
    # First run populates the cache.
    runner1, _ = _build_runner_with_mocks(tmp_path, two_pass=False)
    assert runner1.run_document(tmp_path / "fake.pdf", "PMC1") is True

    # Second run uses a FRESH PipelineRunner instance with raise-on-call
    # stage mocks. If the cache fails to short-circuit, the mocks would
    # raise and the test fails loudly.
    runner2, _ = _build_runner_with_mocks(tmp_path, two_pass=False,
                                          raise_stages=True)
    with caplog.at_level(logging.INFO,
                          logger="nlp_histo.pipeline.stages.pdf_text_extraction.stage_cache"):
        assert runner2.run_document(tmp_path / "fake.pdf", "PMC1") is True

    hits = [r for r in caplog.records if "CACHE HIT" in r.getMessage()]
    assert len(hits) == 3, [r.getMessage() for r in caplog.records]


def test_runner_first_run_writes_three_caches_two_pass_path(tmp_path):
    runner, counters = _build_runner_with_mocks(tmp_path, two_pass=True)
    assert runner.run_document(tmp_path / "fake.pdf", "PMC1") is True
    assert counters == {"detect": 1, "filter": 1, "assemble": 1}
    for p in _expected_artifacts(tmp_path, "PMC1"):
        assert p.exists(), p


def test_runner_two_pass_to_standard_invalidates(tmp_path, caplog):
    runner1, _ = _build_runner_with_mocks(tmp_path, two_pass=True)
    assert runner1.run_document(tmp_path / "fake.pdf", "PMC1") is True

    # Mode flip: build a fresh runner with two_pass=False. The hash should
    # change and all three stages should log CACHE STALE.
    runner2, counters2 = _build_runner_with_mocks(tmp_path, two_pass=False)
    with caplog.at_level(logging.INFO,
                          logger="nlp_histo.pipeline.stages.pdf_text_extraction.stage_cache"):
        assert runner2.run_document(tmp_path / "fake.pdf", "PMC1") is True

    stales = [r for r in caplog.records if "CACHE STALE" in r.getMessage()]
    assert len(stales) == 3, [r.getMessage() for r in caplog.records]
    assert counters2 == {"detect": 1, "filter": 1, "assemble": 1}


def test_runner_skip_false_does_not_use_cache(tmp_path):
    runner1, _ = _build_runner_with_mocks(tmp_path, two_pass=False)
    runner1.run_document(tmp_path / "fake.pdf", "PMC1")

    runner2, counters2 = _build_runner_with_mocks(tmp_path, two_pass=False)
    runner2._cfg.runtime.skip_existing_outputs = False
    runner2.run_document(tmp_path / "fake.pdf", "PMC1")
    assert counters2 == {"detect": 1, "filter": 1, "assemble": 1}


def test_runner_pass1_layout_feeds_table_detection(tmp_path):
    """Pin the assumption that two-pass table detection runs against the
    `pass1_layout`-derived `LayoutResult` (not a fresh extract). Future
    refactors that diverge the input will break this test."""
    runner, _ = _build_runner_with_mocks(tmp_path, two_pass=True)
    # Inspect what _run_table_detection sees by capturing its first arg.
    seen = {}
    real_detect = runner._run_table_detection

    def spy(layout, pdf_path):
        seen["pages"] = sorted({el.page for el in layout.elements})
        return real_detect(layout, pdf_path)

    runner._run_table_detection = spy  # type: ignore[assignment]
    runner.run_document(tmp_path / "fake.pdf", "PMC1")
    # Our fake pass1 layout has page=1 only; if a future refactor swaps in
    # a fresh extract for two-pass detection this assertion exposes it.
    assert seen["pages"] == [1]
