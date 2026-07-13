"""
PipelineRunner — single-document orchestrator for the PDF text-extraction pipeline.

Standard 8-step flow (`PipelineConfig.two_pass.enabled=False`):

  1. Extract layout from original PDF (Docling)
  2. Detect tables (Docling | TATR | Hybrid — `cfg.table_detector`)
  3. Mask detected regions with white rectangles
  4. Re-extract layout from masked PDF (Docling, optionally with
     `cfg.docling_text` overriding OCR/structure options)
  5. Filter layout artifacts (rule-based + optional scispaCy NER)
  6. Assemble hierarchical text rows
  7. Crop figure / table images
  8. Write outputs (text file, media JSON, optional PostgreSQL ingest)

Two-pass flow (default, `PipelineConfig.two_pass.enabled=True`):

  Steps 1/3/4 are replaced by `TwoPassTextExtractor`, which runs Docling
  twice — once on the original PDF (Pass 1) to score every element using
  pixel-rendering + fitz word evidence, then once on a header-masked copy
  (Pass 2) — producing the cleaned `pass1_layout` / `pass2_layout`.
  Step 2 (table detection) runs against `pass1_layout` after the two-pass
  call.  Steps 5–8 are unchanged.

Additional behaviour:

  * Steps 2, 5, 6 are cached on disk under
    `<output_root>/stage_cache/<stage_name>/<pmcid>.json` with a
    config-hash sidecar; cache hits skip recomputation
    (`RuntimeConfig.skip_existing_outputs=True`).
  * Step 2 input PDF can be pre-masked with figure bboxes when
    `MaskingConfig.mask_figures_before_table_detection=True`, so the
    pixel-based detectors never see table-grid pixels embedded in figures.
  * Step 2 output can be post-filtered to drop table regions that sit
    inside FIGURE/PICTURE elements (`MaskingConfig.drop_tables_inside_figures`).
  * Step 7 optionally writes a second crop set under `<tables_dir>/docling`
    and `<tables_dir>/docling_recon` for sweep comparisons
    (`RuntimeConfig.multi_source_crops`).
  * Every run can emit per-document JSON stats and a per-batch manifest
    (see `outputs/stats_writer.py`, `outputs/manifest_writer.py`).

The module also exposes a thin argparse `main()` for shell invocations and
a `_retarget_paths` helper that repoints every `PathConfig` directory under
a sweep-specific root (used by `scripts/eval/run_all_sweeps.py`).

Run it from the repository root as a module::

    python -m pipeline.stages.pdf_text_extraction.runner --help

Usage::

    from pathlib import Path
    from pipeline.stages.pdf_text_extraction import PipelineConfig, PipelineRunner

    cfg    = PipelineConfig()
    cfg.database.enabled = True
    cfg.database.db_url  = "postgresql://user:pw@localhost/nlp_histo"
    cfg.prepare()                       # validate + create output dirs

    runner = PipelineRunner(cfg)
    runner.run_document(Path("files/organized_pdfs/PMC123.pdf"), pmcid="PMC123")

    # Sequential batch (single thread).  For a thread pool use
    # `ParallelBatchRunner` from pipeline.stages.pdf_text_extraction.batch.
    runner.run_batch(pdf_dir=Path("files/organized_pdfs"))
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import List, Optional

from pipeline.stages.pdf_text_extraction.blacklist import BlacklistManager
from pipeline.stages.pdf_text_extraction.config import PipelineConfig, TableDetectorType
from pipeline.stages.pdf_text_extraction.models.dto import LayoutResult
from pipeline.stages.pdf_text_extraction.outputs.manifest_writer import (
    RunManifestWriter,
    make_run_id,
)
from pipeline.stages.pdf_text_extraction.outputs.stats_writer import (
    DocStatsCollector,
    config_digest,
    config_snapshot,
)
from pipeline.stages.pdf_text_extraction.stage_cache import (
    STAGE_CACHE_VERSION,
    StageName,
    _StageCache,
    dump_hierarchical_rows,
    dump_layout_elements,
    dump_table_detection,
    load_hierarchical_rows,
    load_layout_elements,
    load_table_detection,
)
from pipeline.stages.knowledge_extraction.persistence import compute_pipeline_config_hash

logger = logging.getLogger(__name__)


def _count_region_sources(regions) -> dict:
    """Tally TableRegion.source values for stats reporting."""
    counts: dict = {}
    for r in regions or []:
        src = getattr(r, "source", "unknown") or "unknown"
        counts[src] = counts.get(src, 0) + 1
    return counts


_FIGURE_ELEMENT_TYPES = frozenset({"FIGURE", "PICTURE"})


def _bbox_area_pts(bbox) -> float:
    return abs((bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1))


def _bbox_intersect_area(a, b) -> float:
    """Area of the intersection of two BoundingBoxes on the same page.

    Coord-orientation-agnostic: works for Docling (y₁ > y₂, bottom-origin)
    and fitz (y₀ < y₁, top-origin) alike by min/max-normalising each axis.
    """
    if getattr(a, "page", None) != getattr(b, "page", None):
        return 0.0
    x1 = max(min(a.x1, a.x2), min(b.x1, b.x2))
    x2 = min(max(a.x1, a.x2), max(b.x1, b.x2))
    y1 = max(min(a.y1, a.y2), min(b.y1, b.y2))
    y2 = min(max(a.y1, a.y2), max(b.y1, b.y2))
    if x2 > x1 and y2 > y1:
        return (x2 - x1) * (y2 - y1)
    return 0.0


def _drop_tables_inside_figures(
    detection,
    layout,
    *,
    threshold: float = 0.8,
):
    """Filter ``detection.regions`` by figure containment.

    A detected table region is dropped when at least ``threshold`` fraction
    of its bbox area lies inside any FIGURE/PICTURE layout element on the
    same page.  Mutates a new ``TableDetectionResult`` — the input is left
    untouched so the stage cache key stays stable per its existing config
    hash.

    Returns ``(filtered_detection, n_dropped)``.
    """
    from dataclasses import replace
    figure_bboxes = [
        el.bbox for el in (layout.elements if layout is not None else [])
        if el.type in _FIGURE_ELEMENT_TYPES
    ]
    if not figure_bboxes:
        return detection, 0

    kept = []
    dropped = 0
    for region in detection.regions:
        area = _bbox_area_pts(region.bbox)
        if area <= 0:
            kept.append(region)
            continue
        # Sum of intersection-with-figures.  Conservative: if any single
        # figure contains ≥threshold of the table, drop.  We don't OR the
        # overlaps across figures — usually a "table inside figure" sits
        # inside ONE figure entirely.
        max_frac = 0.0
        for fb in figure_bboxes:
            if fb.page != region.bbox.page:
                continue
            frac = _bbox_intersect_area(region.bbox, fb) / area
            if frac > max_frac:
                max_frac = frac
            if max_frac >= threshold:
                break
        if max_frac >= threshold:
            dropped += 1
        else:
            kept.append(region)

    return replace(detection, regions=kept), dropped


def _drop_tables_in_top_pts(
    detection,
    layout,
    *,
    threshold_pts: float,
):
    """Filter ``detection.regions`` whose top edge sits within ``threshold_pts``
    of the page top.  Catches page-header bands that pixel detectors
    (TATR / Hybrid) misclassify as tables.

    Uses ``layout.page_dims[page] = (width, height)`` to convert each
    region's top (Docling coords: y1 = top, y1 is high when near top of
    page) into a distance-from-top measurement.

    Returns ``(filtered_detection, n_dropped)``.
    """
    from dataclasses import replace
    if threshold_pts <= 0:
        return detection, 0
    page_dims = getattr(layout, "page_dims", None) or {}

    kept = []
    dropped = 0
    for region in detection.regions:
        page = region.bbox.page
        dims = page_dims.get(page) or page_dims.get(str(page))
        if not dims:
            kept.append(region)
            continue
        page_height = dims[1] if isinstance(dims, (list, tuple)) else dims.get("height")
        if not page_height:
            kept.append(region)
            continue
        top_from_top = page_height - max(region.bbox.y1, region.bbox.y2)
        if top_from_top < threshold_pts:
            dropped += 1
        else:
            kept.append(region)

    return replace(detection, regions=kept), dropped


def _detector_name(detector) -> str:
    """TableDetectorType enum → short name (HYBRID/TATR/DOCLING/VLM)."""
    return getattr(detector, "name", str(detector))


class PipelineRunner:
    """
    Orchestrates the pipeline for one or many documents.

    Parameters
    ----------
    config:
        Master PipelineConfig.  Call ``config.prepare()`` before first use to
        validate settings and create output directories.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self._cfg = config or PipelineConfig()
        self._blacklist  = BlacklistManager(self._cfg.paths.blacklist_file)
        self._completed  = BlacklistManager(self._cfg.paths.completed_file) if self._cfg.paths.completed_file else None

        # Lazy stage instances — created on first use
        self._layout_extractor  = None
        self._region_masker     = None
        self._text_assembler    = None
        self._artifact_filter   = None
        self._media_cropper     = None
        self._table_detector    = None
        self._visualizer        = None
        self._two_pass_extractor = None
        self._nlp               = None
        self._outputs: list     = []

        # Per-run state — populated at the top of `run_document`.
        self._current_config_hash: Optional[str] = None
        self._current_stats: Optional[DocStatsCollector] = None

        # Seed pipeline-owned randomness (B-027). Idempotent within the
        # process; safe to call from each PipelineRunner constructor.
        self._seed_pipeline()

    # ── Seeding ───────────────────────────────────────────────────────────────

    def _seed_pipeline(self) -> None:
        """Seed pipeline-owned randomness.

        Seeds `random`, `numpy`, and (when installed) `torch` /
        `torch.cuda` with `cfg.runtime.seed`. Setting `seed=None` opts out.

        NOTE: this does NOT guarantee deterministic output from external
        document/layout libraries (Docling, TATR weights, OCR engines,
        scispaCy). Cached stage outputs are content-keyed by
        `cfg.runtime.skip_existing_outputs`, not by seed.

        `PYTHONHASHSEED` is set best-effort; it only fully governs hash
        randomisation when set BEFORE Python interpreter startup. Future
        child processes (none today in the PDF pipeline) inherit it.
        """
        seed = self._cfg.runtime.seed
        if seed is None:
            logger.info("Seeding skipped (runtime.seed is None)")
            return

        import os
        import random

        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        try:
            import numpy as np  # type: ignore
            np.random.seed(seed)
        except ImportError:
            pass
        try:
            import torch  # type: ignore
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # Intentionally NOT setting torch.use_deterministic_algorithms
            # or cuDNN deterministic flags — those carry perf and op-coverage
            # costs and are out of scope for this patch.
        except ImportError:
            pass
        logger.info(
            "Seeded pipeline-owned randomness (seed=%d). Note: this does not "
            "guarantee deterministic output from external libraries (Docling, "
            "TATR, OCR, scispaCy).", seed,
        )

    # ── Config hash (used to invalidate stage cache) ──────────────────────────

    def _compute_config_hash(self) -> str:
        """Compute a stable hash of the output-affecting config surface.

        Excludes RuntimeConfig (audited as gating-only: `skip_blacklisted`,
        `update_blacklist_on_failure`, `blacklist_if_rows_exceed`,
        `num_workers`, log level, `seed` — none touch element/page-level
        output for cached stages 2/5/6 today). Includes the scispaCy model
        name when NER paths are active. Includes `STAGE_CACHE_VERSION` so
        format / behaviour changes invalidate older artifacts.

        Not in the hash but documented in `docs/readmes/HOW_TO_RUN.md`: scispaCy /
        spaCy package + model-weights versions. Run `rm -rf out/stage_cache/`
        after dependency upgrades that affect NLP output.
        """
        cfg = self._cfg
        nlp_model = (
            "en_core_sci_sm"
            if (
                cfg.filtering.apply_ner_filtering
                or cfg.filtering.apply_paragraph_relevance_filtering
                or cfg.masking.mask_header_footer_sidebar
            )
            else None
        )
        return compute_pipeline_config_hash(
            config={
                "docling":        cfg.docling,
                "docling_text":   cfg.docling_text,
                "tatr":           cfg.tatr,
                "masking":        cfg.masking,
                "filtering":      cfg.filtering,
                "text":           cfg.text,
                "two_pass":       cfg.two_pass,
                "table_detector": cfg.table_detector,
                "nlp_model":      nlp_model,
                "stage_version":  STAGE_CACHE_VERSION,
            },
            schema_version="pdf_v1",
        )

    def _get_stage_cache(self) -> _StageCache:
        return _StageCache(
            root=self._cfg.paths.output_root / "stage_cache",
            enabled=self._cfg.runtime.skip_existing_outputs,
        )

    # ── Stage factory helpers ─────────────────────────────────────────────────

    def _get_layout_extractor(self):
        if self._layout_extractor is None:
            from pipeline.stages.pdf_text_extraction.components.layout_extractor import DoclingLayoutExtractor
            self._layout_extractor = DoclingLayoutExtractor(
                config=self._cfg.docling,
                cache_dir=self._cfg.paths.docling_full_dir if self._cfg.docling.export_intermediate_json else None,
            )
        return self._layout_extractor

    def _get_table_detector(self):
        if self._table_detector is None:
            dtype = self._cfg.table_detector
            if dtype == TableDetectorType.DOCLING:
                from pipeline.stages.pdf_text_extraction.table_detectors import DoclingTableDetector
                self._table_detector = DoclingTableDetector()
            elif dtype == TableDetectorType.TATR:
                from pipeline.stages.pdf_text_extraction.table_detectors import TATRTableDetector
                self._table_detector = TATRTableDetector(self._cfg.tatr)
            else:  # HYBRID or VLM (fallback to hybrid)
                from pipeline.stages.pdf_text_extraction.table_detectors import HybridTableDetector
                self._table_detector = HybridTableDetector(tatr_config=self._cfg.tatr)
        return self._table_detector

    def _get_region_masker(self):
        if self._region_masker is None:
            from pipeline.stages.pdf_text_extraction.components.region_masker import PyMuPDFRegionMasker
            self._region_masker = PyMuPDFRegionMasker(
                config=self._cfg.masking,
                output_dir=self._cfg.paths.masked_pdf_dir,
            )
        return self._region_masker

    def _get_two_pass_extractor(self):
        if self._two_pass_extractor is None:
            from pipeline.stages.pdf_text_extraction.components.two_pass_extractor import TwoPassTextExtractor
            docling_text_cfg = self._cfg.docling_text  # None → reuse docling cfg
            self._two_pass_extractor = TwoPassTextExtractor(
                config=self._cfg.two_pass,
                docling_config=self._cfg.docling,
                docling_text_config=docling_text_cfg,
                cache_dir=self._cfg.paths.docling_full_dir if self._cfg.docling.export_intermediate_json else None,
                masked_pdf_dir=self._cfg.paths.masked_pdf_dir,
            )
        return self._two_pass_extractor

    def _get_masked_extractor(self):
        """Second layout extractor for the masked PDF (separate cache dir).

        Uses ``config.docling_text`` if set, otherwise falls back to ``config.docling``.
        This lets you run OCR-enabled detection in Step 1 while keeping OCR off for
        the clean text re-extraction in Step 4.
        """
        from pipeline.stages.pdf_text_extraction.components.layout_extractor import DoclingLayoutExtractor
        cfg = self._cfg.docling_text if self._cfg.docling_text is not None else self._cfg.docling
        return DoclingLayoutExtractor(
            config=cfg,
            cache_dir=self._cfg.paths.docling_masked_dir if cfg.export_intermediate_json else None,
        )

    def _get_artifact_filter(self):
        if self._artifact_filter is None:
            from pipeline.stages.pdf_text_extraction.components.artifact_filter import ArtifactFilter
            nlp = self._get_nlp() if self._cfg.filtering.apply_ner_filtering else None
            self._artifact_filter = ArtifactFilter(config=self._cfg.filtering, nlp=nlp)
        return self._artifact_filter

    def _get_text_assembler(self):
        if self._text_assembler is None:
            from pipeline.stages.pdf_text_extraction.components.text_assembler import HierarchicalTextAssembler
            nlp = self._get_nlp() if self._cfg.filtering.apply_paragraph_relevance_filtering else None
            self._text_assembler = HierarchicalTextAssembler(
                config=self._cfg.text,
                skip_references_section=True,
                nlp=nlp,
            )
        return self._text_assembler

    def _get_media_cropper(self):
        if self._media_cropper is None:
            from pipeline.stages.pdf_text_extraction.components.media_cropper import PyMuPDFMediaCropper
            self._media_cropper = PyMuPDFMediaCropper(
                config=self._cfg.cropping,
                figures_dir=self._cfg.paths.figures_dir,
                tables_dir=self._cfg.paths.tables_dir,
            )
        return self._media_cropper

    def _get_outputs(self) -> list:
        if not self._outputs:
            from pipeline.stages.pdf_text_extraction.outputs.writer import TextFileWriter
            from pipeline.stages.pdf_text_extraction.outputs.media_json_writer import MediaJsonWriter
            self._outputs.append(TextFileWriter(
                output_dir=self._cfg.paths.text_dir,
            ))
            self._outputs.append(MediaJsonWriter(
                output_dir=self._cfg.paths.json_dir,
            ))
            if self._cfg.database.enabled:
                from pipeline.stages.pdf_text_extraction.outputs.db_ingester import PostgresDatabaseIngester
                self._outputs.append(PostgresDatabaseIngester(db_url=self._cfg.database.db_url))
        return self._outputs

    def _already_in_db(self, pmcid: str) -> bool:
        try:
            from database import get_db_connection, Document  # type: ignore
            db = get_db_connection(database_url=self._cfg.database.db_url)
            with db.session_scope() as session:
                return session.query(Document).filter_by(pmcid=pmcid).first() is not None
        except Exception:
            return False  # if DB is unreachable, let the run proceed

    def _get_nlp(self):
        needs_nlp = (
            self._cfg.masking.mask_header_footer_sidebar
            or self._cfg.filtering.apply_ner_filtering
            or self._cfg.filtering.apply_paragraph_relevance_filtering
        )
        if self._nlp is None and needs_nlp:
            # B-029 fix: route through the process-wide small-model singleton
            # in knowledge_extraction/umls_resources rather than calling spacy.load
            # directly. Prevents double-loading when this pipeline runs in the
            # same process as the summarisation pipeline.
            try:
                from pipeline.stages.knowledge_extraction.entities.umls_resources import get_small_nlp
                self._nlp = get_small_nlp("en_core_sci_sm")
                if self._nlp is None:
                    logger.warning("scispaCy not available — NER features disabled")
                    self._nlp = False  # falsy sentinel so we don't retry
                else:
                    logger.info("scispaCy small model in use (via shared singleton).")
            except ImportError as exc:
                logger.warning("scispaCy not importable — NER features disabled: %s", exc)
                self._nlp = False
        return self._nlp or None

    def _get_visualizer(self):
        if self._visualizer is None and self._cfg.visualization.enabled:
            from pipeline.stages.pdf_text_extraction.components.visualizer import DetectionVisualizer
            self._visualizer = DetectionVisualizer(
                config=self._cfg.visualization,
                output_dir=self._cfg.paths.vis_dir,
            )
        return self._visualizer

    # ── Single document ───────────────────────────────────────────────────────

    def run_document(
        self,
        pdf_path: Path,
        pmcid: str,
        run_id: Optional[str] = None,
    ) -> bool:
        """
        Process a single PDF document end-to-end.

        Args:
            pdf_path: Path to the source PDF.
            pmcid:    PubMed Central ID for the document.
            run_id:   Optional batch-run identifier embedded in the per-doc
                      stats file.  Default ``None`` (used in single-doc runs).

        Returns:
            True on success, False on failure.
        """
        if self._cfg.runtime.skip_blacklisted and self._blacklist.contains(pmcid):
            logger.info("⚡ %s — skipped (blacklisted)", pmcid)
            return False

        if self._completed and self._completed.contains(pmcid):
            logger.info("⚡ %s — skipped (already completed)", pmcid)
            return True

        if self._cfg.runtime.skip_existing_in_db and self._cfg.database.enabled:
            if self._already_in_db(pmcid):
                logger.info("⚡ %s — skipped (already in database)", pmcid)
                return True

        if self._cfg.runtime.skip_existing_media_json:
            media_json = self._cfg.paths.json_dir / f"{pmcid}_media.json"
            if media_json.exists():
                logger.info("⚡ %s — skipped (media JSON exists)", pmcid)
                return True

        # Observability collector — created BEFORE the first fail-prone step
        # so failed documents still get a stats file.  Best-effort: if init
        # fails, we proceed without stats rather than break extraction.
        stats: Optional[DocStatsCollector] = None
        try:
            stats = DocStatsCollector(
                pmcid=pmcid,
                pdf_path=pdf_path,
                run_metadata_dir=self._cfg.paths.run_metadata_dir,
                run_id=run_id,
                config_digest=config_digest(self._cfg),
                config_used=config_snapshot(self._cfg),
            )
        except Exception:
            logger.exception("DocStatsCollector init failed — continuing without stats")
            stats = None
        self._current_stats = stats

        try:
            try:
                result = self._process(pdf_path, pmcid)
                logger.info("✅ %s — done (%d rows)", pmcid, len(result))
                if stats is not None:
                    stats.set_count("text_rows", len(result))
                    stats.mark_ok()
                if self._completed:
                    self._completed.add(pmcid)
                limit = self._cfg.runtime.blacklist_if_rows_exceed
                if limit is not None and len(result) > limit:
                    self._blacklist.add(pmcid, reason=f"too large ({len(result)} rows > {limit})")
                    logger.info("🚫 %s — blacklisted (too large: %d rows)", pmcid, len(result))
                return True
            except Exception as exc:
                logger.error("❌ %s — failed: %s", pmcid, exc)
                if self._cfg.runtime.save_error_traces:
                    logger.debug(traceback.format_exc())
                if stats is not None:
                    stats.mark_failed(exc)
                if self._cfg.runtime.update_blacklist_on_failure:
                    self._blacklist.add(pmcid, reason=str(exc))
                if self._cfg.runtime.fail_fast:
                    raise
                return False
        finally:
            if stats is not None:
                stats.write()
            self._current_stats = None

    # ── Stats helpers ─────────────────────────────────────────────────────────

    def _stage(self, name: str):
        """Context manager that times a stage if stats are enabled, else no-op."""
        import contextlib
        if self._current_stats is None:
            @contextlib.contextmanager
            def _noop():
                yield
            return _noop()
        return self._current_stats.stage(name)

    def _record(self, fn) -> None:
        """Run a stats-recording closure when stats are enabled; swallow errors."""
        if self._current_stats is None:
            return
        try:
            fn(self._current_stats)
        except Exception:
            logger.exception("stats record op failed (suppressed)")

    def _patch_section_header_types(
        self,
        masked_layout: LayoutResult,
        full_layout: LayoutResult,
    ) -> None:
        """
        Restore SECTION_HEADER types lost during masked re-extraction.

        Builds a bbox-keyed lookup of every SECTION_HEADER in the full layout
        and promotes any matching TEXT element in the masked layout back to
        SECTION_HEADER.  Matching is done on (page, x1, y1, x2, y2) rounded
        to 2 decimal places to absorb floating-point noise.
        """
        full_headers: set = set()
        for el in full_layout.elements:
            if el.type == "SECTION_HEADER":
                b = el.bbox
                full_headers.add((
                    el.page,
                    round(b.x1, 2), round(b.y1, 2),
                    round(b.x2, 2), round(b.y2, 2),
                ))

        patched = 0
        for el in masked_layout.elements:
            if el.type != "SECTION_HEADER":
                b = el.bbox
                key = (
                    el.page,
                    round(b.x1, 2), round(b.y1, 2),
                    round(b.x2, 2), round(b.y2, 2),
                )
                if key in full_headers:
                    el.type = "SECTION_HEADER"
                    patched += 1

        if patched:
            logger.info("  Patched %d element(s) TEXT → SECTION_HEADER from full layout", patched)

    def _detection_pdf_path(self, layout: LayoutResult, pdf_path: Path) -> Path:
        """Return the PDF path to feed to the Step-2 table detector.

        When ``MaskingConfig.mask_figures_before_table_detection`` is True,
        builds a `<stem>_fig_masked.pdf` where PICTURE/FIGURE bboxes from
        ``layout`` are whited out, so TATR / Hybrid never see table-grid
        pixels embedded in figures.  Falls back to ``pdf_path`` if the
        layout contains no figures, or if the flag is disabled.

        Only affects detector input.  Downstream cropping (Step 7) still
        uses the original ``pdf_path``.
        """
        if not self._cfg.masking.mask_figures_before_table_detection:
            return pdf_path
        masker = self._get_region_masker()
        masked = masker.mask_figures_only(pdf_path, layout)
        return masked if masked is not None else pdf_path

    def _run_table_detection(self, layout: LayoutResult, pdf_path: Path):
        """Run the configured table detector and return a TableDetectionResult.

        ``pdf_path`` here is the *detection input* — may be a figure-masked
        copy when ``MaskingConfig.mask_figures_before_table_detection`` is
        on.  Callers must pass the original PDF to downstream cropping.

        If ``MaskingConfig.drop_tables_inside_figures`` is True, post-filters
        the detector output to drop regions whose bbox is mostly contained in
        any FIGURE/PICTURE element (eliminates "table-inside-figure" FPs).
        """
        detector = self._get_table_detector()
        from pipeline.stages.pdf_text_extraction.table_detectors.hybrid_detector import HybridTableDetector
        from pipeline.stages.pdf_text_extraction.table_detectors.docling_detector import DoclingTableDetector
        if isinstance(detector, HybridTableDetector):
            result = detector.detect_with_layout(layout, pdf_path)
        elif isinstance(detector, DoclingTableDetector):
            result = detector.detect_from_layout(layout)
        else:
            result = detector.detect(pdf_path)

        if self._cfg.masking.drop_tables_inside_figures:
            result, n_dropped = _drop_tables_inside_figures(result, layout)
            if n_dropped:
                logger.info(
                    "  filter: dropped %d table detection(s) inside FIGURE/PICTURE bboxes",
                    n_dropped,
                )
            self._record(lambda s: s.set_count("table_regions_dropped_inside_figures", n_dropped))
        if self._cfg.masking.drop_tables_in_top_pts > 0:
            result, n_dropped = _drop_tables_in_top_pts(
                result, layout,
                threshold_pts=self._cfg.masking.drop_tables_in_top_pts,
            )
            if n_dropped:
                logger.info(
                    "  filter: dropped %d table detection(s) within top %.1fpt of page",
                    n_dropped, self._cfg.masking.drop_tables_in_top_pts,
                )
            self._record(lambda s: s.set_count("table_regions_dropped_in_top_pts", n_dropped))
        return result

    def _steps_1_3_4_standard(
        self, pdf_path: Path, pmcid: str, cache: _StageCache,
    ):
        """
        Standard Steps 1–4: extract → detect tables → mask → re-extract.

        Returns (layout, layout_pre_recon, masked_layout, detection).
        Detection runs here (before masking) because the region masker needs it.

        Reminder: bump STAGE_CACHE_VERSION[StageName.TABLE_DETECTION] in
        stage_cache.py if `_run_table_detection` behaviour changes.
        """
        from pipeline.stages.pdf_text_extraction.components.table_reconstructor import reconstruct_tables_from_lists

        logger.info("[%s] Step 1 — layout extraction", pmcid)
        with self._stage("STEP1_LAYOUT"):
            layout: LayoutResult = self._get_layout_extractor().extract(pdf_path)
        self._record(lambda s: s.set_count("layout_elements_full", len(layout.elements)))
        layout_pre_recon = layout

        if self._cfg.docling.reconstruct_tables_from_lists:
            layout = reconstruct_tables_from_lists(layout)

        logger.info("[%s] Step 2 — table detection (%s)", pmcid, self._cfg.table_detector)
        detection_pdf = self._detection_pdf_path(layout, pdf_path)
        with self._stage("STEP2_TABLE_DETECTION"):
            detection = cache.get_or_compute(
                stage_name=StageName.TABLE_DETECTION,
                pmcid=pmcid,
                config_hash=self._current_config_hash,
                compute_fn=lambda: self._run_table_detection(layout, detection_pdf),
                loader_fn=load_table_detection,
                dumper_fn=dump_table_detection,
                summarise_fn=lambda r: f"{len(r.regions)} regions",
            )
        self._record(lambda s: s.record_table_detection(
            detector=_detector_name(self._cfg.table_detector),
            n_regions=len(detection.regions),
            per_source_counts=_count_region_sources(detection.regions),
        ))

        masker = self._get_region_masker()
        regions_to_mask = masker.collect_regions(
            detection, layout, nlp=self._get_nlp()
        ) if self._cfg.masking.enabled else []
        self._record(lambda s: s.set_count("mask_regions", len(regions_to_mask)))
        if regions_to_mask:
            logger.info("[%s] Step 3 — masking %d regions", pmcid, len(regions_to_mask))
            with self._stage("STEP3_MASKING"):
                masked_path = masker.mask(pdf_path, regions_to_mask)
        else:
            masked_path = pdf_path

        logger.info("[%s] Step 4 — re-extraction from masked PDF", pmcid)
        with self._stage("STEP4_REEXTRACT"):
            masked_layout: LayoutResult = self._get_masked_extractor().extract(masked_path)
        self._record(lambda s: s.set_count("layout_elements_masked", len(masked_layout.elements)))
        self._patch_section_header_types(masked_layout, layout)
        return layout, layout_pre_recon, masked_layout, detection

    def _steps_1_3_4_two_pass(self, pdf_path: Path, pmcid: str):
        """
        Two-pass Steps 1, 3, 4: ghost-text scoring → header masking → re-extract.

        Table detection does NOT run here — it runs in _process() (Step 2) after
        this returns, using pass1_layout as the canonical layout.

        Returns (layout, layout_pre_recon, masked_layout, detection=None).
        """
        from pipeline.stages.pdf_text_extraction.models.dto import LayoutResult as _LR
        from pipeline.stages.pdf_text_extraction.components.table_reconstructor import reconstruct_tables_from_lists

        logger.info("[%s] Steps 1,3,4 — two-pass extraction", pmcid)
        with self._stage("STEP1_3_4_TWO_PASS"):
            tp = self._get_two_pass_extractor().process(pdf_path)

        self._record(lambda s: s.record_scored_nodes(tp.scored_nodes))
        self._record(lambda s: s.set_count("layout_elements_full", len(tp.pass1_layout)))
        self._record(lambda s: s.set_count("layout_elements_masked", len(tp.pass2_layout)))

        layout = _LR(
            elements=tp.pass1_layout,
            page_dims=tp.page_dims,
            pdf_path=pdf_path,
            source="docling",
        )
        layout_pre_recon = layout

        if self._cfg.docling.reconstruct_tables_from_lists:
            layout = reconstruct_tables_from_lists(layout)

        masked_layout = _LR(
            elements=tp.pass2_layout,
            page_dims=tp.page_dims,
            pdf_path=tp.masked_pdf_path or pdf_path,
            source="docling",
        )
        self._patch_section_header_types(masked_layout, layout)
        return layout, layout_pre_recon, masked_layout, None  # detection deferred

    def _process(self, pdf_path: Path, pmcid: str):
        # Compute the per-run config hash used to invalidate stage cache.
        # Stays on `self` so multiple stages share the same value within a
        # `run_document` call without recomputing.
        self._current_config_hash = self._compute_config_hash()
        cache = self._get_stage_cache()

        if self._cfg.two_pass.enabled:
            layout, layout_pre_recon, masked_layout, detection = self._steps_1_3_4_two_pass(pdf_path, pmcid)
            # Step 2: table detection deferred to here in two-pass mode.
            # Same cache slot as the standard branch — `cfg.two_pass.enabled`
            # is in the config hash so a mode flip invalidates correctly.
            logger.info("[%s] Step 2 — table detection (%s)", pmcid, self._cfg.table_detector)
            detection_pdf = self._detection_pdf_path(layout, pdf_path)
            with self._stage("STEP2_TABLE_DETECTION"):
                detection = cache.get_or_compute(
                    stage_name=StageName.TABLE_DETECTION,
                    pmcid=pmcid,
                    config_hash=self._current_config_hash,
                    compute_fn=lambda: self._run_table_detection(layout, detection_pdf),
                    loader_fn=load_table_detection,
                    dumper_fn=dump_table_detection,
                    summarise_fn=lambda r: f"{len(r.regions)} regions",
                )
            self._record(lambda s: s.record_table_detection(
                detector=_detector_name(self._cfg.table_detector),
                n_regions=len(detection.regions),
                per_source_counts=_count_region_sources(detection.regions),
            ))
        else:
            layout, layout_pre_recon, masked_layout, detection = self._steps_1_3_4_standard(pdf_path, pmcid, cache)

        # ── Step 2b: Visualization ────────────────────────────────────────────
        if vis := self._get_visualizer():
            with self._stage("STEP2B_VISUALIZATION"):
                vis.visualize_layout(layout, pmcid)
                vis.visualize_detections(detection, layout, pmcid)

        # ── Step 5: Artifact filtering ────────────────────────────────────────
        # Reminder: bump STAGE_CACHE_VERSION[StageName.FILTERING] in
        # stage_cache.py if this stage's behaviour OR the on-disk format
        # changes (e.g. new module-level constant in
        # parsers/layout_utils.filter_artifacts).
        if self._cfg.filtering.enabled:
            logger.info("[%s] Step 5 — artifact filtering", pmcid)
            with self._stage("STEP5_FILTERING"):
                masked_layout.elements = cache.get_or_compute(
                    stage_name=StageName.FILTERING,
                    pmcid=pmcid,
                    config_hash=self._current_config_hash,
                    compute_fn=lambda: self._get_artifact_filter().filter_elements(
                        masked_layout.elements
                    ),
                    loader_fn=load_layout_elements,
                    dumper_fn=dump_layout_elements,
                    summarise_fn=lambda els: f"{len(els)} elements",
                )
            self._record(lambda s: s.set_count("after_filter", len(masked_layout.elements)))

        # ── Step 5b: Raw text dump (pre-assembly) ─────────────────────────────
        if self._cfg.text.write_raw_text:
            raw_path = self._cfg.paths.text_raw_dir / f"{pmcid}_raw.txt"
            self._cfg.paths.text_raw_dir.mkdir(parents=True, exist_ok=True)
            with raw_path.open("w", encoding="utf-8") as f:
                for el in masked_layout.elements:
                    text = (el.text or "").strip()
                    if text:
                        f.write(f"[{el.type}] {text}\n\n")
            logger.info("[%s] Raw text written to %s", pmcid, raw_path)

        # ── Step 6: Text assembly ─────────────────────────────────────────────
        # Reminder: bump STAGE_CACHE_VERSION[StageName.TEXT_ASSEMBLY] in
        # stage_cache.py if this stage's behaviour OR the on-disk format
        # changes (e.g. ContextAwareStitcher logic changes).
        logger.info("[%s] Step 6 — text assembly", pmcid)
        with self._stage("STEP6_TEXT_ASSEMBLY"):
            rows = cache.get_or_compute(
                stage_name=StageName.TEXT_ASSEMBLY,
                pmcid=pmcid,
                config_hash=self._current_config_hash,
                compute_fn=lambda: self._get_text_assembler().assemble(masked_layout),
                loader_fn=load_hierarchical_rows,
                dumper_fn=dump_hierarchical_rows,
                summarise_fn=lambda r: f"{len(r)} rows",
            )
        self._record(lambda s: s.set_count("text_rows", len(rows)))

        # ── Step 7: Media cropping ────────────────────────────────────────────
        logger.info("[%s] Step 7 — media cropping", pmcid)
        with self._stage("STEP7_MEDIA_CROPPING"):
            cropper = self._get_media_cropper()
            figures, tables = cropper.crop(
                pdf_path, layout, detection=detection,
                drop_tables_inside_figures=self._cfg.masking.drop_tables_inside_figures,
            )
        self._record(lambda s: s.set_count("figures_cropped", len(figures)))
        self._record(lambda s: s.set_count("tables_cropped", len(tables)))

        if self._cfg.runtime.multi_source_crops:
            from dataclasses import replace
            from pipeline.stages.pdf_text_extraction.outputs.media_json_writer import MediaJsonWriter
            from pipeline.stages.pdf_text_extraction.components.media_cropper import PyMuPDFMediaCropper
            json_dir    = self._cfg.paths.json_dir
            tables_root = self._cfg.paths.tables_dir.parent

            # Docling-only variant is a pure detection baseline — no expansion.
            # Figures are identical across all variants so disable figure cropping
            # here and reuse the figures already cropped in the main pass.
            cropping_no_expand = replace(self._cfg.cropping,
                                         expand_tables_with_footnotes=False,
                                         save_figure_crops=False)
            cropping_no_figures = replace(self._cfg.cropping, save_figure_crops=False)

            # Docling TABLE only (no reconstruction, no expansion)
            cropper_docling = PyMuPDFMediaCropper(
                config=cropping_no_expand,
                figures_dir=self._cfg.paths.figures_dir,
                tables_dir=tables_root / "docling",
            )
            _, tables_docling = cropper_docling.crop(
                pdf_path, layout_pre_recon, detection=None,
                docling_table_types=("TABLE",),
                drop_tables_inside_figures=self._cfg.masking.drop_tables_inside_figures,
            )
            MediaJsonWriter(json_dir.parent / "docling").write(
                pmcid, rows, figures, tables_docling,
            )

            # Docling TABLE + RECONSTRUCTED_TABLE (with expansion)
            cropper_docling_recon = PyMuPDFMediaCropper(
                config=cropping_no_figures,
                figures_dir=self._cfg.paths.figures_dir,
                tables_dir=tables_root / "docling_recon",
            )
            _, tables_docling_recon = cropper_docling_recon.crop(
                pdf_path, layout, detection=None,
                docling_table_types=("TABLE", "RECONSTRUCTED_TABLE"),
                drop_tables_inside_figures=self._cfg.masking.drop_tables_inside_figures,
            )
            MediaJsonWriter(json_dir.parent / "docling_recon").write(
                pmcid, rows, figures, tables_docling_recon,
            )
            logger.info("[%s] Multi-source crops written (docling / docling_recon / full)", pmcid)

        # ── Step 8: Outputs ───────────────────────────────────────────────────
        logger.info("[%s] Step 8 — writing outputs", pmcid)
        with self._stage("STEP8_OUTPUTS"):
            for output in self._get_outputs():
                output.write(pmcid, rows, figures, tables, pdf_path=pdf_path)

        return rows

    # ── Batch processing ──────────────────────────────────────────────────────

    def run_batch(
        self,
        pdf_dir: Path,
        glob: str = "*.pdf",
        pmcid_fn=None,
        max_docs: Optional[int] = None,
    ) -> dict:
        """
        Process all PDFs in ``pdf_dir``.

        Args:
            pdf_dir:  Directory containing PDF files.
            glob:     Filename glob pattern (default ``*.pdf``).
            pmcid_fn: Optional callable that maps a Path to a PMCID string.
                      Defaults to using the file stem (e.g. ``PMC123.pdf`` → ``PMC123``).

        Returns:
            Dict with keys ``processed``, ``failed``, ``skipped``.
        """
        self._cfg.prepare()

        pdfs: List[Path] = sorted(pdf_dir.glob(glob))
        if max_docs is not None:
            pdfs = pdfs[:max_docs]
        logger.info("Batch: processing %d PDFs in %s", len(pdfs), pdf_dir)

        # Observability manifest — best-effort.
        manifest: Optional[RunManifestWriter] = None
        try:
            run_id = make_run_id()
            manifest = RunManifestWriter(
                cfg=self._cfg,
                run_metadata_dir=self._cfg.paths.run_metadata_dir,
                run_id=run_id,
            )
            attempted = [
                (pmcid_fn(p) if pmcid_fn else p.stem) for p in pdfs
            ]
            manifest.set_input(
                runner="PipelineRunner.run_batch",
                pdf_dir=str(pdf_dir),
                glob=glob,
                max_docs=max_docs,
                n_paths=len(pdfs),
            )
            manifest.set_attempted(attempted)
            logger.info("PipelineRunner.run_batch: run_id=%s", run_id)
        except Exception:
            logger.exception("RunManifestWriter init failed — continuing without manifest")
            manifest = None
            run_id = None

        stats = {"processed": 0, "failed": 0, "skipped": 0}

        for pdf_path in pdfs:
            pmcid = pmcid_fn(pdf_path) if pmcid_fn else pdf_path.stem
            ok = self.run_document(pdf_path, pmcid, run_id=run_id)
            if ok:
                stats["processed"] += 1
            elif self._blacklist.contains(pmcid):
                stats["skipped"] += 1
            else:
                stats["failed"] += 1

        logger.info(
            "Batch complete: %d processed, %d failed, %d skipped",
            stats["processed"], stats["failed"], stats["skipped"],
        )

        if manifest is not None:
            try:
                manifest.write(batch_stats=dict(stats))
            except Exception:
                logger.exception("RunManifestWriter.write failed")

        return stats


def _retarget_paths(cfg, out_root: Path) -> None:
    """Repoint every PathConfig output directory under ``out_root``.

    Used by the ``--out-root`` CLI flag so a thesis sweep run lands in
    ``out/sweeps/<name>/`` instead of overwriting the canonical ``out/``.
    Path defaults are not modified — only the per-run instance.
    """
    p = cfg.paths
    p.output_root      = out_root
    p.masked_pdf_dir   = out_root / "masked_pdfs"
    p.docling_full_dir = out_root / "docling_full"
    p.docling_masked_dir = out_root / "docling_masked"
    p.text_dir         = out_root / "text"
    p.text_raw_dir     = out_root / "text_raw"
    p.json_dir         = out_root / "json"
    p.vis_dir          = out_root / "visualization"
    p.figures_dir      = out_root / "figures"
    p.tables_dir       = out_root / "tables"
    p.blacklist_file   = out_root / "failed_pdfs_blacklist.json"
    p.run_metadata_dir = out_root / "run_metadata"


def _parse_args(argv=None):
    """Optional argparse layer.  When called with no arguments the parsed
    namespace matches the canonical defaults below (preserving prior
    ``python runner.py`` behaviour byte-for-byte)."""
    import argparse
    from pipeline.stages.pdf_text_extraction.config import TableDetectorType

    p = argparse.ArgumentParser(
        description="PDF text-extraction pipeline (thesis-stabilization runner).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pdf-dir", type=Path, default=Path("files/organized_pdfs"),
                   help="Directory of PDFs to process.")
    p.add_argument("--glob", default="*.pdf", help="Filename glob.")
    p.add_argument("--max-docs", type=int, default=None,
                   help="Cap on number of PDFs.")
    p.add_argument("--workers", type=int, default=None,
                   help="Worker thread count (default: cfg.runtime.num_workers).")
    p.add_argument("--out-root", type=Path, default=None,
                   help="Override output root (e.g. out/sweeps/tatr090).  "
                        "Leaves the canonical out/ untouched.")
    p.add_argument("--detector",
                   choices=["tatr", "docling", "hybrid"],
                   default=None,
                   help="Table detector override.")
    p.add_argument("--tatr-threshold", type=float, default=None,
                   help="TATR detection confidence threshold (0–1).")
    p.add_argument("--render-dpi", type=int, default=None,
                   help="TATR / two-pass render DPI.")
    two_pass = p.add_mutually_exclusive_group()
    two_pass.add_argument("--two-pass", dest="two_pass", action="store_true",
                          default=None, help="Enable two-pass ghost-text detection.")
    two_pass.add_argument("--no-two-pass", dest="two_pass", action="store_false",
                          help="Disable two-pass ghost-text detection.")
    p.add_argument("--visualization", dest="visualization",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Write annotated audit PDFs to out/visualization/ "
                        "(`VisualizationConfig.enabled`). Default on; "
                        "`--no-visualization` skips them — saves disk + per-page "
                        "render time and is not needed for the ILP / silver / "
                        "cascade downstream.")
    p.add_argument("--db", dest="db_enabled", action=argparse.BooleanOptionalAction,
                   default=None, help="Enable / disable database ingestion.")
    p.add_argument("--write-raw-text", dest="write_raw_text",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Dump pre-assembly text to out/text_raw/.")
    p.add_argument("--skip-existing-in-db", dest="skip_existing_in_db",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Skip documents already present in the DB.")
    p.add_argument("--reconstruct-tables-from-lists",
                   dest="reconstruct_tables_from_lists",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Promote list-shaped layouts to RECONSTRUCTED_TABLE elements "
                        "(`DoclingConfig.reconstruct_tables_from_lists`).")
    p.add_argument("--merge-tables-by-caption",
                   dest="merge_tables_by_caption",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Union table regions sharing the same caption number "
                        "(`CroppingConfig.merge_tables_by_caption`).")
    p.add_argument("--expand-tables-with-footnotes",
                   dest="expand_tables_with_footnotes",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Extend table crops downward to absorb nearby footnotes "
                        "(`CroppingConfig.expand_tables_with_footnotes`).")
    p.add_argument("--drop-tables-inside-figures",
                   dest="drop_tables_inside_figures",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Drop table-detector regions whose bbox is ≥80%% inside "
                        "any FIGURE/PICTURE element (eliminates 'table inside "
                        "figure' false positives).  "
                        "(`MaskingConfig.drop_tables_inside_figures`).")
    p.add_argument("--footnote-proximity-pts", type=float, default=None,
                   dest="footnote_proximity_pts",
                   help="Max gap (pts) between table bottom and nearest "
                        "FOOTNOTE/LIST_ITEM to absorb when "
                        "--expand-tables-with-footnotes is on.  "
                        "(`CroppingConfig.footnote_proximity_pts`, default 20.0)")
    p.add_argument("--text-footnote-proximity-pts", type=float, default=None,
                   dest="text_footnote_proximity_pts",
                   help="Tighter max gap (pts) for TEXT-typed elements when "
                        "expanding tables with footnotes.  "
                        "(`CroppingConfig.text_footnote_proximity_pts`, default 8.0)")
    p.add_argument("--footnote-threshold-multiplier", type=float, default=None,
                   dest="footnote_threshold_multiplier",
                   help="After first footnote absorption, max gap becomes "
                        "first_gap * this multiplier.  Higher values absorb "
                        "more cascading elements.  "
                        "(`CroppingConfig.footnote_threshold_multiplier`, default 1.2)")
    p.add_argument("--max-pages", type=int, default=None,
                   help="Skip PDFs with more than N pages (tractability cap). "
                        "Counts pages with PyMuPDF before processing.")
    p.add_argument("--min-pages", type=int, default=None,
                   help="Skip PDFs with fewer than N pages.")
    p.add_argument("--sort-by-pages", action=argparse.BooleanOptionalAction, default=True,
                   help="Process PDFs shortest-first (ascending page count). Default "
                        "ON; pass --no-sort-by-pages for filesystem order. Quick wins "
                        "early and lower peak memory; unreadable PDFs sort last. Being "
                        "ON triggers a one-pass PyMuPDF page count before extraction.")
    p.add_argument("--main-pdf-only", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep only single-PDF PMC packages, dropping ENTIRE papers "
                        "that shipped supplementary PDFs (any PMC id with >1 PDF — "
                        "main and supplements both dropped). Default ON; pass "
                        "--no-main-pdf-only to process every PDF. Prevents B-067 and "
                        "split-across-files (partial-content) papers.")
    return p.parse_args(argv), TableDetectorType


def main(argv=None) -> None:
    from pipeline.stages.pdf_text_extraction.config import PipelineConfig

    args, TableDetectorType = _parse_args(argv)

    # ── Canonical defaults (preserved from prior behaviour) ────────────────────
    cfg = PipelineConfig()
    cfg.database.enabled = True if args.db_enabled is None else args.db_enabled
    cfg.text.write_raw_text = True if args.write_raw_text is None else args.write_raw_text
    cfg.two_pass.enabled = True if args.two_pass is None else args.two_pass
    cfg.runtime.skip_existing_in_db = (
        True if args.skip_existing_in_db is None else args.skip_existing_in_db
    )

    # ── Sweep overrides (no-op when flag omitted) ──────────────────────────────
    if args.detector is not None:
        cfg.table_detector = {
            "tatr":    TableDetectorType.TATR,
            "docling": TableDetectorType.DOCLING,
            "hybrid":  TableDetectorType.HYBRID,
        }[args.detector]
    if args.tatr_threshold is not None:
        cfg.tatr.threshold = args.tatr_threshold
    if args.render_dpi is not None:
        cfg.tatr.render_dpi = args.render_dpi
        cfg.two_pass.render_dpi = args.render_dpi
    if args.workers is not None:
        cfg.runtime.num_workers = args.workers
    if args.visualization is not None:
        cfg.visualization.enabled = args.visualization
    if args.reconstruct_tables_from_lists is not None:
        cfg.docling.reconstruct_tables_from_lists = args.reconstruct_tables_from_lists
    if args.merge_tables_by_caption is not None:
        cfg.cropping.merge_tables_by_caption = args.merge_tables_by_caption
    if args.expand_tables_with_footnotes is not None:
        cfg.cropping.expand_tables_with_footnotes = args.expand_tables_with_footnotes
    if args.drop_tables_inside_figures is not None:
        cfg.masking.drop_tables_inside_figures = args.drop_tables_inside_figures
    if args.footnote_proximity_pts is not None:
        cfg.cropping.footnote_proximity_pts = args.footnote_proximity_pts
    if args.text_footnote_proximity_pts is not None:
        cfg.cropping.text_footnote_proximity_pts = args.text_footnote_proximity_pts
    if args.footnote_threshold_multiplier is not None:
        cfg.cropping.footnote_threshold_multiplier = args.footnote_threshold_multiplier
    if args.out_root is not None:
        _retarget_paths(cfg, args.out_root)

    cfg.prepare()

    logging.basicConfig(
        level=cfg.runtime.log_level.value,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    from pipeline.stages.pdf_text_extraction.batch import ParallelBatchRunner

    pdfs = sorted(args.pdf_dir.glob(args.glob))

    # ── Single-PDF papers only: drop entire packages that shipped supplements ───
    if args.main_pdf_only:
        from collections import defaultdict as _dd
        _by = _dd(list)
        for _p in pdfs:
            _by[_p.stem.split("_", 1)[0]].append(_p)
        pdfs = sorted(g[0] for g in _by.values() if len(g) == 1)
        _dropped = sum(1 for g in _by.values() if len(g) > 1)
        logger.info(
            "--main-pdf-only: %d single-PDF papers kept; %d multi-PDF (supplemented) "
            "packages dropped entirely", len(pdfs), _dropped,
        )

    # ── Page counting (shared by the page filter and the shortest-first sort) ──
    if args.max_pages is not None or args.min_pages is not None or args.sort_by_pages:
        import fitz  # type: ignore  # optional dep — import inside the function

        def _page_count(_path) -> int:
            try:
                with fitz.open(str(_path)) as _doc:
                    return _doc.page_count
            except Exception:  # noqa: BLE001 — unreadable PDF treated as +inf below
                return -1

        _pages = {_p: _page_count(_p) for _p in pdfs}   # one PyMuPDF pass

        if args.max_pages is not None or args.min_pages is not None:
            _lo = args.min_pages if args.min_pages is not None else 0
            _hi = args.max_pages if args.max_pages is not None else 10 ** 9
            _kept = [_p for _p in pdfs if _lo <= _pages[_p] <= _hi]
            logger.info("page filter [%s, %s]: kept %d / %d PDFs",
                        _lo, _hi, len(_kept), len(pdfs))
            pdfs = _kept

        if args.sort_by_pages:
            pdfs = sorted(pdfs, key=lambda _p: _pages[_p] if _pages[_p] >= 0 else 10 ** 9)
            logger.info("sorted %d PDFs by page count (shortest first)", len(pdfs))

    if args.max_docs is not None:
        pdfs = pdfs[: args.max_docs]

    ParallelBatchRunner(cfg, max_workers=cfg.runtime.num_workers).run_paths(pdfs)


if __name__ == "__main__":
    main()
