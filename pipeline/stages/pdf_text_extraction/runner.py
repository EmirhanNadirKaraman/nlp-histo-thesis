"""
PipelineRunner

Orchestrates the full PDF-processing pipeline for a single document:

  1. Extract layout from original PDF (Docling)
  2. Detect tables (Docling | TATR | Hybrid — configurable)
  3. Mask detected regions with white rectangles
  4. Re-extract layout from masked PDF (Docling)
  5. Filter layout artifacts
  6. Assemble hierarchical text
  7. Crop figure / table images
  8. Write outputs (text file, database, visualizations)

Usage::

    from pathlib import Path
    from pipeline.stages.pdf_text_extraction import PipelineConfig, PipelineRunner

    cfg    = PipelineConfig()
    cfg.database.enabled = True
    cfg.database.db_url  = "postgresql://user:pw@localhost/nlp_histo"

    runner = PipelineRunner(cfg)
    runner.run_document(Path("files/organized_pdfs/PMC123.pdf"), pmcid="PMC123")

    # Batch processing
    runner.run_batch(pdf_dir=Path("files/organized_pdfs"))
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path as _Path

# Allow running as `python pipeline/stages/pdf_text_extraction/runner.py` from the project root
sys.path.insert(0, str(_Path(__file__).parent.parent.parent.parent))
import traceback
from pathlib import Path
from typing import List, Optional

from pipeline.stages.pdf_text_extraction.blacklist import BlacklistManager
from pipeline.stages.pdf_text_extraction.config import PipelineConfig, TableDetectorType
from pipeline.stages.pdf_text_extraction.models.dto import LayoutResult

logger = logging.getLogger(__name__)


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
            # in summarization/umls_resources rather than calling spacy.load
            # directly. Prevents double-loading when this pipeline runs in the
            # same process as the summarisation pipeline.
            try:
                from pipeline.stages.summarization.umls_resources import get_small_nlp
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

    def run_document(self, pdf_path: Path, pmcid: str) -> bool:
        """
        Process a single PDF document end-to-end.

        Args:
            pdf_path: Path to the source PDF.
            pmcid:    PubMed Central ID for the document.

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

        try:
            result = self._process(pdf_path, pmcid)
            logger.info("✅ %s — done (%d rows)", pmcid, len(result))
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
            if self._cfg.runtime.update_blacklist_on_failure:
                self._blacklist.add(pmcid, reason=str(exc))
            if self._cfg.runtime.fail_fast:
                raise
            return False

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

    def _run_table_detection(self, layout: LayoutResult, pdf_path: Path):
        """Run the configured table detector and return a TableDetectionResult."""
        detector = self._get_table_detector()
        from pipeline.stages.pdf_text_extraction.table_detectors.hybrid_detector import HybridTableDetector
        from pipeline.stages.pdf_text_extraction.table_detectors.docling_detector import DoclingTableDetector
        if isinstance(detector, HybridTableDetector):
            return detector.detect_with_layout(layout, pdf_path)
        elif isinstance(detector, DoclingTableDetector):
            return detector.detect_from_layout(layout)
        else:
            return detector.detect(pdf_path)

    def _steps_1_3_4_standard(self, pdf_path: Path, pmcid: str):
        """
        Standard Steps 1–4: extract → detect tables → mask → re-extract.

        Returns (layout, layout_pre_recon, masked_layout, detection).
        Detection runs here (before masking) because the region masker needs it.
        """
        from pipeline.stages.pdf_text_extraction.components.table_reconstructor import reconstruct_tables_from_lists

        logger.info("[%s] Step 1 — layout extraction", pmcid)
        layout: LayoutResult = self._get_layout_extractor().extract(pdf_path)
        layout_pre_recon = layout

        if self._cfg.docling.reconstruct_tables_from_lists:
            layout = reconstruct_tables_from_lists(layout)

        logger.info("[%s] Step 2 — table detection (%s)", pmcid, self._cfg.table_detector)
        detection = self._run_table_detection(layout, pdf_path)

        masker = self._get_region_masker()
        regions_to_mask = masker.collect_regions(
            detection, layout, nlp=self._get_nlp()
        ) if self._cfg.masking.enabled else []
        if regions_to_mask:
            logger.info("[%s] Step 3 — masking %d regions", pmcid, len(regions_to_mask))
            masked_path = masker.mask(pdf_path, regions_to_mask)
        else:
            masked_path = pdf_path

        logger.info("[%s] Step 4 — re-extraction from masked PDF", pmcid)
        masked_layout: LayoutResult = self._get_masked_extractor().extract(masked_path)
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
        tp = self._get_two_pass_extractor().process(pdf_path)

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
        if self._cfg.two_pass.enabled:
            layout, layout_pre_recon, masked_layout, detection = self._steps_1_3_4_two_pass(pdf_path, pmcid)
            # Step 2: table detection deferred to here in two-pass mode
            logger.info("[%s] Step 2 — table detection (%s)", pmcid, self._cfg.table_detector)
            detection = self._run_table_detection(layout, pdf_path)
        else:
            layout, layout_pre_recon, masked_layout, detection = self._steps_1_3_4_standard(pdf_path, pmcid)

        # ── Step 2b: Visualization ────────────────────────────────────────────
        if vis := self._get_visualizer():
            vis.visualize_layout(layout, pmcid)
            vis.visualize_detections(detection, layout, pmcid)

        # ── Step 5: Artifact filtering ────────────────────────────────────────
        if self._cfg.filtering.enabled:
            logger.info("[%s] Step 5 — artifact filtering", pmcid)
            masked_layout.elements = self._get_artifact_filter().filter_elements(
                masked_layout.elements
            )

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
        logger.info("[%s] Step 6 — text assembly", pmcid)
        rows = self._get_text_assembler().assemble(masked_layout)

        # ── Step 7: Media cropping ────────────────────────────────────────────
        logger.info("[%s] Step 7 — media cropping", pmcid)
        cropper = self._get_media_cropper()
        figures, tables = cropper.crop(pdf_path, layout, detection=detection)

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
            )
            MediaJsonWriter(json_dir.parent / "docling_recon").write(
                pmcid, rows, figures, tables_docling_recon,
            )
            logger.info("[%s] Multi-source crops written (docling / docling_recon / full)", pmcid)

        # ── Step 8: Outputs ───────────────────────────────────────────────────
        logger.info("[%s] Step 8 — writing outputs", pmcid)
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

        stats = {"processed": 0, "failed": 0, "skipped": 0}

        for pdf_path in pdfs:
            pmcid = pmcid_fn(pdf_path) if pmcid_fn else pdf_path.stem
            ok = self.run_document(pdf_path, pmcid)
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
        return stats


def main() -> None:
    from pipeline.stages.pdf_text_extraction.config import PipelineConfig

    # ── Configuration ──────────────────────────────────────────────────────────
    # Edit pipeline/stages/pdf_text_extraction/config.py to change defaults
    cfg = PipelineConfig()
    cfg.database.enabled = True  # set to True to ingest; db_url auto-loaded from .env
    cfg.text.write_raw_text = True
    cfg.two_pass.enabled = True  # use two-pass ghost-text detection instead of standard masking
    cfg.runtime.skip_existing_in_db = True
    cfg.prepare()

    logging.basicConfig(level=cfg.runtime.log_level.value, format="%(asctime)s [%(levelname)s] %(message)s")

    # ── Single document ────────────────────────────────────────────────────────
    # PipelineRunner(cfg).run_document(
    #     pdf_path=Path("files/organized_pdfs/PMC10047158_dermatopathology-10-00017.pdf"),
    #     pmcid="PMC10047158",
    # )

    # PipelineRunner(cfg).run_document(
    #     pdf_path=Path("files/organized_pdfs/PMC10047213_dermatopathology-10-00018.pdf"),
    #     pmcid="PMC10047213",
    # )

    # ── Sequential batch ───────────────────────────────────────────────────────
    # PipelineRunner(cfg).run_batch(pdf_dir=Path("files/organized_pdfs"), max_docs=5)

    # ── Parallel batch (use pipeline/stages/pdf_text_extraction/batch.py instead) ────────
    from pipeline.stages.pdf_text_extraction.batch import ParallelBatchRunner
    ParallelBatchRunner(cfg, max_workers=cfg.runtime.num_workers).run(
        pdf_dir=Path("files/organized_pdfs"),
    )


if __name__ == "__main__":
    main()
