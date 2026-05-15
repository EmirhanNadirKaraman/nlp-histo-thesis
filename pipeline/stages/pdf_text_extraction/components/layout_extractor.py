"""
DoclingLayoutExtractor

Wraps the Docling DocumentConverter to produce a typed LayoutResult.
Model is loaded lazily on first use.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from pipeline.stages.pdf_text_extraction.config import DoclingConfig
from pipeline.stages.pdf_text_extraction.models.dto import BoundingBox, LayoutElement, LayoutResult
from parsers.layout_utils import CAPTION_PATTERN, _deduplicate_caption

logger = logging.getLogger(__name__)


def _deduplicate_text(text: str) -> str:
    """Remove exact repetitions produced by ghost text layers.

    Delegates to ``_deduplicate_caption`` from layout_utils, which handles
    2–5× repetitions with or without a space separator (character-level).
    Logs when a change is made.
    """
    deduped = _deduplicate_caption(text)
    if deduped != text:
        logger.debug("Deduplicated repeated text: %.60r → %.60r", text, deduped)
    return deduped


class DoclingLayoutExtractor:
    """
    Layout extractor backed by Docling.

    Parameters
    ----------
    config:
        DoclingConfig controlling OCR, table structure, etc.
    cache_dir:
        If supplied, extracted layouts are cached as JSON here and reloaded
        on subsequent calls for the same PDF (keyed by stem).
    """

    def __init__(
        self,
        config: Optional[DoclingConfig] = None,
        cache_dir: Optional[Path] = None,
        max_caption_chars_per_pt: float = 0.0,
    ) -> None:
        self._config = config or DoclingConfig()
        self._cache_dir = cache_dir
        self._converter = None
        # When > 0, TEXT elements that match CAPTION_PATTERN but are denser
        # than this threshold (chars / bbox_height in pts) are NOT reclassified.
        # Hidden text layers often start with "Figure N." but span a full
        # paragraph squeezed into a ~10pt bbox — catching them here keeps them
        # as TEXT so NodeScorer R3 can reject and mask them.
        self._max_caption_chars_per_pt = max_caption_chars_per_pt

    # ── Lazy model loading ────────────────────────────────────────────────────

    def _get_converter(self):
        if self._converter is not None:
            return self._converter
        from docling.document_converter import DocumentConverter, PdfFormatOption  # type: ignore
        from docling.datamodel.pipeline_options import PdfPipelineOptions          # type: ignore
        from docling.datamodel.base_models import InputFormat                       # type: ignore
        from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice  # type: ignore

        opts = PdfPipelineOptions()
        opts.do_table_structure = self._config.do_table_structure
        opts.do_ocr = self._config.do_ocr
        opts.images_scale = self._config.images_scale

        if self._config.do_ocr:
            if hasattr(opts, "force_full_page_ocr"):
                opts.force_full_page_ocr = self._config.force_full_page_ocr
            elif self._config.force_full_page_ocr:
                logger.warning("This version of Docling does not support force_full_page_ocr — ignored")
            self._apply_ocr_engine(opts)

        device_map = {
            "cpu":  AcceleratorDevice.CPU,
            "cuda": AcceleratorDevice.CUDA,
            "mps":  AcceleratorDevice.MPS,
        }
        device = device_map.get(self._config.accelerator_device.lower(), AcceleratorDevice.CPU)
        opts.accelerator_options = AcceleratorOptions(device=device)

        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        return self._converter

    def _apply_ocr_engine(self, opts) -> None:
        """Configure the OCR engine on ``opts`` based on ``self._config.ocr_engine``."""
        from pipeline.stages.pdf_text_extraction.config import OcrEngine
        engine = self._config.ocr_engine
        try:
            if engine == OcrEngine.EASYOCR:
                from docling.datamodel.pipeline_options import EasyOcrOptions  # type: ignore
                opts.ocr_options = EasyOcrOptions()
            elif engine == OcrEngine.TESSERACT:
                from docling.datamodel.pipeline_options import TesseractCliOcrOptions  # type: ignore
                opts.ocr_options = TesseractCliOcrOptions()
            elif engine == OcrEngine.RAPIDOCR:
                from docling.datamodel.pipeline_options import RapidOcrOptions  # type: ignore
                opts.ocr_options = RapidOcrOptions()
        except ImportError as exc:
            logger.warning("OCR engine '%s' not available, falling back to default: %s", engine, exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, pdf_path: Path) -> LayoutResult:
        """
        Extract full layout from a PDF.

        Checks the cache directory first; falls back to running Docling and
        writes the result to cache if a cache_dir was configured.

        Args:
            pdf_path: Path to the input PDF.

        Returns:
            LayoutResult with all elements and page metadata.
        """
        cache_path = self._cache_path(pdf_path)
        if cache_path and cache_path.exists():
            return self._load_from_cache(cache_path, pdf_path)

        result = self._run_docling(pdf_path)

        if cache_path:
            self._save_to_cache(result, cache_path)

        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _convert_with_timeout(self, converter, pdf_path: Path):
        """Run ``converter.convert()`` under ``DoclingConfig.timeout_sec``.

        Pathological PDFs (very large / OCR-heavy / corrupt) can hang the
        Docling pipeline indefinitely. We isolate the call in a single-worker
        thread pool and raise TimeoutError on the configured deadline — the
        runner's per-paper try/except blacklists the pmcid and the batch
        moves on. The runaway thread is abandoned; this is acceptable for
        short batches and far better than a hung run.

        Setting ``timeout_sec <= 0`` disables the guard.
        """
        timeout_sec = self._config.timeout_sec
        if timeout_sec is None or timeout_sec <= 0:
            return converter.convert(str(pdf_path))

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"docling-{pdf_path.stem}",
        ) as ex:
            future = ex.submit(converter.convert, str(pdf_path))
            try:
                return future.result(timeout=timeout_sec)
            except concurrent.futures.TimeoutError as exc:
                logger.error(
                    "Docling timed out on %s after %ds — abandoning worker thread",
                    pdf_path.name, timeout_sec,
                )
                raise TimeoutError(
                    f"Docling exceeded {timeout_sec}s on {pdf_path.name}"
                ) from exc

    def _run_docling(self, pdf_path: Path) -> LayoutResult:
        logger.info("Running Docling on %s", pdf_path.name)
        converter = self._get_converter()
        doc_result = self._convert_with_timeout(converter, pdf_path)
        doc = doc_result.document

        raw_elements = []
        for element, level in doc.iterate_items():
            label = str(getattr(element, "label", "UNKNOWN")).split(".")[-1].upper()
            if not (hasattr(element, "prov") and element.prov):
                continue
            prov = element.prov[0]
            bbox_raw = prov.bbox
            text = ""
            if hasattr(element, "text"):
                text = element.text or ""
            elif hasattr(element, "caption") and element.caption:
                text = element.caption.text or ""
            text = _deduplicate_text(text.strip())
            raw_elements.append({
                "type":  label,
                "page":  prov.page_no,
                "level": level,
                "bbox":  {"x1": bbox_raw.l, "y1": bbox_raw.t,
                          "x2": bbox_raw.r, "y2": bbox_raw.b},
                "text":  text or None,
            })

        # Reclassify TEXT elements that look like captions, unless they are
        # suspiciously dense (likely a hidden text layer that starts with
        # "Figure N." or "Table N." but contains a full paragraph of content).
        reclassified = 0
        skipped_dense = 0
        for el in raw_elements:
            if el["type"] != "TEXT":
                continue
            text = el.get("text") or ""
            if not CAPTION_PATTERN.match(text):
                continue
            if self._max_caption_chars_per_pt > 0:
                bbox   = el.get("bbox", {})
                bbox_h = abs(bbox.get("y1", 0.0) - bbox.get("y2", 0.0))
                if bbox_h > 0 and len(text) / bbox_h > self._max_caption_chars_per_pt:
                    skipped_dense += 1
                    logger.debug(
                        "  Skipped dense TEXT→CAPTION reclassify: "
                        "%d chars / %.1fpt = %.1f chars/pt  text=%.60r",
                        len(text), bbox_h, len(text) / bbox_h, text,
                    )
                    continue
            el["type"] = "CAPTION"
            reclassified += 1
        if reclassified or skipped_dense:
            logger.debug(
                "  Reclassified %d TEXT → CAPTION (%d skipped — too dense)",
                reclassified, skipped_dense,
            )

        page_dims = {
            no: {"width": p.size.width, "height": p.size.height}
            for no, p in doc.pages.items()
        }

        elements = [
            LayoutElement(
                type=e["type"],
                page=e["page"],
                bbox=BoundingBox.from_dict(e["bbox"], e["page"]),
                text=e["text"],
                level=e["level"],
            )
            for e in raw_elements
        ]

        return LayoutResult(
            elements=elements,
            page_dims=page_dims,
            pdf_path=pdf_path,
            source="docling",
        )

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_path(self, pdf_path: Path) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        key = self._config.content_key()
        return self._cache_dir / f"{pdf_path.stem}_{key}_layout.json"

    def _save_to_cache(self, result: LayoutResult, path: Path) -> None:
        data = {
            "source":    result.source,
            "page_dims": {str(k): v for k, v in result.page_dims.items()},
            "elements":  [el.to_dict() for el in result.elements],
        }
        path.write_text(json.dumps(data, indent=2))

    def _load_from_cache(self, cache_path: Path, pdf_path: Path) -> LayoutResult:
        logger.debug("Loading cached layout from %s", cache_path)
        data = json.loads(cache_path.read_text())
        page_dims = {int(k): v for k, v in data["page_dims"].items()}
        elements = [LayoutElement.from_dict(e) for e in data["elements"]]
        return LayoutResult(
            elements=elements,
            page_dims=page_dims,
            pdf_path=pdf_path,
            source=data.get("source", "docling"),
        )
