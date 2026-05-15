"""
TATRTableDetector

Uses the Table Transformer (TATR) model from Microsoft to detect table
bounding boxes in each page of a PDF.  Model is loaded lazily on first use.

Reference model: microsoft/table-transformer-detection
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from pipeline.stages.pdf_text_extraction.config import TATRConfig
from pipeline.stages.pdf_text_extraction.models.dto import BoundingBox, DetectedRegion, TableDetectionResult

logger = logging.getLogger(__name__)

# The TATR model is loaded once per process and shared across all instances.
# from_pretrained is not thread-safe when called concurrently on the same
# cached weights, so we serialise with a lock and reuse the result globally.
_LOAD_LOCK = threading.Lock()
_SHARED_PROCESSOR = None
_SHARED_MODEL = None

# Points-per-inch for PDF coordinates, and the DPI we render pages at for TATR
_PDF_PPI = 72
_RENDER_DPI = 150
_SCALE = _PDF_PPI / _RENDER_DPI   # pixel → PDF points


class TATRTableDetector:
    """
    Table detector backed by the TATR object-detection model.

    The model and processor are loaded once and cached on the instance.
    """

    def __init__(self, config: Optional[TATRConfig] = None) -> None:
        self._config = config or TATRConfig()
        self._processor = None
        self._model = None

    # ── Lazy model loading ────────────────────────────────────────────────────

    def _load_model(self) -> None:
        global _SHARED_PROCESSOR, _SHARED_MODEL

        # Fast path — model already loaded by this or another instance
        if _SHARED_MODEL is not None:
            self._processor = _SHARED_PROCESSOR
            self._model = _SHARED_MODEL
            return

        with _LOAD_LOCK:
            # Another thread may have loaded it while we waited for the lock
            if _SHARED_MODEL is not None:
                self._processor = _SHARED_PROCESSOR
                self._model = _SHARED_MODEL
                return

            from transformers import AutoImageProcessor, AutoModelForObjectDetection  # type: ignore

            logger.info("Loading TATR model (%s)…", self._config.model_name)
            _SHARED_PROCESSOR = AutoImageProcessor.from_pretrained(self._config.model_name)
            device = self._config.device
            _SHARED_MODEL = AutoModelForObjectDetection.from_pretrained(self._config.model_name)
            _SHARED_MODEL = _SHARED_MODEL.to(device)
            _SHARED_MODEL.eval()
            logger.info("TATR model loaded.")

            self._processor = _SHARED_PROCESSOR
            self._model = _SHARED_MODEL

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, pdf_path: Path) -> TableDetectionResult:
        """
        Run TATR table detection on every page of ``pdf_path``.

        Args:
            pdf_path: Path to the PDF to analyse.

        Returns:
            TableDetectionResult with one DetectedRegion per detected table.
        """
        import fitz          # type: ignore  (PyMuPDF)
        import torch         # type: ignore
        from PIL import Image as PILImage  # type: ignore

        # Limit PyTorch's intra-op threads to 1 so multiple worker threads
        # don't all compete for the full CPU core count simultaneously.
        torch.set_num_threads(1)

        self._load_model()

        doc = fitz.open(str(pdf_path))
        regions = []
        page_dims: dict = {}

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_no = page_num + 1
            page_dims[page_no] = {"width": page.rect.width, "height": page.rect.height}

            mat = fitz.Matrix(_RENDER_DPI / _PDF_PPI, _RENDER_DPI / _PDF_PPI)
            pix = page.get_pixmap(matrix=mat)
            img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)

            inputs = self._processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)

            results = self._processor.post_process_object_detection(
                outputs,
                threshold=self._config.threshold,
                target_sizes=[(img.height, img.width)],
            )[0]

            page_h = page.rect.height
            for score, label_id, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                x1, y1_screen, x2, y2_screen = [v * _SCALE for v in box.tolist()]
                # Convert from screen coords to Docling PDF coords
                bbox = BoundingBox(
                    x1=x1,
                    y1=page_h - y1_screen,
                    x2=x2,
                    y2=page_h - y2_screen,
                    page=page_no,
                )
                label = self._model.config.id2label[label_id.item()]
                regions.append(
                    DetectedRegion(
                        bbox=bbox,
                        score=round(score.item(), 3),
                        source="tatr",
                        label=label,
                    )
                )

        doc.close()
        return TableDetectionResult(
            regions=regions,
            source="tatr",
            pdf_path=pdf_path,
            page_dims=page_dims,
        )
