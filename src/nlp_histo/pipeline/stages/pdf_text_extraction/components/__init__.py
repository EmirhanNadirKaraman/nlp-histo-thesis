"""Concrete pipeline stage implementations."""
from nlp_histo.pipeline.stages.pdf_text_extraction.components.layout_extractor import DoclingLayoutExtractor
from nlp_histo.pipeline.stages.pdf_text_extraction.components.region_masker import PyMuPDFRegionMasker
from nlp_histo.pipeline.stages.pdf_text_extraction.components.text_assembler import HierarchicalTextAssembler
from nlp_histo.pipeline.stages.pdf_text_extraction.components.artifact_filter import ArtifactFilter
from nlp_histo.pipeline.stages.pdf_text_extraction.components.media_cropper import PyMuPDFMediaCropper
from nlp_histo.pipeline.stages.pdf_text_extraction.components.visualizer import DetectionVisualizer
from nlp_histo.pipeline.stages.pdf_text_extraction.table_detectors import (
    DoclingTableDetector,
    TATRTableDetector,
    HybridTableDetector,
)

__all__ = [
    "DoclingLayoutExtractor",
    "PyMuPDFRegionMasker",
    "HierarchicalTextAssembler",
    "ArtifactFilter",
    "PyMuPDFMediaCropper",
    "DetectionVisualizer",
    "DoclingTableDetector",
    "TATRTableDetector",
    "HybridTableDetector",
]
