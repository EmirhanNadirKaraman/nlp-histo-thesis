"""Table detector implementations."""
from nlp_histo.pipeline.stages.pdf_text_extraction.table_detectors.docling_detector import DoclingTableDetector
from nlp_histo.pipeline.stages.pdf_text_extraction.table_detectors.tatr_detector import TATRTableDetector
from nlp_histo.pipeline.stages.pdf_text_extraction.table_detectors.hybrid_detector import HybridTableDetector

__all__ = ["DoclingTableDetector", "TATRTableDetector", "HybridTableDetector"]
