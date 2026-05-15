"""
pipeline — modular PDF-processing pipeline for nlp-histo.

Stages
------
1. Layout extraction   — DoclingLayoutExtractor
2. Table detection     — DoclingTableDetector | TATRTableDetector | HybridTableDetector
3. Region masking      — PyMuPDFRegionMasker
4. Re-extraction       — DoclingLayoutExtractor (on masked PDF)
5. Artifact filtering  — ArtifactFilter
6. Text assembly       — HierarchicalTextAssembler
7. Media cropping      — PyMuPDFMediaCropper
8. Output              — TextFileWriter | PostgresDatabaseIngester | DetectionVisualizer

Quick start
-----------
    from pipeline.stages.pdf_text_extraction import PipelineRunner, PipelineConfig

    runner = PipelineRunner(PipelineConfig())
    runner.run_document(pdf_path=Path("files/organized_pdfs/PMC123.pdf"), pmcid="PMC123")
"""
from pipeline.stages.pdf_text_extraction.config import (
    PipelineConfig,
    PathConfig,
    DoclingConfig,
    TATRConfig,
    MaskingConfig,
    FilteringConfig,
    CroppingConfig,
    TextAssemblyConfig,
    VisualizationConfig,
    DatabaseConfig,
    RuntimeConfig,
    TableDetectorType,
)
from pipeline.stages.pdf_text_extraction.runner import PipelineRunner
from pipeline.stages.pdf_text_extraction.batch import ParallelBatchRunner
from pipeline.stages.pdf_text_extraction.blacklist import BlacklistManager

__all__ = [
    # Config
    "PipelineConfig",
    "PathConfig",
    "DoclingConfig",
    "TATRConfig",
    "MaskingConfig",
    "FilteringConfig",
    "CroppingConfig",
    "TextAssemblyConfig",
    "VisualizationConfig",
    "DatabaseConfig",
    "RuntimeConfig",
    "TableDetectorType",
    # Runtime
    "PipelineRunner",
    "ParallelBatchRunner",
    "BlacklistManager",
]
