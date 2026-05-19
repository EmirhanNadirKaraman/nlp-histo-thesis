"""
pipeline.stages.pdf_text_extraction — modular PDF text-extraction pipeline.

`PipelineRunner.run_document` drives the canonical 8-step flow for a single
document:

1. Layout extraction   — DoclingLayoutExtractor
2. Table detection     — DoclingTableDetector | TATRTableDetector | HybridTableDetector
3. Region masking      — PyMuPDFRegionMasker (detected table/figure regions whited out)
4. Re-extraction       — DoclingLayoutExtractor on the masked PDF
5. Artifact filtering  — ArtifactFilter (delegates to parsers.layout_utils.filter_artifacts)
6. Text assembly       — HierarchicalTextAssembler
7. Media cropping      — PyMuPDFMediaCropper
8. Output              — TextFileWriter, MediaJsonWriter, optional PostgresDatabaseIngester

Two-pass mode (`PipelineConfig.two_pass.enabled=True`, default) replaces steps
1/3/4 with a single TwoPassTextExtractor call that uses pixel-rendering
evidence to filter out invisible / ghost-text Docling elements before
producing the cleaned layout.  Step 2 (table detection) then runs against
that layout.

Stages 2, 5, and 6 are individually cached on disk under
`<output_root>/stage_cache/<stage_name>/<pmcid>.json` with a config-hash
sidecar (see `stage_cache.py`); cache hits skip recomputation when
`RuntimeConfig.skip_existing_outputs=True`.

`ParallelBatchRunner` wraps `PipelineRunner` with a thread pool for batch
processing.  `BlacklistManager` persists failed PMCIDs to disk so a
subsequent run can skip them.

Quick start
-----------
    from pathlib import Path
    from pipeline.stages.pdf_text_extraction import PipelineRunner, PipelineConfig

    cfg = PipelineConfig()
    cfg.prepare()                         # validate + create output dirs
    runner = PipelineRunner(cfg)
    runner.run_document(Path("files/organized_pdfs/PMC123.pdf"), pmcid="PMC123")

    # Or batch-process a directory with a thread pool:
    from pipeline.stages.pdf_text_extraction import ParallelBatchRunner
    ParallelBatchRunner(cfg, max_workers=4).run(pdf_dir=Path("files/organized_pdfs"))
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
