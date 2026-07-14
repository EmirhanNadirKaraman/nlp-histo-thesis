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
    from nlp_histo.pipeline.stages.pdf_text_extraction import PipelineRunner, PipelineConfig

    cfg = PipelineConfig()
    cfg.prepare()                         # validate + create output dirs
    runner = PipelineRunner(cfg)
    runner.run_document(Path("files/organized_pdfs/PMC123.pdf"), pmcid="PMC123")

    # Or batch-process a directory with a thread pool:
    from nlp_histo.pipeline.stages.pdf_text_extraction import ParallelBatchRunner
    ParallelBatchRunner(cfg, max_workers=4).run(pdf_dir=Path("files/organized_pdfs"))
"""
from typing import TYPE_CHECKING

from nlp_histo.pipeline.stages.pdf_text_extraction.config import (
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
from nlp_histo.pipeline.stages.pdf_text_extraction.blacklist import BlacklistManager

# `PipelineRunner` and `ParallelBatchRunner` are resolved lazily. Importing them here
# would make `python -m pipeline.stages.pdf_text_extraction.runner` re-execute an
# already-imported module (runpy RuntimeWarning), and would pull `.runner` / `.batch`
# into every consumer that only wants the config dataclasses. The package-root names
# keep working — `from pipeline.stages.pdf_text_extraction import PipelineRunner`
# resolves through `__getattr__` (PEP 562) to the exact same class object.
if TYPE_CHECKING:  # static analysers still see the real symbols
    from nlp_histo.pipeline.stages.pdf_text_extraction.batch import ParallelBatchRunner
    from nlp_histo.pipeline.stages.pdf_text_extraction.runner import PipelineRunner

_LAZY_EXPORTS = {
    "PipelineRunner": "nlp_histo.pipeline.stages.pdf_text_extraction.runner",
    "ParallelBatchRunner": "nlp_histo.pipeline.stages.pdf_text_extraction.batch",
}


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_path), name)
    globals()[name] = value          # cache: subsequent lookups skip __getattr__
    return value


def __dir__():
    return sorted(__all__)


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
