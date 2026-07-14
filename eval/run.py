"""
eval/run.py — Pipeline run for evaluation (precision / recall benchmarking).

All outputs are written to eval/out/ so they don't pollute the main out/
directory. Visualization is enabled by default so detections can be audited
alongside the extracted text.

Usage (from project root):
    python eval/run.py
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from nlp_histo.pipeline.stages.pdf_text_extraction.batch import ParallelBatchRunner  # noqa: E402
from nlp_histo.pipeline.stages.pdf_text_extraction.config import (  # noqa: E402
    CroppingConfig,
    DatabaseConfig,
    DoclingConfig,
    PathConfig,
    PipelineConfig,
    RuntimeConfig,
    TableDetectorType,
    TextAssemblyConfig,
    VisualizationConfig,
)

HERE = Path(__file__).parent
OUT  = HERE / "out"

PDF_DIR        = ROOT / "files/organized_pdfs"
N_SAMPLES      = 10
SEED           = 42
MIN_FILE_MB    = 0.5   # skip PDFs smaller than this (too short / not full papers)
MAX_FILE_MB    = 10.0  # skip PDFs larger than this (too long / image-heavy)


def make_config() -> PipelineConfig:
    cfg = PipelineConfig()

    # ── Redirect all outputs into eval/out/ ───────────────────────────────────
    cfg.paths = PathConfig(
        output_root        = OUT,
        files_root         = ROOT / "files",
        masked_pdf_dir     = OUT / "masked_pdfs",
        docling_full_dir   = OUT / "docling_full",
        docling_masked_dir = OUT / "docling_masked",
        text_dir           = OUT / "text",
        text_raw_dir       = OUT / "text_raw",
        json_dir           = OUT / "json" / "full",
        vis_dir            = OUT / "visualization",
        figures_dir        = OUT / "figures",
        tables_dir         = OUT / "tables" / "full",
        blacklist_file     = HERE / "blacklist.json",
        completed_file     = HERE / "completed.json",
        run_metadata_dir   = OUT / "run_metadata",
    )

    # ── Visualization: annotated PDFs for detection auditing ──────────────────
    cfg.visualization = VisualizationConfig(
        enabled                     = True,
        save_tatr_visualization     = True,
        save_combined_visualization = True,
        max_pages                   = None,
    )

    # ── Text: also dump raw pre-assembly elements for debugging ───────────────
    cfg.text = TextAssemblyConfig(
        write_raw_text = True,
    )

    cfg.database = DatabaseConfig(enabled=True)

    # ── Cropping: merge same-caption same-page detections from TATR + Docling ──
    cfg.cropping = CroppingConfig(
        merge_tables_by_caption=True,
        expand_tables_with_footnotes=True,
    )

    # ── Table detector: hybrid (Docling + TATR) ───────────────────────────────
    cfg.table_detector = TableDetectorType.HYBRID

    # ── Reconstruct tables from list elements where Docling misses them ───────
    cfg.docling = DoclingConfig(reconstruct_tables_from_lists=True)

    # ── Runtime: expose seed explicitly ──────────────────────────────────────
    cfg.runtime = RuntimeConfig(
        seed=SEED,
        blacklist_if_rows_exceed=200,
        multi_source_crops=True,
    )

    return cfg


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cfg = make_config()
    cfg.prepare()

    # ── Sample N_SAMPLES PDFs reproducibly ───────────────────────────────────
    all_pdfs  = sorted(PDF_DIR.glob("*.pdf"))
    min_bytes = MIN_FILE_MB * 1024 * 1024
    max_bytes = MAX_FILE_MB * 1024 * 1024
    eligible  = [
        p for p in all_pdfs
        if min_bytes <= p.stat().st_size <= max_bytes
    ]
    logging.getLogger(__name__).info(
        "Eligible PDFs: %d / %d (%.1f–%.1f MB)",
        len(eligible), len(all_pdfs), MIN_FILE_MB, MAX_FILE_MB,
    )
    rng    = random.Random(cfg.runtime.seed)
    sample = rng.sample(eligible, min(N_SAMPLES, len(eligible)))
    sample.sort()  # stable order within the sample for readable logs

    # ── Skip already-completed documents ─────────────────────────────────────
    json_dir   = OUT / "json"
    remaining  = [p for p in sample if not (json_dir / f"{p.stem}_media.json").exists()]
    n_done     = len(sample) - len(remaining)
    log = logging.getLogger(__name__)
    if n_done:
        log.info("Checkpoint: %d / %d already done — skipping", n_done, len(sample))
    sample = remaining

    log.info(
        "Eval batch: %d / %d PDFs (seed=%d)", len(sample), len(all_pdfs), cfg.runtime.seed
    )

    ParallelBatchRunner(cfg, max_workers=4).run_paths(sample)


if __name__ == "__main__":
    main()
