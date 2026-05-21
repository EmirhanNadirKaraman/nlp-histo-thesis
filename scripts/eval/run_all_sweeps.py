#!/usr/bin/env python3
"""
run_all_sweeps.py — run every PDF-extraction sweep variant in a single
Python process so Docling + TATR models load once and stay cached.

Equivalent to ``/tmp/rerun_sweeps.sh`` but ~5 min faster on a full
cycle: each shell ``run`` invocation in that script spawns a fresh
Python process which reloads Docling (~20s) and TATR (~10s).  Here we
amortize both: TATR uses its module-level ``_SHARED_PROCESSOR/_MODEL``
already, and Docling now caches via ``_DOCLING_CONVERTER_CACHE`` keyed
by the tuple of options that influence converter state.

Variants are defined inline below; each is a ``SweepSpec`` with a name
and a ``configure(cfg)`` callable that mutates the canonical
``PipelineConfig`` to express the variant.

Outputs land under ``out/sweeps/<name>/`` exactly like the shell script.
After all sweeps finish, share-map rebuild and per-variant seeding run
automatically (same as the old shell flow's tail).

Usage::

    python scripts/eval/run_all_sweeps.py
    python scripts/eval/run_all_sweeps.py --stage detector_docling
    python scripts/eval/run_all_sweeps.py --stage table_in_figure --list-variants
    python scripts/eval/run_all_sweeps.py --pdf-dir eval/pdfs --workers 2
    python scripts/eval/run_all_sweeps.py --skip-share-map
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from tqdm import tqdm

# Make ``pipeline.*`` importable when this script is run directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.stages.pdf_text_extraction import PipelineConfig, PipelineRunner  # noqa: F401  (kept for backwards-compat imports)
from pipeline.stages.pdf_text_extraction.batch import ParallelBatchRunner
from pipeline.stages.pdf_text_extraction.config import TableDetectorType
from pipeline.stages.pdf_text_extraction.runner import _retarget_paths


_DEFAULT_PDF_DIR = _REPO_ROOT / "eval" / "pdfs"
_DEFAULT_SWEEPS_ROOT = _REPO_ROOT / "out" / "sweeps"


@dataclass
class SweepSpec:
    name: str
    configure: Callable[[PipelineConfig], None]
    workers: int = 2
    stage: str = "detector_docling"  # one of STAGE_CHOICES (minus "all")


# Ordered: each stage builds on the previous stage's winner.
STAGE_ORDER = (
    "detector_docling",     # Stage 1.1 — Docling baseline
    "detector_tatr",        # Stage 1.2 — TATR threshold selection
    "detector_hybrid",      # Stage 1.3 — Hybrid threshold selection
    "table_in_figure",      # Stage 2   — table_in_figure fixes for TATR / Hybrid
    "footnote_screen",      # Stage 3   — footnote expansion per detector family
    "footnote_multiplier",  # Stage 4   — multiplier sweep on the Stage-3 winner
    "merge_flags",          # Stage 5   — merge_tables / merge_figures on BEST_BASE
    "reconstruction",       # Stage 6   — reconstruction × expand on BEST_BASE
)
STAGE_CHOICES = STAGE_ORDER + ("all",)

# Human-readable one-liners for the stage menu printed when --stage is omitted.
STAGE_BLURBS: dict = {
    "detector_docling":    "Stage 1.1 — Docling baseline",
    "detector_tatr":       "Stage 1.2 — TATR threshold selection",
    "detector_hybrid":     "Stage 1.3 — Hybrid threshold selection",
    "table_in_figure":     "Stage 2 — table_in_figure fixes for TATR / Hybrid",
    "footnote_screen":     "Stage 3 — footnote_expand=1.2 per detector family",
    "footnote_multiplier": "Stage 4 — footnote_threshold_multiplier sweep on BEST_BASE",
    "merge_flags":         "Stage 5 — merge_tables / merge_figures on BEST_BASE",
    "reconstruction":      "Stage 6 — reconstruction × expand setting on BEST_BASE",
}


# ---------------------------------------------------------------------------
# Staged-sweep knobs.  Each constant is the "winner" chosen after the
# corresponding stage and feeds into every later stage.  Edit them in place
# as each stage's scoring lands; the variants below pick them up.
# ---------------------------------------------------------------------------

# Stage 1 winners — detector / threshold per family.  Drive Stages 2-5.
STAGE1_BASE_DOCLING = "01_docling"
STAGE1_BASE_TATR    = "04_tatr_099"
STAGE1_BASE_HYBRID  = "07_hybrid_099"

# Stage 2 winners — table_in_figure mode per detector family.  Each is
# one of: "none", "drop", "premask", "drop_plus_premask".  Drives Stages 3-5
# whenever the selected family is TATR or Hybrid.  Docling skips Stage 2
# entirely (premasking is a no-op for Docling; see runner.py:651-653 and
# media_cropper.py B-058 fix — for Docling, only `drop_tables_inside_figures`
# actually changes behaviour, and it lives in the table_in_figure mode dial).
BEST_TATR_TABLE_IN_FIGURE_MODE   = "drop"
BEST_HYBRID_TABLE_IN_FIGURE_MODE = "drop"

# Stage 3 winner — final detector family after footnote expansion.  Drives
# Stages 4-5.  One of the seven Stage-1 base names.
BEST_BASE = "01_docling"

# Stage 5 winners — merge flags chosen on BEST_BASE.  Drive Stage 6.
BEST_MERGE_TABLES_BY_CAPTION  = False
BEST_MERGE_FIGURES_BY_CAPTION = False

# Stage 6 winners — chosen after the reconstruction × expand sweep.
# These don't feed any later variant (Stage 5 is terminal); they're
# documentation for the freeze step that bakes defaults into
# PipelineConfig.  See docs/THESIS.md Decisions log.
BEST_RECONSTRUCTION_SETTING = False
BEST_EXPAND_SETTING         = True

# Pinned knobs (no sweep).
BEST_TWO_PASS          = True   # ghost-text safety; see 2026-05-13 decision
BEST_EXPAND_MULTIPLIER = 1.2    # Stage-2 winner from earlier sweep (94% missed-footnote reduction)


_STAGE1_BASES = {
    "01_docling":    (TableDetectorType.DOCLING, None),
    "02_tatr_090":   (TableDetectorType.TATR,    0.90),
    "03_tatr_095":   (TableDetectorType.TATR,    0.95),
    "04_tatr_099":   (TableDetectorType.TATR,    0.99),
    "05_hybrid_090": (TableDetectorType.HYBRID,  0.90),
    "06_hybrid_095": (TableDetectorType.HYBRID,  0.95),
    "07_hybrid_099": (TableDetectorType.HYBRID,  0.99),
}

_VALID_TIF_MODES = ("none", "drop", "premask", "drop_plus_premask")


# ---------------------------------------------------------------------------
# Helper functions.
# ---------------------------------------------------------------------------

def _apply_stage1_baseline(cfg: PipelineConfig) -> None:
    """Canonical Stage 1 baseline: every helper flag explicit.

    Every field is set explicitly so a future change to PipelineConfig
    defaults cannot silently shift the experiment baseline.
    """
    cfg.two_pass.enabled                              = True
    cfg.two_pass.render_dpi                           = 150
    cfg.tatr.render_dpi                               = 150
    cfg.docling.reconstruct_tables_from_lists         = False
    cfg.cropping.expand_tables_with_footnotes         = False
    cfg.cropping.merge_tables_by_caption              = False
    cfg.cropping.merge_figures_by_caption             = False
    cfg.masking.drop_tables_inside_figures            = False
    cfg.masking.mask_figures_before_table_detection   = False


def _apply_base(cfg: PipelineConfig, base_name: str) -> None:
    """Apply Stage 1 baseline + the named detector/threshold base."""
    _apply_stage1_baseline(cfg)
    if base_name not in _STAGE1_BASES:
        raise ValueError(
            f"base={base_name!r} must be one of {sorted(_STAGE1_BASES)}"
        )
    detector, threshold = _STAGE1_BASES[base_name]
    cfg.table_detector = detector
    if threshold is not None:
        cfg.tatr.threshold = threshold


def _apply_table_in_figure_mode(cfg: PipelineConfig, mode: str) -> None:
    """Set drop / premask flags from a single dial.

    ``mode`` ∈ {"none", "drop", "premask", "drop_plus_premask"}.
    """
    if mode not in _VALID_TIF_MODES:
        raise ValueError(f"mode={mode!r} must be one of {_VALID_TIF_MODES}")
    cfg.masking.drop_tables_inside_figures          = mode in ("drop", "drop_plus_premask")
    cfg.masking.mask_figures_before_table_detection = mode in ("premask", "drop_plus_premask")


def _base_family(base_name: str) -> str:
    """Return 'docling' | 'tatr' | 'hybrid' from a Stage-1 base name."""
    if "docling" in base_name: return "docling"
    if "tatr" in base_name:    return "tatr"
    if "hybrid" in base_name:  return "hybrid"
    raise ValueError(f"unrecognised base family: {base_name!r}")


def _tif_mode_for_base(base_name: str) -> str:
    """Pick the right Stage-2-winner mode for a given base."""
    fam = _base_family(base_name)
    if fam == "docling": return "none"
    if fam == "tatr":    return BEST_TATR_TABLE_IN_FIGURE_MODE
    if fam == "hybrid":  return BEST_HYBRID_TABLE_IN_FIGURE_MODE
    return "none"


def _apply_best_base_with_mode(cfg: PipelineConfig) -> None:
    """Apply BEST_BASE + its corresponding table_in_figure mode."""
    _apply_base(cfg, BEST_BASE)
    _apply_table_in_figure_mode(cfg, _tif_mode_for_base(BEST_BASE))


def _workers_for_base(base_name: str) -> int:
    """Docling-only base avoids previously-observed OOM by using 1 worker.

    Tolerates placeholder values (e.g. ``"none"`` before Stage 3 picks
    BEST_BASE) by returning the safe default of 1.  The actual variant
    will still fail at configure-time when ``_apply_base`` tries to look
    up the placeholder in ``_STAGE1_BASES`` — that's the right time for
    the error (the user is actively trying to run the stage), not at
    module-import time.
    """
    if base_name not in _STAGE1_BASES:
        return 1
    return 1 if _base_family(base_name) == "docling" else 2


# ---------------------------------------------------------------------------
# Stage 1.1 — Docling baseline.
# ---------------------------------------------------------------------------

def _stage1_variant(base_name: str, stage_slug: str):
    """Build a configure() callable for a Stage-1 variant by base name."""
    def configure(cfg: PipelineConfig) -> None:
        _apply_base(cfg, base_name)
    return configure


ALL_SWEEPS: List[SweepSpec] = [
    SweepSpec("01_docling", _stage1_variant("01_docling", "detector_docling"),
              workers=1, stage="detector_docling"),
]


# ---------------------------------------------------------------------------
# Stage 1.2 — TATR threshold selection.
# ---------------------------------------------------------------------------

ALL_SWEEPS.extend([
    SweepSpec("02_tatr_090", _stage1_variant("02_tatr_090", "detector_tatr"),
              workers=2, stage="detector_tatr"),
    SweepSpec("03_tatr_095", _stage1_variant("03_tatr_095", "detector_tatr"),
              workers=2, stage="detector_tatr"),
    SweepSpec("04_tatr_099", _stage1_variant("04_tatr_099", "detector_tatr"),
              workers=2, stage="detector_tatr"),
])


# ---------------------------------------------------------------------------
# Stage 1.3 — Hybrid threshold selection.
# ---------------------------------------------------------------------------

ALL_SWEEPS.extend([
    SweepSpec("05_hybrid_090", _stage1_variant("05_hybrid_090", "detector_hybrid"),
              workers=2, stage="detector_hybrid"),
    SweepSpec("06_hybrid_095", _stage1_variant("06_hybrid_095", "detector_hybrid"),
              workers=2, stage="detector_hybrid"),
    SweepSpec("07_hybrid_099", _stage1_variant("07_hybrid_099", "detector_hybrid"),
              workers=2, stage="detector_hybrid"),
])


# ---------------------------------------------------------------------------
# Stage 2 — table_in_figure fixes for TATR / Hybrid.
# Docling skips this stage; premask is a no-op for Docling and the drop
# post-filter is part of the table_in_figure mode dial we'll sweep here.
# ---------------------------------------------------------------------------

def _stage2(base_constant_name: str, mode: str):
    """Stage-2 variant: apply STAGE1_BASE_{TATR,HYBRID} + a TIF mode.

    ``base_constant_name`` is the *name* of the module-level constant
    (e.g. "STAGE1_BASE_TATR") so the variant tracks edits to the
    constant — the actual base name is dereferenced at config-build
    time (not at variant-spec build time).
    """
    def configure(cfg: PipelineConfig) -> None:
        base = globals()[base_constant_name]
        _apply_base(cfg, base)
        _apply_table_in_figure_mode(cfg, mode)
    return configure


ALL_SWEEPS.extend([
    SweepSpec("08_tatr_099_drop_tables_in_figures",
              _stage2("STAGE1_BASE_TATR", "drop"),
              workers=2, stage="table_in_figure"),
    SweepSpec("09_tatr_099_premask_figures_for_tables",
              _stage2("STAGE1_BASE_TATR", "premask"),
              workers=2, stage="table_in_figure"),
    SweepSpec("10_tatr_099_drop_plus_premask",
              _stage2("STAGE1_BASE_TATR", "drop_plus_premask"),
              workers=2, stage="table_in_figure"),
    SweepSpec("11_hybrid_099_drop_tables_in_figures",
              _stage2("STAGE1_BASE_HYBRID", "drop"),
              workers=2, stage="table_in_figure"),
    SweepSpec("12_hybrid_099_premask_figures_for_tables",
              _stage2("STAGE1_BASE_HYBRID", "premask"),
              workers=2, stage="table_in_figure"),
    SweepSpec("13_hybrid_099_drop_plus_premask",
              _stage2("STAGE1_BASE_HYBRID", "drop_plus_premask"),
              workers=2, stage="table_in_figure"),
])


# ---------------------------------------------------------------------------
# Stage 3 — footnote_expand=1.2 per detector family.
# Each variant: best-of-its-family base + that family's Stage-2 TIF mode +
# expand=ON, multiplier=1.2.  Compare 14 / 15 / 16 to pick BEST_BASE.
# ---------------------------------------------------------------------------

def _stage3_for_family(base_constant_name: str, family: str):
    """Stage-3 variant: family-best base + that family's TIF mode + expand=1.2."""
    def configure(cfg: PipelineConfig) -> None:
        base = globals()[base_constant_name]
        _apply_base(cfg, base)
        if family == "docling":
            mode = "none"
        elif family == "tatr":
            mode = BEST_TATR_TABLE_IN_FIGURE_MODE
        elif family == "hybrid":
            mode = BEST_HYBRID_TABLE_IN_FIGURE_MODE
        else:
            mode = "none"
        _apply_table_in_figure_mode(cfg, mode)
        cfg.cropping.expand_tables_with_footnotes  = True
        cfg.cropping.footnote_threshold_multiplier = BEST_EXPAND_MULTIPLIER
    return configure


def _stage3_docling_multiplier(multiplier: float):
    """Stage-3 docling-base variant with explicit footnote_threshold_multiplier override.

    Used to sweep the multiplier on Docling (the Stage-3 winner) so we can
    pick the value that best balances footnote recall against
    ``crop too big`` over-expansion.  Compare against variant 14 (1.2,
    the prior pinned default).
    """
    def configure(cfg: PipelineConfig) -> None:
        _apply_base(cfg, STAGE1_BASE_DOCLING)
        _apply_table_in_figure_mode(cfg, "none")
        cfg.cropping.expand_tables_with_footnotes  = True
        cfg.cropping.footnote_threshold_multiplier = multiplier
    return configure


ALL_SWEEPS.extend([
    SweepSpec("14_docling_footnote_expand_1_2",
              _stage3_for_family("STAGE1_BASE_DOCLING", "docling"),
              workers=1, stage="footnote_screen"),
    SweepSpec("15_tatr_best_tif_fix_footnote_expand_1_2",
              _stage3_for_family("STAGE1_BASE_TATR", "tatr"),
              workers=2, stage="footnote_screen"),
    SweepSpec("16_hybrid_best_tif_fix_footnote_expand_1_2",
              _stage3_for_family("STAGE1_BASE_HYBRID", "hybrid"),
              workers=2, stage="footnote_screen"),
])


# ---------------------------------------------------------------------------
# Stage 4 — footnote_threshold_multiplier sweep on the Stage-3 winner.
# Compares against variant 14 (current 1.2 pinned default).  Smaller
# multipliers mean stricter cascade — fewer "crop too big" cases at the
# cost of potentially missing some far-spaced footnotes.  Sweeps run only
# on the Docling base; the helper just hard-codes it.
# ---------------------------------------------------------------------------

ALL_SWEEPS.extend([
    SweepSpec("17_docling_footnote_expand_1_0",
              _stage3_docling_multiplier(1.0),
              workers=1, stage="footnote_multiplier"),
    SweepSpec("18_docling_footnote_expand_1_1",
              _stage3_docling_multiplier(1.1),
              workers=1, stage="footnote_multiplier"),
    SweepSpec("19_docling_footnote_expand_1_15",
              _stage3_docling_multiplier(1.15),
              workers=1, stage="footnote_multiplier"),
])


# ---------------------------------------------------------------------------
# Stage 5 — single-flag flips on BEST_BASE.
# Base: BEST_BASE + corresponding TIF mode + expand=ON, multiplier=1.2.
# Each variant flips exactly one extra flag.
#   20: merge_tables_by_caption=ON
#   21: merge_figures_by_caption=ON
#   22: drop_tables_inside_figures=ON (forced even if BEST_BASE's family TIF
#       mode would otherwise be "none"; meaningful only when BEST_BASE is
#       docling, since TATR/Hybrid bases already get drop=ON via the family
#       TIF mode from Stage 2.  Variant 14 had a `table_in_figure` FP that
#       drop should suppress post-B-058 fix.)
# ---------------------------------------------------------------------------

def _stage4(*, merge_tables: bool = False, merge_figures: bool = False,
            force_drop_tif: bool = False):
    def configure(cfg: PipelineConfig) -> None:
        _apply_best_base_with_mode(cfg)
        cfg.cropping.expand_tables_with_footnotes  = True
        cfg.cropping.footnote_threshold_multiplier = BEST_EXPAND_MULTIPLIER
        cfg.cropping.merge_tables_by_caption  = merge_tables
        cfg.cropping.merge_figures_by_caption = merge_figures
        if force_drop_tif:
            cfg.masking.drop_tables_inside_figures = True
    return configure


ALL_SWEEPS.extend([
    SweepSpec("20_best_merge_tables_by_caption",
              _stage4(merge_tables=True),
              workers=_workers_for_base(BEST_BASE), stage="merge_flags"),
    SweepSpec("21_best_merge_figures_by_caption",
              _stage4(merge_figures=True),
              workers=_workers_for_base(BEST_BASE), stage="merge_flags"),
    SweepSpec("22_best_drop_tables_inside_figures",
              _stage4(force_drop_tif=True),
              workers=_workers_for_base(BEST_BASE), stage="merge_flags"),
])


# ---------------------------------------------------------------------------
# Stage 6 — reconstruction × expand setting on BEST_BASE.
# Base: BEST_BASE + corresponding TIF mode + BEST_MERGE_*_BY_CAPTION.
#   23: reconstruct=ON, expand=OFF
#   24: reconstruct=ON, expand=ON (multiplier=BEST_EXPAND_MULTIPLIER)
# Compare 23 vs 24 vs Stage-3 winner to pick BEST_RECONSTRUCTION_SETTING
# and BEST_EXPAND_SETTING.
# ---------------------------------------------------------------------------

def _stage5(*, expand: bool):
    def configure(cfg: PipelineConfig) -> None:
        _apply_best_base_with_mode(cfg)
        cfg.docling.reconstruct_tables_from_lists  = True
        cfg.cropping.merge_tables_by_caption       = BEST_MERGE_TABLES_BY_CAPTION
        cfg.cropping.merge_figures_by_caption      = BEST_MERGE_FIGURES_BY_CAPTION
        cfg.cropping.expand_tables_with_footnotes  = expand
        if expand:
            cfg.cropping.footnote_threshold_multiplier = BEST_EXPAND_MULTIPLIER
    return configure


ALL_SWEEPS.extend([
    SweepSpec("23_best_reconstruct_only",
              _stage5(expand=False),
              workers=_workers_for_base(BEST_BASE), stage="reconstruction"),
    SweepSpec("24_best_reconstruct_plus_selected_expand",
              _stage5(expand=True),
              workers=_workers_for_base(BEST_BASE), stage="reconstruction"),
])


# ---------------------------------------------------------------------------
# Build / run plumbing (unchanged from the prior layout).
# ---------------------------------------------------------------------------

def _build_config(spec: SweepSpec, *, pdf_dir: Path, sweeps_root: Path,
                  workers_override: Optional[int],
                  ensure_dirs: bool = True) -> PipelineConfig:
    cfg = PipelineConfig()
    # Sweep defaults: no DB writes, no raw text dumps (saves disk + time).
    cfg.database.enabled = False
    cfg.text.write_raw_text = False
    # Resume semantics: skip PDFs whose media JSON already exists, and
    # enable the per-stage cache so a PDF interrupted mid-pipeline can
    # resume from the last completed stage on the next run.
    cfg.runtime.skip_existing_media_json = True
    cfg.runtime.skip_existing_outputs    = True
    # Apply variant mutations
    spec.configure(cfg)
    cfg.runtime.num_workers = workers_override if workers_override is not None else spec.workers
    # Retarget every per-variant path under out/sweeps/<name>/
    _retarget_paths(cfg, sweeps_root / spec.name)
    if ensure_dirs:
        cfg.prepare()
    else:
        # Dry-run path: validate flag values without creating directories.
        cfg.validate()
    return cfg


_DONE_MARKER = "_DONE.json"


def _count_pdfs(pdf_dir: Path) -> int:
    return sum(1 for _ in pdf_dir.glob("*.pdf"))


def _variant_already_done(out_dir: Path, expected_pdf_count: int) -> bool:
    """Return True iff the variant has a _DONE marker stamped for the
    same number of PDFs that currently live in the source directory.

    A PDF-count mismatch (e.g. the user added more PDFs since last run)
    invalidates the marker so the new PDFs get processed on resume.
    Already-processed PDFs are still skipped via
    `cfg.runtime.skip_existing_media_json`.
    """
    marker = out_dir / _DONE_MARKER
    if not marker.exists():
        return False
    try:
        data = json.loads(marker.read_text())
    except Exception:
        return False
    return data.get("pdf_count") == expected_pdf_count


def _mark_variant_done(out_dir: Path, pdf_count: int) -> None:
    marker = out_dir / _DONE_MARKER
    marker.write_text(json.dumps({
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pdf_count": pdf_count,
    }, indent=2))


def _clean_partial_pdf_outputs(out_dir: Path, pdf_dir: Path) -> int:
    """Delete crop artifacts for any PDF that started but did not finish.

    A PDF is considered "completed" iff ``json/<pmcid>_media.json`` exists.
    For each PDF in ``pdf_dir`` whose media JSON is missing, this scans
    ``figures/`` and ``tables/`` for filenames starting with ``<pmcid>_``
    and deletes them.

    Necessary because ``PyMuPDFMediaCropper`` appends ``_1``, ``_2`` suffixes
    on filename collisions (``components/media_cropper.py:334``) — without
    this cleanup, a Ctrl-C mid-crop on the first attempt would silently
    leave behind half a crop set, and the next attempt's crops would be
    written alongside (doubling the share-map entries) instead of
    overwriting.

    Idempotent and cheap: only touches PDFs whose ``_media.json`` is
    missing, so completed work is preserved. Returns the number of
    artifacts deleted (informational).
    """
    json_dir    = out_dir / "json"
    figures_dir = out_dir / "figures"
    tables_dir  = out_dir / "tables"
    if not figures_dir.exists() and not tables_dir.exists():
        return 0

    deleted = 0
    for pdf in pdf_dir.glob("*.pdf"):
        pmcid = pdf.stem
        if (json_dir / f"{pmcid}_media.json").exists():
            continue  # finished PDF — keep its crops
        for d in (figures_dir, tables_dir):
            if not d.exists():
                continue
            for stale in d.glob(f"{pmcid}_*"):
                try:
                    stale.unlink()
                    deleted += 1
                except OSError:
                    pass
    return deleted


def _run_one(spec: SweepSpec, *, pdf_dir: Path, sweeps_root: Path,
             workers_override: Optional[int],
             force_restart: bool = False,
             ) -> tuple[str, float, bool, str]:
    """Run a single sweep variant.

    Returns (name, wall_seconds, ok, status_label) where status_label is
    one of {"ran", "resumed", "skipped"} for the post-run summary.

    Resume semantics:
      * `force_restart=True` removes the output directory before running.
      * Otherwise, if a `_DONE.json` marker is present for the current
        PDF count, the variant is skipped entirely.
      * Otherwise the variant runs; `cfg.runtime.skip_existing_media_json`
        causes already-processed PDFs to be skipped, and the per-stage
        cache lets partially-processed PDFs resume from the last
        completed stage.
    """
    out_dir = sweeps_root / spec.name
    if force_restart and out_dir.exists():
        logging.info("--restart: removing %s", out_dir)
        shutil.rmtree(out_dir)

    pdf_count = _count_pdfs(pdf_dir)

    if _variant_already_done(out_dir, pdf_count):
        logging.info("[%s] already done (%d PDFs) — skipping", spec.name, pdf_count)
        return spec.name, 0.0, True, "skipped"

    status = "resumed" if out_dir.exists() else "ran"
    if status == "resumed":
        n_deleted = _clean_partial_pdf_outputs(out_dir, pdf_dir)
        if n_deleted:
            logging.info(
                "[%s] cleaned %d stale crop file(s) from interrupted runs",
                spec.name, n_deleted,
            )
    t0 = time.perf_counter()
    cfg = _build_config(spec, pdf_dir=pdf_dir, sweeps_root=sweeps_root,
                         workers_override=workers_override)
    # Use ParallelBatchRunner so num_workers, the per-PDF tqdm bar, and the
    # `[N/total]` log line all activate. PipelineRunner.run_batch is a
    # sequential for-loop that bypasses all three.
    try:
        ParallelBatchRunner(cfg, max_workers=cfg.runtime.num_workers).run(
            pdf_dir=pdf_dir,
        )
        ok = True
    except Exception as exc:
        logging.exception("sweep %s failed: %s", spec.name, exc)
        ok = False
    elapsed = time.perf_counter() - t0
    if ok:
        _mark_variant_done(out_dir, pdf_count)
    return spec.name, elapsed, ok, status


def _print_variants_table(
    sweeps: List[SweepSpec],
    *,
    pdf_dir: Path,
    sweeps_root: Path,
    workers_override: Optional[int],
) -> None:
    """Print one row per variant with its resolved config.

    Builds the PipelineConfig for each variant (without running extraction)
    so the printed flags match exactly what a real run would see.
    """
    header = (
        "variant", "stage", "detector", "tatr", "two_pass",
        "recon", "expand", "ftn_x",
        "merge_tables", "merge_figures",
        "drop_tables_inside_figures", "premask_figures_before_table_detection",
        "out_dir",
    )
    rows: List[tuple] = []
    for spec in sweeps:
        cfg = _build_config(spec, pdf_dir=pdf_dir, sweeps_root=sweeps_root,
                             workers_override=workers_override,
                             ensure_dirs=False)
        detector = cfg.table_detector.value
        tatr_thr = (f"{cfg.tatr.threshold:.2f}"
                    if detector in ("tatr", "hybrid") else "-")
        ftn_x = (f"{cfg.cropping.footnote_threshold_multiplier:.2f}"
                 if cfg.cropping.expand_tables_with_footnotes else "-")
        rows.append((
            spec.name,
            spec.stage,
            detector,
            tatr_thr,
            "ON" if cfg.two_pass.enabled else "OFF",
            "ON" if cfg.docling.reconstruct_tables_from_lists else "OFF",
            "ON" if cfg.cropping.expand_tables_with_footnotes else "OFF",
            ftn_x,
            "ON" if cfg.cropping.merge_tables_by_caption else "OFF",
            "ON" if cfg.cropping.merge_figures_by_caption else "OFF",
            "ON" if cfg.masking.drop_tables_inside_figures else "OFF",
            "ON" if cfg.masking.mask_figures_before_table_detection else "OFF",
            str(cfg.paths.output_root),
        ))

    widths = [max(len(str(c)) for c in col)
              for col in zip(header, *rows)]

    def _fmt(row: tuple) -> str:
        return "  ".join(str(c).ljust(w) for c, w in zip(row, widths))

    print(_fmt(header))
    print(_fmt(tuple("-" * w for w in widths)))
    for row in rows:
        print(_fmt(row))


def _print_stage_menu() -> None:
    """List every defined stage with its description and member variants."""
    by_stage: dict = {s: [] for s in STAGE_ORDER}
    for spec in ALL_SWEEPS:
        by_stage.setdefault(spec.stage, []).append(spec.name)

    print("Available --stage values:")
    print()
    for stage in STAGE_ORDER:
        print(f"  {stage}")
        print(f"      {STAGE_BLURBS.get(stage, '')}")
        for name in by_stage.get(stage, []):
            print(f"        - {name}")
        print()
    print("  all")
    print(f"      every variant ({len(ALL_SWEEPS)} total)")
    print()
    print("Examples:")
    print("  python scripts/eval/run_all_sweeps.py --stage detector_docling")
    print("  python scripts/eval/run_all_sweeps.py --stage table_in_figure --list-variants")
    print("  python scripts/eval/run_all_sweeps.py --stage all --only 07_hybrid_099")


def _post_run_share_map() -> None:
    """Rebuild share-map and re-seed per-variant labels (the shell tail)."""
    import subprocess
    PY = sys.executable
    here = Path(__file__).parent
    logging.info("rebuilding share_map...")
    r1 = subprocess.run([PY, str(here / "build_share_map.py")],
                         capture_output=True, text=True)
    print(r1.stdout)
    if r1.returncode != 0:
        print(r1.stderr, file=sys.stderr)
    logging.info("re-seeding per-variant labels...")
    r2 = subprocess.run([PY, str(here / "seed_variant_labels.py")],
                         capture_output=True, text=True)
    print(r2.stdout)
    if r2.returncode != 0:
        print(r2.stderr, file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdf-dir", type=Path, default=_DEFAULT_PDF_DIR,
                   help="Source PDF directory.  Default: eval/pdfs/")
    p.add_argument("--sweeps-root", type=Path, default=_DEFAULT_SWEEPS_ROOT,
                   help="Where to write each variant's outputs.")
    p.add_argument("--stage", default=None, choices=STAGE_CHOICES,
                   help="Which experiment stage to run (required). "
                        "'all' = every variant; otherwise filters ALL_SWEEPS "
                        "to variants tagged with that stage. "
                        "Combine with --only to narrow further. "
                        "Omit to print the stage menu.")
    p.add_argument("--only", nargs="+", default=None,
                   help="Run only these sweep names within the selected --stage "
                        "(e.g. --only 07_hybrid_099). Intersects with --stage.")
    p.add_argument("--workers", type=int, default=None,
                   help="Override per-variant worker count.")
    p.add_argument("--skip-share-map", action="store_true",
                   help="Skip the share-map rebuild + label seed at the end.")
    p.add_argument("--list-variants", "--dry-run", dest="list_variants",
                   action="store_true",
                   help="Print the resolved config for each variant and exit "
                        "without running extraction.")
    p.add_argument("--restart", action="store_true",
                   help="Wipe each selected variant's output directory before "
                        "running (force fresh re-run). Default: resume — "
                        "variants stamped with _DONE.json are skipped and "
                        "partially-completed variants pick up from where they "
                        "left off via per-PDF + per-stage caching.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.stage is None:
        _print_stage_menu()
        return 0

    if not args.list_variants and not args.pdf_dir.exists():
        print(f"error: pdf-dir not found: {args.pdf_dir}", file=sys.stderr)
        return 2

    sweeps = ALL_SWEEPS
    if args.stage != "all":
        sweeps = [s for s in sweeps if s.stage == args.stage]
        if not sweeps:
            print(f"error: no variants defined for --stage {args.stage}", file=sys.stderr)
            return 2
    if args.only:
        wanted = set(args.only)
        sweeps = [s for s in sweeps if s.name in wanted]
        all_in_stage = {s.name for s in ALL_SWEEPS
                        if args.stage == "all" or s.stage == args.stage}
        missing = wanted - all_in_stage
        if missing:
            print(f"error: unknown sweep name(s) for --stage {args.stage}: "
                  f"{sorted(missing)}", file=sys.stderr)
            print(f"  available: {sorted(all_in_stage)}", file=sys.stderr)
            return 2

    if args.list_variants:
        _print_variants_table(
            sweeps,
            pdf_dir=args.pdf_dir,
            sweeps_root=args.sweeps_root,
            workers_override=args.workers,
        )
        return 0

    args.sweeps_root.mkdir(parents=True, exist_ok=True)

    print(f"=== running {len(sweeps)} sweep(s) in one process ===")
    print(f"  pdf-dir:     {args.pdf_dir}")
    print(f"  sweeps-root: {args.sweeps_root}")
    print(f"  variants:    {[s.name for s in sweeps]}")
    print()

    results: List[tuple[str, float, bool, str]] = []
    total_t0 = time.perf_counter()
    with tqdm(
        total=len(sweeps),
        unit="variant",
        desc="Sweep progress",
        position=0,
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} variants  [{elapsed}<{remaining}]  {postfix}",
    ) as outer:
        for i, spec in enumerate(sweeps, 1):
            outer.set_postfix_str(f"running {spec.name}")
            tqdm.write(
                f"[{time.strftime('%H:%M:%S')}] === [{i}/{len(sweeps)}] {spec.name} starting ==="
            )
            name, elapsed, ok, status = _run_one(
                spec,
                pdf_dir=args.pdf_dir,
                sweeps_root=args.sweeps_root,
                workers_override=args.workers,
                force_restart=args.restart,
            )
            results.append((name, elapsed, ok, status))
            if status == "skipped":
                flag = "SKIPPED (already done)"
            elif ok:
                flag = f"OK ({status})"
            else:
                flag = "FAILED"
            tqdm.write(
                f"[{time.strftime('%H:%M:%S')}] === [{i}/{len(sweeps)}] {name} {flag}  ({elapsed:.1f}s) ==="
            )
            outer.update(1)
        outer.set_postfix_str("done")

    total_elapsed = time.perf_counter() - total_t0
    print()
    print("=== summary ===")
    for name, elapsed, ok, status in results:
        if status == "skipped":
            flag = "SKIP  "
        elif ok:
            flag = "OK    "
        else:
            flag = "FAILED"
        print(f"  {flag}  {elapsed:>7.1f}s  {name}  ({status})")
    print(f"  total:  {total_elapsed:>7.1f}s  ({len(results)} sweeps)")

    if not args.skip_share_map:
        print()
        _post_run_share_map()

    return 0 if all(ok for _, _, ok, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
