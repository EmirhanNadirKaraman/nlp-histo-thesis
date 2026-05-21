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
    "header_fix",           # Stage 3   — header-band drop filter for TATR / Hybrid
    "footnote_screen",      # Stage 4   — footnote expansion per detector family
    "footnote_multiplier",  # Stage 5   — multiplier sweep on the Stage-4 winner
    "merge_flags",          # Stage 6   — merge_tables / merge_figures on BEST_BASE
    "reconstruction",       # Stage 7   — reconstruction × expand on BEST_BASE
)
STAGE_CHOICES = STAGE_ORDER + ("all",)

# Human-readable one-liners for the stage menu printed when --stage is omitted.
STAGE_BLURBS: dict = {
    "detector_docling":    "Stage 1.1 — Docling baseline",
    "detector_tatr":       "Stage 1.2 — TATR threshold selection",
    "detector_hybrid":     "Stage 1.3 — Hybrid threshold selection",
    "table_in_figure":     "Stage 2 — table_in_figure fixes for TATR / Hybrid",
    "header_fix":          "Stage 3 — drop_tables_in_top_pts on TATR / Hybrid",
    "footnote_screen":     "Stage 4 — footnote_expand=1.2 per detector family",
    "footnote_multiplier": "Stage 5 — footnote_threshold_multiplier sweep on BEST_BASE",
    "merge_flags":         "Stage 6 — merge_tables / merge_figures on BEST_BASE",
    "reconstruction":      "Stage 7 — reconstruction × expand setting on BEST_BASE",
}


# ---------------------------------------------------------------------------
# Staged-sweep knobs.  Each constant is the "winner" chosen after the
# corresponding stage and feeds into every later stage.  Edit them in place
# as each stage's scoring lands; the variants below pick them up.
# ---------------------------------------------------------------------------

# Stage 1 winners — detector / threshold per family.  Drive Stages 2-7.
STAGE1_BASE_DOCLING = "01_docling"
STAGE1_BASE_TATR    = "04_tatr_099"
STAGE1_BASE_HYBRID  = "07_hybrid_099"

# Stage 2 winners — table_in_figure mode per detector family.  Each is
# one of: "none", "drop", "premask", "drop_plus_premask".  Drives Stages 3-7
# whenever the selected family is TATR or Hybrid.  Docling skips Stage 2
# entirely (premasking is a no-op for Docling; see runner.py:651-653 and
# media_cropper.py B-058 fix — for Docling, only `drop_tables_inside_figures`
# actually changes behaviour, and it lives in the table_in_figure mode dial).
BEST_TATR_TABLE_IN_FIGURE_MODE   = "drop"
BEST_HYBRID_TABLE_IN_FIGURE_MODE = "drop"

# Stage 3 winners — drop_tables_in_top_pts threshold per TATR / Hybrid.
# 0.0 = disabled (don't apply the header-band drop filter).  Typical
# activation value is ~50 (real scientific tables almost never start in
# the top 50pt of a page; TATR/Hybrid misdetect running headers as
# tables in this zone — see PMC4945808).  Docling has no header_fix
# variants because its layout classifies headers correctly enough that
# pixel-zone bogus tables don't appear.  Drives Stages 4-7.
BEST_TATR_HEADER_ZONE_PTS   = 50.0
BEST_HYBRID_HEADER_ZONE_PTS = 50.0

# Stage 4 winner — *provisional* final detector family after footnote
# expansion + best Stages 2+3 family fixes.  Stage 4 picks BEST_BASE at
# multiplier=1.2 across the three families.  Stage 5's 9-variant grid
# (3 multipliers × 3 families) may surface a non-Docling family that
# beats Docling at a non-1.2 multiplier — when that happens, re-pick
# BEST_BASE + BEST_EXPAND_MULTIPLIER from the Stage-5 results.  Drives
# Stages 5-7.  One of the seven Stage-1 base names.
BEST_BASE = "07_hybrid_099"

# Stage 5 winners — best footnote_threshold_multiplier per family.  After
# Stage 5 the BEST_BASE-family value gets copied into BEST_EXPAND_MULTIPLIER
# (the production knob used by all downstream stages).  Defaults match
# BEST_EXPAND_MULTIPLIER's pinned value until Stage 5 lands.  Three
# constants instead of one keeps the family-vs-family comparison
# traceable for thesis writeup.
BEST_DOCLING_EXPAND_MULTIPLIER = 1.2
BEST_TATR_EXPAND_MULTIPLIER    = 1.2
BEST_HYBRID_EXPAND_MULTIPLIER  = 1.2

# Stage 6 winners — merge flags chosen on BEST_BASE.  Drive Stage 7.
BEST_MERGE_TABLES_BY_CAPTION  = False
BEST_MERGE_FIGURES_BY_CAPTION = False

# Stage 7 winners — chosen after the reconstruction × expand sweep.
# These don't feed any later variant (Stage 7 is terminal); they're
# documentation for the freeze step that bakes defaults into
# PipelineConfig.  See docs/THESIS.md Decisions log.
BEST_RECONSTRUCTION_SETTING = False
BEST_EXPAND_SETTING         = True

# Pinned knobs (no sweep).
BEST_TWO_PASS          = True   # ghost-text safety; see 2026-05-13 decision
BEST_EXPAND_MULTIPLIER = 1.2    # Set to the BEST_BASE-family multiplier
                                # after Stage 5 (i.e. = BEST_DOCLING_EXPAND_MULTIPLIER
                                # if BEST_BASE family is Docling, etc.).
                                # Used by Stages 6-7 (merge_flags + reconstruction).


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
    cfg.masking.drop_tables_in_top_pts                = 0.0


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


def _header_zone_pts_for_base(base_name: str) -> float:
    """Pick the right Stage-3-winner header-zone threshold for a given base.

    Docling always returns 0.0 (no header_fix sweep — Docling's layout
    classifies headers correctly).  TATR / Hybrid return the value picked
    after Stage 3 (default 0.0 = disabled before that decision lands).
    """
    fam = _base_family(base_name)
    if fam == "docling": return 0.0
    if fam == "tatr":    return BEST_TATR_HEADER_ZONE_PTS
    if fam == "hybrid":  return BEST_HYBRID_HEADER_ZONE_PTS
    return 0.0


def _apply_family_fixes(cfg: PipelineConfig, base_name: str) -> None:
    """Apply the per-family Stage-2 + Stage-3 fixes (TIF mode + header zone)."""
    _apply_table_in_figure_mode(cfg, _tif_mode_for_base(base_name))
    cfg.masking.drop_tables_in_top_pts = _header_zone_pts_for_base(base_name)


def _apply_best_base_with_mode(cfg: PipelineConfig) -> None:
    """Apply BEST_BASE + its corresponding Stage 2 + Stage 3 family fixes."""
    _apply_base(cfg, BEST_BASE)
    _apply_family_fixes(cfg, BEST_BASE)


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
# Stage 3 — header_fix: drop_tables_in_top_pts sweep on TATR / Hybrid.
# Each variant: family Stage-1 base + family Stage-2 TIF mode +
# drop_tables_in_top_pts > 0.  Compare 14 / 15 to pick
# BEST_*_HEADER_ZONE_PTS per family.  Docling skips this stage because
# its layout doesn't produce header-band TABLE detections.
# ---------------------------------------------------------------------------

def _stage3_header_fix(base_constant_name: str, threshold_pts: float):
    """Stage-3 variant: STAGE1_BASE_{TATR,HYBRID} + family TIF mode + header clip."""
    def configure(cfg: PipelineConfig) -> None:
        base = globals()[base_constant_name]
        _apply_base(cfg, base)
        _apply_table_in_figure_mode(cfg, _tif_mode_for_base(base))
        cfg.masking.drop_tables_in_top_pts = threshold_pts
    return configure


ALL_SWEEPS.extend([
    SweepSpec("14_tatr_099_header_clip_50pts",
              _stage3_header_fix("STAGE1_BASE_TATR", 50.0),
              workers=2, stage="header_fix"),
    SweepSpec("15_hybrid_099_header_clip_50pts",
              _stage3_header_fix("STAGE1_BASE_HYBRID", 50.0),
              workers=2, stage="header_fix"),
])


# ---------------------------------------------------------------------------
# Stage 4 — footnote_expand=1.2 per detector family.
# Each variant: best-of-its-family base + that family's Stage-2 TIF mode +
# Stage-3 header_fix (TATR/Hybrid) + expand=ON, multiplier=1.2.
# Compare 16 / 17 / 18 to pick BEST_BASE.
# ---------------------------------------------------------------------------

def _stage4_for_family(base_constant_name: str, family: str):
    """Stage-4 variant: family-best base + family Stage-2 TIF + Stage-3 header_fix + expand=1.2."""
    def configure(cfg: PipelineConfig) -> None:
        base = globals()[base_constant_name]
        _apply_base(cfg, base)
        _apply_family_fixes(cfg, base)
        cfg.cropping.expand_tables_with_footnotes  = True
        cfg.cropping.footnote_threshold_multiplier = BEST_EXPAND_MULTIPLIER
    return configure


def _stage5_family_multiplier(base_constant_name: str, multiplier: float):
    """Stage-5 variant: family base + family Stage-2/3 fixes + explicit multiplier override.

    Sweep the multiplier on each family so we can pick the value that
    best balances footnote recall against ``crop too big`` over-expansion
    PER family.  Compare against the matching Stage-4 family variant
    (16/17/18, all at multiplier=1.2).
    """
    def configure(cfg: PipelineConfig) -> None:
        base = globals()[base_constant_name]
        _apply_base(cfg, base)
        _apply_family_fixes(cfg, base)
        cfg.cropping.expand_tables_with_footnotes  = True
        cfg.cropping.footnote_threshold_multiplier = multiplier
    return configure


ALL_SWEEPS.extend([
    SweepSpec("16_docling_footnote_expand_1_2",
              _stage4_for_family("STAGE1_BASE_DOCLING", "docling"),
              workers=1, stage="footnote_screen"),
    SweepSpec("17_tatr_best_family_fixes_footnote_expand_1_2",
              _stage4_for_family("STAGE1_BASE_TATR", "tatr"),
              workers=2, stage="footnote_screen"),
    SweepSpec("18_hybrid_best_family_fixes_footnote_expand_1_2",
              _stage4_for_family("STAGE1_BASE_HYBRID", "hybrid"),
              workers=2, stage="footnote_screen"),
])


# ---------------------------------------------------------------------------
# Stage 5 — footnote_threshold_multiplier sweep on ALL three families.
# 3 multipliers × 3 families = 9 variants.  Compare each family's three
# multiplier variants against the corresponding Stage-4 family variant
# (16/17/18, all at multiplier=1.2) to pick BEST_*_EXPAND_MULTIPLIER per
# family.  After Stage 5 the BEST_BASE choice can be revisited if a
# non-Docling family's best multiplier setting beats Docling's best.
# Smaller multipliers mean stricter cascade — fewer "crop too big" cases
# at the cost of potentially missing some far-spaced footnotes.
# ---------------------------------------------------------------------------

ALL_SWEEPS.extend([
    SweepSpec("19_docling_footnote_expand_1_0",
              _stage5_family_multiplier("STAGE1_BASE_DOCLING", 1.0),
              workers=1, stage="footnote_multiplier"),
    SweepSpec("20_docling_footnote_expand_1_1",
              _stage5_family_multiplier("STAGE1_BASE_DOCLING", 1.1),
              workers=1, stage="footnote_multiplier"),
    SweepSpec("21_docling_footnote_expand_1_15",
              _stage5_family_multiplier("STAGE1_BASE_DOCLING", 1.15),
              workers=1, stage="footnote_multiplier"),
    SweepSpec("22_tatr_best_family_fixes_footnote_expand_1_0",
              _stage5_family_multiplier("STAGE1_BASE_TATR", 1.0),
              workers=2, stage="footnote_multiplier"),
    SweepSpec("23_tatr_best_family_fixes_footnote_expand_1_1",
              _stage5_family_multiplier("STAGE1_BASE_TATR", 1.1),
              workers=2, stage="footnote_multiplier"),
    SweepSpec("24_tatr_best_family_fixes_footnote_expand_1_15",
              _stage5_family_multiplier("STAGE1_BASE_TATR", 1.15),
              workers=2, stage="footnote_multiplier"),
    SweepSpec("25_hybrid_best_family_fixes_footnote_expand_1_0",
              _stage5_family_multiplier("STAGE1_BASE_HYBRID", 1.0),
              workers=2, stage="footnote_multiplier"),
    SweepSpec("26_hybrid_best_family_fixes_footnote_expand_1_1",
              _stage5_family_multiplier("STAGE1_BASE_HYBRID", 1.1),
              workers=2, stage="footnote_multiplier"),
    SweepSpec("27_hybrid_best_family_fixes_footnote_expand_1_15",
              _stage5_family_multiplier("STAGE1_BASE_HYBRID", 1.15),
              workers=2, stage="footnote_multiplier"),
])


# ---------------------------------------------------------------------------
# Stage 6 — merge-flag flips on BEST_BASE.
# Base: BEST_BASE + corresponding TIF mode + expand=ON, multiplier=1.2.
# Each variant flips exactly one merge flag.
#   28: merge_tables_by_caption=ON
#   29: merge_figures_by_caption=ON
#
# Original variant 30 (`drop_tables_inside_figures=ON` forced via
# force_drop_tif) was removed on 2026-05-21 because with BEST_BASE =
# "07_hybrid_099", the family TIF mode is already "drop" — variant 30's
# resolved config was bbox-identical to variant 18.  If BEST_BASE ever
# flips to Docling (family TIF = "none"), restore variant 30 to test
# the equivalent of variant 24 from Stage 5.
# ---------------------------------------------------------------------------

def _stage6(*, merge_tables: bool = False, merge_figures: bool = False):
    def configure(cfg: PipelineConfig) -> None:
        _apply_best_base_with_mode(cfg)
        cfg.cropping.expand_tables_with_footnotes  = True
        cfg.cropping.footnote_threshold_multiplier = BEST_EXPAND_MULTIPLIER
        cfg.cropping.merge_tables_by_caption  = merge_tables
        cfg.cropping.merge_figures_by_caption = merge_figures
    return configure


ALL_SWEEPS.extend([
    SweepSpec("28_best_merge_tables_by_caption",
              _stage6(merge_tables=True),
              workers=_workers_for_base(BEST_BASE), stage="merge_flags"),
    SweepSpec("29_best_merge_figures_by_caption",
              _stage6(merge_figures=True),
              workers=_workers_for_base(BEST_BASE), stage="merge_flags"),
])


# ---------------------------------------------------------------------------
# Stage 7 — reconstruction × expand setting on BEST_BASE.
# Base: BEST_BASE + corresponding TIF mode + BEST_MERGE_*_BY_CAPTION.
#   31: reconstruct=ON, expand=OFF
#   32: reconstruct=ON, expand=ON (multiplier=BEST_EXPAND_MULTIPLIER)
# Compare 31 vs 32 vs Stage-4 winner to pick BEST_RECONSTRUCTION_SETTING
# and BEST_EXPAND_SETTING.
# ---------------------------------------------------------------------------

def _stage7(*, expand: bool):
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
    SweepSpec("31_best_reconstruct_only",
              _stage7(expand=False),
              workers=_workers_for_base(BEST_BASE), stage="reconstruction"),
    SweepSpec("32_best_reconstruct_plus_selected_expand",
              _stage7(expand=True),
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
        "drop_tables_in_top_pts",
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
            (f"{cfg.masking.drop_tables_in_top_pts:.1f}"
             if cfg.masking.drop_tables_in_top_pts > 0 else "OFF"),
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
