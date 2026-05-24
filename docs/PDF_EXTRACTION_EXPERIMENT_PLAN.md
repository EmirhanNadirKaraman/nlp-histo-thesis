# PDF Extraction — Sweep Experiment Plan

The full staged plan for the PDF text-extraction pipeline calibration sweep.
Source of truth for variant configs is
[`scripts/eval/run_all_sweeps.py`](../scripts/eval/run_all_sweeps.py);
this document describes what each stage answers, how variants are wired,
and the order in which the `BEST_*` knobs get frozen.

For run / resume / scoring commands see
[`HOW_TO_RUN.md` §2.2](HOW_TO_RUN.md#22-stage-1-detector--threshold-variants-run_all_sweepspy).

> **Outcome (2026-05-21).** The sweep completed. The final frozen config is
> **hybrid @ 0.99 (variant 18)**, strict F1 74.4 % — *not* the Docling base
> that some per-stage descriptions below assume (`BEST_BASE = 01_docling` and
> "Docling-only" are design-time assumptions, superseded by the full
> variant-matrix result). For the actual outcome see
> [`thesis/12_final_config.md`](thesis/12_final_config.md) and
> [`PDF_EXTRACTION_RESULTS.md`](PDF_EXTRACTION_RESULTS.md); the `BEST_*` knobs
> are now baked into `pipeline/stages/pdf_text_extraction/config.py` defaults.

---

## Overview

Seven sub-stages run in order.  Each stage builds on previous stages'
winners by editing the matching module-level constant at the top of
`scripts/eval/run_all_sweeps.py`.  Variants explicitly set every relevant
flag — no reliance on `PipelineConfig` defaults — so downstream
default changes can never silently shift the experiment baseline.

| Stage | Slug | Variants | Constants set after this stage |
|---|---|---|---|
| 1.1 | `detector_docling` | 01 | `STAGE1_BASE_DOCLING` |
| 1.2 | `detector_tatr` | 02–04 | `STAGE1_BASE_TATR` |
| 1.3 | `detector_hybrid` | 05–07 | `STAGE1_BASE_HYBRID` |
| 2 | `table_in_figure` | 08–13 | `BEST_TATR_TABLE_IN_FIGURE_MODE`, `BEST_HYBRID_TABLE_IN_FIGURE_MODE` |
| 3 | `header_fix` | 14–15 | `BEST_TATR_HEADER_ZONE_PTS`, `BEST_HYBRID_HEADER_ZONE_PTS` (Docling skipped — no header-band FPs) |
| 4 | `footnote_screen` | 16–18 | `BEST_BASE` |
| 5 | `footnote_multiplier` | 19–27 | `BEST_*_EXPAND_MULTIPLIER` per family — 3 multipliers × 3 families (Docling 19–21, TATR 22–24, Hybrid 25–27).  May trigger a `BEST_BASE` re-pick if a non-Docling family's best multiplier beats Docling's. |
| 6 | `merge_flags` | 28–30 | `BEST_MERGE_TABLES_BY_CAPTION`, `BEST_MERGE_FIGURES_BY_CAPTION` (variant 30 = docling-specific drop_tables_inside_figures check) |
| 7 | `reconstruction` | 31–32 | `BEST_RECONSTRUCTION_SETTING`, `BEST_EXPAND_SETTING` |

Pinned constants (no sweep):

* `BEST_TWO_PASS = True` — ghost-text safety (2026-05-13 decision; the
  crop rubric is blind to two-pass effects).
* `BEST_EXPAND_MULTIPLIER = 1.2` — validated in an earlier footnote
  screen (94% missed-footnote reduction; raising it risks crop
  overshoot).

Reserved / unused variant IDs: none in the current numbering.  The old
21/22/23 `figure_premask` stage was removed when the
`drop_tables_inside_figures` ↔ `mask_figures_before_table_detection`
question moved into Stage 2's table_in_figure mode dial.

### Global default per variant (set explicitly in every `configure()`)

```
two_pass                          = ON
render_dpi                        = 150
reconstruct_tables_from_lists     = OFF
expand_tables_with_footnotes      = OFF
footnote_threshold_multiplier     = 1.2 (only used when expand=ON)
merge_tables_by_caption           = OFF
merge_figures_by_caption          = OFF
drop_tables_inside_figures        = OFF
mask_figures_before_table_detection = OFF
```

Each stage overrides exactly the flags relevant to its decision.

---

## Stage 1.1 — `detector_docling` (Docling baseline)

```
01_docling   detector=docling, tatr_threshold=n/a, all helpers OFF
```

Locks `STAGE1_BASE_DOCLING` (one variant, but kept as its own stage so
the docling-only run can be triggered independently of TATR / Hybrid).

---

## Stage 1.2 — `detector_tatr` (TATR threshold selection)

```
02_tatr_090   detector=tatr,   tatr_threshold=0.90
03_tatr_095   detector=tatr,   tatr_threshold=0.95
04_tatr_099   detector=tatr,   tatr_threshold=0.99
```

Locks `STAGE1_BASE_TATR`.

---

## Stage 1.3 — `detector_hybrid` (Hybrid threshold selection)

```
05_hybrid_090   detector=hybrid, tatr_threshold=0.90
06_hybrid_095   detector=hybrid, tatr_threshold=0.95
07_hybrid_099   detector=hybrid, tatr_threshold=0.99
```

Locks `STAGE1_BASE_HYBRID`.

---

## Stage 2 — `table_in_figure` (TATR / Hybrid only)

Sweep `drop_tables_inside_figures` × `mask_figures_before_table_detection`
on the per-family Stage-1 winner.  Docling is excluded: `premask` is a
no-op for Docling because `DoclingTableDetector.detect_from_layout`
reads from the Step-1 LayoutResult, not the masked PDF
(see `runner.py:651-653`).  For Docling, `drop` is the only flag that
behaves at all and it's covered implicitly by `mode = "none"` (the
Stage-1 baseline already has both flags OFF).

```
08_tatr_099_drop_tables_in_figures        TATR base + drop=ON
09_tatr_099_premask_figures_for_tables    TATR base + premask=ON
10_tatr_099_drop_plus_premask             TATR base + drop=ON + premask=ON
11_hybrid_099_drop_tables_in_figures      Hybrid base + drop=ON
12_hybrid_099_premask_figures_for_tables  Hybrid base + premask=ON
13_hybrid_099_drop_plus_premask           Hybrid base + drop=ON + premask=ON
```

Locks `BEST_TATR_TABLE_IN_FIGURE_MODE` and
`BEST_HYBRID_TABLE_IN_FIGURE_MODE` — each ∈
`{"none", "drop", "premask", "drop_plus_premask"}`.

---

## Stage 3 — `footnote_screen` (footnote expansion per detector family)

Each variant: that family's Stage-1 base + that family's Stage-2 TIF
mode + `expand_tables_with_footnotes = ON`, multiplier = 1.2.

```
14_docling_footnote_expand_1_2                 docling base, mode=none
15_tatr_best_tif_fix_footnote_expand_1_2       TATR base + BEST_TATR_TABLE_IN_FIGURE_MODE
16_hybrid_best_tif_fix_footnote_expand_1_2     Hybrid base + BEST_HYBRID_TABLE_IN_FIGURE_MODE
```

Compare 14 / 15 / 16 to pick the best detector family **under footnote
expansion + best TIF fix**.  Locks `BEST_BASE` (one of the seven Stage-1
base names).

---

## Stage 4 — `footnote_multiplier` (multiplier sweep on BEST_BASE — Docling-only)

After Stage 3 picks `BEST_BASE`, sweep the footnote-cascade multiplier
to balance footnote recall against `crop too big` over-expansion.
All three variants run on the Docling base (`BEST_BASE = 01_docling` is
the current Stage-3 winner; if a future Stage 3 picks a different
family, refactor this stage to use `BEST_BASE` instead of hard-coded
docling).

```
17_docling_footnote_expand_1_0     multiplier=1.00 (no cascade growth)
18_docling_footnote_expand_1_1     multiplier=1.10
19_docling_footnote_expand_1_15    multiplier=1.15
```

Compare 17 / 18 / 19 / 14 (1.2) to pick `BEST_EXPAND_MULTIPLIER`.

---

## Stage 5 — `merge_flags` (single-flag flips on BEST_BASE)

Base: `BEST_BASE` + corresponding TIF mode + expand=ON, multiplier=1.2.
Each variant flips exactly one extra flag.

```
20_best_merge_tables_by_caption     + merge_tables_by_caption = ON
21_best_merge_figures_by_caption    + merge_figures_by_caption = ON
22_best_drop_tables_inside_figures  + drop_tables_inside_figures = ON
```

Locks `BEST_MERGE_TABLES_BY_CAPTION` and `BEST_MERGE_FIGURES_BY_CAPTION`
independently — each flag picked if it clearly wins on its own.

Variant 19 forces `drop_tables_inside_figures = ON` regardless of the
family's Stage-2 TIF mode.  Meaningful only when `BEST_BASE` is the
Docling family (where the family TIF mode is otherwise locked at
`"none"` because Stage 2 skipped Docling).  When `BEST_BASE` is TATR
or Hybrid, the family TIF mode from Stage 2 may already be `"drop"`,
making variant 19 a duplicate of variant 15 / 16 — note and skip
analysis if so.  See [`docs/BUGS.md`](BUGS.md#bug-58--drop_tables_inside_figures-bypassed-by-cropper-supplementary-source)
for the B-058 fix that made this flag actually work on Docling.

---

## Stage 6 — `reconstruction` (reconstruction × expand on BEST_BASE)

Base: `BEST_BASE` + corresponding TIF mode + selected merge flags +
`reconstruct_tables_from_lists = ON`.

```
23_best_reconstruct_only                  expand=OFF
24_best_reconstruct_plus_selected_expand  expand=ON, ftn_x=BEST_EXPAND_MULTIPLIER
```

Locks `BEST_RECONSTRUCTION_SETTING` and `BEST_EXPAND_SETTING` — the
final production knobs frozen into `PipelineConfig` defaults.

---

## Walking through the plan

```bash
# Stage 1.1 — Docling baseline.
python scripts/eval/run_all_sweeps.py --stage detector_docling

# Stage 1.2 — TATR threshold sweep.  Pick STAGE1_BASE_TATR.
python scripts/eval/run_all_sweeps.py --stage detector_tatr

# Stage 1.3 — Hybrid threshold sweep.  Pick STAGE1_BASE_HYBRID.
python scripts/eval/run_all_sweeps.py --stage detector_hybrid

# Stage 2 — TATR/Hybrid table_in_figure fixes.  Pick the two BEST_*_TIF modes.
python scripts/eval/run_all_sweeps.py --stage table_in_figure

# Stage 3 — per-family footnote expansion.  Pick BEST_BASE.
python scripts/eval/run_all_sweeps.py --stage footnote_screen

# Stage 4 — merge flags on BEST_BASE.  Pick BEST_MERGE_*.
python scripts/eval/run_all_sweeps.py --stage merge_flags

# Stage 5 — reconstruction × expand on BEST_BASE.  Pick BEST_RECONSTRUCTION_SETTING + BEST_EXPAND_SETTING.
python scripts/eval/run_all_sweeps.py --stage reconstruction

# Or run everything end-to-end (e.g. after all knobs are frozen):
python scripts/eval/run_all_sweeps.py --stage all

# No --stage? Prints the menu (stages, blurbs, variant names) and exits.
python scripts/eval/run_all_sweeps.py
```

`--stage <name> --list-variants` confirms the resolved config for each
variant after each constant edit.  The printed columns are: `variant`,
`stage`, `detector`, `tatr`, `two_pass`, `recon`, `expand`, `ftn_x`,
`merge_tables`, `merge_figures`, `drop_tables_inside_figures`,
`premask_figures_before_table_detection`, `out_dir`.

Caching is per-variant (independent of `--stage`), so picking up later —
e.g. `--stage footnote_screen` after `--stage detector_tatr` — reuses
every `_DONE.json` marker and per-stage cache already on disk.  See
[`HOW_TO_RUN.md` §2.2](HOW_TO_RUN.md#22-stage-1-detector--threshold-variants-run_all_sweepspy)
for the full resume / checkpoint semantics.

---

## Freezing the production config

Once every constant is set from scored data, follow the
"Pipeline runner — freeze BEST_* defaults after sweep" TODO in
[`THESIS.md`](THESIS.md#todos) to bake the winners into
`pipeline/stages/pdf_text_extraction/config.py`'s defaults and drop
the matching `_apply_stage1_baseline` overrides from
`run_all_sweeps.py`.

---

## See also

- [`HOW_TO_RUN.md` §2.2](HOW_TO_RUN.md#22-stage-1-detector--threshold-variants-run_all_sweepspy) — runtime / resume / scoring commands
- [`THESIS.md`](THESIS.md) — TODOs + decisions log
- [`STRUCTURE.md`](STRUCTURE.md) — pipeline architecture index
- [`scripts/eval/run_all_sweeps.py`](../scripts/eval/run_all_sweeps.py) — variant config source of truth
- [`scripts/eval/score_pdf_variants.py`](../scripts/eval/score_pdf_variants.py) — per-variant P/R/F1 scorer
