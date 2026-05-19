# PDF Extraction — Sweep Experiment Plan

The full staged plan for the PDF text-extraction pipeline calibration sweep.
Source of truth for variant configs is
[`scripts/eval/run_all_sweeps.py`](../scripts/eval/run_all_sweeps.py);
this document describes what each stage answers, how variants are wired,
and the order in which the `BEST_*` knobs get frozen.

For run / resume / scoring commands see
[`HOW_TO_RUN.md` §2.2](HOW_TO_RUN.md#22-stage-1-detector--threshold-variants-run_all_sweepspy).

---

## Overview

Seven stages run in order.  Each stage builds on the previous stage's
"winner" by editing the matching `BEST_*` module constant at the top of
`scripts/eval/run_all_sweeps.py`:

| Stage | Slug | Variants | Knob set after this stage | Drives |
|---|---|---|---|---|
| 1 | `detector` | 01–07 | `BEST_BASE` | Stages 2–7 detector/threshold |
| 2 | `footnote_screen` | 08–10 | (validates `BEST_BASE`) | confirms detector choice under footnote expansion |
| 3 | `footnote_tuning` | 11–13 | `BEST_EXPAND_MULTIPLIER` | Stages 4–7 footnote multiplier |
| 4 | `two_pass` | 14–15 | `BEST_TWO_PASS` | Stages 5–7 two-pass flag |
| 5 | `merge_drop` | 16–18 | `BEST_STAGE4` (dict) | Stages 6–7 kept merge/drop flags |
| 6 | `reconstruction` | 19–20 | `BEST_EXPAND_SETTING` | Stage 7 expand on/off |
| 7 | `figure_premask` | 21–23 | — | (terminal) |

Stage 7 auto-skips when `BEST_BASE == "01_docling"` — pre-masking
targets pixel detectors (TATR / Hybrid), so it's a no-op for
Docling-only.

Implicit run-wide defaults (matching
`run_all_sweeps.py::_apply_stage1_baseline`) unless a variant flips
them:

```
two_pass = ON                          render_dpi = 150
reconstruct_tables_from_lists = OFF    merge_tables_by_caption = OFF
expand_tables_with_footnotes = OFF     merge_figures_by_caption = OFF
drop_tables_inside_figures = OFF       mask_figures_before_table_detection = OFF
```

---

## Stage 1 — `detector` (detector / TATR threshold selection)

What this stage answers:

- Should the base detector be **Docling**, **TATR**, or **Hybrid**?
- If TATR or Hybrid, what TATR threshold (`0.90` / `0.95` / `0.99`)
  gives the best precision / recall tradeoff?

Stage 1 deliberately keeps every helper flag OFF
(`reconstruct_tables_from_lists`, `merge_tables_by_caption`,
`merge_figures_by_caption`, `expand_tables_with_footnotes`,
`drop_tables_inside_figures`) so the comparison isolates the detector.
Two-pass extraction is ON and `render_dpi` is `150` across all seven
variants.

```
01_docling     — detector=docling
02_tatr_090    — detector=tatr,   tatr_threshold=0.90
03_tatr_095    — detector=tatr,   tatr_threshold=0.95
04_tatr_099    — detector=tatr,   tatr_threshold=0.99
05_hybrid_090  — detector=hybrid, tatr_threshold=0.90
06_hybrid_095  — detector=hybrid, tatr_threshold=0.95
07_hybrid_099  — detector=hybrid, tatr_threshold=0.99
```

The best of each family was selected: 

- docling
- tatr_099
- hybrid_099

---

## Stage 2 — `footnote_screen` (does footnote expansion shift the top detector ranking?)
Stage 2 tests whether footnote expansion changes the detector-family ranking, using the best threshold for each detector family selected in Stage 1.

Runs the top-of-bracket variant from each detector family with footnote
expansion ON at the default multiplier (1.2), to confirm that the
Stage 1 detector ranking holds under expansion.

```
08_docling_footnote_expand_1_2     base=01_docling,    expand=ON, ftn_x=1.2
09_tatr_099_footnote_expand_1_2    base=04_tatr_099,   expand=ON, ftn_x=1.2
10_hybrid_099_footnote_expand_1_2  base=07_hybrid_099, expand=ON, ftn_x=1.2
```

If the ranking moves, revisit `BEST_BASE` before moving on.

---

## Stage 3 — `footnote_tuning` (tune `footnote_threshold_multiplier` on `BEST_BASE`)

```
11_best_footnote_expand_1_2        BEST_BASE + expand=ON, ftn_x=1.2
12_best_footnote_expand_1_3        BEST_BASE + expand=ON, ftn_x=1.3
13_best_footnote_expand_1_5        BEST_BASE + expand=ON, ftn_x=1.5
```

After this, pick `BEST_EXPAND_MULTIPLIER`.

---

## Stage 4 — `two_pass` ablation

```
14_best_twopass_on                  BEST_BASE + expand + BEST_EXPAND_MULTIPLIER + two_pass=ON
15_best_twopass_off                 BEST_BASE + expand + BEST_EXPAND_MULTIPLIER + two_pass=OFF
```

After this, pick `BEST_TWO_PASS`.

---

## Stage 5 — `merge_drop` (independent merge/drop flag ablations)

Each variant runs on `BEST_BASE + expand + BEST_EXPAND_MULTIPLIER +
BEST_TWO_PASS` and flips exactly one extra flag.  Do not combine yet —
pick winners into `BEST_STAGE4` only if one clearly helps.

```
16_best_merge_tables_by_caption     + merge_tables_by_caption = ON
17_best_merge_figures_by_caption    + merge_figures_by_caption = ON
18_best_drop_tables_in_figures      + drop_tables_inside_figures = ON
```

---

## Stage 6 — `reconstruction` interaction

Builds on `BEST_BASE + BEST_TWO_PASS + BEST_STAGE4` plus
`reconstruct_tables_from_lists = ON`:

```
19_best_reconstruct_only                  expand=OFF
20_best_reconstruct_plus_selected_expand  expand=ON, ftn_x=BEST_EXPAND_MULTIPLIER
```

After this, pick `BEST_EXPAND_SETTING` (whether the final
`expand_tables_with_footnotes` should stay ON in production).

---

## Stage 7 — `figure_premask` (pre-mask figures before table detection)

Skipped automatically if `BEST_BASE == "01_docling"` (pre-masking only
affects pixel-based detection).  Base is `BEST_BASE + selected expand +
BEST_TWO_PASS + BEST_STAGE4`.

```
21_best_drop_only_for_premask_control   drop=ON,  premask=OFF
22_best_premask_figures_for_tables      drop=OFF, premask=ON
23_best_drop_plus_premask_figures       drop=ON,  premask=ON
```

`21` re-runs Stage 5's `drop_tables_inside_figures` flag as a control
under the final Stage 6 base, so the pre-mask comparison is fair.

---

## Walking through the plan

```bash
# Stage 1 — label + score + pick BEST_BASE → edit constant.
python scripts/eval/run_all_sweeps.py --stage detector

# Stage 2 — top-of-bracket footnote screen for each detector family.
python scripts/eval/run_all_sweeps.py --stage footnote_screen

# Stage 3 → set BEST_EXPAND_MULTIPLIER.
python scripts/eval/run_all_sweeps.py --stage footnote_tuning

# Stage 4 → set BEST_TWO_PASS.
python scripts/eval/run_all_sweeps.py --stage two_pass

# Stage 5 → fold winners into BEST_STAGE4.
python scripts/eval/run_all_sweeps.py --stage merge_drop

# Stage 6 → set BEST_EXPAND_SETTING.
python scripts/eval/run_all_sweeps.py --stage reconstruction

# Stage 7 (TATR/Hybrid only; auto-skipped if BEST_BASE == "01_docling").
python scripts/eval/run_all_sweeps.py --stage figure_premask

# Or run everything end-to-end (e.g. after all BEST_* are frozen):
python scripts/eval/run_all_sweeps.py --stage all

# No --stage? Prints the menu (stages, blurbs, variant names) and exits.
python scripts/eval/run_all_sweeps.py
```

`--stage <name> --list-variants` confirms the resolved config for that
stage's variants after each `BEST_*` edit.

Caching is per-variant (independent of `--stage`), so picking up later —
e.g. `--stage footnote_tuning` after `--stage detector` — reuses every
`_DONE.json` marker and per-stage cache already on disk.  See
[`HOW_TO_RUN.md` §2.2](HOW_TO_RUN.md#22-stage-1-detector--threshold-variants-run_all_sweepspy)
for the full resume / checkpoint semantics.

---

## Freezing the production config

Once every `BEST_*` knob is set from scored data, follow the
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
