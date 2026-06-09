# Document-extraction sweep — per-stage analysis tables

Source: `reports/figtable_extraction_sweep_rerun_27pdf_20260604_PR.json`  ·  7 variants scored.  
Tables show the **table** rubric (figure cropping is detector-invariant). ⭐ marks each stage's leading variant; the deciding metric is **bold**.

_Figures are detector-invariant across all variants: crop F1 89.9, mask F1 100.0, strict F1 84.0, 76 emitted._

## Stage 1 — detector / threshold selection

*Which table detector (Docling / TATR / Hybrid) and TATR threshold crops tables best.*

| variant | crop P | crop R | crop F1 | mask F1 | strict F1 | emitted | unlab |
|---|--:|--:|--:|--:|--:|--:|--:|
| 01_docling ⭐ | 97.0 | 86.5 | **91.4** | 94.3 | 40.0 | 33 | 0 |
| 04_tatr_099 | 79.5 | 94.6 | **86.4** | 97.7 | 42.0 | 44 | 0 |
| 07_hybrid_099 | 79.5 | 94.6 | **86.4** | 97.7 | 42.0 | 44 | 0 |
| 02_tatr_090 | 75.0 | 97.3 | **84.7** | 94.5 | 40.0 | 48 | 0 |
| 05_hybrid_090 | 75.0 | 97.3 | **84.7** | 94.5 | 40.0 | 48 | 0 |
| 03_tatr_095 | 74.5 | 94.6 | **83.3** | 94.4 | 40.5 | 47 | 0 |
| 06_hybrid_095 | 74.5 | 94.6 | **83.3** | 94.4 | 40.5 | 47 | 0 |

**Winner: `01_docling`** — table crop F1 91.4%.

## Stage 2 — table-in-figure handling

*Does dropping/pre-masking tables that fall inside figure regions help?*

_No scored variants for this stage yet (run the stage and re-score)._

## Stage 3 — header-band drop

*Does dropping detected tables in the top page band (running-header false positives) help?*

_No scored variants for this stage yet (run the stage and re-score)._

## Stage 4 — footnote expansion (family pick)

*Footnote expansion ON at multiplier 1.2, compared across detector families.*

_No scored variants for this stage yet (run the stage and re-score)._

## Stage 5 — footnote multiplier

*Best footnote_threshold_multiplier per family.*

_No scored variants for this stage yet (run the stage and re-score)._

## Stage 6 — caption-merge flags

*Do merge_tables_by_caption / merge_figures_by_caption help?*

_No scored variants for this stage yet (run the stage and re-score)._

## Stage 7 — table reconstruction

*Does reconstruct_tables_from_lists help?*

_No scored variants for this stage yet (run the stage and re-score)._
