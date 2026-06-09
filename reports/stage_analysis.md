# Document-extraction sweep — per-stage analysis tables

Source: `reports/figtable_extraction_sweep_rerun_27pdf_20260604_PR.json`  ·  31 variants scored.  
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

| variant | crop P | crop R | crop F1 | mask F1 | strict F1 | emitted | unlab |
|---|--:|--:|--:|--:|--:|--:|--:|
| 08_tatr_099_drop_tables_in_figures ⭐ | 81.4 | 94.6 | 87.5 | 97.6 | **42.5** | 43 | 0 |
| 11_hybrid_099_drop_tables_in_figures | 81.4 | 94.6 | 87.5 | 97.6 | **42.5** | 43 | 0 |
| 13_hybrid_099_drop_plus_premask | 81.4 | 94.6 | 87.5 | 97.6 | **42.5** | 43 | 0 |
| 12_hybrid_099_premask_figures_for_tables | 79.5 | 94.6 | 86.4 | 97.7 | **42.0** | 44 | 0 |
| 10_tatr_099_drop_plus_premask | 81.4 | 94.6 | 87.5 | 97.6 | **40.0** | 43 | 0 |
| 09_tatr_099_premask_figures_for_tables | 79.5 | 94.6 | 86.4 | 97.7 | **39.5** | 44 | 0 |

**Winner: `08_tatr_099_drop_tables_in_figures`** — table strict F1 42.5%.

## Stage 3 — header-band drop

*Does dropping detected tables in the top page band (running-header false positives) help?*

| variant | crop P | crop R | crop F1 | mask F1 | strict F1 | emitted | unlab |
|---|--:|--:|--:|--:|--:|--:|--:|
| 14_tatr_099_header_clip_50pts ⭐ | 94.6 | 94.6 | 94.6 | 94.6 | **45.9** | 37 | 0 |
| 15_hybrid_099_header_clip_50pts | 94.6 | 94.6 | 94.6 | 94.6 | **45.9** | 37 | 0 |

**Winner: `14_tatr_099_header_clip_50pts`** — table strict F1 45.9%.

## Stage 4 — footnote expansion (family pick)

*Footnote expansion ON at multiplier 1.2, compared across detector families.*

| variant | crop P | crop R | crop F1 | mask F1 | strict F1 | emitted | unlab |
|---|--:|--:|--:|--:|--:|--:|--:|
| 18_hybrid_best_family_fixes_footnote_expand_1_2 ⭐ | 91.9 | 91.9 | 91.9 | 91.9 | **83.8** | 37 | 0 |
| 16_docling_footnote_expand_1_2 | 93.9 | 83.8 | 88.6 | 91.4 | **80.0** | 33 | 0 |
| 17_tatr_best_family_fixes_footnote_expand_1_2 | 91.9 | 91.9 | 91.9 | 91.9 | **64.9** | 37 | 0 |

**Winner: `18_hybrid_best_family_fixes_footnote_expand_1_2`** — table strict F1 83.8%.

## Stage 5 — footnote multiplier

*Best footnote_threshold_multiplier per family.*

| variant | crop P | crop R | crop F1 | mask F1 | strict F1 | emitted | unlab |
|---|--:|--:|--:|--:|--:|--:|--:|
| 25_hybrid_best_family_fixes_footnote_expand_1_0 ⭐ | 91.9 | 91.9 | 91.9 | 91.9 | **83.8** | 37 | 0 |
| 26_hybrid_best_family_fixes_footnote_expand_1_1 | 91.9 | 91.9 | 91.9 | 91.9 | **83.8** | 37 | 0 |
| 27_hybrid_best_family_fixes_footnote_expand_1_15 | 91.9 | 91.9 | 91.9 | 91.9 | **83.8** | 37 | 0 |
| 19_docling_footnote_expand_1_0 | 93.9 | 83.8 | 88.6 | 91.4 | **80.0** | 33 | 0 |
| 20_docling_footnote_expand_1_1 | 93.9 | 83.8 | 88.6 | 91.4 | **80.0** | 33 | 0 |
| 21_docling_footnote_expand_1_15 | 93.9 | 83.8 | 88.6 | 91.4 | **80.0** | 33 | 0 |
| 22_tatr_best_family_fixes_footnote_expand_1_0 | 91.9 | 91.9 | 91.9 | 91.9 | **64.9** | 37 | 0 |
| 23_tatr_best_family_fixes_footnote_expand_1_1 | 91.9 | 91.9 | 91.9 | 91.9 | **64.9** | 37 | 0 |
| 24_tatr_best_family_fixes_footnote_expand_1_15 | 91.9 | 91.9 | 91.9 | 91.9 | **64.9** | 37 | 0 |

**Winner: `25_hybrid_best_family_fixes_footnote_expand_1_0`** — table strict F1 83.8%.

## Stage 6 — caption-merge flags

*Do merge_tables_by_caption / merge_figures_by_caption help?*

| variant | crop P | crop R | crop F1 | mask F1 | strict F1 | emitted | unlab |
|---|--:|--:|--:|--:|--:|--:|--:|
| 29_best_merge_figures_by_caption ⭐ | 91.9 | 91.9 | 91.9 | 91.9 | **83.8** | 37 | 0 |
| 28_best_merge_tables_by_caption | 88.9 | 86.5 | 87.7 | 87.7 | **79.5** | 36 | 0 |

**Winner: `29_best_merge_figures_by_caption`** — table strict F1 83.8%.

## Stage 7 — table reconstruction

*Does reconstruct_tables_from_lists help?*

| variant | crop P | crop R | crop F1 | mask F1 | strict F1 | emitted | unlab |
|---|--:|--:|--:|--:|--:|--:|--:|
| 32_best_reconstruct_plus_selected_expand ⭐ | 85.0 | 91.9 | 88.3 | 88.3 | **80.5** | 40 | 0 |
| 31_best_reconstruct_only | 87.5 | 94.6 | 90.9 | 90.9 | **44.2** | 40 | 0 |

**Winner: `32_best_reconstruct_plus_selected_expand`** — table strict F1 80.5%.
