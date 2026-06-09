# Document-extraction sweep — per-stage analysis tables
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
