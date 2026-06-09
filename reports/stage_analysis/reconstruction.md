# Document-extraction sweep — per-stage analysis tables
## Stage 7 — table reconstruction

*Does reconstruct_tables_from_lists help?*

| variant | crop P | crop R | crop F1 | mask F1 | strict F1 | emitted | unlab |
|---|--:|--:|--:|--:|--:|--:|--:|
| 32_best_reconstruct_plus_selected_expand ⭐ | 85.0 | 91.9 | 88.3 | 88.3 | **80.5** | 40 | 0 |
| 31_best_reconstruct_only | 87.5 | 94.6 | 90.9 | 90.9 | **44.2** | 40 | 0 |

**Winner: `32_best_reconstruct_plus_selected_expand`** — table strict F1 80.5%.
