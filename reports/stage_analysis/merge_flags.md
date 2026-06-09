# Document-extraction sweep — per-stage analysis tables
## Stage 6 — caption-merge flags

*Do merge_tables_by_caption / merge_figures_by_caption help?*

| variant | crop P | crop R | crop F1 | mask F1 | strict F1 | emitted | unlab |
|---|--:|--:|--:|--:|--:|--:|--:|
| 29_best_merge_figures_by_caption ⭐ | 91.9 | 91.9 | 91.9 | 91.9 | **83.8** | 37 | 0 |
| 28_best_merge_tables_by_caption | 88.9 | 86.5 | 87.7 | 87.7 | **79.5** | 36 | 0 |

**Winner: `29_best_merge_figures_by_caption`** — table strict F1 83.8%.
