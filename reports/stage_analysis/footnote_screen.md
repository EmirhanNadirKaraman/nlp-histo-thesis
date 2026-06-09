# Document-extraction sweep — per-stage analysis tables
## Stage 4 — footnote expansion (family pick)

*Footnote expansion ON at multiplier 1.2, compared across detector families.*

| variant | crop P | crop R | crop F1 | mask F1 | strict F1 | emitted | unlab |
|---|--:|--:|--:|--:|--:|--:|--:|
| 18_hybrid_best_family_fixes_footnote_expand_1_2 ⭐ | 91.9 | 91.9 | 91.9 | 91.9 | **83.8** | 37 | 0 |
| 16_docling_footnote_expand_1_2 | 93.9 | 83.8 | 88.6 | 91.4 | **80.0** | 33 | 0 |
| 17_tatr_best_family_fixes_footnote_expand_1_2 | 91.9 | 91.9 | 91.9 | 91.9 | **64.9** | 37 | 0 |

**Winner: `18_hybrid_best_family_fixes_footnote_expand_1_2`** — table strict F1 83.8%.
