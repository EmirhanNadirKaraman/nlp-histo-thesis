# Document-extraction sweep — per-stage analysis tables
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
