# Document-extraction sweep — per-stage analysis tables
## Stage 3 — header-band drop

*Does dropping detected tables in the top page band (running-header false positives) help?*

| variant | crop P | crop R | crop F1 | mask F1 | strict F1 | emitted | unlab |
|---|--:|--:|--:|--:|--:|--:|--:|
| 14_tatr_099_header_clip_50pts ⭐ | 94.6 | 94.6 | 94.6 | 94.6 | **45.9** | 37 | 0 |
| 15_hybrid_099_header_clip_50pts | 94.6 | 94.6 | 94.6 | 94.6 | **45.9** | 37 | 0 |

**Winner: `14_tatr_099_header_clip_50pts`** — table strict F1 45.9%.
