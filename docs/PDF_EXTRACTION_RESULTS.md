# PDF Extraction — Sweep Results

> **Final config = `variant 18`: HYBRID @ 0.99, strict F1 74.4 %** (2026-05-21,
> baked into `pipeline/stages/pdf_text_extraction/config.py`). The per-stage
> "`BEST_BASE = 01_docling`" verdicts below are the **2026-05-20 intermediate**
> run (docling, 71.6 %); the completed matrix flipped the winner to hybrid. See
> [§ Final winning config](#final-winning-config) and the 2026-05-21
> Decisions-log entry in [`THESIS.md`](THESIS.md#decisions-log).

Empirical results from `scripts/eval/run_all_sweeps.py`.  Companion to
[`PDF_EXTRACTION_EXPERIMENT_PLAN.md`](PDF_EXTRACTION_EXPERIMENT_PLAN.md)
(stage plan) and
[`HOW_TO_RUN.md` §2.2](HOW_TO_RUN.md#22-stage-1-detector--threshold-variants-run_all_sweepspy)
(run / resume commands).

Corpus: 28 PDFs in `eval/pdfs/`.  Scoring rubric:
[`eval/label_rubric.yaml`](../eval/label_rubric.yaml).

Each stage's verdict drives the matching `BEST_*` constant at the top
of `scripts/eval/run_all_sweeps.py`.

---

## Stage 1 — `detector` (Stage 1 winner → `BEST_BASE`)

Detector / TATR-threshold selection.  All helper flags OFF; two-pass ON;
`render_dpi=150`.

**Tables** (figures are detector-invariant — identical 88.9% crop F1
across all 7 variants):

| variant | crop F1 | crop P | crop R | mask F1 | strict F1 |
|---|---|---|---|---|---|
| **`01_docling`** | **91.4%** | **97.4%** | 86.0% | 93.8% | 34.6% |
| `02_tatr_090` | 85.7% | 76.4% | 97.7% | 92.2% | 36.7% |
| `03_tatr_095` | 84.5% | 75.9% | 95.3% | 92.0% | 37.1% |
| `04_tatr_099` | 87.2% | 80.4% | 95.3% | 94.8% | **38.3%** |
| `05_hybrid_090` | 83.3% | 75.5% | 93.0% | 94.0% | 35.4% |
| `06_hybrid_095` | 82.1% | 75.0% | 90.7% | 93.9% | 35.8% |
| `07_hybrid_099` | 84.8% | 79.6% | 90.7% | **96.8%** | 37.0% |

Within each detector family, the tightest TATR threshold (`0.99`) wins
crop F1 monotonically.

**Verdict:** `BEST_BASE = "01_docling"`.  Wins crop F1 by ~4 pp on the
biggest margin in the table; precision lead is ~17 pp.  Trade-off: lowest
recall of the seven (86.0%).

---

## Stage 2 — `footnote_screen` (validates `BEST_BASE` under footnote expansion)

Top-of-bracket from each detector family with
`expand_tables_with_footnotes = ON`, `footnote_threshold_multiplier = 1.2`.

### Crop F1

| base | no expand (S1) | with expand (S2) | Δ |
|---|---|---|---|
| docling (`01` → `08`) | 91.4% | 88.9% | −2.5 |
| tatr_099 (`04` → `09`) | 87.2% | 83.0% | −4.2 |
| hybrid_099 (`07` → `10`) | 84.8% | 81.7% | −3.1 |

Small drop — expansion occasionally overshoots into "crop too big
minor / major".

### Strict F1 — biggest signal of this stage

Strict F1 requires every dimension (crop + caption + footnote + mask)
correct.

| base | no expand | with expand | Δ |
|---|---|---|---|
| docling | 34.6% | **71.6%** | **+37.0** |
| tatr_099 | 38.3% | 55.3% | +17.0 |
| hybrid_099 | 37.0% | 68.8% | +31.8 |

### Footnote precision (per-detected-item)

| base | no expand | with expand | Δ |
|---|---|---|---|
| docling | 48.6% | **91.9%** | +43.3 |
| tatr_099 | 53.7% | 75.0% | +21.3 |
| hybrid_099 | 51.3% | **92.3%** | +41.0 |

### Missing-footnote labels (rubric-tag count, lower = better)

| base | Stage 1 | Stage 2 | reduction |
|---|---|---|---|
| docling | 17 | **1** | −94% |
| tatr_099 | 17 | 8 | −53% |
| hybrid_099 | 17 | **1** | −94% |

### All footnote-related labels

(`missing footnotes` + `wrong caption (footnotes matched to captions, rotated image)`)

| base | Stage 1 | Stage 2 | % of crops affected |
|---|---|---|---|
| docling | 19 | 3 | 50% → **8%** |
| tatr_099 | 19 | 10 | 37% → 20% |
| hybrid_099 | 19 | 3 | 38% → **6%** |

The 2 `wrong caption (..., rotated image)` labels are identical across
all variants — rotated-image footnote attachment is a separate
unfixed bug, not addressable by `expand_tables_with_footnotes`.

### Verdicts

1. **`BEST_BASE = "01_docling"` confirmed** — docling still tops crop F1
   *and* strict F1 under expansion.  Family ranking did not shift.
2. **`expand_tables_with_footnotes = ON` is a major win** — strict F1
   roughly doubles for docling, footnote precision nearly doubles.
   Lock on.
3. **TATR is the laggard** — still 8 missed footnotes vs 1 for
   docling/hybrid.  TATR-detected bboxes presumably extend close enough
   to the table edge that the default `footnote_proximity_pts=20` doesn't
   absorb every cascading footnote.  Not pursued — docling is the
   selected base.

### Stage 3 (`footnote_tuning`) skipped

The original Stage 3 (variants `11`/`12`/`13`, multiplier ∈ {1.2, 1.3, 1.5})
was removed on 2026-05-20.  Docling at 1.2 already eliminated 94% of
missed footnotes (17 → 1); the remaining 1 miss plus 2 rotated-image
labels are not addressable by raising the multiplier.  Higher multipliers
only enlarge the crop and risk "crop too big" labels.
`BEST_EXPAND_MULTIPLIER` is pinned at `1.2`.  See the 2026-05-20 row in
[`THESIS.md` Decisions log](THESIS.md#decisions-log).

### Stage 3 (`two_pass`) inconclusive — stage removed

A second Stage-3 attempt swept `two_pass` on/off (variants `14`/`15`).
Variant 15 (two_pass=OFF) scored **identically** to variant 08
(two_pass=ON) across every crop metric — same figures emitted (80),
same tables (38), same crop F1, strict F1, footnote precision.  The
rubric scores figure/table crops; two_pass operates on body-text nodes
via the R1 pixel rule and per-page header masking (see
`pipeline/stages/pdf_text_extraction/components/two_pass_extractor.py`
docstring), so the crop sweep has no signal to measure.

Stage removed 2026-05-20.  `BEST_TWO_PASS = True` stands on the
2026-05-13 ghost-text evidence (`scripts/verify_ghost_text_detection.py`),
not on this sweep.  Variant IDs 14/15 reserved and unused.  See the
2026-05-20 Decisions-log row in
[`THESIS.md`](THESIS.md#decisions-log).

---

## Stage 3 — `merge_drop` (single-flag flips on top of variant 08)

Three independent flag flips, each scored against variant 08 (the base
config with all merge/drop flags OFF).

| Variant | Crop F1 | Strict F1 | Tables emitted | Δ vs 08 |
|---|---|---|---|---|
| `08` (baseline) | 88.9% | 71.6% | 38 | — |
| `16` `merge_tables_by_caption` | **85.0%** | **67.5%** | 37 | regression |
| `17` `merge_figures_by_caption` | 88.9% | 71.6% | 38 | no effect |
| `18` `drop_tables_inside_figures` | 88.9% | 71.6% | 38 | no effect |

Figures stayed identical (88.9% / 83.2%) across all four — none of the
flags affect figure detection on this corpus.

### Per-flag verdicts

* **`merge_tables_by_caption` (16)** — actively hurts.  −3.9pp crop F1,
  −4.1pp strict F1.  Tables 38 → 37, and the merged crop becomes an FP
  (TP 29 → 27, FN 14 → 16).  Heuristic wrongly merges distinct tables
  that happen to share a caption stub.  **Lock False.**
* **`merge_figures_by_caption` (17)** — zero impact.  No figure-pair
  shares a caption on this 28-PDF corpus, so nothing merged.
  **Lock False (no signal to flip).**
* **`drop_tables_inside_figures` (18)** — zero impact.  No detected
  table lives inside a figure bbox on this corpus.
  **Lock False (no signal to flip).**

### No combinations tested

17 and 18 are null on this corpus, so combining them with anything
yields the same result as the singleton.  16 regresses on its own and
combining only inherits the regression.

### Outcome

`BEST_STAGE3 = {merge_tables_by_caption: False, merge_figures_by_caption: False,
drop_tables_inside_figures: False}` — every flag stays at its
`_apply_stage1_baseline` default.  See the 2026-05-20 row in
[`THESIS.md` Decisions log](THESIS.md#decisions-log).

---

## Stage 4 — `reconstruction` (does `reconstruct_tables_from_lists` add real tables?)

Two variants on top of variant 08, both with
`reconstruct_tables_from_lists = ON`.

| Variant | Crop F1 | Strict F1 | TP / FP / FN | Tables emitted | Footnote P |
|---|---|---|---|---|---|
| `08` (baseline, recon OFF, expand ON) | **88.9%** | **71.6%** | 29 / 9 / 14 | 38 | 91.9% |
| `19_best_reconstruct_only` (expand OFF) | 88.4% | **34.9%** | 15 / 28 / 28 | 43 | 50.0% |
| `20_best_reconstruct_plus_selected_expand` (expand ON) | 86.0% | 69.8% | 30 / 13 / 13 | 43 | 92.3% |

### Per-variant read

**Variant 19 — disaster (−36.7pp strict F1).**  Reconstruction adds five
new tables, but turning expand OFF collapses footnote precision
91.9% → 50.0% — every reconstructed table loses its footnotes too.
This is the experiment that proves `expand_tables_with_footnotes = ON`
is non-negotiable.  Recall climbs (+4.7pp on crop, +4.7pp on mask) but
strict F1 plummets because crops fail the footnote dim.

**Variant 20 — marginal regression (−1.8pp strict F1).**  Reconstruction
+ expand gives a clean test of whether reconstruction adds real tables.
The five new tables decompose as:

* +1 TP — one real table recovered that docling missed
* +4 FP — four list-like regions misclassified as tables
* −1 FN — the recovered TP offsets one prior miss

Net precision drops (94.7% → 86.0% crop, 97.4% → 88.4% mask); recall
gains (+2.3pp) don't compensate.  Reconstruction adds mostly noise on
this corpus.

### Verdicts

1. **`reconstruct_tables_from_lists` — lock OFF.**  4-FP/1-TP trade is
   not worth it; the +2.3pp recall is bought at −8.7pp precision.
2. **`BEST_EXPAND_SETTING = True` — confirmed.**  Variant 19's collapse
   is the empirical proof.
3. **Matrix extended on 2026-05-21.**  The run documented above concluded on
   the *docling* base. The matrix was then **completed** with the TATR/hybrid
   families (Stages 3–7), which **flipped the winner to hybrid** — see below.

### Final winning config

> **Two distinct results — do not conflate.** Everything above is the
> **2026-05-20 *intermediate*** run on the docling base. The **2026-05-21
> completed matrix** is the authoritative final, and it is what is frozen in
> the code.

**Intermediate winner (2026-05-20, superseded):**
`08_docling_footnote_expand_1_2` — `table_detector = docling`, **strict F1 = 71.6 %**.

**Final winner (2026-05-21, baked into `config.py` — authoritative): `variant 18`**

```
table_detector                              = HYBRID    # Docling + TATR, merged
tatr.threshold                              = 0.99
two_pass.enabled                            = True
cropping.expand_tables_with_footnotes       = True
cropping.footnote_threshold_multiplier      = 1.2
cropping.merge_tables_by_caption            = False
cropping.merge_figures_by_caption           = False
masking.drop_tables_inside_figures          = True
masking.drop_tables_in_top_pts              = 50.0
docling.reconstruct_tables_from_lists       = False
masking.mask_figures_before_table_detection = False
```

**Strict F1 = 74.4 %** (+2.8 pp over the docling intermediate). These defaults
are now frozen in `pipeline/stages/pdf_text_extraction/config.py` (verified
against the code); see the **2026-05-21** entry in the
[`THESIS.md`](THESIS.md#decisions-log) Decisions log and
`reports/stage4_PR.md … stage7_PR.md` for the full per-stage reasoning. The
earlier "freeze these into config.py" TODO is closed.

---

## How to reproduce these reports

All commands run from the repo root.  `caffeinate -dimsu` keeps the Mac
awake for long sweeps; drop it on Linux / shorter runs.  Use `python3`
explicitly to avoid macOS shell-alias resolution falling back to system
Python 3.9, which lacks `dataclass(slots=True)`.

### 0. Once per sweep — rebuild the share map after a new sweep finishes

`scripts/eval/run_all_sweeps.py` auto-runs this in its post-run tail.
Manual invocation if you skipped it (`--skip-share-map`):

```bash
python3 scripts/eval/build_share_map.py
python3 scripts/eval/seed_variant_labels.py
```

If new variants were sweeped but their per-variant label files already
exist (e.g. with a stray manual entry), shared labels from peer Stage 1
variants don't backfill automatically — `annotate.py`'s propagation only
writes forward at label time.  One-shot fix:

```bash
python3 scripts/eval/backfill_shared_labels.py
# scope to a single variant:
python3 scripts/eval/backfill_shared_labels.py --variant 15_best_twopass_off
# preview without writing:
python3 scripts/eval/backfill_shared_labels.py --dry-run
```

### 1. Stage 1 — `detector`

```bash
# Run all 7 detector / threshold variants.
caffeinate -dimsu python3 scripts/eval/run_all_sweeps.py --stage detector

# Label (per-variant; share-map auto-propagates figures + shared bboxes).
for v in 01_docling 02_tatr_090 03_tatr_095 04_tatr_099 \
         05_hybrid_090 06_hybrid_095 07_hybrid_099; do
  # tables_full for TATR/Hybrid, tables_docling for docling-only — annotator
  # auto-resolves from manifest.
  python3 eval/annotate.py json_figures        --sweep out/sweeps/$v --variant $v
  python3 eval/annotate.py json_tables_full    --sweep out/sweeps/$v --variant $v
  python3 eval/annotate.py json_tables_docling --sweep out/sweeps/$v --variant $v
done

# Score → reports/stage1_PR.md
python3 scripts/eval/score_pdf_variants.py --md-out reports/stage1_PR.md
```

### 2. Stage 2 — `footnote_screen`

```bash
# Run 3 footnote-expansion variants on the top of each Stage-1 family.
caffeinate -dimsu python3 scripts/eval/run_all_sweeps.py --stage footnote_screen

# Label only the unique (footnote-expanded) bboxes — ~17/19/25 fresh
# prompts after share-map backfill.
python3 eval/annotate.py json_tables_docling \
    --sweep out/sweeps/08_docling_footnote_expand_1_2 \
    --variant 08_docling_footnote_expand_1_2

python3 eval/annotate.py json_tables_full \
    --sweep out/sweeps/09_tatr_099_footnote_expand_1_2 \
    --variant 09_tatr_099_footnote_expand_1_2

python3 eval/annotate.py json_tables_full \
    --sweep out/sweeps/10_hybrid_099_footnote_expand_1_2 \
    --variant 10_hybrid_099_footnote_expand_1_2

# Score → reports/stage2_PR.md (same script — includes every variant
# with labels on disk, so the Stage 1 rows also appear).
python3 scripts/eval/score_pdf_variants.py --md-out reports/stage2_PR.md
```

### 3. Footnote-issue breakdown (used in this report)

```bash
python3 - <<'PY'
import json
from collections import Counter
from pathlib import Path
ann = Path('eval/annotations')
pairs = [
    ('Stage 1', '01_docling',                       'json_tables_docling'),
    ('Stage 2', '08_docling_footnote_expand_1_2',   'json_tables_docling'),
    ('Stage 1', '04_tatr_099',                      'json_tables_full'),
    ('Stage 2', '09_tatr_099_footnote_expand_1_2',  'json_tables_full'),
    ('Stage 1', '07_hybrid_099',                    'json_tables_full'),
    ('Stage 2', '10_hybrid_099_footnote_expand_1_2','json_tables_full'),
]
for stage, v, m in pairs:
    p = ann / v / f'{m}.json'
    if not p.exists(): continue
    d = json.loads(p.read_text())
    cnt = Counter(x for x in d.values() if x and 'footnote' in x.lower())
    total = sum(1 for x in d.values() if x)
    miss_total = sum(cnt.values())
    print(f"\n{stage} — {v} ({m}, total={total}):")
    for lbl, n in cnt.most_common():
        print(f"  {n:>3}  {lbl}")
    print(f"  ---   {miss_total} footnote-related issues ({miss_total/total*100:.0f}% of all crops)")
PY
```

---

## See also

- [`PDF_EXTRACTION_EXPERIMENT_PLAN.md`](PDF_EXTRACTION_EXPERIMENT_PLAN.md) — what each stage tests
- [`HOW_TO_RUN.md` §2.2](HOW_TO_RUN.md#22-stage-1-detector--threshold-variants-run_all_sweepspy) — run / resume / checkpoint semantics
- [`THESIS.md`](THESIS.md) — TODO to bake `BEST_*` defaults into `PipelineConfig` after Stage 6
- [`scripts/eval/run_all_sweeps.py`](../scripts/eval/run_all_sweeps.py) — variant config source of truth
- [`scripts/eval/score_pdf_variants.py`](../scripts/eval/score_pdf_variants.py) — per-variant P/R/F1 scorer
