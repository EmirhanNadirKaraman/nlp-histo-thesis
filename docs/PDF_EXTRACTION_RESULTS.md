# PDF Extraction — Sweep Results

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

---

## Stages 3–6 — pending

Will be appended as each stage's scoring lands.  See
[`PDF_EXTRACTION_EXPERIMENT_PLAN.md`](PDF_EXTRACTION_EXPERIMENT_PLAN.md)
for what each stage answers.

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
variants don't backfill automatically.  One-shot fix:

```bash
python3 - <<'PY'
import json
from pathlib import Path
from collections import defaultdict
ANN = Path('eval/annotations')
SHARE_MAP = ANN / 'share_map.json'
MODES = ('json_figures', 'json_tables_full', 'json_tables_docling')
labels = defaultdict(lambda: defaultdict(dict))
for vdir in ANN.iterdir():
    if not vdir.is_dir() or vdir.name.startswith('_'): continue
    for m in MODES:
        p = vdir / f'{m}.json'
        if p.exists():
            try: labels[vdir.name][m] = json.loads(p.read_text())
            except json.JSONDecodeError: pass
sm = json.loads(SHARE_MAP.read_text())
copies = defaultdict(int)
for fname, groups in sm.items():
    for g in groups:
        for m in MODES:
            canon = None
            for v in g['variants']:
                lbl = labels[v][m].get(fname)
                if lbl: canon = lbl; break
            if canon is None: continue
            for v in g['variants']:
                if fname in labels[v][m]: continue
                labels[v][m][fname] = canon
                copies[(v, m)] += 1
for v, by_m in labels.items():
    for m, content in by_m.items():
        if content:
            p = ANN / v / f'{m}.json'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(content, indent=2, ensure_ascii=False) + '\n')
print(f"Backfilled {sum(copies.values())} labels across {len(copies)} (variant, mode) buckets")
PY
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
