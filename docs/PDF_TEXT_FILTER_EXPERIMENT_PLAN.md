# PDF Text Filtering — Experiment Plan

Calibration plan for the **body-text** stages of the PDF extraction
pipeline: artifact filtering, paragraph-relevance filtering, narrative
stitching, and citation removal — i.e. "getting rid of wrong texts" and
"text merging."

This is the text-side companion to
[`PDF_EXTRACTION_EXPERIMENT_PLAN.md`](PDF_EXTRACTION_EXPERIMENT_PLAN.md)
(which covers table/figure detection, masking, cropping) and
[`SUMMARIZATION_EXPERIMENT_PLAN.md`](SUMMARIZATION_EXPERIMENT_PLAN.md)
(which covers the LLM pipeline). Freeze-tier rules live in
[`CALIBRATION_EXECUTION_PLAN.md`](CALIBRATION_EXECUTION_PLAN.md) §11.3–11.8.

---

## 1. Purpose & scope

The detector/mask/crop side is calibrated and scored (crop / caption /
footnote / **mask** F1 against `eval/annotations/`). What is **not** yet
calibrated is what happens to the *body text* after masking:

- which paragraphs get dropped as artifacts (`ArtifactFilter`),
- which get dropped as irrelevant (`is_relevant_para`),
- how split narratives get stitched back (`ContextAwareStitcher`),
- whether citation markers get stripped (`remove_citations`).

These decide whether the text handed to summarization is clean body
prose or contaminated with headers, captions, reference lists, and
page furniture. This plan calibrates them.

**Honest framing up front:** unlike the detector-threshold sweep, this
area is **mostly A/B flips, not numerical sweeps**. There are only ~5
genuine numeric knobs (§4.B). Everything else is binary on/off compared
on proxy metrics or a small paragraph-level manual gold. The answer to
"is it only benchmarking?" is *mostly yes* — with a handful of real
sweeps and one cheap manual-gold metric that turns it into actual P/R.

## 2. What is already covered elsewhere — do not duplicate

These knobs touch text/region handling but are **already scored** by the
existing crop/mask F1 in the PDF plan. Reference them; do not re-run here.

| Knob | Scored by | Where |
|---|---|---|
| `mask_header_footer_sidebar` (region effect) | mask F1 | PDF plan Stage 3 `header_fix` (variants 14/15) |
| `merge_tables_by_caption` / `merge_figures_by_caption` | crop F1 | PDF plan Stage 5 `merge_flags` |
| `drop_tables_inside_figures` / `mask_figures_before_table_detection` | crop/mask F1 | PDF plan Stage 2 `table_in_figure` |
| `reconstruct_tables_from_lists` / `expand_tables_with_footnotes` | crop F1 | PDF plan Stage 6 `reconstruction` |

The **`mask` dimension** (`eval/label_rubric.yaml`) is **crop-anchored**:
it scores whether a *detected figure/table region* was correctly kept
out of body text. It does **not** measure paragraph-level body-text
contamination (header leak, caption leak, reference-list leak). That gap
is exactly what §5 below targets.

## 3. Gold & metrics — three layers

| Layer | What it measures | Gold needed | Cost | Cite in thesis? |
|---|---|---|---|---|
| **A — proxy metrics** | quantitative effect of each filter on the corpus (char delta, citation count, paragraph length dist, NER density, cross-page repeat leak) | none | $0 | as descriptive stats only |
| **B — paragraph manual gold** | per-paragraph precision: did the filter drop *wrong text* and keep *body text*? | ~50 hand-labelled paragraphs | ~1 h | **yes — primary evidence** |
| **C — downstream integration** | did the text variant change summarization silver F1? | silver findings (exists) | **paid** (re-prime per variant) | yes, but deferred to one confirmation run |

Layer B is the defensible "wrong text" metric. Layer A is sanity /
debugging. Layer C is the integration test but expensive (§5, TD1).

## 4. Knob inventory

### 4.A Binary flips (the core of this plan)

| Knob | File | Default | Compared via |
|---|---|---|---|
| `apply_ner_filtering` | `config.py::FilteringConfig` | True | A / B, layers A+B |
| `apply_paragraph_relevance_filtering` | `config.py::FilteringConfig` | True | A / B, layers A+B |
| `pre_filter_relevance` | `config.py::TextAssemblyConfig` | True | A / B, layers A+B |
| `ContextAwareStitcher` on/off | `parsers/text_processing.py` | on (hard-wired) | A / B, layer A + manual stitch audit |
| `remove_citations` on/off | `parsers/text_processing.py` | on (hard-wired) | A / B, layer A |

`ContextAwareStitcher` and `remove_citations` have **no thresholds** —
they are regex/heuristic. The only experiment is on/off, scored on proxy
metrics (+ a manual stitch-correctness spot check for the stitcher).

### 4.B Genuine numeric sweeps (text-safety, ride existing crop/mask gold)

| Knob | File | Default | Grid | Scored by |
|---|---|---|---|---|
| `expand_box_px` | `MaskingConfig` | 2 | `{0,2,3,4,5}` | mask F1 (glyph-remnant leak) |
| `drop_tables_in_top_pts` | `MaskingConfig` | 0.0 | `{0,30,50,80}` | crop/mask F1 (header-band FP) |
| `min_figure_pts` | `CroppingConfig` | 50 | `{20,50,80}` | crop F1 |
| `subfigure_proximity_pts` | `CroppingConfig` | 20 | `{10,20,30,40}` | crop F1 |
| two-pass `blank_brightness_threshold` | `TwoPassConfig` | 245 | `{235,240,245,250,254}` | ghost-text fixtures |
| two-pass `blank_dark_pixel_max_fraction` | `TwoPassConfig` | 0.02 | `{0.005,0.01,0.02,0.05}` | ghost-text fixtures |
| two-pass `max_chars_per_bbox_pt` (R3) | `TwoPassConfig` | 15 | `{5,10,15,20,30}` | ghost-text fixtures |

These are real sweeps but score on **existing** annotation gold
(crop/mask) or the synthetic ghost-text fixtures — no new gold. They can
be wired as additional variants in `scripts/eval/run_all_sweeps.py` and
scored by `scripts/eval/score_pdf_variants.py` exactly like the detector
sweeps. Per `CALIBRATION_EXECUTION_PLAN.md` §11, these are **spot-check
priority**, not deep sweeps — masking/filtering changes invalidate every
downstream label.

## 5. Per-experiment specs

Template: **Goal · Knobs · Layer · Metric · Decision criterion · Output ·
Commands · Cost.**

### TF1 — body-text filter flips (`apply_ner_filtering`, `apply_paragraph_relevance_filtering`, `pre_filter_relevance`)

- **Goal:** decide whether each filter earns its place — does it drop
  more noise than body text?
- **Knobs:** three booleans, flipped one at a time from the all-on
  baseline (4 variants: baseline + 3 single-offs).
- **Layer:** A (proxy) + B (manual gold).
- **Metric:**
  - Layer A — Δ char count, Δ paragraph count, NER density of dropped
    vs kept paragraphs, cross-page repeated-string leak count.
  - Layer B — precision of *drop* decision: of paragraphs the filter
    dropped, what fraction were truly non-body (`*_leak` / `reference_leak`
    / `other`)? And recall: of truly non-body paragraphs, what fraction
    were dropped?
- **Decision criterion:** keep a filter on if drop-precision ≥ 0.90 (it
  rarely eats body text) **and** it removes a non-trivial share of
  leaks. Turn off any filter that drops body text at > 10 %.
- **Output:** `eval/results/text_filter_pr.csv` (one row per variant ×
  metric), `eval/results/text_proxy_metrics.csv`.
- **Commands:**
  ```bash
  # Produce text for each variant (4 runs on a ~10-paper subset).
  python scripts/eval/run_text_filter_variants.py --subset eval/data/text_filter_subset.txt
  # Layer A proxy metrics (no gold):
  python scripts/eval/compute_text_proxy_metrics.py \
      --variants out/text_filter --out eval/results/text_proxy_metrics.csv
  # Layer B P/R against manual paragraph gold (after §6 labelling):
  python scripts/eval/compute_text_filter_pr.py \
      --sample eval/results/paragraph_manual_sample.jsonl \
      --out eval/results/text_filter_pr.csv
  ```
- **Cost:** Layer A/B `$0` (no LLM). The variant runs are local Docling
  passes (CPU/GPU time, no API).

### TF2 — narrative stitching (`ContextAwareStitcher` on/off)

- **Goal:** confirm stitching rejoins split narratives without merging
  unrelated paragraphs.
- **Knobs:** stitcher on (default) vs off.
- **Layer:** A (proxy) + a 20-case manual stitch audit.
- **Metric:**
  - Layer A — count of mid-sentence-truncated paragraphs (ending without
    terminal punctuation) before vs after; average paragraph length.
  - Manual audit — of 20 stitched paragraphs (where `len(sources) > 1`),
    how many are correct joins vs wrong merges? (precision-only).
- **Decision criterion:** keep stitching on if join-precision ≥ 0.90 and
  it reduces truncated-paragraph count materially. Off otherwise.
- **Output:** `eval/results/stitch_audit.md` (20 hand-checked cases),
  proxy rows in `text_proxy_metrics.csv`.
- **Cost:** `$0`.
- **Gotcha:** `ContextAwareStitcher.reconstruct_with_sources()` already
  exposes the pre-stitch chunks per output paragraph — the audit reads
  `sources` directly, no new instrumentation.

### TF3 — citation removal (`remove_citations` on/off)

- **Goal:** verify citation stripping removes `[1, 2, 3]`-style markers
  without eating real bracketed content (e.g. `[Ca2+]`, ranges).
- **Knobs:** on (default) vs off.
- **Layer:** A only.
- **Metric:** citation-marker count removed; false-removal spot check on
  a regex-targeted sample (bracketed non-citations that survived / were
  wrongly stripped).
- **Decision criterion:** keep on if false-removal rate < 2 % on the
  bracket sample.
- **Output:** proxy rows in `text_proxy_metrics.csv` + a short
  false-removal note.
- **Cost:** `$0`.

### TS1–TS3 — numeric text-safety sweeps (existing gold)

- **Goal:** tune the §4.B numeric knobs against existing crop/mask F1
  (TS1: `expand_box_px`, `drop_tables_in_top_pts`, `min_figure_pts`,
  `subfigure_proximity_pts`) and the ghost-text fixtures (TS2: two-pass
  R1; TS3: two-pass R3).
- **Layer:** existing annotation gold — no new gold.
- **Metric:** crop/mask F1 from `score_pdf_variants.py`; ghost-text
  recall from `scripts/verify_ghost_text_detection.py`.
- **Decision criterion:** pick the value maximising the relevant F1 with
  no regression on the others.
- **Commands:** add variants to `scripts/eval/run_all_sweeps.py`
  (mirroring the detector stages), then:
  ```bash
  python scripts/eval/run_all_sweeps.py --stage text_safety
  python scripts/eval/score_pdf_variants.py --md-out reports/text_safety_PR.md
  # Two-pass thresholds (synthetic fixtures, not real papers):
  python scripts/verify_ghost_text_detection.py
  ```
- **Cost:** `$0` (local).
- **Note:** these are **spot-check priority** per
  `CALIBRATION_EXECUTION_PLAN.md` §11.3/§11.6/§11.8 — do not deep-sweep
  unless a regression shows up. Each new variant needs a fresh
  annotation pass on the affected crops (the metric exists; the human
  labelling is the cost).

### TD1 — downstream integration (optional, paid, deferred)

- **Goal:** the only test of "did cleaner text actually improve the rule
  output" — feed the winning text variant(s) through summarization and
  compare silver F1.
- **Layer:** C.
- **Metric:** summarization silver F1 (per
  `SUMMARIZATION_EXPERIMENT_PLAN.md` M-series) on variant text vs
  baseline text.
- **Decision criterion:** confirmation only — the TF1/TF2 winner should
  not *lose* downstream F1.
- **Cost:** **paid.** Text filtering invalidates the MAP primer cache
  (label-invalidation matrix: PDF filtering → MAP labels stale), so each
  text variant needs a re-prime (~$20–$60). **Therefore: run TD1 on the
  single best variant only**, not the full grid. 1 confirmation re-prime,
  not N.
- **Recommendation:** defer TD1 until TF1/TF2/TF3 pick a winner on the
  cheap layers; then one paid confirmation run.

## 6. Manual paragraph-gold workflow (Layer B)

### 6.1 Sample

50 paragraphs, **stratified by filter decision** so every variant is
compared on the same paragraphs:

- 15 dropped only by `apply_ner_filtering`
- 15 dropped only by `apply_paragraph_relevance_filtering`
- 10 dropped only by `pre_filter_relevance`
- 10 kept by all (control — should be body text)

Sample from the `out/text_raw/` (pre-filter) vs `out/text/` (post-filter)
diff on the ~10-paper subset.

### 6.2 Schema

```
sample_id        str   # stable id
pmcid            str
paragraph_text   str   # the candidate paragraph
dropped_by       list  # which filters dropped it (may be empty = kept by all)
label            null  ← FILL THIS
label_options    ["body", "caption_leak", "header_leak", "footer_leak",
                  "table_row_leak", "reference_leak", "other"]
notes            ""
```

`body` = real narrative prose that must be kept. Everything else is
*wrong text* that should be dropped. The metric collapses the non-`body`
classes into "should-drop" for P/R, but keeps the fine class so we can
attribute *which kind* of leak each filter catches.

### 6.3 Commands

```bash
# Build the stratified sample (deterministic; seed-stable):
python scripts/eval/sample_paragraphs_for_manual_labeling.py \
    --raw out/text_raw --filtered out/text \
    --n 50 --seed 42 --out eval/results/paragraph_manual_sample.jsonl

# (hand-label the `label` field, then)
python scripts/eval/compute_text_filter_pr.py \
    --sample eval/results/paragraph_manual_sample.jsonl \
    --out eval/results/text_filter_pr.csv
```

Mirrors the grounding-label workflow in
`SUMMARIZATION_EXPERIMENT_PLAN.md` §7 — same disclaimer rules: manual-gold
P/R is real evidence; proxy metrics are sanity only.

## 7. Proxy-metric definitions (Layer A)

`scripts/eval/compute_text_proxy_metrics.py` (new, ~80 LOC) reads
`out/text_raw/*_raw.txt` and `out/text/*_text.txt` per variant and emits:

| Metric | Definition |
|---|---|
| `char_delta` | chars(filtered) − chars(raw); large negative = aggressive filtering |
| `para_delta` | paragraph count delta |
| `truncated_para_frac` | fraction of paragraphs ending without terminal punctuation (stitch signal) |
| `citation_markers` | count of `[\d, ]+`-style markers remaining |
| `mean_ner_density` | entities per 100 tokens (body text scores higher than boilerplate) |
| `cross_page_repeat_leak` | count of identical short strings appearing on ≥3 pages (header/footer leak) |

All `$0`, no LLM/NLI/embedding imports (same Layer-A invariant as
`eval/sweeps/`).

## 8. Execution order

All Day-1 work is `$0`. There is no paid step until the optional TD1
confirmation.

1. **Build the 10-paper subset** (`eval/data/text_filter_subset.txt`) —
   reuse the PDFs already annotated in `eval/annotations/` so TS1–TS3 can
   piggyback on existing crop/mask gold. — 15 min.
2. **Run the 4 TF1 variants + stitcher/citation variants** locally. —
   ~1 h Docling time.
3. **Layer A proxy metrics** across all variants. — 5 min.
4. **Build + hand-label the 50-paragraph gold** (§6). — ~1 h.
5. **Layer B P/R** (`compute_text_filter_pr.py`). — 5 min.
6. **TF2 stitch audit** (20 cases). — 30 min.
7. **TS1–TS3 numeric sweeps** *only if* a proxy metric or the mask F1
   flags a problem — otherwise leave at defaults and record "spot-checked,
   no regression." — variable.
8. **TD1** — only after a winner emerges; one paid re-prime. — deferred.

## 9. Freezing

Once TF1/TF2/TF3 have winners:

1. Record the chosen filter booleans in the same
   `configs/profiles/profile-pdf-frozen-<date>.yaml` that the detector
   sweep produces (these are PDF-tier knobs). Add an `overrides` block
   for `apply_ner_filtering`, `apply_paragraph_relevance_filtering`,
   `pre_filter_relevance`, and (if changed) the stitcher/citation
   toggles.
2. Bake winners into `config.py` (`FilteringConfig`, `TextAssemblyConfig`)
   defaults **only after** the profile YAML is committed.
3. Add a `docs/STRUCTURE.md ## Pipeline changelog` row + a
   `docs/THESIS.md` Decisions-log row ("Body-text filter freeze: NER=…,
   relevance=…, stitch=…, citations=…; layer-B P/R = …").
4. If TD1 ran, cite the single confirmation F1; otherwise state that
   downstream confirmation is deferred and why.

## 10. Appendix — experiment matrix

| ID | Knobs | Variants | Layer | Metric | Cost | Priority |
|---|---|---|---|---|---|---|
| TF1 | 3 filter booleans | 4 | A+B | drop precision/recall vs paragraph gold | $0 | P0 |
| TF2 | stitcher on/off | 2 | A + audit | join precision (20 cases) | $0 | P1 |
| TF3 | citations on/off | 2 | A | false-removal rate | $0 | P1 |
| TS1 | expand_box_px, drop_tables_in_top_pts, min_figure_pts, subfigure_proximity_pts | numeric | existing gold | crop/mask F1 | $0 | P2 (spot-check) |
| TS2 | two-pass R1 (brightness, dark-frac) | numeric | fixtures | ghost-text recall | $0 | P2 (spot-check) |
| TS3 | two-pass R3 (max_chars_per_bbox_pt) | numeric | fixtures | ghost-text recall | $0 | P2 (spot-check) |
| TD1 | best variant | 1 | C | summarization silver F1 | paid (1 re-prime) | P1 (confirmation) |

Priority: **P0** ship before defending; **P1** ship if budget allows;
**P2** spot-check only, deep-sweep only on regression.

The single sentence to keep in mind: **body-text filtering is mostly
binary A/B scored on proxy metrics + a 50-row paragraph gold; the only
real numeric sweeps ride existing crop/mask annotation gold, and the only
paid step is one optional downstream confirmation run.**

---

## See also

- [`PDF_EXTRACTION_EXPERIMENT_PLAN.md`](PDF_EXTRACTION_EXPERIMENT_PLAN.md)
  — detector / mask / crop sweep (crop, caption, footnote, mask F1).
- [`SUMMARIZATION_EXPERIMENT_PLAN.md`](SUMMARIZATION_EXPERIMENT_PLAN.md)
  — downstream LLM pipeline (the TD1 integration target).
- [`CALIBRATION_EXECUTION_PLAN.md`](CALIBRATION_EXECUTION_PLAN.md) §11.3–11.8
  — per-stage tune-or-freeze guidance for these exact knobs.
- [`eval/label_rubric.yaml`](../eval/label_rubric.yaml) — the crop-anchored
  `mask` dimension (text-safety for *detected regions*, not paragraphs).
- [`eval/sweeps/README.md`](../eval/sweeps/README.md) — Layer-A invariant
  (no LLM/NLI imports) that the proxy scripts must honour.
