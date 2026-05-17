# THESIS_MATERIAL.md

Working notebook for Stage-1 PDF-extraction stabilization.  Records every
configuration we run, what we observed, and which decisions came out of it.

Cross-references:
* TODOs and the formal Decisions log live in [`THESIS.md`](THESIS.md).
* Bugs (with evidence and fix) live in [`BUGS.md`](BUGS.md).
* Reproducible commands for each comparison live in
  [`HOW_TO_RUN.md`](HOW_TO_RUN.md#21-reproducible-sweep-runs).
* Pipeline architecture changes are appended to the changelog at the bottom
  of [`STRUCTURE.md`](STRUCTURE.md#pipeline-changelog).

Routing reminder: bug-driven rationales belong in `BUGS.md`; permanent design
calls belong in `THESIS.md`'s Decisions log.  **Only sweep observations and
their interpretation belong here.**

---

## Configurations compared

First experiment batch — 2026-05-17.  Fixed sample set: first 5 PDFs from
`files/organized_pdfs` (alphabetical) — PMC10047158, PMC10047213,
PMC10047408, PMC10047897, PMC10082646.  Per-experiment reports under
[`reports/`](../reports/) (e.g. `reports/E3a_tatr095_vs_baseline.md`).
Sweep commands documented in
[`HOW_TO_RUN.md §2.1`](HOW_TO_RUN.md#21-reproducible-sweep-runs).

| Exp | run_id | digest | detector | tatr_thr | two_pass | wall | n_ok / n_failed | text_rows | figures | tables | R1 | R3 | header-zone |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E1** baseline | `20260517T164702Z_ba8f9982` | `424591746b68` | HYBRID | 0.99 | on | 164.5s | 5 / 0 | 117 | 18 | 6 | 64 | 11 | 29 |
| **E2a** docling-only | `20260517T164845Z_57e2c142` | `1759103cf8da` | DOCLING | n/a | on | 146.0s | 5 / 0 | 117 | 18 | 6 | 64 | 11 | 29 |
| **E2b** tatr-only | `20260517T165018Z_f0a9dd77` | `83da244b2b0f` | TATR | 0.99 | on | 163.5s | 5 / 0 | 117 | 18 | 6 | 64 | 11 | 29 |
| **E3a** tatr@0.95 | `20260517T165203Z_b58b0545` | `a42fa3e065aa` | HYBRID | 0.95 | on | 164.1s | 5 / 0 | 117 | 18 | **7** | 64 | 11 | 29 |
| **E3b** tatr@0.90 | `20260517T165347Z_c8b644ee` | `89ccc472fb84` | HYBRID | 0.90 | on | 162.6s | 5 / 0 | 117 | 18 | **8** | 64 | 11 | 29 |
| **E4** no-two-pass | `20260517T165529Z_501ebda0` | `56ba69966f12` | HYBRID | 0.99 | **off** | 163.8s | 5 / 0 | **129** | 18 | 6 | **0** | **0** | **0** |

All six sweeps: 5/5 ok, **0 failed documents** across all variants.

---

## Observations

One subsection per comparison.  Each entry should answer:

1. What was changed (knob + values)?
2. What measurable output changed (counts, reason histogram, wall time)?
3. What did NOT change?  (Be explicit — invariance is also a finding.)
4. Working hypothesis or interpretation.

Template:

```
### {short title} ({YYYY-MM-DD})

* **Compared:** `{run_id_A}` (digest `…`) vs `{run_id_B}` (digest `…`).
* **Changed:** `{knob}` `{value_A}` → `{value_B}`.
* **Result:**
  - `counts_sum.{key}`: A=… B=… (Δ=…)
  - `reason_histogram_sum`: A=… B=… (Δ=…)
  - mean wall: A=…s B=…s
* **Invariant:** {what stayed the same}
* **Interpretation:** {one paragraph}
* **Implications for the thesis:** {one sentence}
```

### Detector swap — E1 (HYBRID) vs E2a (DOCLING) vs E2b (TATR-only) (2026-05-17)

* **Compared:** baseline (HYBRID) vs detector_docling vs detector_tatr.
* **Changed:** `table_detector` `HYBRID` → `DOCLING` → `TATR`.
* **Result:**
  - `counts_sum.text_rows`: 117 / 117 / 117 (Δ=0)
  - `counts_sum.figures_cropped`: 18 / 18 / 18 (Δ=0)
  - `counts_sum.tables_cropped`: **6 / 6 / 6 (Δ=0)** — all three detectors emit the same six tables on this 5-PDF set.
  - `reason_histogram_sum`: identical across all three (R1=64, R3=11)
  - wall: 164.5s / 146.0s / 163.5s — DOCLING is ~10 % faster (no TATR forward pass).
* **Invariant:** every per-document `n_text_rows` / `n_figures` / `n_tables` is identical across all three sweeps.  Even at the byte level there is no detector-driven divergence in this sample.
* **Interpretation:** On this 5-PDF sample, Docling's intrinsic TABLE / RECONSTRUCTED_TABLE elements already cover every region TATR finds at `threshold=0.99`, and the HYBRID merger emits the same union as either branch alone — so the detector choice is unobservable at the output level here.  Two scenarios are still possible: (a) the sample doesn't contain a paper TATR helps on; (b) TATR @ 0.99 is too strict to ever contribute additional regions that survive the merge.  E3a / E3b results below disambiguate.
* **Implication for thesis:** Defending `table_detector=HYBRID` over a single-source detector currently requires either a larger sample or a paper class where the two detectors are known to disagree (e.g. faint / multi-column / borderless tables).  Recommend labelling Rank 1 + Rank 2 on **borderless / multi-column / faint** papers specifically before deciding.

### TATR threshold sweep — E1 (0.99) vs E3a (0.95) vs E3b (0.90) (2026-05-17)

* **Compared:** baseline vs `tatr_095` vs `tatr_090` (HYBRID detector, two-pass on, only TATR threshold changes).
* **Changed:** `tatr.threshold` `0.99` → `0.95` → `0.90`.
* **Result:**
  - `tables_cropped`: 6 → **7** → **8** (+1 at 0.95, +2 at 0.90)
  - text / figures / R1 / R3 / header-zone: **invariant**
  - per-doc: at `0.95`, PMC10047158 picks up a new table (was 0 → 1); at `0.90`, PMC10047158 keeps that table **and** PMC10047408 picks up a second one (was 1 → 2).
  - wall: ≈ 163-164s across all three — threshold change is free.
* **Invariant:** every non-table output is identical.  Lowering the TATR threshold only changes which regions survive the score cutoff; the rest of the pipeline doesn't see it.
* **Interpretation:** TATR is producing borderline detections at `score ∈ [0.90, 0.99)` on PMC10047158 and PMC10047408 that the baseline rejects.  Whether the additional detections are TPs (the table really is there but the model is unsure) or FPs (e.g. a figure caption mistaken for a table) **cannot be decided from stats alone** — must be labelled via Rank 1 (`eval/annotate.py json_tables_full` on the three threshold variants).  Worst case is a regression in precision: an extra two FPs over five PDFs would lift FP-rate by ~33 %.
* **Implication for thesis:** This is the cleanest signal in the first batch — a single knob produces a measurable, monotone effect.  Decision rule from the plan: flip the default only if `0.90` lifts F1 ≥10 pp over `0.99` on the labelled set.  Until labelled, default stays at `0.99`.

### Two-pass on/off — E1 vs E4 (2026-05-17)

* **Compared:** baseline (two_pass.enabled=True) vs `no_two_pass` (two_pass.enabled=False).
* **Changed:** `two_pass.enabled` `True` → `False`.
* **Result:**
  - `text_rows`: 117 → **129 (+12, +10.3 %)**
  - `reason_histogram_sum.R1_blank_pixels`: 64 → 0
  - `reason_histogram_sum.R3_dense_text`: 11 → 0
  - `reason_in_header_zone_sum`: 29 → 0
  - `mask_regions`: not recorded in E1 (two-pass path) → 220 in E4 (standard mask path)
  - per-doc text deltas: PMC10047408 +6 rows, PMC10082646 +3, PMC10047897 +2, PMC10047213 +1, PMC10047158 +0
  - figures / tables: invariant (18 / 6).
* **Invariant:** every figure and table crop is the same.  The two-pass change is entirely a text-side effect.
* **Interpretation:** With two-pass disabled, 12 paragraphs that NodeScorer R1/R3 had been rejecting now flow through to text output.  Per [B-002](BUGS.md#bug-2--docling-phantom-layout-elements), those rejections are by design — they're catching phantom elements from invisible / hidden text layers that Docling emits.  The +12 rows are the union of (a) real paragraphs the two-pass falsely suppressed and (b) phantom elements correctly suppressed.  Without paragraph-level labelling (Rank 3) we cannot tell the split.  The fact that figure/table outputs are bit-identical and `reason_in_header_zone_sum=29` in baseline (most rejections sit in the header zone) suggests the majority of the suppressed paragraphs are headers/footnotes, not body text.
* **Implication for thesis:** `two_pass.enabled=True` is already defended in B-002 via the ghost-text fixture (`scripts/verify_ghost_text_detection.py`).  Adding a paragraph-correctness label set on the +12 row delta would let us quote a real precision/recall number rather than the fixture-only argument.  Until labelled, default stays at `True`.

### Stats-only signals — common to all six sweeps (2026-05-17)

* **`mask_regions=220` (E4) vs not-emitted (E1/E2/E3):** This is a known coverage gap in the stats wiring — `mask_regions` is populated only in the standard four-step branch (`_steps_1_3_4_standard`); the two-pass branch builds its mask set differently and currently doesn't emit an equivalent count.  Tracked as a low-priority observability follow-up; does not affect the experiment interpretation here.
* **Wall-time:** 146–165s across all sweeps.  DOCLING-only is the fastest at 146 s (saves the TATR forward pass).  No experiment is wall-time-bound; we can run more sweeps cheaply.
* **R1 / R3 / header-zone:** identical across E1, E2a, E2b, E3a, E3b — confirming the NodeScorer reasons are not detector- or threshold-dependent (as expected: the scorer runs in two-pass extraction, before any table-detector branch).

---

## Open questions

Items the sweeps surface but don't resolve.  Once an item turns into a
permanent decision, move the rationale to `THESIS.md` and link back from
here; once it turns into a defect, file in `BUGS.md` and link back here.

* **Is the detector invariance (E1=E2a=E2b on the 5-PDF set) a sample artifact or a structural property?**  Need to re-run on a stratified set including ≥1 borderless table paper and ≥1 multi-column paper before concluding the HYBRID merge is unobservable.
* **Are the two extra TATR detections at threshold 0.90 (PMC10047158 + PMC10047408) TPs or FPs?**  Requires Rank 1 labelling against the three table-detector annotation modes.
* **Are the 12 paragraphs suppressed by two-pass (E1 → E4 delta) real body text or phantom elements?**  Requires Rank 3 paragraph labelling.  Header-zone breakdown (29 rejections inside the header zone) hints "mostly headers/footnotes" but doesn't prove it.
* **`mask_regions` count not emitted under two-pass mode** — minor observability gap.  Not Stage-1 blocking; would need a small additive change to `two_pass_extractor.process()` to report a comparable mask-region count.  Defer to a follow-up patch.

---

## Decisions made (cross-references into `THESIS.md`)

For each Stage-1 design call, add a row pointing at the Decisions-log entry.

| Date | Decision (short) | Link |
|------|------------------|------|
| _(populated when sweeps lead to a permanent call)_ | | |
