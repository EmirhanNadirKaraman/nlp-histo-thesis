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

## ⏭ Carry-forward — pick up here (2026-05-18 evening continuation)

### Tonight (2026-05-18 ~13:50) — code infrastructure complete

Built and smoke-tested:

* **`runner.py`**: 3 new CLI flags `--reconstruct-tables-from-lists`, `--merge-tables-by-caption`, `--expand-tables-with-footnotes` (each with `--no-*` counterpart, default `None` → don't override config).
* **`scripts/eval/build_share_map.py`**: writes `eval/annotations/share_map.json` mapping crop filename → list of variants that emitted it.  Run after every new sweep.
* **`eval/annotate.py`** extended: new `--sweep <dir>` (read crops directly from a sweep, no `eval/out/` symlinks) and `--variant <name>` (write labels to `eval/annotations/<name>/<mode>.json`).  When the share map is present, labelling a shared crop auto-propagates to every peer variant's annotation file.  Default behavior (no `--sweep` / `--variant`) unchanged.
* **`scripts/eval/score_pdf_variants.py`**: per-variant P/R/F1, reads per-variant labels with legacy fallback.  Output: `reports/variants_PR_partial.md` + `.json`.
* **E1_evalcfg sweep complete** (`out/sweeps/baseline_evalcfg/`, 30/30, 0 failed) using `--reconstruct-tables-from-lists --merge-tables-by-caption --expand-tables-with-footnotes`.

### Tonight's headline finding — eval-config does NOT improve F1

| Variant | Cropping flags | Tables emitted | Labelled | P | R | F1 |
|---|---|---|---|---|---|---|
| **baseline** (production defaults) | recon/merge/foot = False | 64 | 48 | 68.9% | 72.1% | **70.5%** |
| **baseline_evalcfg** (eval/run.py-style) | recon/merge/foot = True | 65 | 51 | 64.6% | 72.1% | **68.1%** |
| detector_docling | DOCLING-only detector | 52 | 37 | 88.2% | 69.8% | **77.9%** |
| detector_tatr | TATR-only detector | 65 | 43 | 73.2% | 69.8% | **71.4%** |

Eval-config emits exactly 1 more crop than baseline (65 vs 64) on this 30-PDF set, and that 1 extra is an FP on the existing label set.  Recall is identical (72.1% → same TPs).  On this corpus the three "kitchen-sink" cropping flags **do not improve table extraction**.  Caveat: this is partial-label scoring; the unlabelled crops (14 in eval-config, 16 in baseline) could swing the comparison once filled.

DOCLING-only continues to lead on F1 (77.9% > everything else) by being more precise.

### Pre-flight checks (2026-05-18 morning) — all green ✅

* **PMCID alignment**: 30 PDFs in `eval/pdfs/`, all 30 in `eval/ground_truth.csv`, zero mismatch.
* **share_map.json**: 198 unique crops across 7 sweep variants.
* **Per-variant label files pre-seeded** from the legacy `annotations_json_tables_*.json` + `annotations_json_figures.json`, filtered by each variant's emitted crops.  Each `eval/annotations/<variant>/<mode>.json` already contains the labels for every crop that matches a legacy filename.  Annotator's resume-cursor will skip those and prompt only for the *new* crops.
* **`scripts/eval/seed_variant_labels.py`** built + run for all 7 sweeps.  Re-run with `--force` after any new sweep to keep the seed in sync.

### Tomorrow's actual work (labelling, then decision)

**Propagation-optimal labelling order (greedy-simulated 2026-05-18 noon)**: ~74-89 keypresses total across 4 distinct sessions (down from ~162 if labelled naively per-variant).

```bash
# 0. Refresh share map (only needed if you ran a new sweep) and re-seed.
python scripts/eval/build_share_map.py
python scripts/eval/seed_variant_labels.py     # --force to overwrite

# Session 1 — HYBRID-family tables.  tatr_090 is a superset of HYBRID
# table crops, so propagation reaches baseline, baseline_evalcfg, tatr_095,
# no_two_pass.  Inside the annotator press [p] to open the source PDF.
python eval/annotate.py json_tables_full \
    --sweep out/sweeps/tatr_090 --variant tatr_090            # 24 prompts

# Session 2 — TATR-only's 8 crops that no other HYBRID variant emits.
python eval/annotate.py json_tables_full \
    --sweep out/sweeps/detector_tatr --variant detector_tatr  # 8 prompts

# Session 3 — Figures.  Identical 115 crops across every variant; one
# session covers all eight via share-map propagation.
python eval/annotate.py json_figures \
    --sweep out/sweeps/baseline --variant baseline            # 27 prompts

# Session 4 — DOCLING-only tables (separate label-file family).
python eval/annotate.py json_tables_docling \
    --sweep out/sweeps/detector_docling --variant detector_docling   # 15 prompts

# Session 5 — DOCLING-with-reconstruction (after task #41 sweep + task #42
# share-map refresh).  Only the new recon-promoted crops will prompt;
# shared crops propagate from session 4.
python eval/annotate.py json_tables_docling \
    --sweep out/sweeps/docling_recon --variant docling_recon  # ~N (TBD)

# After labelling, recompute per-variant P/R/F1
python scripts/eval/score_pdf_variants.py \
    --md-out reports/variants_PR.md \
    --json-out reports/variants_PR.json
```

Total: **~74-89 keypresses** across 5 sessions (vs ~162 if labelled per-variant naively).  Propagation auto-deduplicates shared crops; the cursor jumps straight to the next un-labelled item each session.

Keys: `y` = correct, `n` = incorrect, `o` = other, `l` = custom label, `s` = skip, `b` = back, `space` = next, **`p` = open the source PDF**, `r` = metrics so far, `q` = quit (auto-saves).

2. **#35 → re-run scorer** after each labelling session:

   ```bash
   python scripts/eval/score_pdf_variants.py \
       --md-out reports/variants_PR.md \
       --json-out reports/variants_PR.json
   ```

3. **#36 — decision**: with full coverage, compare baseline vs baseline_evalcfg and HYBRID vs DOCLING.  If DOCLING-only still beats HYBRID by ≥10 pp F1, flip `PipelineConfig.table_detector` default to `DOCLING`.  If eval-config still loses, document why and keep the three cropping flags `False` in defaults.

### State at end of 2026-05-17 session (for reference):

* **30-PDF sweeps complete** on `eval/pdfs/` for all six variants (E1, E2a, E2b, E3a, E3b, E4).  Manifests in `out/sweeps/<name>/run_metadata/run_*.json`, fresh comparison reports in `reports/*_vs_baseline_n30.{md,json}`.
* **5-PDF sweeps (earlier today)** are superseded — keep the previous "Configurations compared" + "Observations" sections below as historical record.  Updated 30-PDF table goes in the next session.
* **Existing labels surveyed** — `eval/annotations/annotations_json_tables_full.json` (52 labels), `_docling.json` (52), `_docling_recon.json` (50), `_json_figures.json` (115).  Created against an `eval/run.py`-style run with `multi_source_crops=True`, `reconstruct_tables_from_lists=True`, `merge_tables_by_caption=True`, `expand_tables_with_footnotes=True`.  Our new sweeps use `PipelineConfig` defaults (all three flags `False`), which explains why E1 baseline has 16/64 unlabelled tables and 27/115 unlabelled figures.
* **`tatr.threshold = 0.99` decision LOCKED on 5-PDF set** (`THESIS.md` Decisions log 2026-05-17) — needs re-validation on the 30-PDF set with proper labels.

### Tomorrow's order (matches Claude task list #27 → #36)

1. **#27** — Add three CLI flags to `pipeline/stages/pdf_text_extraction/runner.py:main` (argparse only):
   * `--reconstruct-tables-from-lists` / `--no-reconstruct-tables-from-lists`
   * `--merge-tables-by-caption`   / `--no-merge-tables-by-caption`
   * `--expand-tables-with-footnotes` / `--no-expand-tables-with-footnotes`
   Default `None` → don't override config.  Mirrors `--two-pass` / `--no-two-pass` pattern.

2. **#28** — Re-run E1 with all three flags ON, output to `out/sweeps/baseline_evalcfg/` — matches the historical `eval/run.py` config so existing labels apply almost 1:1.  Expected ~3-5 new labels needed instead of 16+27.

3. **#29** — `scripts/eval/build_share_map.py`: scan every `out/sweeps/*/json/*_media.json`, emit `eval/annotations/share_map.json` as `{crop_filename: [variants_that_emitted]}`.

4. **#30** — Extend `eval/annotate.py`:
   * `--sweep <path>` — read media JSON from that sweep's `json/` dir (no `eval/out/` symlinks).
   * `--variant <name>` — write labels to `eval/annotations/<name>/<mode>.json` (per-variant subdir).
   * On every label save, propagate to all peer-variant files via the share map.

5. **#31–#34** — Label sessions in this order (each uses `--sweep <dir> --variant <name>`):
   1. **E1_evalcfg** tables + figures (~3-5 prompts each if eval-config matches labels well).
   2. **E1_baseline** (production defaults) tables (~16) + figures (~27).
   3. **E2a_docling** tables (~15).
   4. **E3a + E3b** threshold-relaxation new crops (~20 + ~4 net new).  Share-map auto-propagates shared crops from E1.
   * **Defer**: E2b TATR-only (no existing label file).  E4 paragraph-level (separate `annotate.py text` mode; large new label set).

6. **#35** — `scripts/eval/score_pdf_variants.py`: read per-variant label files + `ground_truth.csv` + each variant's manifest → emit per-variant P/R/F1 table to `reports/`.

7. **#36** — Decide whether to flip `PipelineConfig` defaults for the three cropping flags.  Rule: flip only if `E1_evalcfg` beats `E1_baseline` by ≥10 pp F1 with no precision regression ≥2 pp.  Either way, record in `THESIS.md` Decisions log.

### Quick stats heading into tomorrow (raw 30-PDF manifests)

| | E1 HYB | E2a DOC | E2b TATR | E3a 0.95 | E3b 0.90 | E4 no-2p |
|---|---|---|---|---|---|---|
| n_ok / failed | 30 / 0 | 30 / 0 | 30 / 0 | 30 / 0 | 30 / 0 | 30 / 0 |
| text_rows | 1946 | 1946 | 1946 | 1946 | 1946 | **2331** |
| figures | 115 | 115 | 115 | 115 | 115 | 115 |
| tables | 64 | 52 | 65 | 68 | 72 | 64 |
| R1 / R3 | 88 / 34 | 88 / 34 | 88 / 34 | 88 / 34 | 88 / 34 | 0 / 0 |

Partial F1 from existing labels (incomplete coverage, conservative):
* **E1 HYBRID tables**: P=64.6% R=72.1% F1=68.1% (TP 31 / FP 17 / FN 12)
* **E1 HYBRID figures**: P=81.8% R=100% F1=90.0%
* **E2a docling tables**: P=88.2% R=69.8% F1=77.9% — docling-only beats HYBRID on labelled subset; needs full coverage to confirm

### Tomorrow's open question

**Does HYBRID actually beat DOCLING-only on tables, or is the apparent win an artifact of incomplete label coverage?**  Resolved by completing #32 + #33.  If DOCLING-only still beats HYBRID after the gaps are filled, that's a `table_detector` default candidate flip to record in the Decisions log.

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
* **Implication for thesis:** This is the cleanest signal in the first batch — a single knob produces a measurable, monotone effect.  Decision rule from the plan: flip the default only if `0.90` lifts F1 ≥10 pp over `0.99` on the labelled set.
* **Manual labelling result (2026-05-17, by hand against the source PDFs):**
  - `tatr_095` extra detection (PMC10047158 Table 1, page 1) — **false positive** (not a real table).
  - `tatr_090` extra detection (PMC10047408 Table 2, page 6) — **false positive**.
  - No new true positives at either looser threshold.  Precision strictly drops; recall unchanged.
* **Decision:** **Keep `tatr.threshold = 0.99`.**  Both relaxations added only FPs on this 5-PDF sample, so the labelled-evidence decision rule rejects the flip.  Logged in `docs/THESIS.md` Decisions log (2026-05-17).  Caveat: sample is small (only 2 borderline detections); the conclusion is "no evidence to lower the threshold" rather than "the threshold cannot be lowered ever".

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
* ~~**Are the two extra TATR detections at threshold 0.90 (PMC10047158 + PMC10047408) TPs or FPs?**~~  **Resolved 2026-05-17: both FPs.** See decision in TATR-threshold-sweep observation above; default kept at 0.99.
* **Are the 12 paragraphs suppressed by two-pass (E1 → E4 delta) real body text or phantom elements?**  Requires Rank 3 paragraph labelling.  Header-zone breakdown (29 rejections inside the header zone) hints "mostly headers/footnotes" but doesn't prove it.
* **`mask_regions` count not emitted under two-pass mode** — minor observability gap.  Not Stage-1 blocking; would need a small additive change to `two_pass_extractor.process()` to report a comparable mask-region count.  Defer to a follow-up patch.

---

## Decisions made (cross-references into `THESIS.md`)

For each Stage-1 design call, add a row pointing at the Decisions-log entry.

| Date | Decision (short) | Link |
|------|------------------|------|
| 2026-05-17 | Keep `tatr.threshold = 0.99` (loosening to 0.95 / 0.90 surfaces only false positives on the 5-PDF set). | [THESIS.md Decisions log](THESIS.md#decisions-log) |
