# HOW_TO_RUN.md

End-to-end commands to reproduce the thesis result. Every block is meant to be
copy-paste runnable from the repo root. Update this file whenever a script's
invocation, a config default, or a pipeline stage changes.

---

## 0. Environment

```bash
# Python deps
pip install -r requirements.txt

# Optional — large UMLS-linked scispaCy model (needed by the summarisation
# pipeline unless NLP_HISTO_DISABLE_UMLS=1 is set). Skip on low-RAM machines.
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

# Postgres (local default — edit .env to point elsewhere)
createdb -U postgres nlp_histo
cp .env.example .env
# fill in DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
# also fill the three API keys the summarisation pipeline uses (direct APIs,
# both sync and batch modes — no Azure Foundry or Vertex AI required):
#   OPENAI_API_KEY      — OpenAI (gpt-4o-mini, gpt-4.1-* voters)
#   GOOGLE_API_KEY      — Gemini direct API (Gemini Flash / Flash-Lite voters)
#   ANTHROPIC_API_KEY   — Anthropic direct API (Claude Haiku / Sonnet voters)

# Schema — managed by Alembic. `database/setup_db.py` no longer exists.
alembic upgrade head        # create all base + sum_* tables
alembic current             # sanity: prints latest revision (should be 0011…)
```

Optional kill-switches (useful on low-RAM machines):

```bash
export NLP_HISTO_DISABLE_UMLS=1            # skip scispaCy + UMLS entirely
export NLP_HISTO_SKIP_UMLS_ENRICHMENT=1    # load scispaCy but skip CUI enrichment
```

---

## 1. Acquire papers

The downloader reads `target_pmc_ids.txt` from its own CWD. The canonical
list is checked into `files/target_pmc_ids.txt` — copy (or symlink) it
into `file-selector/` before running:

```bash
cp files/target_pmc_ids.txt file-selector/
cd file-selector
python file_downloader.py
python tarball_extractor.py
python pdf_organizer.py
cd ..
```

Produces `files/organized_pdfs/` and `files/organized_xmls/`.

---

## 2. PDF → text extraction pipeline

```bash
python -c "
from pipeline.stages.pdf_text_extraction import PipelineRunner, PipelineConfig
cfg = PipelineConfig()       # two_pass.enabled defaults to True after May-2026 fix
runner = PipelineRunner(cfg)
runner.run_batch('files/organized_pdfs')
"
```

Or use the CLI in `runner.py` (recommended for sweeps — see §2.1):

```bash
# Canonical run (matches the legacy `python runner.py` behaviour)
python pipeline/stages/pdf_text_extraction/runner.py
```

Outputs land under `out/`:

| Directory             | Contents                                  |
|-----------------------|-------------------------------------------|
| `out/docling_full/`   | Full Docling layout JSON (cached)         |
| `out/docling_masked/` | Docling layout JSON from masked PDFs      |
| `out/masked_pdfs/`    | PDFs with figures/tables whited out       |
| `out/text/`           | Hierarchical text                         |
| `out/figures/`        | Cropped figure images                     |
| `out/tables/`         | Cropped table images                      |
| `out/visualization/`  | Annotated debug PDFs                      |
| `out/json/`           | Detection metadata                        |
| `out/run_metadata/`   | Per-paper `{pmcid}_stats.json` + per-run `run_{ID}.json` manifest |
| `out/failed_pdfs_blacklist.json` | Skip list (thread-safe)        |

### 2.1. Reproducible sweep runs

Every run of the PDF pipeline now writes two observability artifacts (added
2026-05-17, see [`THESIS_MATERIAL.md`](THESIS_MATERIAL.md)):

* **Per-document stats** — `out/run_metadata/{pmcid}_stats.json`
  Contains stage timings, element counts, rejection histogram (R0 / R1 / R2 /
  R3 / R-color, plus a header-zone tally), table-detection summary, status
  (`ok` / `failed`), a 12-char `config_digest`, and a compact `config_used`
  snapshot.  Written even for failed documents.
* **Per-batch manifest** — `out/run_metadata/run_{ISO_TIMESTAMP}_{uuid8}.json`
  Contains `run_id`, git SHA + dirty flag + branch, host, python version,
  full `PipelineConfig` dump, list of attempted PMCIDs, aggregated summary
  (totals, mean wall, summed reason histogram).  One file per batch
  invocation.

The runner's `__main__` exposes opt-in flags so two configurations can be
diffed by inspecting their manifests.  **All flags are optional and a
no-flag invocation reproduces the canonical behaviour.**  Every command
below writes its outputs (and only its outputs) under `--out-root`, leaving
the canonical `out/` untouched.

```bash
# Baseline (canonical defaults) — for reference / regression diffing
python pipeline/stages/pdf_text_extraction/runner.py \
    --pdf-dir files/organized_pdfs --max-docs 10 \
    --out-root out/sweeps/baseline --no-db --workers 2

# Sweep: TATR threshold (0.99 default → 0.95 / 0.90)
python pipeline/stages/pdf_text_extraction/runner.py \
    --pdf-dir files/organized_pdfs --max-docs 10 \
    --tatr-threshold 0.95 --out-root out/sweeps/tatr095 --no-db --workers 2

python pipeline/stages/pdf_text_extraction/runner.py \
    --pdf-dir files/organized_pdfs --max-docs 10 \
    --tatr-threshold 0.90 --out-root out/sweeps/tatr090 --no-db --workers 2

# Sweep: table detector backend (TATR / Docling / Hybrid)
python pipeline/stages/pdf_text_extraction/runner.py \
    --pdf-dir files/organized_pdfs --max-docs 10 \
    --detector docling --out-root out/sweeps/detector_docling --no-db --workers 2

python pipeline/stages/pdf_text_extraction/runner.py \
    --pdf-dir files/organized_pdfs --max-docs 10 \
    --detector tatr --out-root out/sweeps/detector_tatr --no-db --workers 2

# Sweep: two-pass ghost-text detection on vs off
python pipeline/stages/pdf_text_extraction/runner.py \
    --pdf-dir files/organized_pdfs --max-docs 10 \
    --no-two-pass --out-root out/sweeps/no_two_pass --no-db --workers 2

# Sweep: TATR render DPI (recall vs cost)
python pipeline/stages/pdf_text_extraction/runner.py \
    --pdf-dir files/organized_pdfs --max-docs 10 \
    --render-dpi 200 --out-root out/sweeps/dpi200 --no-db --workers 2
```

Read the resulting manifests as JSON:

```bash
# Find the latest manifest under a sweep directory:
ls -t out/sweeps/baseline/run_metadata/run_*.json | head -1

# Diff two configurations knob-by-knob (config_digest will differ):
python - <<'PY'
import json
a = json.load(open("out/sweeps/baseline/run_metadata/run_LATEST.json"))
b = json.load(open("out/sweeps/tatr095/run_metadata/run_LATEST.json"))
print("digests:", a["config_digest"], b["config_digest"])
print("reasons (baseline):", a["summary"]["reason_histogram_sum"])
print("reasons (tatr095):", b["summary"]["reason_histogram_sum"])
print("tables  (baseline):", a["summary"]["counts_sum"].get("tables_cropped"))
print("tables  (tatr095):", b["summary"]["counts_sum"].get("tables_cropped"))
PY
```

Capture the per-comparison observations under
`## Observations` in [`docs/THESIS_MATERIAL.md`](THESIS_MATERIAL.md) when a
sweep informs a thesis decision.

### 2.2. Stage 1 detector / threshold variants (run_all_sweeps.py)

The PDF-extraction evaluation begins with detector / threshold selection.
Seven named variants are defined in `scripts/eval/run_all_sweeps.py`. Each
writes to its own directory under `out/sweeps/<NN_name>/`.

#### What Stage 1 answers

- Should the base detector be **Docling**, **TATR**, or **Hybrid**?
- If TATR or Hybrid is chosen, what TATR threshold (`0.90` / `0.95` /
  `0.99`) gives the best precision / recall tradeoff?

Stage 1 deliberately keeps every helper flag OFF
(`reconstruct_tables_from_lists`, `merge_tables_by_caption`,
`merge_figures_by_caption`, `expand_tables_with_footnotes`,
`drop_tables_inside_figures`) so the comparison isolates the detector.
Two-pass extraction is ON and `render_dpi` is `150` across all seven
variants.

#### Variants

```
01_docling     — detector=docling
02_tatr_090    — detector=tatr,   tatr_threshold=0.90
03_tatr_095    — detector=tatr,   tatr_threshold=0.95
04_tatr_099    — detector=tatr,   tatr_threshold=0.99
05_hybrid_090  — detector=hybrid, tatr_threshold=0.90
06_hybrid_095  — detector=hybrid, tatr_threshold=0.95
07_hybrid_099  — detector=hybrid, tatr_threshold=0.99
```

#### Commands

`--stage` is **required** and selects which experiment stage to run:
one of `{detector_docling, detector_tatr, detector_hybrid,
table_in_figure, header_fix, footnote_screen, footnote_multiplier,
merge_flags, reconstruction, all}`.  Running the script without
`--stage` prints the full menu (name, blurb, member variants) and
exits.  `--only` further narrows to specific variant names *within*
the selected stage.

List resolved configs without running (no Docling/TATR models load, no
output dirs are created):

```bash
python scripts/eval/run_all_sweeps.py --stage all         --list-variants
python scripts/eval/run_all_sweeps.py --stage merge_flags --list-variants
```

Run all seven Stage 1 variants (one per family sub-stage):

```bash
python scripts/eval/run_all_sweeps.py --stage detector_docling
python scripts/eval/run_all_sweeps.py --stage detector_tatr
python scripts/eval/run_all_sweeps.py --stage detector_hybrid
```

Run a single variant (must match the stage):

```bash
python scripts/eval/run_all_sweeps.py --stage detector_hybrid --only 07_hybrid_099
```

Force a fresh re-run (wipes the variant's output dir first):

```bash
python scripts/eval/run_all_sweeps.py --stage footnote_screen --only 17_tatr_best_family_fixes_footnote_expand_1_2 --restart
```

#### Resume / checkpoint semantics

`run_all_sweeps.py` resumes by default. Three layers of checkpointing:

1. **Per-variant marker.** On success, each variant writes
   `out/sweeps/<variant>/_DONE.json` (`{"completed_at": ..., "pdf_count": N}`).
   On the next run, variants whose marker matches the current PDF count
   are skipped entirely. Adding new PDFs to the source dir invalidates
   the marker so the new ones get processed; already-processed PDFs are
   still skipped via the per-PDF cache.
2. **Per-PDF cache.** `cfg.runtime.skip_existing_media_json = True` —
   a PDF whose `<json_dir>/<pmcid>_media.json` already exists is
   skipped before Docling loads. Cheapest possible resume.
3. **Per-stage cache.** `cfg.runtime.skip_existing_outputs = True` —
   `_StageCache` under `out/sweeps/<variant>/stage_cache/` lets a PDF
   interrupted mid-pipeline resume from the last completed stage on the
   next attempt. Invalidates on config-hash mismatch.

So Ctrl-C is safe: finished variants stay finished (marker + skip),
partially-finished variants pick up from the last completed PDF (per-PDF
cache), and a PDF interrupted mid-pipeline picks up from its last
completed stage (per-stage cache).

#### Next steps after Stage 1 (`detector`)

The full staged sweep plan — all sub-stages, the `BEST_*` knob table,
every variant's resolved config, and the end-to-end walkthrough — lives
in [`PDF_EXTRACTION_EXPERIMENT_PLAN.md`](PDF_EXTRACTION_EXPERIMENT_PLAN.md).
That document is the source of truth for what each stage answers;
`scripts/eval/run_all_sweeps.py` is the source of truth for what each
variant actually does.

Quick recap of the stage slugs (run with `--stage <slug>`):

```
detector_docling      Stage 1.1 — Docling baseline                            (01)
detector_tatr         Stage 1.2 — TATR threshold selection                    (02–04)
detector_hybrid       Stage 1.3 — Hybrid threshold selection                  (05–07)
table_in_figure       Stage 2   — TIF fixes for TATR / Hybrid                 (08–13)
header_fix            Stage 3   — drop_tables_in_top_pts on TATR / Hybrid     (14–15)
footnote_screen       Stage 4   — footnote_expand=1.2 per detector family    (16–18)
footnote_multiplier   Stage 5   — multiplier × family grid (3 × 3 = 9)        (19–27)
merge_flags           Stage 6   — merge_tables / merge_figures on BEST_BASE  (28–30)
reconstruction        Stage 7   — reconstruction × expand on BEST_BASE       (31–32)
all                   every variant
```

`--stage <name> --list-variants` confirms the resolved config for that
stage's variants after each `BEST_*` edit.  Caching is per-variant
(independent of `--stage`), so picking up later — e.g.
`--stage merge_flags` after `--stage detector_tatr` — reuses every
`_DONE.json` marker and per-stage cache already on disk.

---

## 3. Summarisation pipeline

Exactly one of `--sync` / `--batch` is **required** — there is no implicit
default, to prevent accidentally launching the wrong mode. `--profile` is
also required (`cheap` | `real`).

Both modes use the same three direct-API keys (`OPENAI_API_KEY`,
`GOOGLE_API_KEY`, `ANTHROPIC_API_KEY` — see §0). No extra env vars are
needed for `--batch`.

```bash
# Single paper, sync (live) mode — pmcid is positional, no --pmcid flag
python scripts/run_paper.py PMC7150310_main --sync  --profile cheap

# Single paper, batch (async) mode
python scripts/run_paper.py PMC7150310_main --batch --profile real

# Corpus run from a YAML selection (sync)
python scripts/run_paper.py --from-selection configs/paper_selection/runA.yaml --sync  --profile cheap

# Corpus run from a YAML selection (batch)
python scripts/run_paper.py --from-selection configs/paper_selection/calibration_set_v1.yaml --batch --profile real
```

Outputs land in `out/summaries/runs/<run_id>/`:

```
canonicalize/<pmcid>/canonical_rules.jsonl
group/<pmcid>/groups.jsonl
map/<pmcid>/findings.jsonl
normalize/<pmcid>/normal_findings.jsonl
relate/<pmcid>/relations.jsonl
resolve/<pmcid>/final_rules.jsonl
cost/cost_report.json
logs/
manifest.json
```

The pooled cross-paper relations live at `out/summaries/corpus_relations.json`.

---

## 4. Regenerating the paper-selection YAML

`configs/paper_selection/calibration_set_v1.yaml` was produced offline by
the ILP selector in `eval/paper_selection/`. To rebuild it (or produce a
new selection version) from the ingested corpus:

```bash
# DB-backed, ILP strategy (recommended). Requires `pip install pulp`.
python -m eval.paper_selection.run_select \
    --strategy ilp \
    --output-version calibration_set_v1

# Greedy fallback — no PuLP needed; faster but order-sensitive at the margins.
python -m eval.paper_selection.run_select --strategy greedy --output-version smoke_v2

# Inspect without writing anything.
python -m eval.paper_selection.run_select --strategy ilp --dry-run
```

Outputs land under `configs/paper_selection/`:

* `{version}.yaml` — consumed by `--from-selection` in §3
* `{version}_rationale.json` — full audit trail (ILP objective, sub-pool
  sizes, hardness breakdown, per-pick reason strings)
* `{version}_summary.csv` — flat per-paper table for spreadsheet review

Full algorithm — formulas, weights, ILP formulations, design rationale —
in [`readmes/PAPER_SELECTION.md`](readmes/PAPER_SELECTION.md).

---

## 5. Cost estimation

`estimate_selection_cost.py` requires `--profile` and either positional
PMCIDs or `--from-selection`:

```bash
# Project MAP cost for the calibration set under the cheap cascade profile
python scripts/estimate_selection_cost.py \
    --from-selection configs/paper_selection/calibration_set_v1.yaml \
    --profile cheap
```

`estimate_pipeline_cost_percentiles.py` takes no arguments — it loads
every ingested paper, picks P50 / P80 / P90 by `n_te`, and writes a
markdown report covering per-paper cost AND cumulative spend at cuts
10 / 25 / 50 / 75 / 80 / 90 / 100 %:

```bash
python scripts/estimate_pipeline_cost_percentiles.py
# → out/cost_percentile_report.md (also printed to stdout)

# Just the cumulative-spend section (answers "what if I run the
# smallest P% of papers?"):
sed -n '/^## Cumulative spend/,/^## Cold-run/p' \
    out/cost_percentile_report.md
```

Both scripts read prices from `configs/model_prices.json` and pull
cascade profiles live from
`pipeline/stages/summarization/batch/voter_configs.py`.

---

## 6. Tests

```bash
# Full summarisation suite (~3 min)
python -m pytest tests/summarization/ -q

# Focused — group/canonicalize hash invariants
python -m pytest tests/summarization/test_phase3_group.py tests/summarization/test_demographics.py -q

# Phase 1 of the evaluation harness
python -m pytest tests/eval/ -q
```

---

## 6a. Evaluation harness (Phase 1 — no-API proxy metrics)

`scripts/eval/compute_proxy_metrics.py` reads the summarisation pipeline's
frozen artifacts and writes no-label proxy metrics. It never calls an LLM,
embedding API, or NLI model, and never re-executes any pipeline stage. See
[`CALIBRATION_EVAL.md`](CALIBRATION_EVAL.md) for the full column reference.

```bash
python scripts/eval/compute_proxy_metrics.py \
    --input out/summaries \
    --out eval/results/proxy_metrics.csv
```

Outputs land alongside the CSV:

```
eval/results/proxy_metrics.csv               one row per pmcid + __aggregate__
eval/results/proxy_metrics_aggregate.json    nested aggregate + status_counts
eval/results/proxy_metrics.meta.json         git_commit, created_at, schema_version
```

---

## 6b. Calibration sweeps and plots (Layer A — no-API)

All Layer A sweeps live under `eval/sweeps/`. They read frozen
summarisation artifacts and never call an LLM / NLI / embedding API. See
[`eval/sweeps/README.md`](../eval/sweeps/README.md) for the Layer A
invariant and the list of which knobs are honestly sweepable today.

### Grounding-threshold retention sweep

```bash
python eval/sweeps/grounding.py
```

Reads `out/summaries/` (configurable via `--input`) and writes:

```
eval/results/grounding_sweep.csv             one row per threshold, with `#`-comment metadata
eval/results/grounding_sweep.md              human-readable report (disclaimer + retention table + per-paper detail)
```

Threshold grid defaults to `0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95`;
override with `--thresholds 0.4,0.5,0.6`. Pass `--strict-single-config`
to fail loudly when the input dir mixes producer configurations
(multiple `pipeline_config_hash` or `run_id` values).

### Grounding-sweep plots (matplotlib)

After running the sweep, generate the two thesis-ready PNGs and embed
them into the sweep markdown:

```bash
python eval/sweeps/grounding_plot.py
```

Outputs:

```
eval/results/grounding_retention.png            retention curve + mean-score-kept / mean-score-rejected overlay
eval/results/grounding_score_distribution.png   stacked histogram of all persisted grounding scores
```

The plotter also injects the two images into `grounding_sweep.md` inside
an HTML-comment fence (`<!-- BEGIN: grounding_plot.py auto-generated -->`
… `<!-- END: ... -->`) so the report and the figures travel together.
Re-running the plotter replaces the block in place rather than appending
duplicates. Pass `--sweep-md ""` to skip the markdown-embed step.

To view the figures inline, open `eval/results/grounding_sweep.md` in any
markdown renderer (VS Code preview, GitHub web UI, JetBrains preview).

### Manual-label sample (Phase 2)

When you want to hand-label findings to turn the retention sweep into a
precision/recall analysis, draw a stratified sample with:

```bash
python scripts/eval/sample_grounding_for_manual_labeling.py
```

Outputs:

```
eval/results/grounding_manual_sample.jsonl   first line is _meta header; remaining lines are placeholder rows
eval/results/grounding_manual_sample.md      bucket allocation + disclaimer
```

Defaults: `--n 100`, `--threshold 0.50`, `--seed 42`. Findings are bucketed
by score (`very_low` → `high`) and the near-threshold buckets are
over-sampled because that's where threshold calibration matters most.
Same `--seed` always produces the same JSONL — safe to regenerate.
Hand-label by setting each row's `label` field to one of
`supported` / `partial` / `unsupported`.

### Eval-harness tests

```bash
python -m pytest tests/eval/ -q
```

Covers all of the above: proxy metrics, grounding sweep, grounding plots,
manual sampler. None of them touch APIs or pipeline code.

---

## 7. Thesis demos (see [`THESIS.md`](THESIS.md))

```bash
# Synthetic Tr=3 / opacity=0 PDF — confirms R1 catches ghost text
python scripts/verify_ghost_text_detection.py

# Real-paper corpus scan (N defaults to 8; pass a positional integer to widen)
python scripts/scan_ghost_text_real_papers.py 12

# Before/after demos that back up THESIS.md (writes PNGs + JSON)
python scripts/thesis_demo_ghost_text.py
```

Artifacts land in `out/thesis_demo/ghost_text/`.

---

## 8. Inspection helpers

```bash
python scripts/inspect_pipeline_output.py --pmcid PMC7150310_main
python scripts/inspect_normalize_group.py --pmcid PMC7150310_main
python scripts/inspect_phase123_pipeline.py --pmcid PMC7150310_main
python scripts/test_map_schema.py
```

### Pipeline telemetry — what to read after a run

Append-only JSONL telemetry from the summarisation pipeline. See
[`STRUCTURE.md` § Log files](STRUCTURE.md#log-files-logs) for the full
contract and producer-site list.

```bash
# Enum coercions / alias repairs / case repairs on Finding fields (B-018, B-019).
jq -r '.field_name + "\t" + (.raw_value|tostring) + "\t" + .reason' \
  logs/enum_observations.jsonl | sort | uniq -c | sort -rn

# Failed Findings / AuditableSummaries / chunk_id repairs.
jq -r '(.context.stage // "-") + "\t" + (.context.level // "-") + "\t" + (.context.provider // "-") + "\t" + .error' \
  logs/bad_findings.jsonl | sort | uniq -c | sort -rn

# Per-chunk cascade decisions for one paper (voter_count, decision, gate_origin).
jq -r '"level=" + .level + " voter_count=" + (.voter_count|tostring) + " decision=" + .decision' \
  out/summaries/cascade_decisions/<PMCID>.jsonl

# Latest cost / escalation report.
ls -t out/summaries/reports/escalation_report_*.json | head -1 | xargs cat
```

Pre-2026-05-15 baselines live under `logs/archive/` for before/after comparison.

---

## 9. Database housekeeping

```bash
# Count text elements for a paper
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "
SELECT COUNT(*) AS total, COUNT(DISTINCT text_content) AS distinct
FROM text_elements
WHERE document_id = (SELECT id FROM documents WHERE pmcid = 'PMC7150310_main');"

# Find a paper's canonical rules
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "
SELECT canonical_id, subject_entity, outcome_entity, relation_type, predicate_text
FROM sum_canonical_rules
WHERE pmcid LIKE 'PMC7150310%';"
```

---

## 10. Per-stage output cache (B-027)

`runtime.skip_existing_outputs` reuses cached outputs for the three
expensive non-Docling stages of the PDF pipeline:

* **Stage 2** — table detection (`out/stage_cache/table_detection/<pmcid>.json`)
* **Stage 5** — artifact filtering (`out/stage_cache/filtering/<pmcid>.json`)
* **Stage 6** — text assembly (`out/stage_cache/text_assembly/<pmcid>.json`)

Each artifact has a `<pmcid>.hash` sidecar carrying the pipeline-config
hash. A cache hit requires both files to exist AND the sidecar to match
the current hash AND the loader to parse cleanly. Final writers (steps
7/8) and Docling extraction (steps 1/4 — already cached at the Docling
layer) always run.

```yaml
# configs/run.yaml
pdf_extraction:
  runtime:
    skip_existing_outputs: true   # opt in
    seed: 42                      # int | None — None opts out of seeding
```

### Clearing the stage cache

The stage cache survives most output deletions. Removing
`out/text/<pmcid>.txt`, `out/json/<pmcid>_media.json`, `out/figures/`,
`out/tables/`, `out/docling_full/`, or `out/summaries/` does **not**
invalidate the per-stage cache for stages 2/5/6. To force those stages
to recompute, either flip the knob:

```bash
# In configs/run.yaml or PipelineConfig
runtime.skip_existing_outputs: false
```

…or wipe the cache directory:

```bash
rm -rf out/stage_cache/
```

### When you MUST clear the stage cache

The pipeline-config hash captures the dataclass-shaped config surface
(Docling, TATR, masking, filtering, text assembly, two-pass, table
detector, scispaCy model name). It does **not** capture:

* **scispaCy / spaCy package or model-weights versions.** A
  `pip install --upgrade scispacy` or a model upgrade can change
  `nlp(text)` output without changing the model name string. After
  any such upgrade, run `rm -rf out/stage_cache/`.
* **Module-level constants** in cached stages (e.g.
  `_NON_BIO_ENT_LABELS`, `_SIDEBAR_METADATA`, `MIN_ANCHOR_H` in
  `parsers/layout_utils.py`). Contributors changing those should bump
  `STAGE_CACHE_VERSION[<stage>]` in
  `pipeline/stages/pdf_text_extraction/stage_cache.py` so the cache
  invalidates automatically. The constant's docstring lists the
  triggers.

### Reproducibility scope

`runtime.seed` initialises pipeline-owned RNGs (`random`, `numpy`,
`torch`, `torch.cuda`) for any work the pipeline recomputes. Existing
cache hits are reused as-is when the content config hash matches —
seed does not affect deterministic cached artifacts. Determinism for
external libraries (Docling, TATR, OCR engines, scispaCy) is not
promised.
