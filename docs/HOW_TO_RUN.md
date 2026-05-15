# HOW_TO_RUN.md

End-to-end commands to reproduce the thesis result. Every block is meant to be
copy-paste runnable from the repo root. Update this file whenever a script's
invocation, a config default, or a pipeline stage changes.

---

## 0. Environment

```bash
# Python deps
pip install -r requirements.txt

# Postgres (local default — edit .env to point elsewhere)
createdb -U postgres nlp_histo
cp .env.example .env
# fill in DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

# Schema
python database/setup_db.py            # create tables
python database/setup_db.py --check    # inspect, no changes
python database/setup_db.py --drop     # destructive: wipe + recreate
```

Optional kill-switches (useful on low-RAM machines):

```bash
export NLP_HISTO_DISABLE_UMLS=1            # skip scispaCy + UMLS entirely
export NLP_HISTO_SKIP_UMLS_ENRICHMENT=1    # load scispaCy but skip CUI enrichment
```

---

## 1. Acquire papers

```bash
cd file-selector
python file_downloader.py     # needs target_pmc_ids.txt
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
| `out/run_metadata/`   | Per-paper timing/processing stats         |
| `out/failed_pdfs_blacklist.json` | Skip list (thread-safe)        |

---

## 3. Summarisation pipeline

```bash
# Single paper, sync (live) mode — pmcid is positional, no --pmcid flag
python scripts/run_paper.py PMC7150310_main --sync

# Single paper, default (batch / async) mode
python scripts/run_paper.py PMC7150310_main

# Cheap-tier corpus run from a YAML selection (matches configs/paper_selection/runA.yaml)
python scripts/run_paper.py --from-selection configs/paper_selection/runA.yaml --sync
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

## 4. Cost estimation

```bash
python scripts/estimate_selection_cost.py
python scripts/estimate_pipeline_cost_percentiles.py
```

---

## 5. Tests

```bash
# Full summarisation suite (~3 min)
python -m pytest tests/summarization/ -q

# Focused — group/canonicalize hash invariants
python -m pytest tests/summarization/test_phase3_group.py tests/summarization/test_demographics.py -q
```

---

## 6. Thesis demos (see [`THESIS.md`](THESIS.md))

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

## 7. Inspection helpers

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

## 8. Database housekeeping

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

## 9. Per-stage output cache (B-027)

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
