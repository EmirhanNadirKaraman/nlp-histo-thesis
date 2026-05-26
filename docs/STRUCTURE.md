# STRUCTURE.md

Authoritative map of the pipeline architecture. **Update this file in the same
commit as any change to pipeline stages, runners, configs, DB schema, or
artifact layout.** If a file move, rename, or new stage isn't reflected here
within the commit that introduced it, the change is incomplete.

The detailed per-file project walkthrough lives in
[`.claude/CLAUDE.md`](../.claude/CLAUDE.md); this file is the higher-level
architectural index — what flows where, and which files are load-bearing.

---

## Top-level layout

```
nlp-histo/
├── file-selector/          Stage 1 — download PMC tarballs, organise PDFs/XMLs
├── parsers/                Shared parsing utilities (layout, text, XML)
├── pipeline/               Stage 2 — two sub-pipelines
│   ├── utils/              Cross-pipeline utilities (memory logging, …)
│   └── stages/
│       ├── pdf_text_extraction/   PDF → hierarchical text + media → DB
│       └── summarization/         Text → structured rules via LLM cascade
├── database/               SQLAlchemy ORM, connection pool, schema setup
├── named_entity_recognition/  scispaCy + UMLS entity extraction
├── eval/                   Evaluation harness, annotation tools, paper selection
├── scripts/                One-off / inspection / thesis-demo scripts
├── tests/                  pytest suite (summarisation-heavy)
├── configs/                YAML run configs, NLI/pricing tables
├── notebooks/              Demo / workflow notebooks
├── out/                    Runtime outputs (cached layouts, summaries, demos)
└── files/                  Input PDFs/XMLs (not in repo)
```

---

## Pipeline A — PDF text extraction

**Orchestrator:** [`pipeline/stages/pdf_text_extraction/runner.py`](../pipeline/stages/pdf_text_extraction/runner.py) — `PipelineRunner.run_document(pdf_path, pmcid)`

**Active 8-step flow:**

```
PDF
  │
  ▼
 (1) Extract full layout (Docling) ─────────── out/docling_full/*.json (cached)
  │
  ▼
 (2) Detect tables (Docling | TATR | Hybrid)  ─ TableDetectionResult
  │
  ▼
 (3) Mask detected regions (PyMuPDF)           ─ out/masked_pdfs/*.pdf
  │
  ▼
 (4) Re-extract layout from masked PDF         ─ out/docling_masked/*.json
  │
  ▼
 (5) Filter artifacts (layout_utils)
  │
  ▼
 (6) Assemble hierarchical text                ─ List[HierarchicalRow]
  │
  ▼
 (7) Crop figure/table images                  ─ out/figures/, out/tables/
  │
  ▼
 (8) Write outputs (text files + Postgres)
```

When `TwoPassConfig.enabled=True` (default since May-2026), steps 1/3/4 are
replaced by a single `TwoPassTextExtractor` call that gathers per-element
evidence and applies `NodeScorer` rules R1/R-color/R2/R3 *before* text reaches
later stages.

### Key files

| Role                              | File                                                                |
|-----------------------------------|---------------------------------------------------------------------|
| Orchestrator                      | `pipeline/stages/pdf_text_extraction/runner.py`                     |
| Config (all sub-configs)          | `pipeline/stages/pdf_text_extraction/config.py`                     |
| Batch driver (ThreadPoolExecutor) | `pipeline/stages/pdf_text_extraction/batch.py`                      |
| Blacklist (skip list)             | `pipeline/stages/pdf_text_extraction/blacklist.py`                  |
| Two-pass extractor                | `pipeline/stages/pdf_text_extraction/components/two_pass_extractor.py` |
| Evidence gatherer (PyMuPDF)       | `pipeline/stages/pdf_text_extraction/components/evidence_gatherer.py` |
| NodeScorer (R0–R3 rules)          | `pipeline/stages/pdf_text_extraction/components/node_scorer.py`     |
| Layout extractor (Docling + cache)| `pipeline/stages/pdf_text_extraction/components/layout_extractor.py` |
| Region masker                     | `pipeline/stages/pdf_text_extraction/components/region_masker.py`   |
| Text assembler                    | `pipeline/stages/pdf_text_extraction/components/text_assembler.py`  |
| Artifact filter                   | `pipeline/stages/pdf_text_extraction/components/artifact_filter.py` |
| Media cropper                     | `pipeline/stages/pdf_text_extraction/components/media_cropper.py`   |
| Visualiser                        | `pipeline/stages/pdf_text_extraction/components/visualizer.py`      |
| Table detectors                   | `pipeline/stages/pdf_text_extraction/table_detectors/{docling,tatr,hybrid}_detector.py` |
| Output writers                    | `pipeline/stages/pdf_text_extraction/outputs/{writer,db_ingester,media_json_writer}.py` |
| Shared layout helpers             | `parsers/layout_utils.py`                                           |
| Text post-processing              | `parsers/text_processing.py`                                        |

### Config sub-configs (`pipeline/stages/pdf_text_extraction/config.py`)

`PathConfig`, `DoclingConfig`, `TATRConfig`, `MaskingConfig`,
`FilteringConfig`, `CroppingConfig`, `TextAssemblyConfig`,
`VisualizationConfig`, `DatabaseConfig`, `RuntimeConfig`, `TwoPassConfig`.

Enums: `TableDetectorType ∈ {TATR, DOCLING, HYBRID, VLM}`,
`BaselineMode ∈ {MASKED, UNMASKED, BOTH}`.

---

## Pipeline B — Summarisation

**Orchestrator:** [`pipeline/stages/summarization/runner.py`](../pipeline/stages/summarization/runner.py) — `SummarizationRunner.process(file_data) -> dict`

**Stage flow (one paper):**

```
Sentences (per paper)
  │
  ▼ MAP             current_stages/map_stage.py             3-tier ABC cascade
  │                                                          → AuditableSummary[] per chunk
  ▼ GROUNDING       helpers/grounding_filter.py             NLI entailment filter
  ▼ NORMALIZE       current_stages/normalize_stage.py       UMLS-normalised entities + dedup
  ▼ GROUP           current_stages/group_stage.py           NormalFinding[] → FindingGroup[]
  │                                                          (group_id includes pmcid since May-2026)
  ▼ CANONICALIZE    current_stages/canonicalize_stage.py    Pick predicate, split by direction → CanonicalRule[]
  ▼ RELATE          current_stages/relate_stage.py          NLI pairwise → Relation[]
  ▼ RESOLVE         current_stages/resolve_stage.py         Score → FinalRule[]

[Optional, off by default]
  REDUCE            old_stages/reduce_stage.py              Tree-reduce chunk summaries → ConsolidatedSummary
  RULES             old_stages/rule_stage.py                IF-THEN extraction → ExtractedRules
```

Cross-paper relations are produced separately by
`helpers/corpus_relate.py:CorpusRelateStage` over the pooled per-paper
`CanonicalRule[]`.

### Three-tier voter cascade (MAP)

| Tier | Models                                                        | Approx. \$/call |
|------|----------------------------------------------------------------|----------------|
| L1   | DeepSeek, Gemini Flash-Lite, Mistral Large                     | ~\$0.001       |
| L2   | Gemini Flash, Kimi K2.5, Claude Haiku — fires on L1 disagreement | mid-tier       |
| L3   | Claude Sonnet 4.6 — final escalation when L1+L2 disagree       | premium        |

### Key files

| Role                                | File                                                               |
|-------------------------------------|--------------------------------------------------------------------|
| Orchestrator                        | `pipeline/stages/summarization/runner.py`                          |
| Persistence helpers                 | `pipeline/stages/summarization/persistence.py`                     |
| Config                              | `pipeline/stages/summarization/config.py`                          |
| All Pydantic models                 | `pipeline/stages/summarization/models.py`                          |
| LLM prompt builders                 | `pipeline/stages/summarization/prompts.py`                         |
| LLM call cache                      | `pipeline/stages/summarization/cache.py`                           |
| LLM provider wiring                 | `pipeline/stages/summarization/llm_providers.py`                   |
| Error classification (retry / fail) | `pipeline/stages/summarization/llm_errors.py`                      |
| UMLS singleton                      | `pipeline/stages/summarization/umls_resources.py`                  |
| UMLS lookup helpers                 | `pipeline/stages/summarization/umls_utils.py`                      |
| Synonym table                       | `pipeline/stages/summarization/synonyms.yaml`                      |
| Cross-paper relate stage            | `pipeline/stages/summarization/helpers/corpus_relate.py`           |
| Grounding NLI filter                | `pipeline/stages/summarization/helpers/grounding_filter.py`        |
| Async batch dispatch                | `pipeline/stages/summarization/batch/{runner,dispatch,models,voter_configs}.py` |
| Trace collector                     | `pipeline/stages/summarization/observability/`                     |
| Usage / cost accounting             | `pipeline/stages/summarization/costing/`                           |
| Agreement scorers                   | `pipeline/stages/summarization/agreement/`                         |
| MAP router (provenance gates)       | `pipeline/stages/summarization/routing/`                           |
| Artifact writers                    | `pipeline/stages/summarization/output/`                            |

### Critical invariants

* `group_id = GRP_{sha8(pmcid)}_{sha8(subject)}_{sha8(outcome)}_{relation}_{sha8(category)}`
  — includes `pmcid` so identical (subject, outcome, relation, category)
  tuples in two different papers produce distinct group IDs (fix for the
  cross-paper `canonical_id` collision; see [`THESIS.md`](THESIS.md) §1).
* `canonical_id = CR_{sha8(group_id)}_{direction}` — inherits the pmcid
  embedding via `group_id`.
* Cross-paper merging happens via the
  `helpers/corpus_relate.py:_should_compare_cross_paper` gate (CUIs first,
  normalised entity strings as fallback) — never via `canonical_id` equality.

---

## Pipeline C — Paper selection

**Orchestrator:** [`eval/paper_selection/run_select.py`](../eval/paper_selection/run_select.py) — CLI driver, no LLM / API calls.

Offline pipeline that builds the YAML selection files consumed by
`scripts/run_paper.py --from-selection`. Produces three buckets — `related`,
`diverse`, `hard` — with rationale + flat summary alongside.

```
RawPaper (DB / JSONL)
  │   loaders.py            DBLoader | JSONLLoader
  ▼
PaperFingerprint            fingerprints.py:build_fingerprints
  │     ─ UMLS-TUI bucketed entity sets (disease/biomarker/gene/tissue/method/outcome)
  │     ─ Workload counters, layout counters, source_stats
  ▼
SelectionResult             selectors.py (greedy)  | ilp_selectors.py (PuLP-CBC, opt-in)
  │     ─ select_related_papers(_ilp)
  │     ─ select_diverse_papers(_ilp)
  │     ─ select_hard_papers(_ilp)
  ▼
configs/paper_selection/{version}.yaml           ← consumed by --from-selection
configs/paper_selection/{version}_rationale.json ← full audit trail
configs/paper_selection/{version}_summary.csv    ← flat per-paper table
                          export.py:write_calibration_set
```

### Key files

| File | Role |
|---|---|
| `eval/paper_selection/run_select.py` | CLI entry point + post-selection validation (count, useful-entities, rel-ordering, hard-ordering sanity checks). |
| `eval/paper_selection/loaders.py` | `DBLoader` (default) + `JSONLLoader` (fallback) → `RawPaper` records. |
| `eval/paper_selection/fingerprints.py` | TUI semantic-type bucketing, regex extractors (`CD-N`, `Ki-67`, gene-like uppercase), curated keyword dicts → `PaperFingerprint`. |
| `eval/paper_selection/models.py` | Pure-dataclass DTOs: `PaperFingerprint`, `HardnessBreakdown`, `SelectionResult`. |
| `eval/paper_selection/metrics.py` | Pluggable Protocols: `Relatedness` (PairMetric), `Diversity` (SetMetric), `Hardness` (PaperMetric → `HardnessBreakdown`). |
| `eval/paper_selection/selectors.py` | Greedy bucket selectors (default). `select_calibration_set` composes the three buckets with default mutual exclusion. |
| `eval/paper_selection/ilp_selectors.py` | ILP bucket selectors via PuLP-CBC. Candidate pruning + edge sparsification; per-bucket time budget; pre-score fallback on infeasibility. |
| `eval/paper_selection/export.py` | YAML / JSON / CSV writers; tiny in-tree YAML emitter — no PyYAML dependency. |

### Critical invariants

* **Strategy-agnostic.** `(papers, rationale)` shape is identical between
  greedy and ILP; downstream consumers (`run_select.py`, `export.py`)
  never branch on strategy.
* **Deterministic.** Both strategies are deterministic for a given
  fingerprint list + metric config. ILP ties broken by PMCID ordering;
  greedy picks broken by score then PMCID.
* **Fallback chain.** ILP infeasible / no usable solution →
  `_prescore_fallback` returns top-`k` by pre-score with
  `ilp_solution_quality="prescore_fallback"` in the rationale; PuLP not
  installed → CLI errors unless `--ilp-fallback-greedy`.
* **YAML is the only consumer-facing artifact.** JSON + CSV exist for
  thesis evidence — picking, hardness breakdowns, ILP objective, sub-pool
  reasons.

Full algorithm spec — formulas, weights, design rationale — in
[`docs/readmes/PAPER_SELECTION.md`](readmes/PAPER_SELECTION.md).

---

## Database

**ORM:** `database/models.py`

| Table                                | Purpose                                              |
|--------------------------------------|------------------------------------------------------|
| `documents`                          | Per-paper metadata (unique `pmcid`)                  |
| `text_elements`                      | One row per paragraph with `unique_path`, `path_list` (TEXT[]), `path_string`, `depth`, `position_in_section` |
| `figures`, `tables`                  | Cropped media with bbox/page metadata                |
| `entities`                           | scispaCy/UMLS named entities                         |
| `text_element_figure_references`     | Many-to-many text ↔ figure                           |
| `text_element_table_references`      | Many-to-many text ↔ table                            |
| `pipeline_runs`                      | Summarisation run manifest                           |
| `sum_map_findings`                   | MAP-stage output (AuditableSummary[] flattened)      |
| `sum_normal_findings`                | NORMALIZE output                                     |
| `sum_normal_finding_spans`           | Provenance spans for NormalFindings                  |
| `sum_finding_groups`                 | GROUP output (`group_id` is the pmcid-namespaced hash)|
| `sum_group_members`                  | NormalFinding ↔ FindingGroup junction                |
| `sum_canonical_rules`                | CANONICALIZE output                                  |
| `sum_relations`                      | RELATE output (intra-paper)                          |
| `sum_corpus_relations`               | CorpusRelateStage output (intra + cross paper)       |
| `sum_final_rules`                    | RESOLVE output                                       |
| `sum_rejected_findings`              | Findings filtered out during NORMALIZE/GROUP         |
| `sum_rejection_summaries`            | Per-paper rejection roll-up                          |
| `llm_judge_cache`                    | LLM judge call cache (semantic agreement scoring)    |

**Connection:** `database/db_connection.py` →
`get_db_connection().session_scope()`.

**Schema setup:** Alembic — `alembic upgrade head` to create the schema,
`alembic current` to inspect the version. (Legacy `database/setup_db.py`
has been removed.)

---

## Output directories (runtime)

```
out/
├── docling_full/         Full Docling layout JSON (cached)
├── docling_masked/       Docling layout JSON from masked PDFs
├── masked_pdfs/          PDFs with detected regions whited out
├── text/                 Hierarchical text
├── text_raw/             Pre-assembly raw elements
├── figures/, tables/     Cropped media
├── visualization/        Annotated debug PDFs
├── json/                 Per-paper media JSON
├── run_metadata/         Per-paper timing + processing stats
├── summaries/            Summarisation artifacts (runs/<run_id>/...)
│   ├── summaries/        Per-paper result cache `<pmcid>.json`
│   │                       (stamped with pipeline_config_hash for cache invalidation; B-007 / B-008)
│   ├── cascade_decisions/ Per-paper `<pmcid>.jsonl` — one JSONL row per L1/L2/L3 chunk decision
│   │                       (always-on; emit schema per STRUCTURE.md 2026-05-14 changelog)
│   └── reports/          `escalation_report_<UTCstamp>.{json,csv}` — chunk counts, token usage,
│                           est_cost_usd, est_saved_usd per paper + totals
├── thesis_demo/          Thesis demo PNG/JSON artifacts
└── failed_pdfs_blacklist.json
```

## Log files (`logs/`)

Lower-volume, line-oriented telemetry that does NOT belong under `out/` (these
survive across runs and aggregate behaviour over time). All paths are relative
to the repo root unless `NLP_HISTO_LOG_DIR` is set.

| File                              | Producer                                                                                                            | Use                                                                                                                                                                                       |
|-----------------------------------|---------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `logs/enum_observations.jsonl`    | `pipeline/stages/summarization/enum_logging.py::log_enum_observation`                                               | One JSONL row per enum coercion / alias repair on a MAP `Finding` field (`category`, `relation_type`, `direction`, `confidence`). Reasons: `alias_repair`, `case_repair`, `missing`, `null_to_no_direction`, `unknown_value`, `invalid_literal_dropped`. Aggregate with `jq` to mine LLM-side label drift (see B-018, B-019). |
| `logs/bad_findings.jsonl`         | `pipeline/stages/summarization/enum_logging.py::log_bad_finding` (called from `AuditableSummary._drop_invalid_findings`, `MapStage._timed_invoke`, `_invoke_l3`, `batch.dispatch.parse_result`, `batch.runner._process_level`) | Full payload of every Finding/AuditableSummary that failed Pydantic validation OR was repaired (chunk_id mismatch). Context block records `pmcid`, `chunk_id`, `level`, `voter_index`, `provider`, `model`, `attempt`. Used to catalogue LLM contract violations and to drive future schema-tightening decisions. |
| `logs/runA*.log`                  | Manual runs of `scripts/run_paper.py` redirected via shell                                                          | Full text log of a pipeline invocation. Format depends on the redirection command — these are not produced by the pipeline itself, they are by-products of the operator. Safe to delete.   |
| `logs/archive/`                   | Operator                                                                                                            | Pre-fix baselines moved here on 2026-05-15 (e.g. `enum_observations_pre_2026-05-15.jsonl`) so post-fix aggregations stay clean while the baseline survives for thesis comparison.            |

Writes are append-only and best-effort — `enum_logging.py::_append_jsonl` swallows
I/O exceptions so telemetry never breaks the pipeline. Override the directory
via `NLP_HISTO_LOG_DIR=<absolute_path>` if you run from different working dirs
and want all logs to coalesce.

---

## Scripts that matter

| Script                                          | Purpose                                                   |
|-------------------------------------------------|-----------------------------------------------------------|
| `scripts/run_paper.py`                          | End-to-end driver for one paper                           |
| `python -m eval.paper_selection.run_select`     | Build a new `configs/paper_selection/{version}.yaml`      |
| `scripts/estimate_selection_cost.py`            | Cheap-tier cost estimate before a run                     |
| `scripts/estimate_pipeline_cost_percentiles.py` | Cost-percentile sweep                                     |
| `scripts/inspect_pipeline_output.py`            | HTML inspector for a single paper                         |
| `scripts/inspect_normalize_group.py`            | Walks NORMALIZE → GROUP for one paper                     |
| `scripts/inspect_phase123_pipeline.py`          | Walks MAP → NORMALIZE → GROUP for one paper               |
| `scripts/verify_ghost_text_detection.py`        | Synthetic Tr=3 / opacity=0 PDF → R1 pixel-check passes    |
| `scripts/scan_ghost_text_real_papers.py`        | Corpus-sample ghost-text scan                             |
| `scripts/thesis_demo_ghost_text.py`             | Before/after demos (writes PNGs + JSON for THESIS.md)     |
| `scripts/test_map_schema.py`                    | Standalone MAP-output schema validator                    |
| `scripts/eval/compute_proxy_metrics.py`         | No-API proxy metrics over frozen summarisation artifacts (Phase 1 of the eval harness, see [`CALIBRATION_EVAL.md`](CALIBRATION_EVAL.md)) |

---

## Coordinate system

| System              | Origin             | Used by                                  |
|---------------------|--------------------|------------------------------------------|
| Docling PDF coords  | y=0 at **bottom**; y1=top, y2=bottom | `LayoutElement.bbox`, `BoundingBox` |
| fitz/screen coords  | y=0 at **top**; rect.y0 < rect.y1   | PyMuPDF masking, cropping, rendering |

Convert via `BoundingBox.to_fitz_rect(page_height)` /
`BoundingBox.from_fitz_rect(rect, page_height, page)`.

---

## When to edit this file

* New pipeline stage, file move, or significant rename → update the relevant
  section here in the **same commit** as the code change.
* New table or schema column → update the Database table.
* New top-level output directory → add it to the Output directories list.
* New thesis-relevant script → add a row to "Scripts that matter".
* Config defaults that affect the whole pipeline (e.g. the May-2026
  `TwoPassConfig.enabled` flip) → mention in the affected pipeline section
  *and* link to the THESIS.md entry that motivated the change.
* **Always** append a row to the Pipeline changelog below so the history is
  visible at a glance even without diffing the file.

---

## Pipeline changelog

Append one row per change that affects pipeline architecture, behaviour, or
artifact layout. Date in ISO format. Link to the THESIS.md bug / decision
that motivated the change when applicable.

| Date       | Area                              | Change                                                                                                                       | Motivated by |
|------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------|--------------|
| 2026-05-13 | Summarisation › GROUP             | `_group_id` now hashes `pmcid` alongside `(subject, outcome, relation, category)`. `canonical_id` inherits the namespace.    | THESIS.md [Bug 1](THESIS.md#bug-1--duplicate-intra-paper-relations-produced-by-canonical_id-collisions) |
| 2026-05-13 | PDF extraction › TwoPassConfig    | `enabled` default flipped `False → True`. NodeScorer now runs on every Docling element by default.                           | THESIS.md [Bug 2](THESIS.md#bug-2--docling-phantom-layout-elements) |
| 2026-05-13 | PDF extraction › TwoPassConfig    | `max_white_char_fraction` default flipped `0.5 → 1.0` — Rule R-color disabled by default.                                    | THESIS.md [Bug 3](THESIS.md#bug-3--r-color-white-text-false-positive-latent) |
| 2026-05-13 | Scripts                           | Added `verify_ghost_text_detection.py`, `scan_ghost_text_real_papers.py`, `thesis_demo_ghost_text.py`.                       | THESIS.md [Topic — ghost-text detection](THESIS.md#topic--ghost-text-detection-empirical-verification-and-policy-fix) |
| 2026-05-13 | Docs                              | New `docs/` folder with `THESIS.md`, `HOW_TO_RUN.md`, `STRUCTURE.md`. CLAUDE.md updated to enforce keeping them in sync.      | Thesis-supporting docs request |
| 2026-05-13 | Docs                              | Consolidated pre-existing root MDs into `docs/`: `REPOSITORY_GUIDE.md`, `PIPELINE_BUGS.md`, `TODO.md`. Moved `readmes/` → `docs/readmes/`. `README.md` stays at repo root. Co-located module docs untouched. | Tidy-up |
| 2026-05-14 | Summarisation › `BatchSummarizationRunner` | `finalize()` now mirrors sync: verbatim-from-DB before grounding, stable `finding_id`, DB persistence (`sum_*` tables), `corpus_relate_incremental`, `rejection_summary` build + persist, optional NER. `__init__` accepts `db`/`force_rerun`/`run_ner`. `BatchHandle` carries `pipeline_run_db_id` + `cached_result_only`. | THESIS.md [Bug 5](THESIS.md#bug-5--batch-runner-missing-sync-parity-features) |
| 2026-05-14 | Summarisation › `BatchSummarizationRunner` | Result caching added: `_load_result`/`_save_result` stamp + check `pipeline_config_hash`; valid cache short-circuits at `submit()` so stale results cannot consume L1/L2/L3 batch dollars. Pre-fix `out/summaries/summaries/*.json` deleted to force regeneration. | THESIS.md [Bug 5](THESIS.md#bug-5--batch-runner-missing-sync-parity-features) |
| 2026-05-14 | Inspector › batch-index template | Fixed `dataset.nilBa` typo (NLI B→A sort) and added missing `.badge-blue` class (SCOPE_QUALIFY rendering). | THESIS.md [Bug 13](THESIS.md#bug-13--inspector-nli-ba-sort-typo), [Bug 14](THESIS.md#bug-14--inspector-badge-blue-class-missing) |
| 2026-05-14 | Summarisation › MAP scorer default | `SummarizationRunner` + `BatchSummarizationRunner` now default to `SemanticAgreementScorer(EmbeddingSimilarityStrategy)` (Soiffer max-consensus + centrality). Previous default `EmbeddingScorer` left `best_index` unset and `AgreementChecker.best()` fell back to a `(mean_evidence_length, n_findings)` heuristic. | [`ABC_IMPLEMENTATION_COMPARISON.md` §5 Gap 1](ABC_IMPLEMENTATION_COMPARISON.md#gap-1-no-centrality-based-output-selection-on-default-path) |
| 2026-05-14 | Summarisation › MAP cascade | `MapOutputRouter` wired into both runners by default (`enable_router=True`). Schema + provenance validation runs before agreement; voters classified UNUSABLE are dropped before scoring. Router-on path escalates **L1 → L3 directly** (skips L2). Opt-out via `enable_router=False`. | [`ABC_IMPLEMENTATION_COMPARISON.md` §5 Gap 2 + Gap 8](ABC_IMPLEMENTATION_COMPARISON.md#gap-2-grounding-does-not-gate-the-cascade) |
| 2026-05-14 | Summarisation › MAP cascade | Per-level KEEP/escalate decision factored into shared `agreement/decision.py::evaluate_chunk`. Both `MapStage._cascade` (sync) and `BatchSummarizationRunner._process_level` (batch) now call the same function, so the cascade decision is identical between sync and batch by construction. | [`ABC_IMPLEMENTATION_COMPARISON.md` §5 Gap 3](ABC_IMPLEMENTATION_COMPARISON.md#gap-3-sync-and-batch-run-different-cascade-code) |
| 2026-05-14 | Summarisation › Observability | New `CascadeDecisionLog` writes one JSONL row per L1/L2/L3 decision to `output_dir/cascade_decisions/{pmcid}.jsonl`. Always-on (independent of `trace_enabled`). Schema: `run_id, pmcid, chunk_id, level, voter_count, eligible_voter_count, decision, gate_origin, reason_codes, deferral_score, best_voter_index, selected_provider, selected_model, cascade_signature, cascade_profile, timestamp`. | [`ABC_IMPLEMENTATION_COMPARISON.md` §9 P0 task 4](ABC_IMPLEMENTATION_COMPARISON.md#p0--must-have-before-expensive-experiments) |
| 2026-05-14 | Docs › Evaluation design | New [`docs/STAGE_EVAL_EXPERIMENTS.md`](STAGE_EVAL_EXPERIMENTS.md): a model-output-agnostic per-stage P/R/F1 battery (M/G/N/GR/C/R/Rs + cross-run X1–X5) using PMC source text, pinned NLI, pinned UMLS, structural replays, and cross-cascade differentials. Complements `eval/llm_judge/STAGE_EVAL_DESIGN.md` (Opus-silver dependent). | Thesis-eval portability — avoid regenerating silver labels every time the MAP cascade changes |
| 2026-05-15 | Summarisation › MAP enum coercion | Added `_RELATION_TYPE_ALIASES = {"prognosis": "prognostic"}` and an alias-repair branch in `_coerce_invalid_relation_type` (`models.py`). Bumped `MAP_SCHEMA_VERSION` → `"map_v2_relation_type_alias_repair"`, invalidating the MAP cache. Prognostic findings that were previously coerced to `unclear` and dropped at `is_groupable()` now flow through GROUP → CANONICALIZE → RELATE → RESOLVE. | BUGS.md [Bug 18](BUGS.md#bug-18--relation_type-prognosis-noun-form-coerced-to-unclear-instead-of-prognostic) |
| 2026-05-15 | Summarisation › MAP enum coercion (II) | Extended enum-repair to cover same-stem and casing variants. Added `_RELATION_TYPE_ALIASES["treatment"]="treatment_response"`; new `_CATEGORY_VALID`/`_CATEGORY_CANONICAL_BY_LOWER` and case-insensitive repair branches in `_repair_category_alias` / `_coerce_invalid_relation_type` / `_coerce_invalid_direction`; new `_repair_confidence_casing` validator; structured logging of category-invalid drops in `AuditableSummary._drop_invalid_findings`. Added MAP-prompt examples for `staging` and `molecular_genetics` (previously had no relation_type mapping); tightened prompt to forbid null `subject_entity`/`outcome_entity`. Bumped `MAP_PROMPT_VERSION` → `"map_prompt_v3_staging_molgen_no_null_subject"`. | BUGS.md [Bug 18](BUGS.md#bug-18--relation_type-prognosis-noun-form-coerced-to-unclear-instead-of-prognostic), [`MAP_PROMPT_AUDIT.md`](MAP_PROMPT_AUDIT.md) |
| 2026-05-15 | Summarisation › `MapOutputRouter` citation regex | Relaxed `_CITATION_RE` in both `routing/schema_validator.py` and `routing/provenance_validator.py` from `PMC\d+` to `PMC[\w\-]+` so suffixed document IDs (`PMC10100421_HIS-82-393`, `PMC7150310_main`, `_his0066-0409`, …) parse cleanly. Pre-fix: every voter on every suffixed-pmcid paper was classified UNUSABLE (`ReasonCode.INVALID_SENTENCE_ID` ∈ `_HARD_CODES`), reducing every L1 decision to `voter_count=1` and silently bypassing the 3-voter consensus design. Cross-document safety check at `provenance_validator.py:116` unchanged. Bumped `MAP_SCHEMA_VERSION` → `"map_v4_citation_regex_suffixed_pmcids"` so cached AuditableSummary rows selected under single-voter routing recompute. | BUGS.md [Bug 19](BUGS.md#bug-19--mapoutputrouter-citation-regex-rejects-suffixed-pmcids-silently-strips-voters) |
| 2026-05-15 | Summarisation › batch dispatch grouping | `dispatch.submit_level` now groups requests by `(provider, model)` instead of `provider` only. Pre-fix: every multi-model OpenAI batch (cheap-profile L1 with `gpt-4o-mini` + `gpt-4.1-nano`) was rejected by the OpenAI Batch API with `BatchError(code='mismatched_model')` — every batched run since the multi-model L1 profile was introduced silently degraded to single-voter (Gemini only) at L1. Bumped `MAP_SCHEMA_VERSION` → `"map_v5_batch_group_by_provider_model"` to invalidate stale batch handles. Also added MAP-stage observability for chunk_id round-trip mismatches and AuditableSummary parse failures: `log_bad_finding` is now called from `MapStage._timed_invoke`, `_invoke_l3`, `batch.dispatch.parse_result`, and `batch.runner._process_level` with `pmcid` / `chunk_id` / `level` / `voter_index` / `provider` / `model` / `attempt` context. Sync L1/L2 voters retry on mismatch via the existing 2-attempt loop; L3 and batch repair in place (cannot re-submit). | BUGS.md [Bug 20](BUGS.md#bug-20--batch-dispatch-groups-by-provider-only-not-provider--model--openai-multi-model-batches-rejected-silently) |
| 2026-05-15 | Summarisation › MAP cross-field bleed | Extended `_RELATION_TYPE_ALIASES` with `morphology→has_feature`, `ihc→expression`, `molecular_genetics→expression` so category-name leaks into `relation_type` recover instead of dropping at GROUP. New `_CATEGORY_NAMES_LOWER` set + `reason="cross_field_bleed"` branch in `_coerce_invalid_relation_type` tags every category-name leak in `enum_observations.jsonl` (recovered or not). `staging` deliberately left unaliased — descriptive vs prognostic split needs claim context, falls through to `unclear` with the bleed tag for measurement. MAP prompt: added a sharp "Invalid relation_type values" anti-pattern enumeration under the `relation_type` field definition + a third molecular-genetics example showing the prognostic crossover (`MYD88 L265P mutation associated with inferior OS` → `relation_type: prognostic`). Bumped `MAP_PROMPT_VERSION` → `"map_prompt_v4_relation_type_anti_pattern"` and `MAP_SCHEMA_VERSION` → `"map_v6_cross_field_bleed_aliases"` so cached `unclear` rows re-validate through the new aliases. | BUGS.md [Bug 44](BUGS.md#bug-44--map-relation_type-bleeds-category-names-and-loses-findings-at-group) |
| 2026-05-15 | Summarisation › CANONICALIZE direction split | Fixed `"None"` → `"no_direction"` typo in `_compute_scope_fields` and `_split_by_direction` (`canonicalize_stage.py:57, 98`). Pre-fix: every NormalFinding with `direction=no_direction` got its own CanonicalRule bin and (in mixed groups) flipped `is_conflicted=True` against any real-polarity finding. Post-fix: `no_direction` is treated like `unclear` — bypasses the polarity split and attaches to the dominant polarity bin. No schema or cache invalidation needed (helpers run at canonicalize time, output JSONLs regenerate next run). Added 9-test regression file `tests/summarization/test_canonicalize_direction_split.py`. | BUGS.md [Bug 21](BUGS.md#bug-21--canonicalize-no_direction-treated-as-real-polarity-due-to-none-string-typo) |
| 2026-05-15 | Cross-pipeline › scispaCy small-model singleton | New `umls_resources.get_small_nlp(model_name)` exposes a thread-safe per-model cache for the small scispaCy model (no UMLS linker attached). Routed `PipelineRunner._get_nlp` (`pipeline/stages/pdf_text_extraction/runner.py`) and `SummarizationRunner.load_paper_from_db` (`pipeline/stages/summarization/runner.py`) through it — both previously called `spacy.load("en_core_sci_sm")` directly, double-loading the model when both pipelines ran in the same process (B-029) and reloading once per paper in batch mode (B-038). Honours `$NLP_HISTO_DISABLE_UMLS` to stay consistent with the large-model loader. Regression test `tests/summarization/test_scispacy_singleton.py` asserts `spacy.load(` only appears in `umls_resources.py` across the `pipeline/stages/` tree. | BUGS.md [Bug 29](BUGS.md#bug-29--pipelinerunner_get_nlp-bypasses-umls_resources-singleton) + [Bug 38](BUGS.md#bug-38--summarizationrunnerload_paper_from_db-bypasses-scispacy-singleton) |
| 2026-05-15 | Summarisation › MAP `FindingScope.scope_parsed` | New `@model_validator(mode="after") _compute_scope_parsed` on `FindingScope` (`models.py`) recomputes `scope_parsed = any(sub_field is not None)`; LLM-emitted value is ignored. MAP prompt updated to tell voters to always emit `false` for this field; OutputFormat block unchanged because OpenAI strict schema requires every property. Bumped `MAP_SCHEMA_VERSION` → `"map_v7_scope_parsed_autocompute"`. Regression test `tests/summarization/test_scope_parsed_autocompute.py`. | BUGS.md [Bug 45](BUGS.md#bug-45--scope_parsed-is-llm-set-but-trivially-derivable), [`MAP_PROMPT_AUDIT.md` Issue 5](MAP_PROMPT_AUDIT.md#issue-5--scopescope_parsed-is-llm-set-but-trivially-derivable-low) |
| 2026-05-15 | Summarisation › MAP `direction` alias-repair | New `_DIRECTION_ALIASES` in `models.py` (`maybe`/`possibly`/`perhaps`/`likely`/`unknown` → `unclear`; `none`/`n/a`/`na` → `no_direction`); alias-repair branch in `_coerce_invalid_direction` runs after case-fold and logs `reason="alias_repair"`. Same shape as the `_RELATION_TYPE_ALIASES` mechanism shipped for B-018 / B-044. Bumped `MAP_SCHEMA_VERSION` → `"map_v8_direction_alias_repair"`. Tests: `tests/summarization/test_enum_alias_repair.py` (14 new parametrised cases). | BUGS.md [Bug 46](BUGS.md#bug-46--direction-hedging-words-coerce-to-unclear-instead-of-alias-repair), [`MAP_PROMPT_AUDIT.md` Issue 8](MAP_PROMPT_AUDIT.md#issue-8--directionmaybe-single-occurrence-low) |
| 2026-05-15 | Summarisation › MAP `direction` rubric (prompt) | Added a "Disambiguating absent vs negative (relation_type=expression only)" block under the `direction` definition (`prompts.py:104-119`): for `expression`, "negative staining"/"no expression detected" → `absent`, "decreased"/"reduced expression" → `negative`; other relation_types default to `negative` unless text says "absent"/"not present"/"lacking". Removes the dual-label ambiguity that was blocking RELATE CONTRADICT signal on expression findings. Bumped `MAP_PROMPT_VERSION` → `"map_prompt_v5_expression_absent_vs_negative"`. | BUGS.md [Bug 47](BUGS.md#bug-47--direction-absent-vs-negative-ambiguity-on-expression-claims), [`MAP_PROMPT_AUDIT.md` Issue 6](MAP_PROMPT_AUDIT.md#issue-6--directionabsent-vs-directionnegative-ambiguity-in-expression-contexts-low) |
| 2026-05-15 | Summarisation › optional RULE block enum casing | `Rule.type` lowered to `Literal["diagnostic","prognostic","management"]`; `RuleCounts` field names matched. New `Rule._lowercase_type` `field_validator(mode="before")` accepts legacy Title-Case payloads. Updated MAP RULE OutputFormat prompt + `_recompute_audit` helper. RULE block is off by default — no DB rows to migrate. Tests in `tests/summarization/test_enum_alias_repair.py` (5 new), `tests/test_inspector.py` fixture updated. | BUGS.md [Bug 48](BUGS.md#bug-48--ruletype-title-case-inconsistent-with-lowercase-convention), [`MAP_PROMPT_AUDIT.md` Issue 7](MAP_PROMPT_AUDIT.md#issue-7--ruletype-is-title-case-diagnosticprognosticmanagement-everything-else-lowercase-low) |
| 2026-05-15 | PDF extraction › `ContextAwareStitcher._is_cut_off` | Reordered `_MID_SENTENCE_ABBREVS` check to run before the `.?!)]"\'»` sentence-final early-return (`parsers/text_processing.py`). Pre-fix: every abbreviation in the frozenset (`fig.`, `et al.`, `e.g.`, `i.e.`, `cf.`, `vs.`, `approx.`, `dept.`, `no.`, `nos.`, `ref.`, `refs.`) was period-terminal, so the early-return shadowed the rule and the frozenset was dead code — paragraphs ending in those abbrevs never stitched with the next narrative paragraph. Added a multi-token `last_two` lookup so `"Smith et al."` (which tokenises as `["Smith", "et", "al."]` → final token `"al"`) joins the last two tokens (`"et al"`) and matches the frozenset entry. Regression coverage: `tests/parsers/test_text_processing_cutoff.py` (21 cases). | BUGS.md [Bug 42](BUGS.md#bug-42--is_cut_off-mid-sentence-abbreviation-rule-is-dead-code) |
| 2026-05-15 | PDF extraction › Docling timeout enforced | New `DoclingLayoutExtractor._convert_with_timeout` helper wraps `converter.convert(...)` in a single-worker `ThreadPoolExecutor` and raises `TimeoutError` on `future.result(timeout=self._config.timeout_sec)`. `_run_docling` calls the helper instead of the converter directly. Pre-fix the `DoclingConfig.timeout_sec=300` knob was advisory; a pathological PDF could hang the entire batch. The runner's per-paper try/except blacklists the pmcid on the raised `TimeoutError`. `timeout_sec <= 0` bypasses the executor (disables the guard). Regression test in `tests/pdf_text_extraction/test_docling_timeout.py`. | BUGS.md [Bug 35](BUGS.md#bug-35--doclingconfigtimeout_sec-never-enforced-by-doclinglayoutextractor) |
| 2026-05-15 | PDF extraction › `remove_citations` year preservation | Capped citation-index runs at 1–3 digits in three branches of `remove_citations` (`parsers/text_processing.py`): after-period (`\. \d{1,3}(?:[,–\-]\d{1,3})*`), after-comma (same shape), standalone-with-separator (same shape, requires ≥1 separator). Bracket-style branch left as `\d+` because brackets disambiguate years from indices. Pre-fix `"Smith et al. 2020 reported …"` became `"Smith et al. reported …"`; post-fix the year survives. Citation indices in pathology papers are practically never ≥1000. Regression test in `tests/parsers/test_remove_citations.py` (9 cases). | BUGS.md [Bug 43](BUGS.md#bug-43--remove_citations-strips-publication-years) |
| 2026-05-15 | Cross-pipeline › config hygiene cluster | Five dead-knob bugs cleaned up in one pass. Deletions: `BaselineMode` enum + `pipeline.stages.pdf_text_extraction.BaselineMode` re-export; `FilteringConfig.{fix_ligatures, remove_reference_markers, min_paragraph_chars}`; `TextAssemblyConfig.{enabled, baseline_mode, use_hierarchical_extraction, use_context_aware_stitching, compare_combinations, save_combination_outputs}`; `CroppingConfig.{include_captions_in_metadata, panel_counting_enabled}`; `TATRConfig.{enabled, max_detections_per_page, batch_size_pages, structure_model_name}`. Wires: promoted hardcoded `tatr_detector._RENDER_DPI = 150` to `TATRConfig.render_dpi: int = 150` (read per-call in `detect()`); added `NormalizeConfig.extra_synonyms: dict[str, str] | None` to `SummarizationConfig` and threaded it through both sync and batch runners into `NormalizeStage(extra_synonyms=...)`. Loader fix discovered while wiring: `pipeline/config_loader.py:_unwrap_optional` now also unwraps PEP-604 `X | None` (`types.UnionType`) — was only matching `typing.Union`; new `_is_mapping_type` helper short-circuits `dict[...]`-typed YAML values out of the nested-dataclass branch so `extra_synonyms: {acme: ACME}` no longer crashes. `configs/run.yaml` template updated: dead-knob comment lines removed, `tatr.render_dpi` and `summarization.normalize.extra_synonyms` added. Regression coverage: 8 tests in `tests/test_config_loader.py`, including new `test_normalize_extra_synonyms_loaded_as_mapping` and `test_tatr_render_dpi_overridable`. | BUGS.md [Bug 30](BUGS.md#bug-30--filteringconfig-dead-knobs-fix_ligatures-remove_reference_markers-min_paragraph_chars), [Bug 31](BUGS.md#bug-31--textassemblyconfig-six-of-eight-fields-unread), [Bug 32](BUGS.md#bug-32--croppingconfig-dead-knobs-include_captions_in_metadata-panel_counting_enabled), [Bug 34](BUGS.md#bug-34--tatrconfig-dead-knobs-render-dpi-hardcoded), [Bug 37](BUGS.md#bug-37--normalizestageextra_synonyms-not-exposed-via-summarizationconfig) |
| 2026-05-15 | Summarisation › CANONICALIZE per-direction binning + group-level `is_conflicted` | `_split_by_direction` rewritten: one bin per observed direction (no folding of `unclear` / `no_direction` into the majority polarity bin); returns `sorted(bins.items())` for determinism. `_compute_scope_fields` shrunk to `_study_coverage`. `is_conflicted` repurposed to a group-level signal (True iff the group emits ≥2 polarity-bearing bins; stamped on every rule from the group). New shared symbols in `pipeline/stages/summarization/models.py`: `direction_value()` normalizer (`DirectionEnum` / raw string / `None` → string), `POLARITY_BEARING_DIRS`, `NON_POLARITY_DIRS`. `RelateStage._should_compare` and `corpus_relate._should_compare_cross_paper` skip pairs where either side's direction is non-polarity (`return False, "non_polarity_direction"`). New `CANONICALIZE_DIRECTION_POLICY_VERSION = "per_direction_no_folding_v2"` fed into `pipeline_config_hash` via both runners' `thresholds` dict — cache flips on bumps. Tests: rewritten `tests/summarization/test_canonicalize_direction_split.py` (16 cases incl. S5 core invariant), new `tests/summarization/test_corpus_relate_non_polarity.py` (6 cases), extended `tests/summarization/test_relate_skipped_pairs.py` (+4), extended `tests/summarization/test_pipeline_config_hash.py` (+2). Supersedes B-026. | BUGS.md [Bug 49](BUGS.md#bug-49--canonicalize-folds-unclear--no_direction-into-majority-polarity-bin) |
| 2026-05-16 | Eval › paper-selection algorithm documented | Full write-up of `eval/paper_selection/`: three buckets (related / diverse / hard), pluggable PairMetric / SetMetric / PaperMetric Protocols (`Relatedness`, `Diversity`, `Hardness`), two interchangeable strategies (greedy default, PuLP-CBC ILP opt-in), ILP scaling levers (candidate pruning + edge sparsification + per-bucket time budget + pre-score fallback), three output artifacts (`{version}.yaml` consumer-facing, `_rationale.json` + `_summary.csv` for thesis evidence). New `docs/readmes/PAPER_SELECTION.md` is the algorithm spec; STRUCTURE.md gained a "Pipeline C — Paper selection" section + scripts table row; HOW_TO_RUN.md §4 documents the regen command. THESIS.md decisions log entry for the bucket structure + ILP-as-default call. No code changes. | [`docs/readmes/PAPER_SELECTION.md`](readmes/PAPER_SELECTION.md) |
| 2026-05-15 | PDF extraction › PipelineRunner seeding + per-stage cache | Closes B-027. New `PipelineRunner._seed_pipeline()` seeds `random` / `numpy` / `torch` (+ `torch.cuda` when available) at `__init__` from `cfg.runtime.seed` (widened to `int \| None = 42`; `None` opts out). New `pipeline/stages/pdf_text_extraction/stage_cache.py` provides `_StageCache.get_or_compute(stage_name, pmcid, config_hash, compute_fn, loader_fn, dumper_fn, summarise_fn)` and per-stage (loader, dumper) pairs for `TableDetectionResult`, `List[LayoutElement]`, `List[HierarchicalRow]`. `runner._process` wraps stages 2 (table detection — both standard and two-pass branches), 5 (artifact filtering), 6 (text assembly) through the cache; final writers (7/8) and Docling extraction (1/4 — already cached at the Docling layer) untouched. Cache hash includes `cfg.docling`, `cfg.docling_text`, `cfg.tatr`, `cfg.masking`, `cfg.filtering`, `cfg.text`, `cfg.two_pass`, `cfg.table_detector`, the scispaCy model name when NER is active, and `STAGE_CACHE_VERSION` (per-stage int dict — bump for serialisation OR behaviour changes). Disk layout: `out/stage_cache/<stage>/<pmcid>.{json,hash}` with atomic temp+rename writes; sidecar / loader corruption logs WARNING and falls through to recompute, unexpected exceptions propagate. Final writers (steps 7/8) always run. Atomic helpers inlined in `stage_cache.py` rather than extracted from `summarization/persistence.py` (planned contingency — keeps summarisation byte-output untouched). `configs/run.yaml` template updated; `docs/HOW_TO_RUN.md` §9 documents cleanup, version-bump triggers, and reproducibility scope. 22-test regression in `tests/pdf_text_extraction/test_b027_seed_and_cache.py`. | BUGS.md [Bug 27](BUGS.md#bug-27--runtimeconfig-knobs-num_workers-log_level-seed-skip_existing_outputs-not-consumed) |
| 2026-05-16 | Eval › Phase 1 proxy metrics                        | New `scripts/eval/` package: `_lib.py` (pure file I/O loaders, git/hash helpers) and `compute_proxy_metrics.py`. Reads `out/summaries/{summaries,traces,cascade_decisions,cost}/*` and emits `eval/results/proxy_metrics.csv` (one row per pmcid + `__aggregate__`), `proxy_metrics_aggregate.json` (sums + rate p50/p90, raw `status_counts`, source distribution), and `proxy_metrics.meta.json` (git commit, created_at, schema_version=v1). Selective `_source` companion columns for ambiguous metrics only. No NLI/LLM/embedding imports — verified by `tests/eval/test_compute_proxy_metrics.py::test_import_safety_no_nli_or_llm_modules` in a subprocess. Phases 2–5 of the eval harness designed but not implemented. | [`docs/CALIBRATION_EVAL.md`](CALIBRATION_EVAL.md) |
| 2026-05-17 | PDF extraction › observability artifacts (Stage-1 thesis stabilization) | New `pipeline/stages/pdf_text_extraction/outputs/stats_writer.py` (`DocStatsCollector` + reason-code classifier + `config_digest` / `config_snapshot` helpers) writes `out/run_metadata/{pmcid}_stats.json` per processed document with stage timings, kept/rejected counts, R0/R1/R2/R3/R-color rejection histogram + header-zone tally, table-detection summary, status (`ok`/`failed`), config digest, and a compact config snapshot. Collector is constructed in `PipelineRunner.run_document` **before** `_process` is entered so failed documents still produce stats. New `pipeline/stages/pdf_text_extraction/outputs/manifest_writer.py` (`RunManifestWriter`) writes one `out/run_metadata/run_{ISO}_{uuid8}.json` per batch invocation with `run_id`, git SHA / branch / dirty, host, python version, full `PipelineConfig` dump, attempted PMCIDs, and aggregated summary. `ParallelBatchRunner.run` / `PipelineRunner.run_batch` thread the `run_id` to `run_document(..., run_id=…)` (only library-signature change is the new optional `run_id` kwarg — default `None` preserves prior behaviour). `runner.py:main` gains an opt-in argparse layer (`--detector` / `--tatr-threshold` / `--render-dpi` / `--two-pass`/`--no-two-pass` / `--out-root` / `--workers` / `--max-docs` / `--pdf-dir` / `--db` / `--write-raw-text` / `--skip-existing-in-db`); flags default to `None` so a no-flag invocation reproduces the canonical run byte-for-byte. `--out-root out/sweeps/<name>` repoints every `PathConfig` output dir under that prefix so sweep runs never touch the canonical `out/`. No `PipelineConfig` defaults touched. No `MediaJsonWriter` change (config metadata lives in stats / manifest only). Resolved pre-existing committed merge-conflict markers in `components/visualizer.py`, `table_detectors/tatr_detector.py`, and `eval/run.py` (B-057) to unblock end-to-end smoke tests. | THESIS_MATERIAL.md (new), HOW_TO_RUN.md §2.1, BUGS.md [Bug 57](BUGS.md#bug-57--committed-merge-conflict-markers-in-visualizerpy) |
| 2026-05-18 | Eval › sweep dispatcher | `scripts/eval/run_all_sweeps.py` rewritten for Stage 1 detector / threshold selection. `ALL_SWEEPS` replaced with 7 stage-numbered variants (`01_docling`, `02_tatr_090`, `03_tatr_095`, `04_tatr_099`, `05_hybrid_090`, `06_hybrid_095`, `07_hybrid_099`); all share an explicit `_apply_stage1_baseline()` that forces every helper flag OFF (`reconstruct_tables_from_lists`, `merge_tables_by_caption`, `merge_figures_by_caption`, `expand_tables_with_footnotes`, `drop_tables_inside_figures`) and pins `two_pass=ON`, `render_dpi=150`. New `--list-variants` / `--dry-run` flag prints a column table of the resolved config per variant without creating output dirs. Old 9 ad-hoc variants (`baseline`, `tatr_090`, …) removed from the dispatcher; their output and annotation dirs remain on disk but become unreferenced. No `runner.py` / `config.py` changes. | HOW_TO_RUN.md §2.2, THESIS.md Decisions log 2026-05-18 |
| 2026-05-23 | Summarisation › MAP-findings persistence robustness | `persist_map_findings` (`pipeline/stages/summarization/persistence.py`) now logs an entry line and **re-raises** on failure instead of swallowing — the only `sum_*` writer that does. A failed bulk insert previously left zero map rows while the run reported `success` (silent corruption). Both runners already mark the `pipeline_run` failed on exception. New `scripts/diagnose_b055.py` replays on-disk batch handles through the loud persist path at zero LLM cost to surface the real DB exception. No schema change. | BUGS.md [Bug 55](BUGS.md#bug-55--sum_map_findings-not-populated-by-batch-runner), THESIS.md Decisions log 2026-05-23 |
| 2026-05-23 | Summarisation › agreement scoring config (H-EMB-01) | New `AgreementConfig` sub-config on `SummarizationConfig` (`tau`, `count_alpha`, `reuse_weight`, `contradiction_weight`); YAML key `summarization.agreement.*`. `EmbeddingSimilarityStrategy.from_config` + `HybridStructuredSimilarity` now read it via `_align()` so both scorers share one knob. Wired at the three production scorer-construction sites (`runner.py`, `batch/runner.py` ×2). Defaults unchanged; weights flow into the per-paper config hash. | CALIBRATION_INVENTORY.md §10, THESIS.md Decisions log 2026-05-23 |
| 2026-05-23 | Eval › MAP cascade calibration sweep | `eval/silver/map_theta_sweep.py` upgraded from θ-only to a joint `scorer × theta × reject_theta` sweep over `SemanticAgreementScorer{EmbeddingSimilarityStrategy, HybridStructuredSimilarity}` (both honour `AgreementConfig`/H-EMB-01). Adds deferral-safety columns `early_accept_rate` / `early_accept_precision` / `escalate_rate`, stamps `cascade_path="legacy_agreement_checker"`, warns on empty-L3 cache. Required fix: `HybridStructuredSimilarity.compute_matrix` now accepts `context=` (forwarded to the embedding sub-signal) so it works as a `SemanticAgreementScorer` strategy. Also bumped `eval/silver/prompts.py` `PROMPT_VERSION` v2→v3 (category `demographics`→`demographic`, B-016). Replays the legacy AgreementChecker path — NOT the production router. | THESIS.md ABC-P1 TODOs, BUGS.md B-016 |
| 2026-05-23 | Summarisation › cascade path pinned in config | `MapConfig` gains `enable_router: bool = False` + `router_single_voter_policy: str = "escalate"` (YAML `summarization.map.*`). `build_runner` and `build_batch_runner` (`scripts/run_paper.py`) now pass `sum_cfg.map.enable_router` / `…router_single_voter_policy` into `SummarizationRunner` / `BatchSummarizationRunner` — previously both used the constructor defaults, so the legacy-vs-router choice was implicit and the sync/batch entry points could silently diverge. Config-load log line now prints `map.enable_router`. Defaults unchanged (False / escalate) so there is no behaviour change at the current YAML; the YAML now *governs* the path for reproducibility (verified via load → flip → reload round-trip). The MAP θ sweep is unaffected — it replays the legacy `AgreementChecker` directly (`CASCADE_PATH='legacy_agreement_checker'`), never instantiating either runner. | THESIS.md Decisions log 2026-05-23 |
| 2026-05-24 | PDF extraction › `--no-visualization` flag | `runner.py:main` gains `--visualization/--no-visualization` (`argparse.BooleanOptionalAction`, default `None` → no override). `--no-visualization` sets `cfg.visualization.enabled=False`, skipping the annotated audit PDFs in `out/visualization/` (~5 GB + per-page render time on a full-corpus run). Crops, text, and DB rows are unaffected — the audit PDFs are not consumed by any downstream stage (ILP / silver / cascade). Added for the 943-paper best-config re-extraction on a disk-constrained box. No behaviour change when the flag is omitted. | HOW_TO_RUN.md §2 |
| 2026-05-24 | Summarisation › sync agreement embedder aligned to Gemini | `build_runner` (`scripts/run_paper.py`) now passes `embed_fn=GeminiEmbedder()` (was `OpenAIEmbedder()`), matching `build_batch_runner`. Both production entry points now use `gemini-embedding-001` for the MAP agreement gate, so a θ calibrated on Gemini (`eval/silver/map_theta_sweep.py --embedder gemini`) is faithful to **both** sync and batch. `GeminiEmbedder` is a drop-in (identical `__call__(texts) -> list[list[float]]`); the OpenAI/Gemini split was undocumented drift with no recorded rationale. Behaviour change for **sync** runs only (agreement similarity now Gemini; batch unchanged); requires `GOOGLE_API_KEY`. No schema/config change. | THESIS.md Decisions log 2026-05-24 |
| 2026-05-25 | Summarisation › `legacy_single_voter_policy` knob | `MapConfig` gains `legacy_single_voter_policy: str = "keep"` and `AgreementChecker` gains a `single_voter_policy: Literal["keep","escalate"]` kwarg (default `"keep"` preserves the prior silent N=1 → KEEP behaviour). The `len(outputs) < 2` branch now consults the policy: `"escalate"` returns `ChunkDecision.ESCALATE` with `confidence=0.0` so a chunk whose other L1 voters' API calls failed is routed up the cascade instead of accepted unvetted. Mirrors `router_single_voter_policy` for the legacy non-router path; the two are independent fields. Hash-gated on `not enable_router`. Sweep harness extended: `eval/silver/map_theta_sweep.py --legacy-single-voter-policy {keep,escalate,both}` (default `keep` preserves historical CSV output); new `--stage map_routing_policy` in `eval/silver/run_summarization_sweeps.py` enumerates the two-cell categorical comparison at the pinned BEST_*. 10 explicit tests (`tests/summarization/agreement/test_legacy_single_voter_policy.py`) pin N=0/1/2/3 behaviour across both policy values and verify independence from `MapOutputRouter.single_voter_policy`. No production default change; no API spend (offline replay over the existing primer cache). | THESIS.md Decisions log 2026-05-25 |
| 2026-05-26 | Summarisation › config layout v2 — `RoutingConfig` split out of `MapConfig` | `enable_router`, `router_single_voter_policy`, `legacy_single_voter_policy` moved from `MapConfig` into a new `RoutingConfig` sub-config; `SummarizationConfig` gains `routing: RoutingConfig`. Defaults preserved verbatim (`False`, `"escalate"`, `"keep"`). YAML migration: `summarization.routing.*` block in `configs/run.yaml` (hard break — the strict loader rejects v1 paths like `summarization.map.enable_router` with a clear `Unknown config field: MapConfig.…` error; one negative test pins the failure mode). Readers updated in `scripts/run_paper.py` (`build_runner`, `build_batch_runner`, config-load log line). New `CONFIG_LAYOUT_VERSION = 2` constant in `config.py` is stamped into `_config_snapshot` on both runners so historical-vs-current `manifest.json` / `PipelineRun.config_snapshot` / TraceCollector JSONL consumers can branch on the layout (absence ⇒ v1). One-time `pipeline_config_hash` invalidation by design (the `dataclasses.asdict(cfg)` payload reshapes); no API spend (MAP voter cache keyed by `cascade_signature`, not `pipeline_config_hash`; downstream stages local). | THESIS.md Decisions log 2026-05-26 |
