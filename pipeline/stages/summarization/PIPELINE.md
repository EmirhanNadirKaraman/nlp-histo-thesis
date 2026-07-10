# Summarization Pipeline — Reference

This document describes the summarization stage — what each stage does, the key design decisions, and known issues. Keep it up to date when making structural changes.

**File layout.** The stage implementations live in subpackages, not at the top level of `pipeline/stages/summarization/`. The `## <stage>.py` section headers below name each stage file; use this map to find it:

- `stages/` — `map_stage.py`, `normalize_stage.py`, `group_stage.py`, `canonicalize_stage.py`, `relate_stage.py`, `resolve_stage.py`
- `helpers/` — `grounding_filter.py`, `contradiction_detector.py`, `corpus_relate.py`, `citation_filter.py`, `paragraph_lookup.py`, `entity_linker.py`
- `old_stages/` — `reduce_stage.py`, `rule_stage.py` (optional secondary block; still imported by `runner.py` but off by default — note the name/location tension)
- top level — `runner.py`, `config.py` (`SummarizationConfig` + per-stage configs; the home of every tunable referenced below), `models.py`, `prompts.py`, `persistence.py`, `cache.py`, `llm_providers.py`, `nli_config.py`, `health_checks.py`
- NER is a **repo-root** module: `named_entity_recognition/ner.py` (not under `summarization/`).

---

## Stage order

```
sentences (from DB)
    │
    ▼
MAP (LLM, per chunk, ABC cascade)
    │
    ▼
GROUNDING FILTER (NLI — drops ungrounded findings)
    │
    ├──────────────────────────────────────────────────────────┐
    ▼                                                          ▼
NORMALIZE (deterministic)                              REDUCE (LLM, recursive)
    │                                                          │
    ▼                                                          ▼
GROUP (deterministic)                                  RULES (LLM)
    │                                                          │
    ▼                                                          ▼
CANONICALIZE (LLM)                             GROUNDING FILTER (NLI — drops ungrounded rules)
    │
    ▼
RELATE (NLI — pairwise comparison)
    │
    ▼
RESOLVE (deterministic — scoring)
    │
    ▼
NER (scispaCy + UMLS — saves to entities table)
    │
    ▼
result JSON + DB persistence
```

MAP/REDUCE/RULES/CANONICALIZE make LLM calls.
GROUNDING FILTER and RELATE share one NLI model instance. The active model is
selected from the registry in `configs/nli_models.yaml` (default
`pubmedbert_mednli` = `pritamdeka/PubMedBERT-MNLI-MedNLI`; overridable via
`$NLP_HISTO_NLI_MODEL`), resolved by `nli_config.py`.
NORMALIZE, GROUP, and RESOLVE are fully deterministic.

---

## `models.py`

Single source of truth for all data shapes. Every stage reads from and writes to these Pydantic models.

### Key types

| Type | Stage | Notes |
|------|-------|-------|
| `Finding` | MAP output | One atomic fact from a chunk. `grounding_score` written in-place by grounding filter. |
| `AuditableSummary` | MAP output | One chunk's `Finding` list + `AuditMetadata`. |
| `NormalFinding` | NORMALIZE output | Post-dedup, post-normalization finding. Has `mean_grounding_score`, `evidence: List[SourceSpan]`. |
| `FindingGroup` | GROUP output | All `NormalFinding`s with the same `(subject, outcome, relation_type, category)`. May have mixed directions. |
| `CanonicalRule` | CANONICALIZE output | One direction-bin of a `FindingGroup`. LLM-selected `predicate_text`. Has `canonical_scope`. |
| `Relation` | RELATE output | NLI-derived relation between two `CanonicalRule`s. Today only SUPPORT and CONTRADICT are emitted by `_classify_pair` (SCOPE_QUALIFY is defined in `RelationTypeLabel` but no branch produces it — see SCOPE_QUALIFY note in `relate_stage.py`). |
| `FinalRule` | RESOLVE output | Scored `CanonicalRule` with `final_score`, `support_count`, `contradict_count`, `is_contradicted`. |
| `ConsolidatedSummary` | REDUCE output | Four-section narrative summary. |
| `ExtractedRules` | RULE stage output | IF-THEN clinical rules with `evidence_chain`. |

### Design decisions
- `Finding.direction` is the structured polarity field (`positive`, `negative`, `absent`, `partial`, `unclear`, `no_direction`). NORMALIZE applies a keyword heuristic (`infer_direction`) as a recovery fallback only when the incoming `direction` is `None` or `unclear` — never as an overwrite.
- `FindingGroup.group_id` is a hash of `(pmcid, subject, outcome, relation_type, category)` — deterministic, per-paper namespaced so the same (subject, outcome, relation, category) tuple in two papers produces distinct group_ids (and therefore distinct `canonical_id`s downstream).
- `Relation.nli_score_a_to_b` stores the **entailment** score for SUPPORT and the **contradiction** score for CONTRADICT. Do not conflate these.
- `AtomicFinding` is defined but unused — reserved schema for a planned refactor.

---

## `runner.py` — `SummarizationRunner`

Orchestrates the full pipeline for one paper. Entry point: `runner.process(file_data)`.

### Steps
1. Check disk cache (`out/summaries/summaries/<pmcid>.json`) — skip if present unless `force_rerun=True`
2. Create `PipelineRun` DB row
3. MAP → GROUNDING FILTER → score findings in-place → persist map findings
4. NORMALIZE → persist normal findings
5. GROUP → persist finding groups
6. CANONICALIZE → persist canonical rules
7. RELATE → persist relations
8. RESOLVE → persist final rules
9. REDUCE → RULES → GROUNDING FILTER (rules)
10. NER (if `run_ner=True`, default)
11. Save result JSON to disk

### Key parameters
| Parameter | Default | Effect |
|-----------|---------|--------|
| `theta` | 0.7 | MAP agreement threshold. 0.0 disables cascading (always accept L1). |
| `grounding_threshold` | 0.3 | NLI entailment threshold for MAP findings filter. `None` disables. |
| `run_ner` | `True` | Whether to run scispaCy NER after RESOLVE. |
| `force_rerun` | `False` | Ignore disk cache and reprocess from scratch. |
| `db` | `None` | `DatabaseConnection` or `None`. All DB persistence is skipped when `None`. |

### Gotchas
- `db=None` is valid and fully supported — pipeline runs without any DB connection.
- The `_scored_map_findings`, `_normal_findings`, etc. dicts are keyed by pmcid — `process_batch()` accumulates across papers in the same runner instance.

### Memory checkpoint logging
Every stage call inside `process()` is wrapped in a `MemoryLogger` context manager (`pipeline/utils/memory_logging.py`). Each run emits a sequence of grep-friendly lines like:

```
MEMORY pmcid=PMC1448691 stage=pipeline event=start rss_mb=… vms_mb=… elapsed_s=0.0
MEMORY pmcid=PMC1448691 stage=MAP event=before rss_mb=… elapsed_s=0.1
MEMORY pmcid=PMC1448691 stage=MAP event=after rss_mb=… elapsed_s=83.2 delta_rss_mb=+263.4
MEMORY pmcid=PMC1448691 stage=NORMALIZE event=before …
…
MEMORY pmcid=PMC1448691 stage=UMLS event=before_scispaCy_load …
MEMORY pmcid=PMC1448691 stage=UMLS event=after_scispaCy_load …
MEMORY pmcid=PMC1448691 stage=pipeline event=end …
```

- On exception: emits `event=failed` for the offending stage (context manager) and again from the outer `except` so the very last MEMORY line in a crashed run always names the killing stage.
- Requires `psutil`. If missing, a one-time warning is logged and the rss/vms/delta fields are omitted — the timeline still works.

---

## `map_stage.py` — `MapStage`

Extracts atomic findings from sentence chunks using ABC (Agreement-Based Cascading).

### Per-chunk flow
1. Run all Level-1 voter LLMs in parallel (one thread per voter)
2. Score pairwise agreement via `AgreementChecker` (default: `EmbeddingScorer`)
3. If agreement ≥ theta → accept best voter output (Level 1 kept)
4. Else → run Level-2 voters in parallel, re-score
5. If L2 agreement ≥ theta → accept best L2 output
6. Else → call single Level-3 (escalation_llm)

### Design decisions
- Voters should be from **different providers** (Gemini + DeepSeek + Mistral) so disagreement reflects genuine uncertainty, not sampling noise.
- The optional router path (`self._router`) bypasses L2 and goes L1 → L3 directly, using a trained routing policy.
- Chunk IDs are positional (`C1`, `C2`, …). Cached chunks get their `chunk_id` overwritten to the current position so stale IDs from prior runs don't leak.
- Sentences are tagged `[S{i}|{pmcid}|{te_id}]` before being sent to the LLM — this is how `verbatim_support` citations are traced back to `TextElement` rows.

---

## `prompts.py`

All LLM prompt templates and LangChain chain factories.

### MAP prompt key rules given to the LLM
- **ZERO LOSS**: extract every dose, p-value, patient count, demographic, and clinical relationship
- **ATOMICITY**: one finding per observation — if a sentence mentions two stains, create two findings
- `relation_type` must be exactly one of: `has_feature | expression | prognostic | comparative | demographic | treatment_response | unclear`
- `direction` is optional — only output when clearly inferable
- `verbatim_support` must be an exact quote from source text, not a paraphrase

The prompt includes field-level examples for each `relation_type` because early versions produced excessive `unclear` classifications and confusion between `expression` and `has_feature`.

---

## `grounding_filter.py` — `GroundingFilter`

NLI-based filter. Applied twice: after MAP (on findings), after RULES (on IF-THEN rules).

### What it does
- Runs NLI on `(verbatim_support, claim)` pairs
- Drops items below `threshold`
- `score_findings()` writes scores in-place without dropping anything (used after filtering to score all survivors)

### Windowed NLI
`_split_windows` segments long premises at sentence boundaries, sized by a per-hypothesis token budget so the joint premise+hypothesis sequence never exceeds the NLI model's 512-token limit. Per-label scores are max-pooled across windows so a supporting sentence near a window boundary is not silently truncated.

### Shared NLI instance
`_NLI_PIPE_CACHE` is a module-level dict. Both `GroundingFilter` and `RelateStage` pull from it via `_get_nli_pipe()` in `relate_stage.py`. Model loads exactly once per process.

### Design decision
The filter runs **before** NORMALIZE, on raw MAP output. This means it acts on LLM-generated `verbatim_support` vs. raw claims — not on normalized predicate text. Intentional: filtering before normalization prevents bad findings from polluting the dedup.

---

## `normalize_stage.py` — `NormalizeStage`

Deterministic entity normalization + conditional deduplication. `Finding` → `NormalFinding`.

### Entity normalization — `_norm()` resolution order
1. **Synonym dict** (`synonyms.yaml`, loaded at startup) — domain knowledge takes priority
2. **UMLS linker** (scispaCy `en_core_sci_lg`, threshold 0.85) — filtered by junk semantic type blocklist
3. **Identity fallback** — stripped input returned unchanged

### Junk semantic type blocklist
UMLS concepts whose types are purely in `_JUNK_SEMANTIC_TYPES` (taxonomy: T001–T016, geography: T083, occupations: T097, etc.) are discarded. Short all-caps acronyms (≤5 chars) linked to any junk type are also discarded. This fixes the CEAN→Cetacea bug (CEAN was being linked to the mammalian order Cetacea via string similarity).

### `direction` handling
`infer_direction()` (keyword heuristic over negation/presence words) is applied **only when** the incoming `direction` is `None` or `DirectionEnum.unclear`. Concrete LLM directions (`positive`/`negative`/`absent`/`partial`/`no_direction`) are preserved unchanged. Applied in `_normalize_entities()`, `_merge()`, and `_wrap_single()`.

### Conditional dedup — `_dedup_key()`
Findings sharing the same `(te_id, subject_entity, outcome_entity, relation_type)` after normalization are merged into one `NormalFinding`. Non-groupable conditions (any field is None, `te_id is None`, `relation_type == unclear`) route to `_wrap_single()` instead.

`te_id` fallback is `None` (not `0`). Using `0` as sentinel caused unrelated findings with missing evidence strings to be falsely merged.

---

## `synonyms.yaml`

Canonical source of entity synonym mappings. Loaded by `_load_synonyms()` at startup with the hardcoded dict as fallback.

**Format:** `lowercase surface form: Canonical Name`

Add new entries here — no code change needed. Covers: survival endpoints (OS/PFS/DFS), IHC markers (CD30/Ki-67/BCL2/MYC/PD-L1/etc.), treatments (R-CHOP/CHOP), disease entities (CEAN), cell variants.

---

## `group_stage.py` — `GroupStage`

Deterministic bucketing of `NormalFinding` → `FindingGroup`. No LLM, no embeddings.

### Grouping key
`(subject_entity, outcome_entity, relation_type, category)` — all four must match. The `category` field was added to the key to prevent IHC and morphology findings about the same entity pair from landing in the same group.

### Groupability
A `NormalFinding` is groupable iff `subject_entity` non-None, `outcome_entity` non-None, `relation_type != unclear`. **Direction is not part of the groupability check** — a group can contain mixed directions. Contradictions are resolved in RELATE (Phase 5), not here.

### `scope_heterogeneity`
Float 0–1 measuring how many of the 8 scope fields have >1 distinct non-None value across group members. High value = group spans different study populations.

---

## `canonicalize_stage.py` — `CanonicalizeStage`

Deterministic reduction of `FindingGroup` → `CanonicalRule`. No LLM call today; `_select_predicate` is a thin wrapper that returns `_pick_best_predicate_deterministic` (highest `mean_grounding_score` wins). The LLM-selection hook is reserved for future use (see PERSISTENCE_TODOS #4).

### Per-group flow
1. Split by direction via `_split_by_direction()` — a group with both positive and negative findings emits two `CanonicalRule` objects
2. Build ranked candidate list of `(mean_grounding_score, predicate_text)`
3. Pick the highest-scored candidate as `predicate_text` (deterministic)
4. Compute `is_conflicted` + `study_coverage` from direction_counts and PMCID coverage

### Scope fields on `CanonicalRule`
- `is_conflicted`: True when ≥2 non-unclear directions appear in the bin
- `study_coverage`:
  - `multi_study`: ≥2 unique PMCIDs across member NFs
  - `single_study`: exactly 1 unique PMCID
  - `unknown`: no PMCID info

### Design decisions
- Predicate selection is deterministic — no LLM hallucination surface here.
- `unclear` findings are assigned to the largest direction bin (known limitation — inflates majority count when there is a meaningful split).

---

## `relate_stage.py` — `RelateStage`

NLI-based pairwise comparison of `CanonicalRule` → `Relation`.

### Comparability gate (`_should_compare`)
A pair reaches NLI only if ALL four match:
1. `category` — exact
2. `relation_type` — exact
3. `subject_entity` — exact string equality
4. `outcome_entity` — lowercase/stripped equality; for `expression` rules, marker suffixes ("expression", "positivity", "staining") are stripped first

### Classification logic (`_classify_pair`)

| Result | Condition |
|--------|-----------|
| CONTRADICT | mutual contradiction score ≥ threshold, AND rules do not share the same polarity direction |
| SUPPORT | mutual entailment score ≥ threshold (both directions) |
| UNRELATED (None) | none of the above — not stored. Asymmetric entailment collapses here today; `SCOPE_QUALIFY` was removed in B-006 along with its RESOLVE filter and log column. `FinalRule.scope_qualify_count` is retained as a hard-zero field for DB/template back-compat. |

### `contradict_allowed` gate (April 2026 fix)
The old gate called `infer_assertion_status()` on predicate text to require one positive + one negative assertion. Since the keyword heuristic returns `uncertain` for most clinical language, this blocked nearly all contradictions (relations was always empty). Replaced with a direction-field check:
- Block CONTRADICT when both rules have `direction ∈ {positive, partial}` OR both have `direction ∈ {negative, absent}`
- Allow CONTRADICT when directions differ, or either is `unclear`/`None`

### Score storage
`nli_score_a_to_b` / `nli_score_b_to_a` store:
- **Entailment score** for SUPPORT
- **Contradiction score** for CONTRADICT

### Known remaining issues
- All eligible pairs are compared (no `MAX_PAIRS` cap today). Large rule sets do O(N²) NLI calls.

---

## `resolve_stage.py` — `ResolveStage`

Deterministic weighted scoring of `CanonicalRule` → `FinalRule`. No LLM.

### Scoring formula — when relations are present

```
base             = mean_grounding_score * 0.60  (default 0.50 if None)
finding_bonus    = min(finding_count / 5, 1.0) * 0.10
support_bonus    = min(support_count * 0.08, 0.20)
single_study_pen = 0.10 if canonical_scope == single_study
contradict_pen   = min(contradict_count * 0.15, 0.30)

final_score = clip(base + finding_bonus + support_bonus
                   - single_study_pen - contradict_pen, 0.0, 1.0)
```

### Scoring formula — when relations are absent (April 2026 addition)

When `len(relations) == 0` (e.g. single-paper run or NLI skipped), grounding weight is raised so scores spread across [0,1] rather than clustering at ~0.51:

```
base             = mean_grounding_score * 0.80
finding_bonus    = min(finding_count / 5, 1.0) * 0.15
single_study_pen = 0.05
support/contradict = 0
```

Switch is automatic: `relations_present = len(relations) > 0`.

---

## `reduce_stage.py` — `ReduceStage`

Recursive tree-collapse of `AuditableSummary` → `ConsolidatedSummary`. LLM-based.

Groups chunk summaries in batches of `batch_size=10`, reduces each batch with the LLM, then recursively reduces intermediates until one remains. Output has four fixed sections: `clinical_significance`, `histopathological_features`, `management_outcomes`, `risk_factors_associations`.

Recursive reduction avoids context window overflow from passing all chunks at once.

---

## `rule_stage.py` — `RuleStage`

Extracts IF-THEN clinical rules from a `ConsolidatedSummary`. LLM-based.

Output: `Rule` objects with `type` (Diagnostic/Prognostic/Management), `condition` (IF clause), `action` (THEN clause), `confidence`, and `evidence_chain` with verbatim quotes and `text_element_id` links.

This is a **separate output track** from the MAP→NORMALIZE→…→RESOLVE track. Both run in parallel within `runner.process()`. The IF-THEN rules are clinical-decision-support oriented; the final rules from RESOLVE are knowledge-graph oriented.

---

## `cache.py` — `PipelineCache`

JSON-backed disk cache for LLM call results, keyed by content hash.

- MAP: keyed by hash of chunk text
- REDUCE: keyed by hash of input summaries
- RULES: keyed by hash of `ConsolidatedSummary` + pmcid

Cache file: `out/summaries/pipeline_cache.json`. Avoids re-running expensive LLM calls when re-running the pipeline with unchanged input (e.g. to test downstream changes without re-running MAP).

---

## `named_entity_recognition/ner.py`

Runs scispaCy `en_core_sci_lg` + UMLS linker on all `TextElement` rows for a paper. Saves to the `entities` table.

Runs as part of `SummarizationRunner` after RESOLVE when `run_ner=True` (default). Use `--skip-ner` to bypass. NER failure is non-fatal — logged as a warning, pipeline continues.

### UMLS filtering
- Threshold: 0.85 (was 0.7 — raised to reduce false positives)
- `_JUNK_SEMANTIC_TYPES`: frozenset of UMLS semantic types for taxonomy, geography, organisms — hits whose types are purely in this set are discarded
- `_ACRONYM_RE`: short all-caps tokens (≤5 chars) linked to any junk type are also discarded
- These settings mirror the same filters in `normalize_stage.py`'s own UMLS instance

---

## `scripts/inspect_pipeline_output.py`

Generates self-contained HTML inspector(s) from pipeline output JSON files.

### Single-run mode
```bash
PYTHONPATH=. python scripts/inspect_pipeline_output.py out/summaries/summaries/PMC10047158.json
# → out/inspector/PMC10047158.html

# With explicit output path:
PYTHONPATH=. python scripts/inspect_pipeline_output.py out/summaries/summaries/PMC10047158.json -o out/inspector/PMC10047158.html

# Export flagged findings to CSV:
PYTHONPATH=. python scripts/inspect_pipeline_output.py out/summaries/summaries/PMC10047158.json --export-flagged-csv out/inspector/PMC10047158_flagged.csv
```

### Cross-run diff mode
Compare two runs of the same PMCID side-by-side. Both JSONs must have matching `pmcid` fields.
```bash
PYTHONPATH=. python scripts/inspect_pipeline_output.py run_a.json --compare run_b.json
# → out/inspector/PMC10047158_diff.html
```
Produces a diff report with: summary stats for both runs, added/removed/changed final rules, canonical rules, relations, and MAP findings. Field-level diffs shown for `final_score`, `support_count`, `contradict_count`, `is_contradicted`, `mean_grounding_score`, `direction`, `category`, `relation_type`.

### Batch mode
```bash
PYTHONPATH=. python scripts/inspect_pipeline_output.py --batch-dir out/summaries/summaries/
# → out/inspector/<pmcid>.html  (one per JSON)
# → out/inspector/index.html    (sortable/searchable index page)
```
Invalid JSON files are skipped with a warning rather than stopping the batch.

### Templates
- `scripts/templates/pipeline_inspector.html.jinja2` — single-run inspector
- `scripts/templates/pipeline_diff.html.jinja2` — cross-run diff view
- `scripts/templates/pipeline_batch_index.html.jinja2` — batch index page

### Features
- Overview stats, final rules with full FINAL→CANONICAL→MAP lineage
- Stage-by-stage field comparison with diff highlights
- Suspicious-case badges and client-side search/filter by category/relation_type/flagged/contradicted
- **Sentence hover**: hovering over evidence tokens (e.g. `S2|PMC10047158|5`) shows the original sentence verbatim text as a tooltip
- **Cross-run diff**: added/removed/changed rules with field-level highlights
- **CSV export**: flagged findings with all metadata columns
- **Batch index**: sortable table linking all inspectors

### Flag types detected
- `taxonomy-leak`: subject_entity matches known taxonomy strings
- `polarity-mismatch`: direction=unclear but claim contains polarity words
- `generic-entity`: subject_entity is a generic term (patient, cell, lesion, etc.)
- `low-grounding`: grounding_score or mean_grounding_score < threshold (default 0.4)
- `empty-outcome`: outcome_entity is empty/None

### Sentence hover implementation
Sentence text is mined from two sources in the JSON (without guessing):
1. `audit_trail.map_chunks[].findings[].verbatim_support` paired with the first `evidence` token
2. `audit_trail.rules_provenance.rules[].evidence_chain[].verbatim` keyed by `sentence_id|pmcid|text_element_id`

The lookup is serialised as a JS object (`SENTENCE_LOOKUP`) embedded in the HTML.

### Export flagged CSV columns
`pmcid`, `run_id`, `stage`, `item_type`, `item_id`, `stable_key`, `category`, `relation_type`, `subject_entity`, `outcome_entity`, `predicate_text`, `direction`, `grounding_score`, `mean_grounding_score`, `final_score`, `flags`, `evidence_refs`, `verbatim_summary`

Note: `--export-flagged-csv` is only supported in single-run mode. In diff mode, run each JSON separately.

---

## Known remaining issues

| # | File | Issue |
|---|------|-------|
| 1 | `canonicalize_stage.py` | `unclear` findings assigned to the largest direction bin — inflates majority count when the split is meaningful |
| 2 | `relate_stage.py` | Same NLI model used for grounding (sentence↔claim) and rule-to-rule comparison — different tasks, different score distributions |
