# STAGE_EVAL_DESIGN.md

Stage-by-stage evaluation design for the summarization pipeline.

This document is the spec for implementing per-stage judge / eval tests in `eval/llm_judge/`. Every claim is anchored to a file:line in the current tree (branch `eval-speedrun`, working tree state). It is implementation-ready: a reader should be able to bring up Q1/Q2/Q3 against the pipeline without re-deriving any thresholds, prompts, or data sources.

Companions: `pipeline/stages/summarization/PIPELINE.md` (architecture), `PERSISTENCE_TODOS.md` (artifact backlog), `KNOWN_ISSUES.md`, `eval/EXPERIMENTS.md` (prior design ideas for Later judges), `eval/README.md` (smoke-test commands).

## §0. Terminology — read this once

Two model selections matter and they are independent.

1. **Summarization model / cascade** — the models that *produce* the pipeline output we are evaluating. Selected via cascade profile (`pipeline/stages/summarization/batch/voter_configs.py`). Profiles include `smoke_haiku` (Haiku at L1/L2/L3 — cheapest), the default 3-tier ABC cascade (DeepSeek / Mistral / Gemini Flash-Lite at L1; Flash / Kimi / Haiku at L2; Sonnet 4.6 at L3), and all-Sonnet / all-Opus profiles used for quality and calibration passes. In the 17-item eval-speedrun plan, "Haiku smoke", "Sonnet quality", "Opus calibration" refer to **this** model.
2. **Judge model** — the model used *inside* `eval/llm_judge` to score the pipeline output. Today hardcoded to `claude-opus-4-7` (`eval/llm_judge/__init__.py:22`, `MODEL = "claude-opus-4-7"`). No CLI flag exposes it. `client.py:48,72` already takes a `model: str = MODEL` parameter, so swapping is one constant + one `--judge-model` CLI flag away — see §8.7.

**Naming guarantee for the rest of this doc:** "Haiku smoke" means *summarization with the smoke_haiku cascade, judged by Opus*. Per-stage "Sonnet quality" means *summarization with all-Sonnet, judged by Opus*. Per-stage "Opus calibration" means *summarization with all-Opus, judged by Opus*. The judge is held fixed for thesis comparability. The small config change to enable a Haiku/Sonnet *judge* is documented in §8.7 — useful only for cost-aware iteration on the judge prompt itself.

---

## §1. Pipeline overview

```
sentences ─┬─▶ MAP ──┬─▶ NORMALIZE ──▶ GROUP ──▶ CANONICALIZE ──▶ RELATE ──▶ RESOLVE ──▶ FinalRule[]
           │         │
           │  AuditableSummary[]
           │  Finding[]
           │
           └──▶ chunk_id, position_in_chunk
```

ID propagation, end-to-end:

```
Finding.finding_id       ──▶ NormalFinding.source_finding_ids[]
NormalFinding.normal_id  ──▶ FindingGroup.member_ids[]
FindingGroup.group_id    ──▶ CanonicalRule.group_id
CanonicalRule.canonical_id ──▶ Relation.rule_id_a / rule_id_b
                           ──▶ FinalRule.canonical_id
NormalFinding.normal_id  ──▶ CanonicalRule.member_normal_ids[] ──▶ FinalRule.member_normal_ids[]
```

Data layout:

- **DB tables** (Postgres; `database/models.py`): `SumMapFinding`, `SumNormalFinding`+`SumNormalFindingSpan`, `SumFindingGroup`+`SumGroupMember`, `SumCanonicalRule`, `SumRelation`, `SumFinalRule`, `SumRejectionSummary`+`SumRejectedFinding`, `SumCorpusRelation`. Keyed by `pipeline_run_id` FK to `PipelineRun`.
- **JSONL artifact tree** (`<artifact_root>/<run_id>/`):
  ```
  manifest.json                  (run-level: schema_version, prompt_version,
                                  cascade_signature, pipeline_config_hash,
                                  git_commit, models, thresholds, papers,
                                  stages_attempted, stages_completed)
  map/{pmcid}/findings.jsonl
  map/{pmcid}/chunks.jsonl
  map/{pmcid}/rejected_findings.jsonl
  map/{pmcid}/bad_findings.jsonl
  map/{pmcid}/enum_observations.jsonl
  normalize/{pmcid}/normal_findings.jsonl
  normalize/{pmcid}/entity_links.jsonl
  normalize/{pmcid}/dedup_trace.jsonl
  group/{pmcid}/groups.jsonl
  group/{pmcid}/non_groupable.jsonl
  canonicalize/{pmcid}/canonical_rules.jsonl
  relate/{pmcid}/relations.jsonl
  relate/{pmcid}/raw_pairs.jsonl
  relate/{pmcid}/skipped_pairs.jsonl
  resolve/{pmcid}/final_rules.jsonl
  resolve/{pmcid}/score_trace.jsonl
  ```

Versioning constants live in:

- `pipeline/stages/summarization/models.py:20-25`: `MAP_SCHEMA_VERSION`, `MAP_PROMPT_VERSION`, `MAP_STAGE_NAME`.
- `pipeline/stages/summarization/persistence.py:230-259`: `compute_pipeline_config_hash(...)` — 16-char SHA256 over `(config, thresholds, models, schema_version, prompt_version, cascade_signature)`. Written to `manifest.extra["pipeline_config_hash"]` (`runner.py:1043-1051`, `batch/runner.py:493-518`).
- `pipeline/stages/summarization/models.py:29-44`: `compute_finding_id(pmcid, chunk_id, position_in_chunk, claim)` — 12-char SHA256, deterministic, populated post-MAP at `runner.py:365-373`.
- `pipeline/stages/summarization/models.py:191-198`: `Finding._finding_id` PrivateAttr (hidden from OpenAI strict schema and cache payloads).

Adapter status: see §0 + §5.4. Today's MVP judges read DB; the JSONL is parallel and used for offline sweeps + thesis archive.

---

## §2. MAP

### 2.1 Purpose

- **Input:** `list[dict]` of sentences, each `{text, pmcid, text_element_id}` (see `runner.py` chunk-building path; pipeline-side dict shape).
- **Output:** `AuditableSummary[]` (one per chunk), each containing `Finding[]`. Schema in `pipeline/stages/summarization/models.py`.
- **Why:** atomic medical-fact extraction with cross-voter agreement, plus audit metadata (cascade level reached, agreement score, voter provenance).

### 2.2 Decisions made

**LLM calls**

- Level 1 voters (default 3): `_voter_chains` built via `build_map_chain(llm)` (`current_stages/map_stage.py:138`). Each chain invokes `{pmcid, chunk_id, text}` → `AuditableSummary` via `with_structured_output` (LangChain). Voter specs flow into the cascade signature (`models.py: compute_cascade_signature`).
- Level 2 voters: same chain factory, mid-tier models, invoked iff agreement decision == ESCALATE (`map_stage.py:491-493`).
- Level 3 voter: single premium model (default Sonnet 4.6 in production, Haiku in `smoke_haiku`), invoked when both L1 and L2 disagree (`map_stage.py:511`).
- Prompt template: `pipeline/stages/summarization/prompts.py` (`build_map_chain`). Prompt version constant `MAP_PROMPT_VERSION = "map_prompt_v1_explicit_enums"` (`models.py:24`).

**Agreement / scoring**

- `AgreementChecker` with default `EmbeddingScorer()` (`map_stage.py:141`). Switchable to other scorers via constructor.
- `theta = 0.7` (`map_stage.py:99`) — accept-threshold; pairwise claim-embedding alignment ≥ θ → KEEP L1 best.
- `reject_theta = 0.2` (`map_stage.py:118`) — hard-reject; ≤ reject_theta forces escalation.
- Decision flow (`compute` → `ScoreBundle.decision ∈ {KEEP, ESCALATE, REJECT}`, `map_stage.py:481-494`).
- Best voter pick: `self._agreement.best(voters, bundle)` (`:486, :502, :512`).

**Routing path (optional)**

- When `self._router` is not None (`MapOutputRouter`), the agreement step is delegated (`:381-426`). Router returns `ChunkDecision(KEEP/REJECT, valid_voter_indices, …)`. REJECT short-circuits to L1→L3 escalation.

**Deterministic heuristics**

- Chunking: `chunk_size = 10` sentences (`map_stage.py:119`), `chunk_overlap = 0` default (`:105`). Stride = `chunk_size − chunk_overlap` (`:796`).
- Evidence string template: `[S{i}|{pmcid}|{te_id}]` per sentence — formatted into the prompt and parsed back during NORMALIZE.
- Voter-spec fallback for cache key when caller omits specs (`:59-73`).

**Caching**

- Cache key: `compute_cascade_signature(L1 + L2 + L3 specs)` — not provider/model alone (`:158`). Schema-version change in `MAP_SCHEMA_VERSION` invalidates.
- Cache hit always re-stamps `chunk_id` to current-run position (`:255`).

**Fallbacks**

- Single-voter `LengthFinishReasonError` (output truncation): voter excluded, no retry (`:719-725`).
- Other non-retryable LLM errors: voter excluded, logged (`:726-735`).
- All voters fail: chunk escalates straight to L3 (`:774`).

**Post-MAP, runner-side**

- `Finding.finding_id` is assigned for every emitted finding *before* grounding (`runner.py:365-373`) so grounding-rejected findings carry the same id as their pre-grounding counterpart.
- Grounding filter: `grounding_filter.GroundingFilter` runs NLI entailment on `(verbatim_support, claim)`. `grounding.threshold` lives in pipeline config (default referenced from `eval/silver/analysis/pipeline_sweep.py` sweep range). Findings below threshold are recorded in `rejected_findings.jsonl` (and `SumRejectedFinding` with `stage="grounding_map"`).

### 2.3 Information lost or transformed

- Per-voter raw outputs (L2 / L3 alternates) are discarded; only `best` voter's `Finding[]` persists.
- Sentence text → `chunk_trace.text_preview` stripped to 200 chars (`map_stage.py:273`).
- `chunk_id` overwritten on cache hit; cached chunk_id semantics are tied to *that run's* sentence ordering, not the cached run's.
- Audit metadata captured: `audit.cascade_level_reached`, agreement score, voter provenance. Persisted in `chunks.jsonl` row.

### 2.4 Failure modes (specific to current code)

1. **All voters emit empty `Finding[]`**: chunk passes through with `findings=[]`. No filter rejects it. Downstream stages see fewer findings; recall drop is invisible from MAP artifacts alone — requires Q3 (recall gap).
2. **Voter truncation silently excluded**: when one voter hits `LengthFinishReasonError`, agreement uses remaining voters. If the remaining voters happen to agree on an incomplete extraction, the chunk is KEPT at L1 with a truncated finding set.
3. **Cache-hit chunk_id drift**: re-running with a different sentence ordering re-stamps chunk_ids; downstream IDs derived from chunk_id (`finding_id` derives from `(pmcid, chunk_id, position, claim)`) shift. Mitigated only when sentence ordering is deterministic.
4. **Evidence string malformation**: NORMALIZE silently drops findings whose evidence cannot be parsed (`normalize_stage.py:365`). A MAP voter that emits `[S1|PMC9|some other format]` is lost downstream.
5. **Hallucinated finding** — claim not entailed by verbatim. Grounding filter is the only line of defense; sensitive to its threshold. Q1 measures this directly.
6. **Wrong `relation_type` / `direction` / `category`** — these are the most-corrected fields in pilot Q1 runs (per `eval/EXPERIMENTS.md` aggregates).
7. **`subject_entity` / `outcome_entity` missing** — voter sometimes omits one; downstream NORMALIZE marks the NormalFinding ungroupable → finding cannot enter GROUP / CANONICALIZE / RELATE. Visible in `non_groupable.jsonl`.

### 2.5 Persisted artifacts

| JSONL artifact | Row fields (eval-relevant) | DB table |
| --- | --- | --- |
| `map/{pmcid}/findings.jsonl` | full `Finding.model_dump()` **+ `finding_id`, `pmcid`, `chunk_id`, `position_in_chunk`** (`persistence.py:537-548`) | `SumMapFinding` (`database/models.py:357-396`) — **no `finding_id` column**; keyed by `(pipeline_run_id, chunk_id, position_in_chunk)` |
| `map/{pmcid}/rejected_findings.jsonl` | subset: `pmcid, chunk_id, finding_id, claim, category, subject_entity, outcome_entity, relation_type, direction, grounding_score, evidence, verbatim_support, reason="grounding_score_below_threshold"` (`persistence.py:565-575`) | `SumRejectedFinding` (`database/models.py:674-712`) — `stage ∈ {"grounding_map","group_non_groupable"}`, no `finding_id` column |
| `map/{pmcid}/chunks.jsonl` | `pmcid, chunk_id, n_findings, summary_text, audit_metadata` | (chunk audit not in DB) |
| `map/{pmcid}/bad_findings.jsonl` | enum-validation drops; copied from log file | — |
| `map/{pmcid}/enum_observations.jsonl` | unknown-enum coercion log; copied from log file | — |

**Primary artifact for eval:** `SumMapFinding` (DB) for Q1; `map/findings.jsonl` if running JSONL-only (adapter path, §7).

**Lineage status:** `finding_id` exists in JSONL artifacts; not in DB. Q1 today keys cache by `(pmcid, chunk_id, position_in_chunk)` — equivalent to the unique constraint `uq_sum_map_finding_pos`. No adapter required for MVP.

### 2.6 Evaluation questions for MAP

| id | name | input | unit | judge prompt idea | output schema | metric | MVP? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Q1** | **MAP grounding & field correctness** | `SumMapFinding` row + `TextElement.text_content` for verbatim_support | one finding | "Is the claim entailed by the verbatim? Correct each metadata field strictly against the verbatim." | `is_grounded: bool, relation_type_correction, direction_correction, category_correction, subject_entity_correction, outcome_entity_correction, scope_field_errors, fields_changed, explanation` (already in `prompts.py:Q1_SCHEMA`) | grounding_rate; per-field correction rate; fields_changed distribution | **MVP** |
| **Q3** | **Recall gap on sampled paragraphs** | paragraphs sampled via `sample_paragraphs(...)` (`sampling.py:128`) + `SumMapFinding.claim` aggregated by `text_element_id` | one paragraph | "List generalizable medical findings missing from the extraction." | `missing_findings: [{claim, category, subject_entity, outcome_entity, relation_type, direction}], has_gaps: bool` (`prompts.py:Q3_SCHEMA`) | has_gaps_rate per stratum; missing_count distribution; corpus recall floor | **MVP** |
| Q5 | Paragraph-level extraction F1 | paragraph + pipeline findings on that paragraph | one paragraph | Opus extracts all findings, aligns to pipeline output | silver findings + alignment + TP/FP/FN | precision / recall / F1 per paragraph; aggregate F1 | Later (already implemented but expensive) |
| MAP-1 | Voter agreement post-hoc audit | `chunks.jsonl` `audit_metadata` | one chunk | "Did the cascade reach the right level? Was the kept voter the best one?" | `correct_level: bool, better_voter_index: int\|None` | escalation efficiency | Later |
| MAP-2 | Empty-findings false-negative rate | `chunks.jsonl` rows with `n_findings == 0` | one chunk | judge re-reads the chunk source; flags missed findings | `should_have_extracted: bool, claims: [...]` | empty-chunk recall | Later |

---

## §3. NORMALIZE

### 3.1 Purpose

- **Input:** `Finding[]` (post-grounding survivors).
- **Output:** `NormalFinding[]`. Entities resolved (canonical name + UMLS CUI), direction inferred when missing, duplicates merged.
- **Why:** collapse surface-form variants of the same `(subject, outcome, relation_type, te_id)` claim so GROUP / RELATE have a clean entity space.

### 3.2 Decisions made

**Entity resolution** (`current_stages/normalize_stage.py:223-250`)

1. Synonym dict lookup. Built from hardcoded fallback (`:40-98`) + optional YAML at `pipeline/stages/summarization/synonyms.yaml` (loaded `:101-118`). Lowercase-stripped match. Hit → take canonical name → follow-up UMLS lookup to fill CUI (`:244`).
2. UMLS linker via scispaCy. Module-level singletons `_SPACY_NLP`, `_SPACY_LINKER` (`:135-170`). `_probe_spacy()` tries `en_core_sci_lg`, falls back to `en_core_sci_sm`. Linker: `linker_name="umls"`, `resolve_abbreviations=True`. Threshold `UMLS_THRESHOLD = 0.85` (`umls_utils.py`).
3. Identity fallback (`:249`): if both miss, return input stripped, CUI=None.

**Direction inference** (`:266-323`)

- Run iff finding's `direction` is None or `unclear` (`:508-510`).
- Negative triggers checked first (`:266-288`), positive next (`:290-303`). Substring match, case-insensitive.
- Hardcoded tuples; not configurable per-paper.

**Dedup** (`:327-350, 439-479`)

- Grouping key (`:328-349`): `f"{text_element_id}|{subject}|{outcome}|{relation_type.value}"`. All four must be present and `relation_type ≠ unclear`. Otherwise key = None → ungroupable.
- Partition (`:453-461`): groupable findings cluster by key, ungroupable findings wrap solo.
- Merge (`_merge`, `:526-573`): representative = highest `grounding_score` (`:533`). Dedup spans by `(sentence_id, text_element_id)` (`:537-548`). PMCID = sorted union (`:550`). Mean grounding score over members (`:572`). Source finding IDs (`:569`) collected via `_collect_source_ids` (`:403-418`).
- `NormalFinding.normal_id` generated via SHA256 of `(pmcid, subject, outcome, relation_type, te_ids)`.

**Evidence span extraction** (`:354-379`)

- Parse `[S{i}|{pmcid}|{te_id}]` (regex split on `|`). Malformed → debug log + skip. `te_id` must be int else skip.

**No LLM call in this stage.**

### 3.3 Information lost or transformed

- Merge picks one rep predicate text; other surface forms discarded.
- Per-finding grounding scores → mean only.
- `FindingScope` of non-rep findings dropped; rep's scope copied.
- **Lineage gain (working-tree change):** `NormalFinding.source_finding_ids` now populated from `Finding.finding_id`. `_collect_source_ids` skips findings whose finding_id was never assigned (legacy cached payloads / tests).
- **DB-side gap:** `SumNormalFinding` (`database/models.py:399-440`) does NOT have a `source_finding_ids` column. The list is JSONL-only (in `dedup_trace.jsonl` row `source_finding_ids` field, `persistence.py:636`). Any judge that wants to backtrack a NormalFinding → MAP Finding through DB must add the column or read JSONL.

### 3.4 Failure modes

1. **UMLS linker unavailable** (scispaCy not installed in env): dict-only resolution → many CUIs stay None → downstream group key uses surface form → preventable splits.
2. **`infer_direction` keyword fallback mis-labels**: e.g. "partial response" might collide with a positive trigger. Mis-set direction propagates through RELATE's polarity logic and into RESOLVE.
3. **Malformed evidence strings**: silently dropped → finding becomes ungroupable (no te_id available) → goes solo into GROUP's non_groupable bucket.
4. **All-unclear merge**: cluster of (positive, unclear, unclear, negative) on same te_id may collapse to one NormalFinding with `direction="unclear"`; conflict information lost.
5. **Synonym dict alias resolves to canonical but UMLS still returns CUI=None**: downstream CUI hashing falls back to text → fine when text is consistent; failure when text varies.
6. **Empty `pmcid` list edge case**: extremely rare; would propagate as empty `pmcids` field on NormalFinding.

### 3.5 Persisted artifacts

| JSONL artifact | Row fields (eval-relevant) | DB |
| --- | --- | --- |
| `normalize/{pmcid}/normal_findings.jsonl` | full `NormalFinding.model_dump()`: `normal_id, subject_entity, outcome_entity, subject_cui, outcome_cui, relation_type, direction, category, predicate_text, scope, evidence, pmcids, mean_grounding_score, source_finding_ids` | `SumNormalFinding` + `SumNormalFindingSpan` (no `source_finding_ids` column) |
| `normalize/{pmcid}/entity_links.jsonl` | `normal_id, subject_entity, subject_cui, outcome_entity, outcome_cui` | — |
| `normalize/{pmcid}/dedup_trace.jsonl` | `normal_id, n_evidence_spans, evidence_coords, source_finding_ids` (`persistence.py:624-639`) | — |

**Primary artifact for eval:** `normal_findings.jsonl` for N-1/N-2/N-3; DB row for any join-heavy query.

### 3.6 Evaluation questions for NORMALIZE

| id | name | input | unit | judge prompt idea | output schema | metric | MVP? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N-1 | Entity-normalization correctness | `NormalFinding` row + the *pre-normalization* entity strings | one finding | "Is the canonical name correct for the source text? Is the CUI correct?" | `subject_correct, outcome_correct, suggested_canonical_subject, suggested_canonical_outcome, suggested_cui_*` | normalization precision | Later — **blocker:** pre-norm entity strings are not currently persisted (`NormalizeStage` overwrites `subject_entity`/`outcome_entity` in-place at merge). Either add `pre_subject_entity` / `pre_outcome_entity` private attrs + persist, or capture during the synonym/UMLS pass into a `entity_resolution_trace.jsonl`. |
| N-2 | False-merge / false-split | merged NormalFinding + its `source_finding_ids`-resolved MAP findings | one normal_id with ≥ 2 sources | "Are these MAP findings actually duplicates? Or were findings that should have merged left separate?" | `merge_correct: bool, should_split: [finding_id], should_have_merged: [normal_id]` | merge P + R | Later — unblocked by `source_finding_ids` (working tree). |
| N-3 | Direction inference correctness on unclear/null findings | NormalFinding rows where MAP's `direction ∈ {None, unclear}` | one finding | "Given the verbatim, what is the correct direction?" | `direction: enum` | direction precision on inferred subset | Later — straightforward; reuses Q1's verbatim. |

---

## §4. GROUP

### 4.1 Purpose

- **Input:** `NormalFinding[]` (caller must pre-filter ungroupable; `current_stages/group_stage.py:128-133` raises `ValueError`).
- **Output:** `FindingGroup[]`. Buckets by `(subject, outcome, relation_type, category)`.
- **Why:** aggregate evidence about the same entity pair regardless of direction. Conflicting directions become CONTRADICT relations in RELATE rather than separate groups.

### 4.2 Decisions made

- **Groupability check** (`is_groupable`, `:39-50`): subject and outcome non-None, relation_type ≠ unclear.
- **Group id** (`_group_id`, `:57-68`): `"GRP_{sha8(subject_cui or subject)}_{sha8(outcome_cui or outcome)}_{relation_type}_{sha8(category)}"`. CUI preferred over surface text.
- **Direction counts** (`:97-102`): per-group tally of all member directions; `None` mapped to `"unclear"` in dict key.
- **Scope heterogeneity** (`:70-94`): fraction of 8 scope fields (`disease_subtype, cohort_n, assay_method, biomarker_cutoff, tissue_site, treatment_context, endpoint, study_design`, `:27-36`) with ≥ 2 distinct non-None values. Range [0.0, 1.0].
- **No LLM / NLI / embedding calls. All deterministic.**

### 4.3 Information lost or transformed

- All evidence/spans dropped from the group view; only counts + `member_ids[]` survive.
- Individual grounding scores discarded (aggregated upstream in NORMALIZE; not re-aggregated here).
- Scope objects not stored on `FindingGroup`; only heterogeneity score.
- CUI used for bucketing but not stored on `FindingGroup` (stored downstream on `CanonicalRule`).

### 4.4 Failure modes

1. **CUI mismatch on truly-equivalent entities**: two strings normalize to the same canonical name but different CUIs (e.g. one resolves "CD30" → C0054721, another resolves "CD30 antigen" → C0054721 but a third hits a sibling CUI). Hash diverges → preventable split.
2. **Caller forgets to pre-filter**: ValueError raised. Visible in tests; not a silent failure.
3. **Categorically incorrect category from MAP**: groups split by `category`. Mis-categorized findings sit in the wrong group and never aggregate.
4. **Same entity pair, different relation_type**: groups separate. This is intentional — `prognostic CD30→survival` and `has_feature CD30→nodular` should NOT merge.

### 4.5 Persisted artifacts

| JSONL | Row fields | DB |
| --- | --- | --- |
| `group/{pmcid}/groups.jsonl` | `group_id, subject_entity, outcome_entity, relation_type, category, member_ids, direction_counts, scope_heterogeneity` | `SumFindingGroup` + `SumGroupMember` |
| `group/{pmcid}/non_groupable.jsonl` | `normal_id, subject_entity, outcome_entity, relation_type, category, predicate_text, reason` (`persistence.py:664-678`) | (within `SumRejectedFinding` with `stage="group_non_groupable"`) |

### 4.6 Evaluation questions for GROUP

| id | name | input | unit | judge prompt idea | output schema | metric | MVP? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gr-1 | Non-groupable correctness | `non_groupable.jsonl` row | one row | "Should this have been groupable? Is the rejection reason accurate?" | `reject_correct: bool, true_reason: enum` | non-groupable precision | Later — cheap. |
| Gr-2 | Group-membership correctness | FindingGroup + its member NormalFindings | one group | "Do all members describe the same entity pair?" | `over_grouped: [normal_id], under_grouped: [normal_id]` | group purity + completeness | Later. |
| Gr-3 | CUI-driven split audit | groups sharing a (subject_entity, outcome_entity) surface but different group_id | one surface pair | "Should these have merged?" | `should_merge: bool` | CUI-split precision | Later. |

---

## §5. CANONICALIZE

### 5.1 Purpose

- **Input:** `FindingGroup[]` + `dict[normal_id → NormalFinding]`.
- **Output:** `CanonicalRule[]` — one per direction bin within a group. Carries canonical predicate text, scope flags (`is_conflicted`, `study_coverage`), and propagated lineage.
- **Why:** pick the single best predicate per direction; expose conflict and coverage as first-class scope fields for RESOLVE.

### 5.2 Decisions made

- **Direction binning** (`current_stages/canonicalize_stage.py:85-120`):
  - One distinct non-unclear direction: single bin (`:111`).
  - Multiple distinct non-unclear: one bin per direction; unclear findings assigned to the largest non-unclear bin (`:119`).
- **Predicate selection** (`_pick_best_predicate_deterministic`, `:75-82, 230-237`): argmax `mean_grounding_score`. Ties broken by first-seen order. LLM-selection path commented out at `:190-191` — currently disabled.
- **`is_conflicted`** (`:54-59`): True iff bin has ≥ 2 distinct non-unclear directions. After binning, this is rarely True per-bin; it captures *intra-bin* conflict, which can happen if unclear findings are folded into a positive bin alongside a partial.
- **`study_coverage`** (`:61-70`): `"single_study"` if 1 PMCID, `"multi_study"` if ≥ 2, `"unknown"` if 0.
- **`canonical_id`** (`:37-38`): `"CR_{sha8(group_id)}_{direction}"`.
- **No LLM / NLI calls.**

### 5.3 Information lost or transformed

- All non-rep predicate strings discarded.
- Scope heterogeneity from GROUP discarded; replaced by `is_conflicted` + `study_coverage` per bin.
- `FindingScope` originals are gone (last surviving copy is on NormalFinding).

### 5.4 Failure modes

1. **Predicate tie picks an over-specific surface form**: argmax + first-seen is stable but not semantic. Two equally-grounded predicates "X expression in nodular sclerosis" and "X expression" → arbitrary winner.
2. **Unclear direction folded into wrong bin**: when a group has positive + negative + unclear, unclear joins the larger; if positive == negative (same size), tie-break is order-dependent.
3. **`relation_type ∈ {unclear}`**: cannot occur (GROUP filters them); but if upstream invariants break, CANONICALIZE will still produce a rule with `relation_type=unclear`.
4. **`direction=no_direction`** (added in commit `ce1bf9a`): the binning logic treats it as a distinct direction; RELATE's polarity logic does NOT include it in `_NEGATIVE_DIRECTIONS` / `_POSITIVE_DIRECTIONS` (see §6.4 failure 5). Potentially-conflicting CanonicalRules with `no_direction` get through to RELATE without a same-polarity guard.

### 5.5 Persisted artifacts

| JSONL | Row fields | DB |
| --- | --- | --- |
| `canonicalize/{pmcid}/canonical_rules.jsonl` | `canonical_id, group_id, subject_entity, outcome_entity, relation_type, category, direction, predicate_text, canonical_scope (is_conflicted, study_coverage, …), member_normal_ids, direction_counts, mean_grounding_score` | `SumCanonicalRule` (`database/models.py:534-567`) |

### 5.6 Evaluation questions for CANONICALIZE

| id | name | input | unit | judge prompt idea | output schema | metric | MVP? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C-1 | Direction-bin correctness | CanonicalRule + member NormalFindings | one rule | "Is this the right direction for these members? Should they have split into more bins?" | `correct: bool, suggested_directions: [enum]` | direction-binning precision | Later. |
| C-2 | Predicate text quality | CanonicalRule + alternate predicate candidates | one rule | "Among these candidates, which is the best canonical predicate?" | `chosen_index: int, reason: str` | predicate-pick agreement with judge | Later. |
| C-3 | `is_conflicted` correctness | CanonicalRule + member NormalFindings | one rule | "Are these members actually in conflict?" | `correct: bool` | conflict-flag precision | Later. |

---

## §6. RELATE

### 6.1 Purpose

- **Input:** `CanonicalRule[]` for a single paper (intra-paper) or for the corpus (corpus-relate, `stages/corpus_relate.py`).
- **Output:** three lists (after working-tree change):
  - `Relation[]` — non-UNRELATED relations only (SUPPORT / CONTRADICT / SCOPE_QUALIFY).
  - `RawNLIPair[]` — every eligible pair including UNRELATED, with all NLI scores. Enables offline threshold sweep.
  - `SkippedPair[]` — every pair rejected by the pre-NLI comparability gate, with reason + discriminator fields (`artifact_models.py`).
- **Why:** infer semantic relationships between rules so RESOLVE can score support/contradict and so the rule graph supports downstream reasoning.

### 6.2 Decisions made

**Pre-NLI comparability gate** (`current_stages/relate_stage.py:211-243`)

Four exact-match conditions:

1. `category` (`:230`).
2. `relation_type` (`:232`).
3. Normalized `subject_entity` (`:234`).
4. Normalized `outcome_entity` (`:237-242`).

For `relation_type == expression`, outcome normalization strips marker suffixes ` expression`, ` positivity`, ` staining`, ` immunoreactivity` before comparing (`:197-208`).

Rejection reasons recorded: `category_mismatch`, `relation_type_mismatch`, `subject_mismatch`, `outcome_incompatible`. Each rejection emits a `SkippedPair` row (`:317, 327-340`).

**NLI model**

- `cross-encoder/nli-deberta-v3-large` (`:43`). Singleton shared with `GroundingFilter` (`:48`).
- `batch_size = 16` (`:269`).
- Sliding-window support (`:67-106`): long premises split via `_split_windows`; per-label scores max-pooled across windows (`:100-104`).

**Pairwise classification** (`:109-174`)

Bidirectional NLI: score `(predicate_a → predicate_b)` and `(predicate_b → predicate_a)`. Output: `{entailment, contradiction, neutral}` per direction.

- **CONTRADICT**: `con_ab ≥ contradiction_threshold AND con_ba ≥ contradiction_threshold AND NOT same_polarity` (`:115, 141-160`). Default `contradiction_threshold = 0.65` (`:268`).
- **SUPPORT**: `ent_ab ≥ entailment_threshold AND ent_ba ≥ entailment_threshold` (`:167`). Default `entailment_threshold = 0.55` (`:267`).
- **SCOPE_QUALIFY**: one direction ≥ threshold, the other not (`:171`).
- **UNRELATED**: none of the above; recorded in `RawNLIPair` only, not in `Relation[]`.

**Same-polarity guard** (`:141-160`)

```
_NEGATIVE_DIRECTIONS = {"negative", "absent"}
_POSITIVE_DIRECTIONS = {"positive", "partial"}
```

If both rules' directions fall in the same set → `contradict_allowed = False`. Rationale (`:122-130`): two positive findings about the same entity pair are coexisting observations, not contradictions.

### 6.3 Information lost or transformed

- Predicate text → discarded after NLI; only score + rule_id_a/b on `Relation`.
- UNRELATED pairs dropped from `Relation[]`. **They live in `RawNLIPair[]`** — eval can recover them.
- SCOPE_QUALIFY direction (which side was the qualifying premise) implicit in the `nli_score_a_to_b` vs `nli_score_b_to_a` asymmetry.

### 6.4 Failure modes

1. **Gate over-filters on subject string drift**: e.g. "CD30" vs "CD30 antigen" with the same CUI — gate compares normalized surface strings, not CUIs. NORMALIZE may have given them different `subject_entity` even if `subject_cui` matched. Eligible-but-rejected pairs show up in `skipped_pairs.jsonl` with `reason="subject_mismatch"` — directly auditable.
2. **Bidirectional asymmetry suppresses SUPPORT**: NLI scores are not symmetric for directional claims ("A causes B" vs "B occurs in A"). One direction ≥ threshold, other not → labelled SCOPE_QUALIFY instead of SUPPORT.
3. **Lexical-overlap false CONTRADICT**: NLI model can score "X expressed in 80%" vs "X expressed in 60%" as contradiction despite both being positive expression findings. The same-polarity guard catches the (positive, positive) case (`:141-160`) but only if NORMALIZE set both directions to `positive`.
4. **Sliding-window boundary effects**: long predicates split; max-pool reduces but does not eliminate boundary loss.
5. **`no_direction` polarity-guard hole**: `_NEGATIVE_DIRECTIONS` and `_POSITIVE_DIRECTIONS` (`relate_stage.py:148-159`) do NOT include `no_direction` (the value added in commit `ce1bf9a`). So `(positive, no_direction)` or `(negative, no_direction)` pairs are NOT same-polarity. The guard allows CONTRADICT in cases that, semantically, may or may not be contradictions. **The Q2 judge prompt should explicitly fail any CONTRADICT label where one direction is `no_direction` unless the judge confirms the contradiction is genuine.**
6. **Empty `predicate_text`**: NLI receives `""`; scores ill-defined. Mitigated by CANONICALIZE invariants (every CanonicalRule has a predicate) but not enforced.
7. **Corpus-relate not on disk**: `CorpusRelateStage` writes to `SumCorpusRelation` only (`persistence.py:24`, `PERSISTENCE_TODOS.md` #5). Corpus-level Q2 has no JSONL today.

### 6.5 Persisted artifacts

| JSONL | Row fields (eval-relevant) | DB |
| --- | --- | --- |
| `relate/{pmcid}/relations.jsonl` | `relation_id, rule_id_a, rule_id_b, category, relation_type, direction_a, direction_b, nli_score_a_to_b, nli_score_b_to_a, is_support, is_contradict, …` | `SumRelation` (`database/models.py:572-590`) |
| `relate/{pmcid}/raw_pairs.jsonl` | per-pair NLI scores (all four), classified_label including UNRELATED | — |
| `relate/{pmcid}/skipped_pairs.jsonl` | `SkippedPair`: `rule_id_a, rule_id_b, reason, stage="gate_pre_nli", pmcid, category_a/b, relation_type_a/b, subject_entity_a/b, outcome_entity_a/b` (`artifact_models.py`) | — |
| **(no `relate/corpus/relations.jsonl`)** | — | `SumCorpusRelation` |

**Primary artifact for eval:** `SumRelation ⋈ SumCanonicalRule` for Q2 (current path). `relations.jsonl` mirrors DB. `raw_pairs.jsonl` is the source for R-2 (recall). `skipped_pairs.jsonl` is the source for R-3.

### 6.6 Evaluation questions for RELATE

| id | name | input | unit | judge prompt idea | output schema | metric | MVP? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Q2** | **RELATE label correctness (precision)** | `SumRelation` row + both rules' predicate + direction + scope | one relation | "Given the two rules, what is the correct label? Blind to pipeline label by default." | `correct_label ∈ {SUPPORT, CONTRADICT, SCOPE_QUALIFY, UNRELATED}, confidence ∈ {low,medium,high}, explanation` (already in `prompts.py:Q2_SCHEMA`) | label-agreement rate per pipeline label; per-class precision; explanation review | **MVP** |
| R-2 | RELATE recall on pairs just below threshold | `raw_pairs.jsonl` rows with `classified_label == UNRELATED` and max(con, ent) within ±0.1 of threshold | one near-miss pair | "Should this pair have been SUPPORT/CONTRADICT/SCOPE_QUALIFY?" | same as Q2 | recall at threshold sweep points | Later — **unblocked** by `raw_pairs.jsonl`. |
| R-3 | Skipped-pair audit | `skipped_pairs.jsonl` row | one skip | "Should these have been comparable? Is the gate reason accurate?" | `should_compare: bool, true_reason: enum` | gate precision | Later — cheap; **unblocked** by `skipped_pairs.jsonl`. |
| R-4 | Corpus-relate Q2 | `SumCorpusRelation` row | one corpus relation | same as Q2 | same as Q2 | corpus label-agreement | Later — DB-only today; needs JSONL persistence to support pure-artifact eval. |

---

## §7. RESOLVE

### 7.1 Purpose

- **Input:** `CanonicalRule[]` + `Relation[]`.
- **Output:** `FinalRule[]`, sorted by `final_score` descending.
- **Why:** combine grounding signal, finding-count, support/contradict counts, and study coverage into a single rank.

### 7.2 Decisions made

Two-mode scoring (`current_stages/resolve_stage.py:100-156`):

**Mode 1: relations_present** (`len(relations) > 0` globally, `:104`)

```
base               = mean_grounding_score × grounding_weight        (default 0.60)
                     (default 0.30 if grounding score missing)
finding_bonus      = min(finding_count / finding_bonus_scale, 1.0)
                       × finding_bonus_max                          (scale 5, cap 0.10)
support_bonus      = min(support_count × support_boost_per_rel,
                         support_boost_cap)                         (per_rel 0.08, cap 0.20)
single_study_pen   = 0.10 if study_coverage == "single_study" else 0
contradict_pen     = min(contradict_count × contradict_pen_per_rel,
                          contradict_pen_cap)                       (per_rel 0.15, cap 0.30)
final_score = clip(base + finding_bonus + support_bonus
                    − single_study_pen − contradict_pen, 0, 1)
```

**Mode 2: relations_absent** (`len(relations) == 0`)

```
base               = mean_grounding_score × no_rel_grounding_weight (default 0.80)
                     (default 0.40 if missing)
finding_bonus      = min(finding_count / scale, 1.0)
                       × no_rel_finding_bonus_max                   (cap 0.15)
support_bonus      = 0
single_study_pen   = no_rel_single_study_pen                        (default 0.05)
contradict_pen     = 0
```

All weights / caps in `ResolveConfig` (imported from `config.py`).

**Contradict tracking** (`:158-162, 184-185`): `contradicted_by = [canonical_id …]`; `is_contradicted = len > 0`.

**SCOPE_QUALIFY** counted (`scope_qualify_count`) but **not used in scoring** (`:183`).

**Sort** (`:190`): descending `final_score`. Ties broken by insertion order.

### 7.3 Information lost or transformed

- Per-relation NLI scores discarded; only counts used.
- Per-member grounding scores discarded; only mean used.
- Component scores (base, finding_bonus, support_bonus, single_study_pen, contradict_pen) **not persisted**. `score_trace.jsonl` duplicates fields already in `final_rules.jsonl` (`persistence.py:755-775`; `PERSISTENCE_TODOS.md` #9). Ablation requires either rerun-with-logging or a tiny RESOLVE patch.

### 7.4 Failure modes

1. **Mode-flip semantics**: `relations_present` is decided globally by `len(relations) > 0`, but an individual rule may touch zero relations even when the run has many. That orphan rule still uses Mode 1 weights (0.60 grounding) instead of Mode 2 (0.80). Comment at `:104` flags it. Result: orphan rules systematically under-scored.
2. **SCOPE_QUALIFY ignored**: a scope-qualified rule pair carries genuine information; RESOLVE drops it. Either reward or penalize; today it's neither.
3. **Support cap saturates at 2.5 supports** (`0.20 / 0.08`): additional supports do not lift score. High-support rules clipped to the cap may rank below high-grounding-low-support rules.
4. **Contradict cap −0.30 saturates at 2 contradicts** (`0.30 / 0.15`): cannot push score to 0 unless grounding also low.
5. **finding_count scaling clips at 5**: `finding_count / 5` saturates at 1. A rule with 50 findings carries the same bonus as 5.
6. **`grounding_default` floor** (0.30 / 0.40): rules with missing mean_grounding_score get a non-trivial base. Could over-rate truly weak rules whose grounding score was never set.

### 7.5 Persisted artifacts

| JSONL | Row fields | DB |
| --- | --- | --- |
| `resolve/{pmcid}/final_rules.jsonl` | `final_id, canonical_id, group_id, subject_entity, outcome_entity, relation_type, direction, category, predicate_text, canonical_scope, member_normal_ids, final_score, support_count, contradict_count, scope_qualify_count, is_contradicted, contradicted_by, mean_grounding_score, finding_count, study_coverage` | `SumFinalRule` (`database/models.py:597-628`) |
| `resolve/{pmcid}/score_trace.jsonl` | duplicates `final_rules.jsonl` — **no component breakdown** | — |

### 7.6 Evaluation questions for RESOLVE

| id | name | input | unit | judge prompt idea | output schema | metric | MVP? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Rs-1 | Ranking quality | top-N FinalRules + a uniformly-sampled tail rule | one pair | "Which of these two rules carries stronger evidence?" | `winner: id, reason` | pairwise win-rate vs Opus silver | Later — useful but expensive. |
| Rs-2 | Weight sweep | full `final_rules.jsonl` + cached components | full corpus | `eval/silver/analysis/pipeline_sweep.py relate` and equivalent grounding sweep | sweep CSV | per-weight P / R / F1 vs silver | Later — **partially unblocked**; full ablation needs component breakdown (TODO). |
| Rs-3 | Contradiction-penalty correctness | FinalRule with `is_contradicted=True` + the contradicting Relation(s) + both rules | one final rule | "Is this contradiction real? Does the penalty seem warranted?" | `contradiction_real: bool` | precision of contradict signal | Later. |

---

## §8. Cross-stage lineage map

Forward chain (clean):

```
Finding.finding_id ─▶ NormalFinding.source_finding_ids[] ─▶ FindingGroup.member_ids[] ─▶ CanonicalRule.member_normal_ids[] ─▶ FinalRule.member_normal_ids[]
```

Backward by id (where the chain breaks today):

- `FinalRule.canonical_id` → `CanonicalRule` ✅
- `CanonicalRule.group_id` → `FindingGroup` ✅
- `FindingGroup.member_ids[]` → `NormalFinding.normal_id` ✅
- `NormalFinding.source_finding_ids[]` → `Finding.finding_id` ✅ in JSONL, ❌ in DB (`SumNormalFinding` lacks the column).
- `Finding.finding_id` → MAP source ✅ in JSONL, ❌ in DB (`SumMapFinding` lacks the column; keyed by `(run, chunk_id, position)`).

Other breaks:

- `relate/corpus/relations.jsonl` does not exist → corpus-relate eval is DB-only.
- `score_trace.jsonl` duplicates `final_rules.jsonl` → component-ablation eval needs more data.
- `pipeline_config_hash` is in `manifest.extra` but **not** mirrored to `PipelineRun` DB row → DB-only cache-invalidation on config change is not automatic.

Implication for the judges:

- **DB-only path** (today's `eval/llm_judge`): every MVP query works, no migration needed. `finding_id` is unavailable as a join key but `(pmcid, chunk_id, position)` is equivalent and the cache key already uses it.
- **JSONL-only path** (artifact-archive eval, e.g. CI without DB): needs the adapter sketched in §9.

---

## §9. Adapter: JSONL artifact → judge request (Later)

Not MVP. Document so we know what to build when needed.

Proposed module: `eval/llm_judge/adapters/artifact_loader.py`. API:

```python
def iter_map_findings(run_dir: Path, pmcid: str) -> Iterable[dict]:
    """Yield rows shaped like SumMapFinding ORM rows from map/{pmcid}/findings.jsonl."""

def iter_relations(run_dir: Path, pmcid: str) -> Iterable[dict]:
    """Yield rows shaped like SumRelation joined to SumCanonicalRule
       from relate/{pmcid}/relations.jsonl and canonicalize/{pmcid}/canonical_rules.jsonl."""

def load_paragraphs(run_dir: Path, pmcid: str) -> list[dict]:
    """OPEN QUESTION: paragraph text lives in TextElement (DB), not in run artifacts.
       Either persist a per-run map/{pmcid}/source_paragraphs.jsonl at pipeline time,
       or require DB for Q3.  MVP keeps the DB requirement for Q3."""
```

Build only if/when we need to evaluate a frozen run without DB access. Not on the eval-speedrun critical path.

---

## §10. MVP evaluation — Q1, Q2, Q3

This is the executable plan. Three questions, all already implemented (in `eval/llm_judge/tests/`), all reading from DB. No new judge code is required for MVP beyond two infrastructure tweaks (§10.5).

### 10.1 Q1 — MAP grounding & field correctness

- **Goal:** For every sampled MAP finding, does the verbatim text actually entail the claim, and are the metadata fields correct?
- **Data path (verified):** `eval/llm_judge/tests/q1_precision.py:66-71` reads `SumMapFinding` filtered by `pipeline_run_id` and `pmcid`. The verbatim text is fetched via `TextElement.text_content` joined through the finding's `evidence_refs` parsing. **No JSONL is read.** The new `finding_id` (`map/findings.jsonl`) is **informational**; cache key today is `(pmcid, chunk_id, position_in_chunk)` (a unique constraint on `SumMapFinding`), which is functionally equivalent.
- **Inputs:**
  - `claim`, `subject_entity`, `outcome_entity`, `relation_type`, `direction`, `category` from `SumMapFinding`.
  - `verbatim_support` reconstructed from `TextElement.text_content` via `evidence_refs`.
  - Scope fields (`scope_disease_subtype`, …, `scope_study_design`).
- **Sampling:** `--q1-findings 20` per paper, `--n 15` papers (selected-15). Deterministic seeding (`sampling.py:53 stable_seed`).
- **Prompt sketch:** existing `Q1_SYSTEM` + `Q1_USER` (`prompts.py:Q1_*`). Strict-entailment judge; corrects each field strictly from verbatim. **Doc note:** `prompts.py:35 DIRECTIONS` currently lists `["positive","negative","absent","partial","unclear"]` — **missing `no_direction`** which was added to the pipeline enum in commit `ce1bf9a`. Update before Sonnet/Opus pass so the judge can produce `no_direction` corrections. (Single-line fix; no code change required by this doc.)
- **Output schema (`Q1_SCHEMA`):** `is_grounded, relation_type_correction, direction_correction, category_correction, subject_entity_correction, outcome_entity_correction, scope_field_errors, fields_changed, explanation`.
- **Metric:**
  - `grounding_rate` = #(`is_grounded=true`) / total.
  - Per-field correction rate.
  - `fields_changed` distribution (histogram).
- **Cache key today:** `JudgeCache.make_key(task, …)` SHA256 over `model, prompt_version, schema_version, task, request_inputs` (`cache.py`). To add `pipeline_config_hash` we read `manifest.json["extra"]["pipeline_config_hash"]` at request-build time and fold it into the cache key inputs. Bump `SCHEMA_VERSION` from `"v1"` → `"v2"` to invalidate prior cache rows.
- **Smoke pass criteria:**
  - On `smoke_haiku`-cascade summarization: `grounding_rate ≥ 0.50` is the floor; below that, investigate the prompt / pipeline before scaling.
  - On Sonnet cascade: `grounding_rate ≥ 0.85`.
  - On Opus cascade: `grounding_rate ≥ 0.90` (used as calibration; below that, the cap is hit and we cannot improve grounding by switching models).

### 10.2 Q2 — RELATE label correctness (precision)

- **Goal:** For sampled persisted relations, is the label correct?
- **Data path (verified):** `eval/llm_judge/tests/q2_relations.py:47-61` reads `SumRelation` joined to `SumCanonicalRule` for the two endpoints' predicate / direction / scope. **No JSONL is read.**
- **Inputs:** `predicate_a`, `predicate_b`, `direction_a`, `direction_b`, `category`, `relation_type`, `subject_entity`, `outcome_entity` (both rules), pipeline label (hidden by default).
- **Sampling:** `--q2-relations 10` per paper.
- **Prompt sketch:** existing `Q2_SYSTEM` + `Q2_USER` (`prompts.py`). Blind by default; CLI flag `--show-pipeline-label` reveals the label.
- **Doc note:** add an explicit instruction in `Q2_USER` reminding the judge that **`(positive, no_direction)` or `(negative, no_direction)` pairs may bypass the pipeline's same-polarity guard** — judge should confirm the contradiction is genuine before labelling CONTRADICT. (See §6.4 failure 5.)
- **Output schema (`Q2_SCHEMA`):** `correct_label ∈ {SUPPORT, CONTRADICT, SCOPE_QUALIFY, UNRELATED}, confidence ∈ {low, medium, high}, explanation`.
- **Metric:**
  - Label-agreement rate (judge label == pipeline label).
  - Per-class precision.
  - Confidence-stratified agreement (high-confidence disagreements are the most actionable).
- **Known precision-only limitation:** UNRELATED recall (R-2) is Later; needs `raw_pairs.jsonl` at near-threshold scores.
- **Smoke pass criteria:**
  - Haiku cascade: label-agreement ≥ 0.60 (pipeline output is noisy; some RELATE precision is on us).
  - Sonnet: ≥ 0.75.
  - Opus: ≥ 0.80.

### 10.3 Q3 — Recall gap on sampled paragraphs

- **Goal:** What's missing from MAP's output on a representative paragraph sample?
- **Data path (verified):** `eval/llm_judge/tests/q3_recall.py:1-80` uses `sample_paragraphs(session, pmcid, run_id, …)` from `sampling.py:128`. That function queries `SumMapFinding` + `TextElement` + `Document`. Two strata: `with_extraction` and `zero_extraction`. Zero-extraction paragraphs are section-filtered via `_is_boilerplate_section` (`sampling.py:123`) to skip acknowledgments, funding, etc. **No JSONL is read.**
- **Inputs:**
  - Paragraph text (`TextElement.text_content`).
  - Already-extracted claims for that paragraph (from `SumMapFinding`).
- **Sampling:** `--q3-with-extraction-paragraphs 3`, `--q3-zero-extraction-paragraphs 2` per paper.
- **Prompt sketch:** existing `Q3_SYSTEM` + `Q3_USER` (`prompts.py`). Shared `FILTER_RULES` (`prompts.py:14`) excludes patient narratives + boilerplate so the judge doesn't false-positive missing findings on case vignettes.
- **Output schema (`Q3_SCHEMA`):** `missing_findings: [{claim, category, subject_entity, outcome_entity, relation_type, direction}], has_gaps: bool`.
- **Metric:**
  - `has_gaps_rate` per stratum.
  - Missing-finding count distribution.
  - Corpus-level **recall floor** = 1 − Σ missing / (Σ missing + Σ extracted-on-sampled-paragraphs). Lower bound because the judge can miss things too.
- **Smoke pass criteria:**
  - Haiku cascade: `has_gaps_rate ≤ 0.85` on with-extraction paragraphs (a high gap rate is expected from cheap models but >0.9 means the pipeline is barely scratching the paragraph).
  - Sonnet: `has_gaps_rate ≤ 0.50`.
  - Opus: `has_gaps_rate ≤ 0.35` — calibration ceiling.

### 10.4 What "no adapter needed for MVP" means concretely

MVP judges consume Postgres rows written by the same `KnowledgeExtractionRunner` invocation that wrote the new JSONL artifacts. Running

```bash
python -m eval.llm_judge --mode sync --tests q1,q2,q3 --n 2 \
  --max-requests 12 --results-dir /tmp/judge_smoke
```

against a run that populated both DB and JSONL will succeed today. The JSONL is supplementary: it enables offline replay (`eval/silver/analysis/pipeline_sweep.py`, `map_theta_sweep.py`), it preserves `finding_id` / `skipped_pairs` / `raw_pairs` for Later judges, and it gives a portable thesis archive. If/when we run eval against a DB-less archive, build §9's adapter.

### 10.5 Two infrastructure tweaks before scaling MVP

Both are required before the Sonnet pass (item 15 of the eval-speedrun plan). Neither is a stage change.

1. **Fold `pipeline_config_hash` into the cache key.** Read `manifest.json["extra"]["pipeline_config_hash"]` at `build_q*_requests` time; pass it into `JudgeCache.make_key(...)` as an additional input. Bump `eval/llm_judge/__init__.py:23 SCHEMA_VERSION` from `"v1"` to `"v2"` to invalidate prior cache rows. **Rationale:** without this, two pipeline runs with different thresholds but identical (chunk_id, position) keys would collide in cache. (Doc-only spec; implementer change not in this PR.)
2. **Patch `prompts.py:35 DIRECTIONS`** to include `"no_direction"`. (`commit ce1bf9a` added it to the pipeline enum.) Without it, the Q1 judge cannot emit `direction_correction = "no_direction"` and instead picks a wrong-but-allowed value. (Doc-only spec.)

---

## §11. Minimal evaluation implementation plan (final action list)

### 11.1 Implementation order

1. Verify smoke prerequisites:
   - `KnowledgeExtractionRunner` completed for ≥ 2 of the 5 related papers.
   - `manifest.json["extra"]["pipeline_config_hash"]` is set.
   - `relate/{pmcid}/skipped_pairs.jsonl` non-empty for at least one paper.
   - DB writes succeeded — `SELECT COUNT(*) FROM sum_map_findings WHERE pipeline_run_id = …` > 0.
2. Apply §10.5 tweak (1): plumb `pipeline_config_hash` into the cache key; bump `SCHEMA_VERSION` → `"v2"`.
3. Apply §10.5 tweak (2): add `"no_direction"` to `prompts.py:DIRECTIONS`.
4. Haiku-pipeline smoke (§11.4).
5. Sonnet-pipeline pass (§11.5).
6. Opus-pipeline calibration (§11.6).
7. Final report (§11.7).

No new judge code is required for MVP beyond steps (2) and (3).

### 11.2 Artifacts consumed per question (final)

| Question | Primary source | Secondary source |
| --- | --- | --- |
| Q1 | `SumMapFinding` + `TextElement.text_content` | `map/{pmcid}/findings.jsonl` (informational, for `finding_id`) |
| Q2 | `SumRelation ⋈ SumCanonicalRule` | `relate/{pmcid}/{relations,raw_pairs,skipped_pairs}.jsonl` for inspection |
| Q3 | `SumMapFinding` + `TextElement` paragraphs | `map/{pmcid}/findings.jsonl` for cross-check |

### 11.3 Output layout

Today (unchanged):

```
<results_dir>/
  q1_precision.jsonl
  q2_relations.jsonl
  q3_recall.jsonl
  q5_f1.jsonl          (only if --tests includes q5)
  skipped_cases.jsonl
  errors.jsonl
  summary.json
  paper_sample.json
  batch_meta.json      (batch mode only)
```

Add (MVP requirement):

```
<results_dir>/
  eval_manifest.json   # judge_model, PROMPT_VERSION, SCHEMA_VERSION,
                       # pipeline_run_id(s), pipeline_config_hash(es),
                       # judge_run_id, timestamps, summarization cascade name
```

Optional for Later: co-locate at `<artifact_root>/<run_id>/eval/judge_<judge_run_id>/…`. Not required by MVP — `--results-dir` is already a CLI arg.

### 11.4 Haiku-pipeline smoke (2 papers)

- **Summarization:** run with `smoke_haiku` profile against 2 papers from the related-5 selection. Produces low-quality output by design — purpose is to validate end-to-end plumbing.
- **Judge:** `claude-opus-4-7` (hardcoded; see §0 + §8.7 for the Haiku-judge option).
- **Command:**
  ```bash
  python -m eval.llm_judge --mode sync \
    --tests q1,q2,q3 --n 2 \
    --q1-findings 5 --q2-relations 3 \
    --q3-with-extraction-paragraphs 1 \
    --q3-zero-extraction-paragraphs 1 \
    --max-requests 12 \
    --results-dir /tmp/judge_haiku_smoke
  ```
- **Pass criteria:**
  - ≥ 1 row in each of `q1_precision.jsonl`, `q2_relations.jsonl`, `q3_recall.jsonl`.
  - Re-running the same command: `eval_manifest.json` shows `cache_hits == requests_total` (i.e. full cache replay).
  - `summary.json` populates `grounding_rate` (Q1), `label_agreement_rate` (Q2), `has_gaps_rate` (Q3).
  - Thresholds in §10.1–§10.3 under "Haiku cascade".
- **Failure triage:**
  - If `q1.grounding_rate < 0.50`: inspect first 3 Q1 rows manually — likely prompt regression (e.g. `DIRECTIONS` missing `no_direction`).
  - If `q3.has_gaps_rate > 0.90`: confirm zero-extraction paragraphs aren't all boilerplate (`_is_boilerplate_section` allow-list).
  - If `errors.jsonl` non-empty: check rate-limit / network; sync mode has 4 retries with exponential backoff (`client.py`).

### 11.5 Sonnet-pipeline quality pass (selected-15)

- **Summarization:** all-Sonnet cascade (or default ABC with Sonnet 4.6 at L3) against the selected-15 papers (item 9 of the eval-speedrun plan).
- **Judge:** Opus (unchanged).
- **Command (batch):**
  ```bash
  python -m eval.llm_judge --mode batch \
    --tests q1,q2,q3 --n 15 \
    --q1-findings 20 --q2-relations 10 \
    --q3-with-extraction-paragraphs 3 \
    --q3-zero-extraction-paragraphs 2 \
    --results-dir eval/reports/sonnet_15
  ```
- **Pass criteria:** §10.1–§10.3 "Sonnet" thresholds.
- **Compare to Haiku:** write `eval/reports/sonnet_15/comparison_vs_haiku.csv` with per-test deltas (`grounding_rate`, `label_agreement_rate`, `has_gaps_rate`, mean `fields_changed`). Any drop > 5 pp triggers triage of the summarization output (the judge is held fixed, so a drop is on the pipeline).

### 11.6 Opus-pipeline calibration pass (selected-15)

- **Summarization:** all-Opus cascade. Most expensive, slowest. Two purposes:
  1. **Silver findings for `eval/silver/`**: Opus-extracted findings on the same paragraphs become the silver set. Reuse `eval/silver/generation/generator.py` (the Opus-based silver generator) for extraction; reuse `eval/silver/analysis/evaluate.py` for P/R/F1 against pipeline findings.
  2. **Threshold sweep inputs**:
     - `eval/silver/analysis/map_theta_sweep.py` for MAP agreement θ.
     - `eval/silver/analysis/pipeline_sweep.py grounding` for the grounding threshold.
     - `eval/silver/analysis/pipeline_sweep.py relate` for RELATE thresholds.
     - The CSVs already at `eval/reports/grounding_sweep_*.csv`, `relate_sweep_*.csv`, `map_theta_sweep_*.csv` (per `git status` and `ls`) gain Opus columns.
- **Pass criteria:** §10.1–§10.3 "Opus" thresholds — these are calibration ceilings, not pass/fail gates.

### 11.7 Final report (item 17)

Deliverable: `eval/reports/final_<YYYYMMDD>.md`. Contents:

1. Per-stage P/R/F1 across the three cascade tiers (Haiku / Sonnet / Opus).
2. Threshold choices justified by the sweep CSVs (which θ values minimize regret).
3. Q1 / Q2 / Q3 number lines + confidence intervals from the 15-paper sample.
4. Explicit "what we *don't* measure yet" list pointing back to §3–§7's Later rows.

### 11.8 Implementation order summary (flat)

1. Bump `eval/llm_judge/__init__.py:SCHEMA_VERSION` to `"v2"`.
2. Plumb `pipeline_config_hash` from `manifest.json` into `JudgeCache.make_key` inputs in each `build_q*_requests`.
3. Add `"no_direction"` to `eval/llm_judge/prompts.py:DIRECTIONS`.
4. Add `eval_manifest.json` writing in `eval/llm_judge/runner.py` (records summarization cascade + judge model + versions + pipeline_config_hash + pipeline_run_id).
5. Haiku-pipeline smoke (§11.4).
6. Sonnet-pipeline pass (§11.5).
7. Opus-pipeline calibration (§11.6).
8. Final report (§11.7).

Steps 1–4 are the only code changes implied by this MVP. Everything else is configuration of existing tooling.

---

## §12. Appendix — open TODOs cited above

| TODO | File / location | Blocks |
| --- | --- | --- |
| `SumMapFinding` lacks `finding_id` column | `database/models.py:357-396` | JSONL-only Q1 (DB-Q1 unaffected). |
| `SumNormalFinding` lacks `source_finding_ids` column | `database/models.py:399-440` | DB-only N-2. |
| `NormalizeStage` does not persist pre-normalization entity strings | `current_stages/normalize_stage.py:223-250` | N-1 entirely. |
| `relate/corpus/relations.jsonl` not written | `persistence.py:24`; `PERSISTENCE_TODOS.md` #5 | Corpus-relate eval from artifacts (R-4). |
| `score_trace.jsonl` duplicates `final_rules.jsonl` (no component breakdown) | `persistence.py:755-775`; `PERSISTENCE_TODOS.md` #9 | Rs-1 ablation. |
| `manifest.extra["pipeline_config_hash"]` not mirrored to `PipelineRun` | `database/models.py:318`, `persistence.py:230-259` | DB-only cache-invalidation on config change. |
| Same-polarity guard in RELATE missing `no_direction` | `current_stages/relate_stage.py:148-159` | Quality of Q2; doc-only mitigation in prompt for now. |
| `prompts.py:DIRECTIONS` missing `no_direction` | `eval/llm_judge/prompts.py:35` | Q1 judge cannot suggest `no_direction` corrections. |
| Q3 paragraph source requires DB | `eval/llm_judge/sampling.py:128-…` | JSONL-only Q3 (DB-Q3 unaffected). |
| Judge model hardcoded to Opus, no CLI flag | `eval/llm_judge/__init__.py:22`, `eval/llm_judge/__main__.py` | Haiku/Sonnet judge cost-saving on prompt iteration. See §0 + §8.7 — single-flag fix: add `--judge-model` to `__main__.py`, thread to `RunConfig`, pass to `call_opus_sync(model=…)` and `build_batch_requests(model=…)` (already accept the param). Doc-only spec. |

---

## §8.7. Enabling a Haiku/Sonnet *judge* (separate from summarization cascade)

When useful: iterating on judge prompts and tool schemas where Opus cost dominates. **Not** for thesis numbers — thesis Q1/Q2/Q3 numbers should always be Opus-judged so they are comparable across runs.

Current state:

- `eval/llm_judge/__init__.py:22`: `MODEL = "claude-opus-4-7"` is a module-level constant.
- `eval/llm_judge/client.py:48,72`: `call_opus_sync(model: str = MODEL, …)` and `build_batch_requests(model: str = MODEL, …)` already accept `model`.
- `eval/llm_judge/__main__.py`: **no `--judge-model` flag** today.
- `eval/llm_judge/cache.py`: cache key includes `MODEL`, so swapping model invalidates cache automatically.

Required config change (doc-only spec — not implemented here):

1. Add `--judge-model {haiku|sonnet|opus|<full-id>}` to `__main__.py` argparser.
2. Add `judge_model: str = MODEL` to `RunConfig` (`runner.py`).
3. Thread `cfg.judge_model` into the `call_opus_sync(model=…)` / `build_batch_requests(model=…)` call sites.
4. Bump `SCHEMA_VERSION` (so old Opus cache rows aren't returned for Haiku queries; the cache key already includes model name, so this is belt-and-suspenders).

That's it. Three files, ~15 lines.
