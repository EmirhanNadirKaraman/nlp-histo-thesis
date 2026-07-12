# Summarization persistence — open TODOs

> **2026-05-10 update**: batch path now writes the full artifact layout via
> `BatchKnowledgeExtractionRunner.finalize()` and runs the modern post-MAP chain
> (NORMALIZE → GROUP → CANONICALIZE → RELATE → RESOLVE). Original "batch
> persistence missing" item is **CLOSED**. Stage persisters live as free
> functions in `persistence.py` and are shared by both runners. DB
> persistence (`Sum*` tables) for batch mode remains a TODO — file artifacts
> only today.



Tracks gaps in `persistence.py` / runner integration. Each item is "decide
whether to do, then do or close". Source files cited inline.

Status legend: **DECIDE** = needs a yes/no, **DO** = decision is to implement,
**CLOSED** = no longer needed (record reason).

---

## 1. MAP `Finding.finding_id` — CLOSED (shipped)

`Finding._finding_id` is now a `PrivateAttr` populated by
`Finding.set_finding_id` after MAP completes; `compute_finding_id` in
`models.py` hashes `(pmcid, chunk_id, position_in_chunk, normalized_claim)`
into a stable 12-char id. Both `KnowledgeExtractionRunner.process` and
`BatchKnowledgeExtractionRunner.finalize` assign ids before grounding. Persistence
writes the id explicitly in `persist_map_artifacts` and to the
`rejected_findings.jsonl` rows.

---

## 2. NORMALIZE `source_finding_ids` population — CLOSED (shipped)

`NormalizeStage._merge` and `_wrap_single` now call `_collect_source_ids` to
populate `NormalFinding.source_finding_ids` from the underlying MAP
`finding_id`s; `dedup_trace.jsonl` already serialises the field.

---

## 3. RELATE skipped/blocking pair trace — CLOSED (shipped)

`RelateStage.relate()` now returns `(relations, raw_pairs, skipped_pairs)`.
`SkippedPair` records (rule ids, reason, category, relation_type, subject,
outcome) are persisted to `skipped_pairs.jsonl` via
`persist_relate_artifacts`.

---

## 4. CANONICALIZE LLM/predicate-selection trace — DECIDE

**Where**: `stages/canonicalize_stage.py::_select_predicate`,
`runner.py::_persist_canonicalize_artifacts` (writes `canonical_rules.jsonl` only).

**Today**: predicate-selection is deterministic (best-grounded fallback). No LLM trace today, so there's nothing to persist beyond the rule itself.

**Action**:
- Close as **CLOSED — N/A** while canonicalize stays deterministic.
- Reopen if/when an LLM is plugged in for predicate canonicalisation. At that point persist `{group_id, candidates[], chosen, fallback_reason, model, prompt_hash}` to `canonicalize_trace.jsonl`.

---

## 5. Corpus-level RELATE artifact — DECIDE

**Where**: `stages/corpus_relate.py::CorpusRelateStage.relate_incremental`,
`runner.py::_corpus_relate_incremental` (DB-only side effect, no return mirror).

**Today**: incremental cross-paper relations land in Postgres only. The runs/.../relate/corpus/ directory is never created.

**Options**:
- (a) Have `relate_incremental` return the relations it produced; runner mirrors them to `runs/{run_id}/relate/corpus/relations.jsonl` (append-mode, since corpus state is cumulative across papers).
- (b) Provide a separate `runner.export_corpus_relations()` post-run helper that reads from DB.
- (c) Skip — DB is source of truth.

**Recommendation**: (a). Append is fine because corpus relate already de-dupes via DB. Keeps file-only debugging viable.

---

## 6. CSV summaries — DECIDE

**Where**: `persistence.py::write_csv` (helper exists, unused).

**Today**: spec lists CSVs as nice-to-have. v1 omits them entirely so the writer surface stays small.

**Options**:
- (a) Wire `_persist_*_artifacts` to also emit `*_summary.csv` per stage. ~30 lines per stage.
- (b) Provide an offline `runs_to_csv.py` script that reads JSONL and writes CSVs on demand. Decouples format churn.

**Recommendation**: (b). CSV schemas drift; JSONL is the durable record.

---

## 7. NLP_HISTO_LOG_DIR contention — DECIDE (low priority)

**Where**: `runner.py::process` (sets env, restores in `finally`).

**Risk**: two `process()` calls running concurrently in the same Python process (if anyone ever multithreads at this level) would race on the env var. Not a concern today (`process_batch` is serial), but worth noting.

**Action**: only fix if/when concurrent `process()` becomes a use case.
Likely fix: pass `log_dir` directly to the enum logger via a contextvar
instead of an env var.

---

## 8. Manifest `pipeline_config_hash` — CLOSED (shipped)

`compute_pipeline_config_hash` in `persistence.py` produces a stable
16-char hash over `(config snapshot, thresholds, models, schema_version,
prompt_version, cascade_signature)`. Both sync and batch runners write it
into `manifest.extra["pipeline_config_hash"]` from `_make_artifact_writer`,
and `BatchKnowledgeExtractionRunner` uses it to invalidate stale cached results
in `_load_result` / `_save_result`.

---

## 9. RESOLVE per-component score breakdown — DEPENDENT on stage exposure

**Where**: `stages/resolve_stage.py::resolve` returns only the final
score and counts. `runner.py::_persist_resolve_artifacts::score_trace.jsonl`
just duplicates `final_rules` fields.

**Action**: when `ResolveStage` is updated to return a per-component breakdown
(grounding contribution, support boost, single-study penalty, contradiction
penalty), surface those fields in `score_trace.jsonl`. No writer change needed.

---

## Quick triage table

| # | Item | Effort | Impact | Status |
|---|------|--------|--------|--------|
| 1 | MAP `finding_id`             | S | High (lineage) | CLOSED — shipped |
| 2 | NF `source_finding_ids` fill | S | High (lineage) | CLOSED — shipped |
| 3 | RELATE skipped trace         | S | High (calibration) | CLOSED — shipped |
| 4 | CANONICALIZE LLM trace       | — | N/A today | Reopen if LLM added |
| 5 | Corpus relate file mirror    | S | Med | Soon |
| 6 | CSV summaries                | M | Low | Defer to offline tool |
| 7 | Log-dir env race             | S | Low | Only if concurrent |
| 8 | `pipeline_config_hash`       | XS | Low | CLOSED — shipped |
| 9 | RESOLVE component breakdown  | — | Med | After stage exposes |
