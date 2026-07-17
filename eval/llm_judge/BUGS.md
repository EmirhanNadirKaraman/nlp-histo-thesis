# eval/llm_judge — Bugs, Logical Mistakes & Fixes

Comprehensive audit of the Phase 1 evaluation harness.

---

## BUG 1 — ~~`sampling.py` pipeline_runs status check is case-sensitive~~ ✅ NOT A BUG

**File:** `sampling.py:74-78`

**Analysis:** The pipeline always writes lowercase string literals (`"success"`, `"failed"`, `"interrupted"`, `"running"`) via hardcoded calls to `_finish_pipeline_run()`. The query `WHERE status = 'success'` matches exactly. No code path writes `"SUCCESS"` or `"Success"`.

The `PaperSample.run_id` naming (stores integer PK, not the human-readable `pipeline_runs.run_id` string) is a readability trap but not a runtime bug — the FK wiring is correct.

---

## BUG 2 — Q2 looks up canonical rules by `canonical_id` but the relation FK points to `rule_id_a`/`rule_id_b` which are string IDs

**File:** `q2_relations.py:60-69`

```python
cr_rows = (
    session.query(SumCanonicalRule)
    .filter_by(pmcid=pmcid, pipeline_run_id=run_id)
    .all()
)
cr_by_id: dict[str, Any] = {r.canonical_id: r for r in cr_rows}

resolvable = [
    r for r in relations
    if r.rule_id_a in cr_by_id and r.rule_id_b in cr_by_id
]
```

**Analysis:** `SumRelation.rule_id_a` and `SumRelation.rule_id_b` are `String(100)` columns. `SumCanonicalRule.canonical_id` is also `String(100)`. The code builds a dict keyed on `canonical_id` and then looks up `rule_id_a` / `rule_id_b` in it. **This is correct only if the pipeline stores the same string value in both places.** If they diverge (e.g. one stores a prefixed version), all relations silently fall into the "no resolvable relations" skip path.

**Fix:** Verify that the pipeline writes the same canonical ID string to both `SumRelation.rule_id_a` and `SumCanonicalRule.canonical_id`. Add a defensive log if the lookup fails:

```python
if not resolvable:
    logger.warning(
        "Q2 [%s] no resolvable relations. "
        "%d relations found, %d canonical rules found. "
        "rule_id_a samples: %s, canonical_id samples: %s",
        pmcid, len(relations), len(cr_rows),
        [r.rule_id_a for r in relations[:3]],
        list(cr_by_id.keys())[:3],
    )
```

---

## BUG 3 — `_score_buckets` in `metrics.py` can crash on `KeyError`

**File:** `metrics.py:24-25`

```python
bucket = f"{int(s * 10) / 10:.1f}"
buckets.setdefault(bucket, []).append(r[label_key] == positive_value)
```

**Problem:** `r[label_key]` uses dict subscript, not `.get()`. If any row is missing the `label_key` (e.g. `"is_grounded"` is absent from a malformed Opus response that was still cached), this crashes with `KeyError`.

**Fix:**
```python
buckets.setdefault(bucket, []).append(r.get(label_key) == positive_value)
```

---

## BUG 4 — ~~Q1 `correct_direction` uses `type: ["string", "null"]` which Anthropic tool_use may not support~~ ✅ FIXED

**File:** `prompts.py`

Tested `anyOf` against the live API — confirmed working. All 7 fields converted from `["type", "null"]` array syntax to `anyOf`:
- Q1: `correct_direction`, `correct_subject_entity`, `correct_outcome_entity`
- Q3: `subject_entity`, `outcome_entity`
- Q5 silver_findings: `subject_entity`, `outcome_entity`, `direction`
- Q5 alignments: `pipeline_index` (`["integer", "null"]` → `anyOf`)

---

## BUG 5 — Q5 F1 computation double-counts partial matches as FP

**File:** `q5_f1.py:117-123`

```python
matched_pipeline_indices = {
    a["pipeline_index"]
    for a in alignments
    if a.get("pipeline_index") is not None and a.get("match_type") in ("match", "partial")
}
fp = max(0, n_pipeline - len(matched_pipeline_indices))
```

**Problem:** A "partial" match is counted as covering the pipeline finding (reducing FP) but is NOT counted as TP. So a partial match simultaneously reduces FP and doesn't contribute to TP — it vanishes from the metrics entirely. This means:
- If Opus reports 3 pipeline findings, all "partial", then: TP=0, FP=0, FN=0, precision=0/0=0, recall=0/0=0, F1=0. This is misleading — it looks like a zero-data paragraph.

**Fix:** Decide on a clear policy. Options:
1. Count partial as 0.5 TP (soft credit)
2. Count partial as TP for recall but flag separately for precision
3. Simplest: exclude partial from `matched_pipeline_indices` so they count as FP, making the metric strict:
```python
matched_pipeline_indices = {
    a["pipeline_index"]
    for a in alignments
    if a.get("pipeline_index") is not None and a.get("match_type") == "match"
}
```

---

## BUG 6 — `sampling.py` DISTINCT ON is PostgreSQL-specific, not standard SQL

**File:** `sampling.py:74-78`

This is not a bug per se (the project uses PostgreSQL), but worth noting: if you ever need to test against SQLite (e.g. in CI), this raw SQL will crash. The query should ideally use SQLAlchemy ORM constructs or be marked with a `# PostgreSQL-only` comment.

---

## BUG 7 — Q5 `pipeline_index` in alignments can produce `KeyError`

**File:** `q5_f1.py:119`

```python
a["pipeline_index"]
```

**Problem:** Uses dict subscript instead of `.get()`. If Opus omits `pipeline_index` from any alignment entry (despite it being "required" in the schema — LLMs don't always comply), this raises `KeyError`.

**Fix:**
```python
a.get("pipeline_index")
```

---

## BUG 8 — ~~`runner.py` imports `anthropic` unconditionally in `run()`~~ ✅ FIXED

**File:** `runner.py:78`

```python
def run(cfg: RunConfig) -> None:
    import anthropic as _anthropic
```

**Problem:** The `import anthropic` runs even when `--dry-run` is specified or `--no-submit` is used. If anthropic isn't installed, dry-run mode crashes even though it doesn't need the API at all.

**Fix:** Move the import into the branches that actually need the client:
```python
def run(cfg: RunConfig) -> None:
    from database import get_db_connection
    # ... setup ...
    if cfg.dry_run:
        _log_dry_run(all_requests, cfg)
        cache.close()
        return
    # Import anthropic only when we actually need the API
    import anthropic as _anthropic
```

---

## BUG 9 — ~~`_write_jsonl` silently skips empty result sets~~ ✅ FIXED

**File:** `runner.py:463-467`

```python
def _write_jsonl(rows: list[dict], path: Path) -> None:
    if not rows:
        return
```

**Problem:** If a test produces zero results (all cached, or all skipped), the corresponding JSONL file from a previous run is never overwritten. A user might see stale results from a prior run and think they're current.

**Fix:** Write an empty file or delete the old one:
```python
def _write_jsonl(rows: list[dict], path: Path) -> None:
    if not rows:
        if path.exists():
            path.unlink()
        return
```

---

## BUG 10 — Q1 `_compute_fields_changed` does case-sensitive string comparison on entities

**File:** `q1_precision.py:37-46`

```python
pipe_val = pipeline.get(pipe_key)
correct_val = judgment.get(correct_key)
# ...
if pipe_val != correct_val:
    changed.append(field_name)
```

**Problem:** Entity comparison is case-sensitive. If the pipeline stores `"Ki-67"` and Opus returns `"ki-67"`, this will count as a field change. For entity names in medical literature, casing is inconsistent and this inflates the error rate.

**Fix:** Normalize casing for entity comparisons:
```python
if field_name in ("subject_entity", "outcome_entity"):
    if pipe_val is not None:
        pipe_val = pipe_val.strip().lower()
    if correct_val is not None:
        correct_val = correct_val.strip().lower()
```

---

## DESIGN ISSUE 1 — ~~`DIRECTIONS` enum is defined in `prompts.py` but never used~~ ✅ FIXED

**File:** `prompts.py:35`

`DIRECTIONS` was unwired in Q1 `correct_direction` and Q5 `silver_findings.direction`. Both now use `anyOf` with the enum:
```python
"anyOf": [{"type": "string", "enum": DIRECTIONS}, {"type": "null"}]
```
All four enum constants (`RELATION_TYPES`, `CATEGORIES`, `RELATION_LABELS`, `DIRECTIONS`) are now wired into schemas.

---

## DESIGN ISSUE 2 — Cache stores only `result_payload`, but `parse_*_result()` needs `request.metadata`

**File:** `cache.py:61-73` vs `runner.py:160-166`

```python
# cache.get() returns only result_payload
raw_judgment = cache.get(req.cache_key)
if raw_judgment is not None:
    parsed_row = _parse_result(req, raw_judgment)
```

This works during normal runs because `req` (the `JudgeRequest`) is built fresh from the DB before checking the cache. But in `_resume_batch()`, if we try to generate output files, we don't have the original `JudgeRequest` objects — we only have what's in `batch_meta.json`. The current resume flow caches results but tells the user to "re-run without --batch-id to generate output files." This is functional but fragile: it requires the DB state to be identical between the submit run and the output-generation run.

**Fix:** Already partially mitigated by storing `request_payload` alongside `result_payload` in the Postgres `LlmJudgeCache` table. But `cache.get()` only returns `result_payload` — it should optionally return `request_payload` too for standalone re-analysis.

---

## SUMMARY TABLE

| #  | Severity | File | Issue |
|----|----------|------|-------|
| 1  | ~~HIGH~~ | ~~sampling.py~~ | ~~Case-sensitive status check → 0 papers matched~~ ✅ NOT A BUG |
| 2  | MEDIUM | q2_relations.py | Silent failure if canonical_id format diverges |
| 3  | LOW | metrics.py | `KeyError` on malformed cached rows |
| 4  | ~~MEDIUM~~ | ~~prompts.py~~ | ~~`["string", "null"]` may be rejected by Anthropic~~ ✅ FIXED |
| 5  | **HIGH** | q5_f1.py | Partial matches vanish from F1 metrics |
| 6  | LOW | sampling.py | PostgreSQL-only raw SQL |
| 7  | LOW | q5_f1.py | `KeyError` on missing `pipeline_index` |
| 8  | ~~MEDIUM~~ | ~~runner.py~~ | ~~`import anthropic` crashes dry-run when not installed~~ ✅ FIXED |
| 9  | ~~LOW~~ | ~~runner.py~~ | ~~Stale JSONL files not cleaned up~~ ✅ FIXED |
| 10 | ~~MEDIUM~~ | ~~q1_precision.py~~ | ~~Case-sensitive entity comparison inflates error rate~~ ✅ FIXED |
| D1 | ~~LOW~~ | ~~prompts.py~~ | ~~Dead `DIRECTIONS` constant~~ ✅ FIXED |
| D2 | LOW | cache.py | `cache.get()` can't return request context |
