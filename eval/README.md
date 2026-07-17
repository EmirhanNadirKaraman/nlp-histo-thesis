# LLM-as-Judge Evaluation Harness (Phase 1)

Uses Claude Opus as a proxy judge to produce silver labels for the histopathology
NLP pipeline. These are **proxy labels, not clinical ground truth**. They measure
whether the pipeline's extracted rules and relations are consistent with what a
capable LLM would extract from the same text.

> **Scope.** This README documents only the LLM-as-Judge track (`eval/llm_judge/`).
> The `eval/` directory also hosts other measurement tracks, each with its own
> entry point:
> - **Figure/table detection P/R** — `eval/run.py`, `eval/precision_recall.py`, `eval/recall.py`
> - **Silver-label loop + thesis experiments E01–E14** — `eval/silver/` (see `eval/silver/experiments/`)
> - **Calibration-set selection** — `eval/paper_selection/`
> - **Frozen-artifact threshold sweeps** — `eval/sweeps/` (see `eval/sweeps/README.md`)
>
> The canonical reproduction commands for all tracks live in
> `docs/HOW_TO_RUN.md`.

---

## Recommended first run

Apply the migration, then start tiny and verify outputs and cache before launching
larger sync or batch runs.

```bash
alembic upgrade head

python -m eval.llm_judge \
  --mode sync --tests q1 --n 1 --max-requests 3 \
  --results-dir /tmp/judge_smoke
```

---

## Tests implemented

| Test | File | What it measures |
|------|------|-----------------|
| Q1 — MAP precision | `eval/llm_judge/tests/q1_precision.py` | Is each extracted claim grounded in its verbatim source? Are subject/outcome/relation/direction/category fields correct? |
| Q2 — Relation accuracy | `eval/llm_judge/tests/q2_relations.py` | Is the Opus-assigned label (SUPPORT / CONTRADICT / SCOPE_QUALIFY / UNRELATED) correct for persisted relation pairs? Blind by default. Phase 1 only evaluates pairs the pipeline persisted — see limitations. |
| Q3 — Recall gap | `eval/llm_judge/tests/q3_recall.py` | What generalizable findings did the pipeline miss from sampled paragraphs? Zero-extraction paragraphs are filtered to skip obvious boilerplate before sampling. |
| Q5 — Paragraph F1 | `eval/llm_judge/tests/q5_f1.py` | Opus returns silver findings and alignment decisions; Python computes TP/FP/FN/precision/recall/F1 from those alignments. Opus-provided counts are not trusted directly. |

## Tests intentionally deferred

| Test | Reason |
|------|--------|
| Q4 — Grounding threshold calibration | `SumRejectedFinding` does not currently store exact source or verbatim evidence. Evaluating a rejected claim against its own claim text would be circular and invalid. Enable Q4 once rejected findings persist the exact source text (or a stable reference to reconstruct it) used by the grounding filter. |
| Q6–Q9 | Future phases. |

---

## How caching works

Results are stored in the `llm_judge_cache` Postgres table. The cache key is a
SHA-256 hash of `(model, prompt_version, schema_version, task, request_inputs)`.

- Cache hits skip the Opus API call entirely.
- `--force-refresh-cache` deletes only rows matching the current model/prompt/schema
  version (and optionally the selected `--tests`), then re-runs.
- The cache stores the **raw Opus judgment** (result_payload). Derived fields
  (`fields_changed`, `is_correct`, F1 metrics) are **always recomputed in Python**
  when a row is read back — they are never stored stale.

`alembic upgrade head` creates or updates the `llm_judge_cache` table. Run it before
first use and after any schema migration.

```bash
alembic upgrade head
```

### Cache versioning

Cache entries are separated by judge model, prompt version, schema version, task, and
request inputs. Changing any of the first three (e.g. bumping `PROMPT_VERSION` in
`eval/llm_judge/__init__.py`) produces new cache keys and does not reuse old judgments.
This means a prompt change safely invalidates only the affected version — old results
remain in the table and can be compared against new ones.

---

## Execution modes

### Sync mode (default, for debugging)

Processes requests one at a time. Automatically retries on transient failures
with exponential backoff (up to 4 attempts, starting at 5 s).

```bash
python -m eval.llm_judge --mode sync --tests q1,q2 --n 5
```

### Batch mode (for production runs)

Submits all requests to the Anthropic Message Batches API in one asynchronous call.
Batch processing is asynchronous — the harness polls at the configured interval until
the batch completes, then retrieves results. If the process is interrupted, resume
with `--batch-id` and the harness will poll and fetch from where it left off.

```bash
python -m eval.llm_judge --mode batch --tests q1,q2,q3,q5 --n 15
```

Resume a batch after interruption:
```bash
python -m eval.llm_judge --batch-id msgbatch_... --poll-interval-seconds 60
```

Dry run — writes `pending_requests.jsonl` without submitting:
```bash
python -m eval.llm_judge --mode batch --no-submit --tests q1 --n 10
```

---

## How to run smoke tests

**Pre-requisite:** Alembic migration applied; DB contains at least one processed paper.

```bash
# 1. Tiny sync run — 2 papers, Q1 only, cap 3 API calls
python -m eval.llm_judge \
  --mode sync --tests q1 --n 2 --max-requests 3 \
  --results-dir /tmp/judge_smoke

# 2. Verify cache hit — re-run same command, expect "0 uncached, 3 cached"
python -m eval.llm_judge \
  --mode sync --tests q1 --n 2 --max-requests 3 \
  --results-dir /tmp/judge_smoke

# 3. Inspect cache in Postgres
psql "$DATABASE_URL" -c \
  "SELECT task, judge_model, prompt_version, created_at FROM llm_judge_cache LIMIT 10;"

# 4. Force refresh and re-run
python -m eval.llm_judge \
  --mode sync --tests q1 --n 2 --max-requests 3 \
  --force-refresh-cache --results-dir /tmp/judge_smoke

# 5. Batch dry run — no API calls, writes pending_requests.jsonl
python -m eval.llm_judge \
  --mode batch --no-submit --tests q1 --n 2 --max-requests 5 \
  --results-dir /tmp/judge_batch
```

---

## Output files

All written to `--results-dir` (default: `eval/llm_judge_results/`).

| File | Contents |
|------|----------|
| `q1_precision.jsonl` | One row per Q1 judgment (finding + Opus verdict + fields_changed) |
| `q2_relations.jsonl` | One row per Q2 judgment (relation pair + correct_label + is_correct) |
| `q3_recall.jsonl` | One row per Q3 judgment (paragraph + missing_findings[]) |
| `q5_f1.jsonl` | One row per Q5 judgment (paragraph + alignments + tp/fp/fn/f1) |
| `skipped_cases.jsonl` | Papers/paragraphs skipped due to missing data |
| `errors.jsonl` | API call failures after all retries |
| `summary.json` | Aggregate metrics (see below) |
| `paper_sample.json` | Papers selected for this run |
| `batch_meta.json` | Batch API metadata for resume (batch mode only) |
| `pending_requests.jsonl` | Requests not yet submitted (--no-submit only) |

---

## summary.json metrics

```jsonc
{
  "q1": {
    "grounded_rate": 0.92,          // fraction of findings Opus considers grounded
    "relation_type_accuracy": 0.88, // fraction with correct relation_type
    "direction_accuracy": 0.91,
    "category_accuracy": 0.95,
    "entity_error_rate": 0.07,      // fraction with subject or outcome changed
    "scope_error_rate": 0.04,
    "score_calibration": { ... }    // grounding_score buckets vs positive rate
  },
  "q2": {
    "accuracy": 0.84,               // fraction where pipeline label == Opus label
    "confusion_matrix": { ... },    // rows=pipeline label, cols=Opus label
    "nli_score_calibration": { ... } // NLI score buckets vs correct rate
    // NOTE: Phase 1 measures correctness of persisted relation pairs only.
    // The pipeline only persists non-UNRELATED pairs; candidate pairs filtered
    // before or labeled UNRELATED by the pipeline are not evaluated here.
  },
  "q3": {
    "gap_rate_overall": 0.31,       // fraction of paragraphs with ≥1 missed finding
    "gap_rate_by_stratum": { "with_extractions": ..., "zero_extractions": ... },
    "total_missing_findings": 47,
    "avg_missing_per_paragraph": 0.8
  },
  "q5": {
    "micro_precision": 0.87,
    "micro_recall": 0.79,
    "micro_f1": 0.83,
    "total_tp": 312,
    "total_fp": 46,
    "total_fn": 83,
    "total_partial_matches": 29
  }
}
```

---

## Known limitations (Phase 1)

- **Silver labels only**: Opus judgments reflect what the model would extract, not
  clinical expert consensus. Do not treat them as ground truth.
- **Q2 measures precision of persisted pairs, not relation recall**: The pipeline
  only persists non-UNRELATED pairs. Relation pairs the pipeline filtered out or
  labeled UNRELATED are not evaluated. False-negative UNRELATED labels are invisible
  to this test.
- **Q2 blind by default**: Opus is not shown the pipeline label unless
  `--show-pipeline-label` is set. Use that flag only for debugging or calibration —
  blind mode avoids anchoring bias.
- **Sync mode is slow**: Use batch mode for full production runs (15+ papers).
- **Retry covers transient errors**: Systematic API errors (bad prompts, schema
  violations) will exhaust retries and log to `errors.jsonl`.
