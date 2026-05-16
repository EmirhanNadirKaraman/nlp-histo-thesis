# Calibration & evaluation harness — Phase 1

Status: Phase 1 only. Later phases (manual sampling, silver labels, threshold
analysis, downstream replay, config comparison) are designed in
`/Users/emir/.claude/plans/i-want-you-to-reactive-quiche.md` but **not
implemented**.

This page documents the first script in the harness, `compute_proxy_metrics.py`.
It reads the artifacts the summarization pipeline already writes and emits
no-label proxy metrics. It never calls an LLM, embedding API, or NLI model, and
never re-executes any pipeline stage.

## When to use it

Run it after `python -m pipeline.stages.summarization.runner` (or any sync /
batch summarization workflow) has populated `out/summaries/`. It is safe to run
as many times as you want; it only reads files.

## Command

```bash
python scripts/eval/compute_proxy_metrics.py \
    --input out/summaries \
    --out eval/results/proxy_metrics.csv
```

Both flags are optional; the defaults match the layout above.

## Inputs read

| Source | Path | Used for |
|---|---|---|
| Per-paper summaries | `out/summaries/summaries/{pmcid}.json` | `canonical_rules`, `final_rules`, `relations`, `audit_trail.map_chunks`, `rejection_summary`, `pipeline_config_hash` |
| Per-paper runs | `out/summaries/traces/runs.jsonl` | `duration_s`, `status` |
| Chunk summary | `out/summaries/traces/chunk_summary.csv` | `cache_hit` (per-chunk) |
| Cascade decisions | `out/summaries/cascade_decisions/{pmcid}.jsonl` | L1/L2/L3 counts, polarity-conflict flag |
| Cost reports | `out/summaries/cost/{run_id}/cost_report.json` | per-paper tokens + `cost_usd` |

If any source is missing, the corresponding metric is reported as empty/NaN
with a `_source = "missing"` flag (see below). The script never raises on a
missing file.

## Outputs

Three files are written into the parent of `--out`:

1. `proxy_metrics.csv` — one row per pmcid, plus one `__aggregate__` row.
2. `proxy_metrics_aggregate.json` — the aggregate row in nested form, plus raw
   `status_counts` and per-column source distributions. NaN values are
   serialised as JSON `null`.
3. `proxy_metrics.meta.json` — run metadata: `git_commit`, `created_at`,
   `schema_version`, `input_dir`, `n_papers`.

## Columns

### Identity

| Column | Description |
|---|---|
| `pmcid` | Paper identifier (matches the summary filename stem). |
| `run_id` | Run identifier from the summary. |
| `status` | Raw status string from the summary (not normalised). |
| `is_success_like` | `1` if status is in `{success, skipped, ok, completed}`; else `0`. |
| `pipeline_config_hash` | Pipeline config hash stamped on the summary. |

### MAP-level

| Column | Description |
|---|---|
| `raw_map_finding_count` | Total findings emitted by MAP (from `rejection_summary.map_findings_total`; falls back to sum over `audit_trail.map_chunks`). |
| `grounded_finding_count` | `raw_map_finding_count − map_grounding_rejected`. |
| `grounding_rejection_rate` | Direct passthrough of `rejection_summary.grounding_rejection_rate`. |
| `grounding_threshold_used` | Direct passthrough of `rejection_summary.grounding_threshold` (often `None`/empty when grounding filter is disabled). |
| `mean_grounding_score` | Mean of `grounding_score` over findings in `audit_trail.map_chunks`. |

### Downstream-stage counts

| Column | Description |
|---|---|
| `normal_finding_count` | From `rejection_summary.normal_findings_total`. |
| `non_groupable_total` | From `rejection_summary.non_groupable_total`. |
| `canonical_rule_count` | `len(canonical_rules)`. |
| `final_rule_count` | `len(final_rules)`. |

### Relations

| Column | Description |
|---|---|
| `relations_total` | `len(relations)`. |
| `support_edge_count` | Count of `relation_type == "SUPPORT"`. |
| `contradiction_edge_count` | Count of `relation_type == "CONTRADICT"`. |
| `scope_qualify_edge_count` | Count of `relation_type == "SCOPE_QUALIFY"`. |
| `unclear_no_direction_rate` | Fraction of canonical_rules with `direction in {unclear, no_direction}`. |

### Dedup

| Column | Description |
|---|---|
| `duplicate_rate` | Reserved; currently always missing — the existing `rejection_summary` schema does not expose a duplicate count. |

### Final-rule slices

| Column | Description |
|---|---|
| `top_k_final_rule_count_score_ge_0_5` | `len([r for r in final_rules if r.final_score >= 0.5])`. |
| `top_k_final_rule_count_top10` | `min(10, len(final_rules))`. |

### Cascade

| Column | Description |
|---|---|
| `l1_chunks`, `l2_chunks`, `l3_chunks` | Counts grouped by `level` in `cascade_decisions/{pmcid}.jsonl`. |
| `polarity_conflict_chunks` | Count of decisions whose `reason_codes` mention `polarity`. |
| `cache_hits`, `cache_misses` | Counted from `chunk_summary.csv` rows filtered by pmcid. |

### Cost

| Column | Description |
|---|---|
| `total_input_tokens`, `total_output_tokens` | Sum of `input_tokens` / `output_tokens` across cost_report `actual_usage.by_paper` entries for the pmcid. |
| `estimated_total_cost_usd` | Sum of `cost_usd` from the same source. |

### Timing

| Column | Description |
|---|---|
| `run_duration_s` | From the most recent `runs.jsonl` entry for the pmcid. |

## `_source` companion columns

Only metrics whose origin may be ambiguous or missing carry a `_source` column.
The current list:

- `grounded_finding_count_source`
- `grounding_rejection_rate_source`
- `mean_grounding_score_source`
- `duplicate_rate_source`
- `cache_hits_source`
- `cache_misses_source`
- `estimated_total_cost_usd_source`
- `token_count_source` (covers both token columns)

Possible values: `summary_json` / `audit_trail` / `chunk_summary` /
`cascade_decisions` / `cost_jsonl` / `missing`. In the aggregate row, the value
is a pipe-joined list of all distinct sources seen across papers.

Direct summary-derived counts (`canonical_rule_count`, `final_rule_count`,
`support_edge_count`, `contradiction_edge_count`, `top_k_final_rule_count_*`)
do **not** carry a `_source` column.

## Aggregate row

The CSV's final row has `pmcid = "__aggregate__"`:

- Sum columns hold the sum across papers (or empty when no finite values).
- Rate columns hold the mean across papers with finite values.
- `is_success_like` holds the count of success-like papers.
- `pipeline_config_hash` is the single shared hash, or `"MIXED"` if papers
  disagree.
- `_source` columns hold a pipe-joined list of distinct sources.

`proxy_metrics_aggregate.json` adds:

- `status_counts` — raw `{status_string: count}` map.
- `pipeline_config_hashes` — sorted list of distinct hashes.
- `sums` and `rate_stats` (mean/median/p90/n) for finer-grained reporting.
- `source_distribution` — `{column: {source_value: count}}` so it is easy to
  see, e.g., that grounding scores were available for 7 of 8 papers.

## What this script will not do

- It does not call any LLM, embedding model, or NLI model.
- It does not re-execute MAP, NORMALIZE, GROUP, CANONICALIZE, RELATE, or
  RESOLVE.
- It does not sweep thresholds or scorer choices.
- It does not write into `out/summaries/`.

Those capabilities are reserved for later phases described in the project
plan; nothing in this script needs them.
