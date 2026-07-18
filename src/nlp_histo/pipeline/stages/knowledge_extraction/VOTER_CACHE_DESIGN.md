# VoterCache Design

## Problem

`PipelineCache` stores only the **final winning `AuditableSummary`** per chunk.
Per-voter raw outputs are computed in memory and discarded after the agreement
checker selects a winner.  This means:

- The ABC cascade theta threshold **cannot** be swept post-hoc.
- Cost-quality trade-offs (e.g. "how often does L2 actually improve on L1?")
  **cannot** be measured without re-running LLMs.
- LLM-level cost attribution (which model, which level, how many tokens)
  is not recoverable from the current cache.

The `AgreementTrace` records voter metadata (count, pairwise agreement score)
but does **not** store the full voter `AuditableSummary` objects.

---

## Proposed Schema

New file: `pipeline/stages/knowledge_extraction/voter_cache.py`

```python
from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime
from .models import AuditableSummary


class VoterRawOutput(BaseModel):
    # Identity
    voter_label: str          # human-readable model name, e.g. "deepseek-v3"
    provider: str             # e.g. "azure", "anthropic", "openai"
    model_id: str             # exact model identifier used in the API call
    prompt_version: str       # map prompt version hash or tag

    # Cascade level
    level: int                # 1, 2, or 3

    # Full output
    summary: AuditableSummary

    # Cost & performance
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None     # estimated at call time
    latency_ms: float | None = None

    # Reliability
    temperature: float | None = None
    input_hash: str = ""              # sha256 of the formatted chunk text sent to this voter
    parse_error: str | None = None    # set if the response was malformed (still store it)


class MapVoterRecord(BaseModel):
    """
    Stores all voter outputs for a single MAP chunk.
    Enables full cascade replay at any theta without re-invoking any LLM.
    """
    # Cache key — same format as PipelineCache._map_key()
    cache_key: str                    # versioned pipe-joined: schema|prompt|stage|cascade|voter_cfg_hash|nli_model|input_hash

    # Chunk content for auditability
    chunk_sentences: list[str]        # formatted sentence strings sent to all voters

    # Per-level outputs
    l1_outputs: list[VoterRawOutput]
    l2_outputs: list[VoterRawOutput]  # empty if L1 agreed above theta
    l3_output: VoterRawOutput | None  # None if L1 or L2 agreed

    # Pre-computed pairwise agreement scores.
    # l1_pairwise_scores[i][j] = embedding agreement score between voter i and j.
    # Storing scores (not raw embeddings) is sufficient for cascade replay.
    # Without this, replay requires re-embedding voter outputs.
    l1_pairwise_scores: list[list[float]] | None = None
    l2_pairwise_scores: list[list[float]] | None = None

    # Metadata
    pipeline_run_id: int              # FK to PipelineRun
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
```

---

## Storage Format

**Per-file, not a single JSONL**: one JSON file per chunk at:

```
pipeline/cache/voter_records/{pipeline_run_id}/{cache_key_prefix}/{cache_key}.json
```

Where `cache_key_prefix` = first 2 chars of `cache_key` (sharding to avoid
directory size limits on large corpora).

Rationale:
- Avoids re-parsing a giant JSONL to retrieve one chunk's record.
- Allows partial replay (only load chunks whose theta decision flips).
- Can be deleted per pipeline run without affecting other runs.

---

## Cascade Replay Logic

```python
def replay_cascade(
    record: MapVoterRecord,
    theta: float,
    reject_theta: float = 0.20,
) -> AuditableSummary:
    """
    Select the cascade winner at a given theta without any LLM calls.
    Requires l1_pairwise_scores (and l2_pairwise_scores if l2_outputs exist).
    Falls back to best_of heuristic if scores are unavailable.
    """
    def mean_pairwise(scores: list[list[float]]) -> float:
        pairs = [(scores[i][j] for j in range(len(scores[i])) if i != j)
                 for i in range(len(scores))]
        flat = [x for row in pairs for x in row]
        return sum(flat) / len(flat) if flat else 0.0

    def best_of(outputs: list[VoterRawOutput]) -> AuditableSummary:
        # Quality heuristic: prefer more findings with longer evidence chains
        return max(
            outputs,
            key=lambda v: (len(v.summary.findings), sum(len(f.evidence) for f in v.summary.findings)),
        ).summary

    if record.l1_pairwise_scores:
        l1_score = mean_pairwise(record.l1_pairwise_scores)
        if l1_score >= theta:
            return best_of(record.l1_outputs)

    if record.l2_outputs and record.l2_pairwise_scores:
        l2_score = mean_pairwise(record.l2_pairwise_scores)
        if l2_score >= theta:
            return best_of(record.l2_outputs)

    if record.l3_output:
        return record.l3_output.summary

    return best_of(record.l1_outputs)   # fallback: best L1
```

---

## Integration Point

In `map_stage.py`, after each level's voter calls complete and before the
agreement checker writes the final decision to `PipelineCache`, write a
`MapVoterRecord` to `VoterCache`.

The existing `PipelineCache` is **not changed** — `VoterCache` is purely additive.
The production serving path continues to use `PipelineCache` for fast lookup.
`VoterCache` is only read by offline calibration tooling.

```python
# Approximate location in map_stage.py (~line 225)
result = self._agreement.best(l1_outputs)
cache.set_map(chunk, result)

# NEW: persist per-voter outputs for offline theta replay
if voter_cache is not None:
    voter_cache.write(MapVoterRecord(
        cache_key=cache._map_key(chunk),
        chunk_sentences=[s["sentence"] for s in chunk],
        l1_outputs=l1_raw_outputs,
        l2_outputs=l2_raw_outputs,
        l3_output=l3_raw_output,
        l1_pairwise_scores=l1_pairwise,
        l2_pairwise_scores=l2_pairwise,
        pipeline_run_id=run_id,
    ))
```

`voter_cache` is `None` by default — zero overhead for runs that don't need it.

---

## Why This Is Required

| Goal | Without VoterCache | With VoterCache |
|------|--------------------|-----------------|
| Sweep theta post-hoc | Re-run all LLMs | Local replay, zero API cost |
| Cost attribution per level | Unknown | Per-voter token + cost fields |
| Measure L2 lift over L1 | Impossible | Compare l1 vs l2 outputs |
| Diagnose escalation rate | Only current theta | Any theta, any split |
| Audit a specific chunk | Final result only | Full voter decision tree |

---

## Decisions Needed Before Implementation

1. **Who creates `VoterCache`?** — passed into `MapStage.__init__` as optional arg,
   or constructed in `KnowledgeExtractionRunner` and passed through.

2. **Should `input_hash` be the hash of the raw sentences or the formatted string?**
   Formatted string is more faithful to what the model actually saw.

3. **Retention policy** — voter records can be large for big corpora. Define whether
   records older than N pipeline runs are pruned automatically.
