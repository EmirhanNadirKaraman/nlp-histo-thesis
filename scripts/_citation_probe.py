"""Throwaway probe (B-080): citation-failure prevalence in the related15 voter_cache.

Runs the ProvenanceValidator over every cached MAP voter output, scoring each
finding's citation against its chunk source index (built from chunk_map). No
scorer / embedder / API / DB needed — validates the chunk_map→SourceIndex
plumbing and reports the real-world hallucinated-citation rate.
"""
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from pipeline.stages.summarization.models import AuditableSummary
from pipeline.stages.summarization.helpers.citation_filter import (
    STRUCTURAL_CITATION_CODES,
)
from pipeline.stages.summarization.routing.provenance_validator import (
    ProvenanceValidator,
    SourceIndex,
)

_cache_arg = sys.argv[1] if len(sys.argv) > 1 else "eval/data/map_primer/voter_cache.json"
_cache_path = Path(_cache_arg)
if not _cache_path.is_absolute():
    _cache_path = _REPO / _cache_path
print(f"probing: {_cache_path}")
cache = json.loads(_cache_path.read_text())

n_findings = 0
n_dropped = 0
reason_counts: Counter = Counter()
chunks_seen = 0
chunks_missing_map = 0

for entry in cache.values():
    pmcid = entry["pmcid"]
    chunk_map = entry.get("chunk_map", {})
    for chunk_id in chunk_map:
        chunk = chunk_map.get(chunk_id) or []
        if not chunk:
            chunks_missing_map += 1
            continue
        chunks_seen += 1
        validator = ProvenanceValidator(SourceIndex(chunk), pmcid,
                                        fabricated_threshold=0.0, weak_threshold=0.0)
        # Score every L1/L2/L3 voter output for this chunk.
        for level in ("l1", "l2", "l3"):
            raw = entry.get(level, {}).get(chunk_id)
            if not raw:
                continue
            outputs = raw if isinstance(raw, list) else [raw]
            for d in outputs:
                if not d:
                    continue
                summary = AuditableSummary.model_validate(d)
                for val in validator.validate(summary):
                    n_findings += 1
                    hard = [c for c in val.reason_codes if c in STRUCTURAL_CITATION_CODES]
                    if hard:
                        n_dropped += 1
                        for c in hard:
                            reason_counts[c.value] += 1

print(f"chunks scored:          {chunks_seen} (missing chunk_map: {chunks_missing_map})")
print(f"voter findings scored:  {n_findings}")
print(f"citation-FAIL findings: {n_dropped}  ({100*n_dropped/max(n_findings,1):.2f}%)")
print("by reason code:")
for code, n in reason_counts.most_common():
    print(f"  {code:32s} {n}")
