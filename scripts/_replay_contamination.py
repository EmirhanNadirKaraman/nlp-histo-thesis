"""Throwaway (B-081): selected-set contamination rate via the real _replay.

Runs the frozen-config legacy cascade over the related15 voter_cache and measures
how many of the SELECTED findings (the ~2280 that drive strict-F1) fail the
structural citation check — i.e. how much Gemini cross-paper contamination
actually survives selection, vs the 37% all-voter rate.
"""
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from dotenv import load_dotenv
load_dotenv(str(_REPO / ".env"))

from eval.silver.map_context import _load_map_context
from eval.silver.map_theta_sweep import _build_scorer, _replay
from eval.silver.experiments.E03_grounding.grounding_sweep_related15 import _frozen_spec
from pipeline.stages.knowledge_extraction.helpers.citation_filter import STRUCTURAL_CITATION_CODES
from pipeline.stages.knowledge_extraction.routing.provenance_validator import (
    ProvenanceValidator,
    SourceIndex,
)

THETA, REJECT = 0.9, 0.1

print("loading map context (gemini embedder, cached) ...")
ctx = _load_map_context("gemini", embed_cache_path=None)
scorer = _build_scorer(_frozen_spec(), ctx.agreement_embed_fn)


def _run(citation_enabled: bool):
    out, *_ = _replay(
        ctx.voter_cache, scorer, THETA, REJECT,
        single_voter_policy="keep", force_escalate_on_polarity_conflict=True,
        citation_enabled=citation_enabled,
    )
    out = [co for co in out if co.case_id in ctx.silver_by_case]
    return out


print(f"replaying frozen cascade theta={THETA}/reject={REJECT} (citation OFF) ...")
base = _run(citation_enabled=False)
n_base = sum(len(co.findings) for co in base)

print("replaying (citation ON) ...")
filt = _run(citation_enabled=True)
n_filt = sum(len(co.findings) for co in filt)

# Reason-code breakdown over the SELECTED baseline set (PipelineFinding ducks as
# Finding for ProvenanceValidator: it has .evidence / .verbatim_support / .claim).
case_to_entry = {e["case_id"]: e for e in ctx.voter_cache.values()}
reason_counts: Counter = Counter()
n_contaminated = 0
n_checked = 0
for co in base:
    entry = case_to_entry.get(co.case_id)
    if not entry:
        continue
    for f in co.findings:
        chunk = entry.get("chunk_map", {}).get(f.chunk_id) or []
        if not chunk:
            continue
        n_checked += 1
        v = ProvenanceValidator(SourceIndex(chunk), co.pmcid,
                                fabricated_threshold=0.0, weak_threshold=0.0)
        val = v._validate_finding(0, f)
        hard = [c for c in val.reason_codes if c in STRUCTURAL_CITATION_CODES]
        if hard:
            n_contaminated += 1
            for c in hard:
                reason_counts[c.value] += 1

print("\n" + "=" * 60)
print(f"SELECTED findings (citation OFF, baseline): {n_base}")
print(f"SELECTED findings (citation ON,  filtered): {n_filt}")
print(f"dropped by filter:                          {n_base - n_filt} "
      f"({100*(n_base-n_filt)/max(n_base,1):.2f}%)")
print("-" * 60)
print(f"selected findings checked:                  {n_checked}")
print(f"selected findings FAILING citation:         {n_contaminated} "
      f"({100*n_contaminated/max(n_checked,1):.2f}%)")
print("by reason code:")
for code, n in reason_counts.most_common():
    print(f"  {code:32s} {n}")
print("=" * 60)
