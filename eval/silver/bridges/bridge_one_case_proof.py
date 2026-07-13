#!/usr/bin/env python3
"""One-case proof: primer→MAP bridge at the frozen θ0.9 cascade (per-case, overlap 0).

Verifies, with NO expensive (voter) API spend, that the frozen-config MAP stage,
fed one case at chunk_size=10 / chunk_overlap=0:

  1. ZERO API   — voter/escalation LLMs are stubbed to raise AND their provider
                  keys are blanked, so any attempt to call one fails loudly
                  (never a charge). Reaching the end = no voter call happened.
  2. CACHE HIT  — when the MAP cache is populated from the validated θ0.9 replay,
                  MapStage.process hits every chunk (cache._hits == n_chunks, 0 misses).
  3. CHUNK ID   — MapStage._make_chunks(case, size=10, overlap=0) == the primer's
                  chunk_map exactly; overlap=2 does NOT (proves we're off the old path).
  4. OUTPUT     — the cache-hit MapStage findings == the sweep's validated θ0.9
                  output for that case (`_replay`, the exact code E06–E12 calibrated on).

Only cheap, cache-warm agreement embeddings stay live (GeminiEmbedder, served from
the prewarmed cache). Run:  python -m eval.silver.bridges.bridge_one_case_proof
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from eval.paths import REPO_ROOT as _REPO_ROOT

load_dotenv(str(_REPO_ROOT / ".env"))

# HARD SPEND GUARD: blank the expensive voter providers. Voters are replayed from
# the primer and must never be invoked; with blank keys, any stray call is an auth
# error, not a charge. (GOOGLE_API_KEY is kept: the agreement embedder needs it, but
# reads from the prewarmed cache — cheap embeddings only, no LLM voter spend.)
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

from langchain_core.runnables import RunnableLambda  # noqa: E402

from eval.silver.analysis.map_context import _load_map_context  # noqa: E402
from eval.silver.analysis.map_theta_sweep import (  # noqa: E402
    ScorerSpec,
    _build_scorer,
    _replay,
)
from pipeline.stages.knowledge_extraction.agreement import AgreementChecker  # noqa: E402
from pipeline.stages.knowledge_extraction.batch.voter_configs import get_profile  # noqa: E402
from pipeline.stages.knowledge_extraction.cache import PipelineCache  # noqa: E402
from pipeline.stages.knowledge_extraction.config import AgreementConfig, HybridConfig  # noqa: E402
from pipeline.stages.knowledge_extraction.stages.map_stage import MapStage  # noqa: E402
from pipeline.stages.knowledge_extraction.interfaces.scoring import ChunkDecision  # noqa: E402
from pipeline.stages.knowledge_extraction.models import AuditableSummary  # noqa: E402
from eval.silver.analysis.run_new_summarization_sweeps import (  # noqa: E402
    BEST_FORCE_ESCALATE_POLARITY, BEST_REJECT_THETA, BEST_SINGLE_VOTER_POLICY,
    BEST_THETA, BEST_VOTER_SUBSET, _filtered_voter_cache,  # noqa: F401  (BEST_VOTER_SUBSET/_filtered used by importers)
)

# Shipped 5-voter / escalate pins (BEST_*). bridge_populate_corpus imports THETA/REJECT from here.
THETA, REJECT = BEST_THETA, BEST_REJECT_THETA
_PROVIDER_SPEC = {"claude": "anthropic", "anthropic": "anthropic",
                  "openai": "openai", "gemini": "gemini"}


def _frozen_spec() -> ScorerSpec:
    """The E05–E08 winner scorer (= run.yaml / E03 `_frozen_spec`)."""
    cfg = AgreementConfig(
        scorer_kind="hybrid", alignment_strategy="greedy",
        tau=0.15, count_alpha=0.25, reuse_weight=0.15, contradiction_weight=0.20,
        hybrid=HybridConfig(w_category=0.15, w_embedding=0.30, w_entity=0.50, w_evidence=0.05),
    )
    return ScorerSpec("hybrid", "hybrid", cfg)


def _raise(*_a, **_k):
    raise RuntimeError("VOTER CALL ATTEMPTED — the proof requires every chunk to be a cache hit")


class _StubLLM:
    """Constructs into a map-chain but raises if the chain is ever invoked."""
    temperature = 0.0
    top_p = None

    def with_structured_output(self, *_a, **_k):
        return RunnableLambda(_raise)


def _sorted_cids(chunk_map: dict) -> list[str]:
    return sorted(chunk_map, key=lambda c: int(c[1:]))


def _chunk_key(chunk: list[dict]) -> list[tuple]:
    return [(s.get("text_element_id"), s.get("sentence")) for s in chunk]


def _claim_key(f) -> str:
    for attr in ("claim", "predicate_text", "predicate"):
        v = getattr(f, attr, None)
        if v is None and isinstance(f, dict):
            v = f.get(attr)
        if v:
            return str(v).strip()
    return repr(f)


def _resolve_chunk(entry: dict, chunk_id: str, checker: AgreementChecker):
    """Faithful copy of _replay's per-chunk legacy cascade → the chunk's selected
    AuditableSummary dict (None if rejected / no data). Same AgreementChecker as
    production, so the result is the validated θ0.9 MAP output for the chunk."""
    source_text = entry.get("source_texts", {}).get(chunk_id, "")
    l1_raw = entry.get("l1", {}).get(chunk_id) or []
    l1_out = [AuditableSummary.model_validate(d) for d in l1_raw if d]
    selected = None
    level = "no_data"
    if l1_out:
        b1 = checker.compute(l1_out, source_text=source_text)
        if b1.decision == ChunkDecision.KEEP:
            selected = checker.best(l1_out, b1).model_dump()
            level = "l1"
        elif b1.decision == ChunkDecision.REJECT:
            level = "reject"
    if selected is None and level == "no_data":
        l2_raw = entry.get("l2", {}).get(chunk_id) or []
        l2_out = [AuditableSummary.model_validate(d) for d in l2_raw if d]
        if l2_out:
            b2 = checker.compute(l2_out, source_text=source_text)
            if b2.decision == ChunkDecision.KEEP:
                selected = checker.best(l2_out, b2).model_dump()
                level = "l2"
            elif b2.decision == ChunkDecision.REJECT:
                level = "reject"
            else:
                selected = entry.get("l3", {}).get(chunk_id)
                level = "l3"
        else:
            selected = entry.get("l3", {}).get(chunk_id)
            level = "l3"
    return selected, level


def _build_mapstage(prof, scorer, *, chunk_overlap: int) -> MapStage:
    l1_specs = [(_PROVIDER_SPEC[v.provider], v.model) for v in prof.l1_voters]
    l2_specs = [(_PROVIDER_SPEC[v.provider], v.model) for v in prof.l2_voters]
    l3_spec = (_PROVIDER_SPEC[prof.l3_voter.provider], prof.l3_voter.model)
    return MapStage(
        [_StubLLM() for _ in prof.l1_voters],
        [_StubLLM() for _ in prof.l2_voters],
        _StubLLM(),
        theta=THETA, reject_theta=REJECT,
        chunk_size=10, chunk_overlap=chunk_overlap, chunk_workers=1,
        scorer=scorer, router=None,
        voter_specs=l1_specs, level2_voter_specs=l2_specs, escalation_spec=l3_spec,
        cascade_profile="real",
        legacy_single_voter_policy=BEST_SINGLE_VOTER_POLICY,
        force_escalate_on_polarity_conflict=BEST_FORCE_ESCALATE_POLARITY,
    )


def main() -> int:
    print("E-bridge one-case proof — frozen θ0.9 cascade, per-case, chunk_size=10, overlap=0\n")
    ctx = _load_map_context("gemini", embed_cache_path=None)
    scorer = _build_scorer(_frozen_spec(), ctx.agreement_embed_fn)
    checker = AgreementChecker(scorer, theta=THETA, reject_theta=REJECT,
                               single_voter_policy=BEST_SINGLE_VOTER_POLICY,
                               force_escalate_on_polarity_conflict=BEST_FORCE_ESCALATE_POLARITY)

    # 1. pick the first MULTI-chunk case (so chunk alignment is actually exercised)
    safe_id, entry = next(
        ((sid, e) for sid, e in ctx.voter_cache.items() if len(e.get("chunk_map", {})) >= 2),
        (None, None),
    )
    if entry is None:
        print("No multi-chunk case in the primer — cannot exercise alignment.")
        return 1
    case_id, pmcid = entry["case_id"], entry["pmcid"]
    cids = _sorted_cids(entry["chunk_map"])
    primer_chunks = [entry["chunk_map"][c] for c in cids]
    sentences = [s for c in cids for s in entry["chunk_map"][c]]  # disjoint → full list
    n_sent = len(sentences)
    print(f"selected case_id : {case_id}")
    print(f"pmcid            : {pmcid}")
    print(f"sentences        : {n_sent}")
    print(f"expected chunks  : {len(primer_chunks)}  (primer chunk_map: {[len(c) for c in primer_chunks]})")

    prof = get_profile("real")

    # 2. CHUNK IDENTITY (#3, #5)
    ms = _build_mapstage(prof, scorer, chunk_overlap=0)
    actual = ms._make_chunks(sentences)
    actual_match = (
        len(actual) == len(primer_chunks)
        and all(_chunk_key(a) == _chunk_key(p) for a, p in zip(actual, primer_chunks))
    )
    ms_ov2 = _build_mapstage(prof, scorer, chunk_overlap=2)
    actual_ov2 = ms_ov2._make_chunks(sentences)
    ov2_differs = not (
        len(actual_ov2) == len(primer_chunks)
        and all(_chunk_key(a) == _chunk_key(p) for a, p in zip(actual_ov2, primer_chunks))
    )
    print(f"\nactual chunks    : {len(actual)}  (overlap=0 sizes: {[len(c) for c in actual]})")
    print(f"overlap=2 chunks : {len(actual_ov2)} (sizes {[len(c) for c in actual_ov2]})")
    print(f"#3 chunk identity (overlap=0 == primer) : {'PASS' if actual_match else 'FAIL'}")
    print(f"#5 overlap=2 differs from primer        : {'PASS' if ov2_differs else 'FAIL'}")

    # 3. validated θ0.9 output for the case (the sweep's own replay) + per-chunk resolve
    f_replay = _replay({safe_id: entry}, scorer, THETA, REJECT,
                       single_voter_policy=BEST_SINGLE_VOTER_POLICY,
                       force_escalate_on_polarity_conflict=BEST_FORCE_ESCALATE_POLARITY)[0][0].findings
    per_chunk = [(_resolve_chunk(entry, c, checker)) for c in cids]

    # 4. populate the MAP cache from the resolved summaries, keyed by MapStage's own chunks
    cache = PipelineCache(Path(tempfile.mkdtemp()) / "proof_cache.json")
    lookup_md = ms.cascade_lookup_metadata()
    populated = 0
    for chunk_sents, (sel, _lvl) in zip(actual, per_chunk):
        if sel is None:
            continue
        cache.set_map(chunk_sents, AuditableSummary.model_validate(sel), lookup_md)
        populated += 1

    # 5. run the production MAP stage against the populated cache (stubs raise on any call)
    try:
        results = ms.process(sentences, pmcid, cache=cache)
        voter_called = False
    except RuntimeError as exc:  # a stub fired ⇒ a voter call was attempted
        print(f"\n#1 ZERO API : FAIL — {exc}")
        return 1
    hits, misses = cache._hits["map"], cache._misses["map"]
    proc_findings = [f for r in results for f in (r.findings or [])]
    output_match = sorted(map(_claim_key, proc_findings)) == sorted(map(_claim_key, f_replay))

    print(f"\ncache populated  : {populated}/{len(actual)} chunks")
    print(f"cache hits/misses: {hits} / {misses}")
    print(f"voter calls      : {'NONE' if not voter_called else 'SOME'}  (stub never fired)")
    print(f"#1 zero API (no voter call)              : {'PASS' if not voter_called else 'FAIL'}")
    print(f"#2 cache hit alignment (hits==chunks,0 miss): "
          f"{'PASS' if hits == len(actual) and misses == 0 else 'FAIL'}")
    print(f"#4 output == sweep θ0.9 (findings: {len(proc_findings)} vs {len(f_replay)}): "
          f"{'PASS' if output_match else 'FAIL'}")

    ok = actual_match and ov2_differs and (not voter_called) and \
        hits == len(actual) and misses == 0 and output_match
    print(f"\n=== ONE-CASE PROOF: {'PASS ✓' if ok else 'FAIL ✗'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
