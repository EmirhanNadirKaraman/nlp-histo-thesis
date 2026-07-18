"""
MAP stage with Agreement-Based Cascading (ABC).

For each chunk of sentences:
  Level 1  — run each voter LLM in parallel (one call per voter).
             Voters should be distinct models / providers so disagreement
             reflects architectural differences rather than sampling
             noise (e.g. gemini-flash-lite + gpt-4o-mini + gpt-4.1-nano).  When using
             same-provider voters, use temperature diversity and lower theta.
             If pairwise claim-embedding alignment >= theta, accept the
             best result.
  Level 2  — escalate to mid-tier voter LLMs (one call per voter, in
             parallel) for chunks where Level 1 voters disagree.  If these
             agree, accept the best result without calling Level 3.
  Level 3  — escalate to the strong model (single call) for chunks where
             both Level 1 and Level 2 voters disagree.

Note: the router path (when self._router is not None) bypasses Level 2 and
escalates directly from Level 1 to Level 3.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING
from ..agreement import (
    AgreementChecker,
    CascadeDecisionLog,
    EmbeddingScorer,
    MapOutputScorer,
    evaluate_chunk,
    make_decision_record,
    producer_from_outcome,
)
from ..cache import PipelineCache
from ..enum_logging import log_bad_finding
from ..llm.llm_errors import is_non_retryable_llm_error
from ..models import (
    AuditableSummary,
    MapRunMetadata,
    compute_cascade_signature,
    compute_voter_config_hash,
)
from ..llm.prompts import build_map_chain
from ..routing import MapOutputRouter
from ..routing.routing_dataset import RoutingDataset, RoutingRecord
from nlp_histo.pipeline.stages.knowledge_extraction.interfaces.scoring import ChunkDecision

if TYPE_CHECKING:
    from ..config import CitationConfig

logger = logging.getLogger(__name__)

# LengthFinishReasonError lives in the `openai` package (the structured-
# output parser raises it when finish_reason="length"). Gemini's
# direct-API ChatOpenAI wrapper goes through the same OpenAI client at
# https://generativelanguage.googleapis.com/v1beta/openai/..., so this
# exception fires for Gemini voters too — not just OpenAI ones.
# Import from the openai SDK, not langchain_core.exceptions — the class does
# not exist there, so the ImportError fallback would leave the guard None and
# length-limit errors would go through the retry loop pointlessly.
try:
    from openai import LengthFinishReasonError as _LengthFinishReasonError
except ImportError:  # openai SDK absent (test contexts)
    _LengthFinishReasonError = None  # type: ignore[assignment,misc]


def _selected_global_index(outcome, *, kept: bool) -> int | None:
    """Translate an agreement outcome's best_index into the ORIGINAL voter
    spec index. Returns None when the level escalated (no winner) or when
    the bundle is missing — both legitimate "no selected voter" cases.

    Mirrors the lookup that ``producer_from_outcome`` uses, but returns the
    integer index instead of the (provider, model) tuple. Used to populate
    ``SumMapVoterOutput.is_selected`` so the row whose ``raw_output``
    became the level's MAP result is identifiable.
    """
    if not kept:
        return None
    bundle = getattr(outcome, "agreement_bundle", None)
    best_idx = getattr(bundle, "best_index", None) if bundle is not None else None
    if best_idx is None:
        return None
    vvi = getattr(outcome, "valid_voter_indices", None)
    if vvi is None:
        return int(best_idx)
    if 0 <= best_idx < len(vvi):
        return int(vvi[best_idx])
    return None


def _format_sentences(chunk: list[dict]) -> str:
    """Tag each sentence with a citation ID: [S{i}|{PMCID}|{te_id}]."""
    lines = []
    for i, item in enumerate(chunk, 1):
        pmcid = item.get("pmcid", "UNKNOWN")
        te_id = item.get("text_element_id", 0)
        text = item.get("sentence", "").strip()
        lines.append(f"[S{i}|{pmcid}|{te_id}] {text}")
    return "\n".join(lines)


def _llm_fallback_spec(llm) -> tuple[str, str]:
    """Extract (provider, model) from the LLM object when no specs were passed.

    The langchain chat clients usually expose ``model_name`` or ``model``;
    we fall back to the class name. The signature is informational; cache
    correctness depends on callers passing real specs.
    """
    name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or type(llm).__name__
    provider = type(llm).__name__.replace("Chat", "").lower() or "unknown"
    return (provider, str(name))


def _llm_fallback_specs(llms: list) -> list[tuple[str, str]]:
    return [_llm_fallback_spec(llm) for llm in llms]


def _voter_grounding(v: AuditableSummary) -> tuple[float, float]:
    """Fallback grounding signals when no AgreementContext is available."""
    if not v.findings:
        return 1.0, 0.0
    pf = sum(1 for f in v.findings if f.evidence) / len(v.findings)
    me = sum(len(f.evidence) for f in v.findings) / len(v.findings)
    return pf, me


class MapStage:
    """
    MAP stage: chunks → AuditableSummary list, with ABC cascading.

    Parameters
    ----------
    voter_llms:
        List of LLMs used as Level-1 voters.  Use models from different
        providers (e.g. [ChatOpenAI(...), ChatAnthropic(...)]) so that
        disagreement reflects uncertainty rather than sampling noise.
        Must have at least one entry.
    escalation_llm:
        LLM for Level-2 escalation.  Only called when voters disagree below
        theta.  Typically a larger / more capable model (the shipped
        escalation is Claude-Sonnet).
    theta:
        Agreement threshold in [0, 1].  Higher = more escalations.
    chunk_size:
        Number of sentences per chunk.
    chunk_overlap:
        Number of sentences shared between adjacent chunks.  Must be >= 0 and
        < chunk_size.  Default 0 preserves the original disjoint behaviour.
    scorer:
        MapOutputScorer used to score voter agreement.  Defaults to
        EmbeddingScorer.  Pass CascadedCompositeScorer for embedding +
        LLM judge cascade, or CategoryJaccardScorer for a fast, API-free
        alternative.
    """

    def __init__(
        self,
        voter_llms: list,
        level2_voter_llms: list,
        escalation_llm,
        theta: float = 0.7,
        reject_theta: float = 0.0,
        chunk_size: int = 10,
        chunk_overlap: int = 0,
        chunk_workers: int = 5,
        scorer: MapOutputScorer | None = None,
        router: MapOutputRouter | None = None,
        routing_collector: RoutingDataset | None = None,
        voter_specs:        list[tuple[str, str]] | None = None,
        level2_voter_specs: list[tuple[str, str]] | None = None,
        escalation_spec:    tuple[str, str] | None = None,
        cascade_profile:    str = "custom",
        legacy_single_voter_policy: str = "keep",
        force_escalate_on_polarity_conflict: bool = True,
        citation_config: "CitationConfig | None" = None,
    ) -> None:
        if not voter_llms:
            raise ValueError("voter_llms must contain at least one LLM.")
        if not level2_voter_llms:
            raise ValueError("level2_voter_llms must contain at least one LLM.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._voter_chains = [build_map_chain(llm) for llm in voter_llms]
        self._level2_voter_chains = [build_map_chain(llm) for llm in level2_voter_llms]
        self._escalation_chain = build_map_chain(escalation_llm)
        # Per-voter temperatures, captured at construction so the in-run voter
        # cache (process(): self._voter_cache) can key on (provider, model, temp).
        # Falls back to 0.0 for LLM types that don't expose .temperature.
        self._l1_temps = [float(getattr(llm, "temperature", 0.0) or 0.0) for llm in voter_llms]
        self._l2_temps = [float(getattr(llm, "temperature", 0.0) or 0.0) for llm in level2_voter_llms]
        self._l3_temp  = float(getattr(escalation_llm, "temperature", 0.0) or 0.0)

        # Per-voter top_p, captured for the MAP cache key (voter_config_hash).
        # Some LangChain LLM wrappers don't expose .top_p; we record None in
        # that case so the hash distinguishes "absent" from any literal value.
        def _capture_top_p(llm) -> float | None:
            value = getattr(llm, "top_p", None)
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        self._l1_top_ps = [_capture_top_p(llm) for llm in voter_llms]
        self._l2_top_ps = [_capture_top_p(llm) for llm in level2_voter_llms]
        self._l3_top_p  = _capture_top_p(escalation_llm)
        self._agreement = AgreementChecker(
            scorer=scorer or EmbeddingScorer(),
            theta=theta,
            reject_theta=reject_theta,
            single_voter_policy=legacy_single_voter_policy,  # type: ignore[arg-type]
            force_escalate_on_polarity_conflict=force_escalate_on_polarity_conflict,
        )
        self._router = router
        # Finding-level citation filter (B-080) — applied to the selected chunk
        # summary in _cascade, covering every cascade level. None → default
        # CitationConfig (enabled). See provenance/citation_filter.py.
        from ..config import CitationConfig  # noqa: PLC0415 — avoid import cycle at module load
        self._citation_cfg = citation_config or CitationConfig()
        self._escalation_lock = threading.Lock()
        self._last_escalation_counts: dict[str, int] = {}
        self._routing_collector = routing_collector
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._chunk_workers = chunk_workers

        # Cascade provenance for cache keys / run artifacts
        # Fallback: when callers don't pass voter_specs we still
        # produce a cascade_signature, but it's based on type+id rather than
        # provider/model so it won't be cross-process stable.
        self._voter_specs = list(voter_specs)        if voter_specs        else _llm_fallback_specs(voter_llms)
        self._l2_specs    = list(level2_voter_specs) if level2_voter_specs else _llm_fallback_specs(level2_voter_llms)
        self._l3_spec     = escalation_spec          if escalation_spec    else _llm_fallback_spec(escalation_llm)
        self._cascade_profile = cascade_profile
        self._cascade_signature = compute_cascade_signature(
            self._voter_specs + self._l2_specs + [self._l3_spec]
        )

        # MAP cache key inputs that go beyond (provider, model) identity:
        # voter sampler settings and the active NLI grounding model id. See
        # ``models.compute_voter_config_hash`` and ``MapRunMetadata`` for the
        # rationale. Captured here so every metadata object built by
        # ``_make_metadata`` carries identical values.
        l1_full_specs = [
            (p, m, t, tp)
            for (p, m), t, tp in zip(self._voter_specs, self._l1_temps, self._l1_top_ps)
        ]
        l2_full_specs = [
            (p, m, t, tp)
            for (p, m), t, tp in zip(self._l2_specs, self._l2_temps, self._l2_top_ps)
        ]
        l3_full_spec = (
            self._l3_spec[0], self._l3_spec[1], self._l3_temp, self._l3_top_p,
        )
        self._voter_config_hash = compute_voter_config_hash(
            l1_full_specs + l2_full_specs + [l3_full_spec]
        )

        # Active NLI grounding/relation checkpoint. Bound at construction so
        # the metadata stamped on every cache entry reflects the model that
        # produced the persisted grounding scores. Resolves ``$NLP_HISTO_NLI_MODEL``.
        try:
            from ..grounding.nli_config import get_active_spec  # noqa: PLC0415 — local import
            self._nli_model_id = get_active_spec().hf_id
        except Exception:  # noqa: BLE001
            self._nli_model_id = ""

        # Invocation-time usage capture
        # Records one InvocationUsage per chain.invoke() call. Populated in
        # _run_voters and _invoke_l3 by reading usage_metadata directly off
        # the raw AIMessage returned by build_map_chain's include_raw=True
        # structured-output chain. Cache hits do not add a record (no API
        # call happened). See pipeline/stages/knowledge_extraction/costing/
        # invocation_usage.py for the rationale.
        from ..costing import InvocationUsage  # noqa: PLC0415 — local import
        self._InvocationUsage = InvocationUsage
        self._invocation_usage_records: list = []
        self._invocation_usage_lock = threading.Lock()

        # Per-voter output buffer
        # Captures every voter's AuditableSummary (or failure) per (chunk,
        # level, voter_index). The runner reads this via
        # ``voter_output_records()`` after MAP completes and persists into
        # ``sum_map_voter_outputs``. Cache hits are skipped (the chunk did
        # not reach any voter this run; the original voter outputs were
        # captured by whichever earlier run populated the cache).
        self._voter_output_records: list[dict] = []
        self._voter_output_lock = threading.Lock()

    # Run-metadata helpers

    def _make_metadata(self, provider: str, model: str) -> MapRunMetadata:
        return MapRunMetadata(
            provider=provider,
            model=model,
            cascade_profile=self._cascade_profile,
            cascade_signature=self._cascade_signature,
            voter_config_hash=self._voter_config_hash,
            nli_model_id=self._nli_model_id,
        )

    def cascade_lookup_metadata(self) -> MapRunMetadata:
        """Metadata used for cache lookups — provider/model are placeholders.

        Cache keys do not include provider/model (they include cascade_signature
        instead) so any (provider, model) pair works for lookups.
        """
        provider, model = self._voter_specs[0] if self._voter_specs else ("unknown", "unknown")
        return self._make_metadata(provider, model)

    def run_metadata_summary(self) -> dict:
        """Canonical run-metadata block for inclusion in MAP run artifacts.

        Lists every cascade member (L1 voters + L2 voters + L3) so a downstream
        consumer can recover the exact configuration that produced the run.
        """
        return {
            "schema_version":    self.cascade_lookup_metadata().schema_version,
            "prompt_version":    self.cascade_lookup_metadata().prompt_version,
            "stage_name":        self.cascade_lookup_metadata().stage_name,
            "cascade_profile":   self._cascade_profile,
            "cascade_signature": self._cascade_signature,
            "l1_voters":         [{"provider": p, "model": m} for p, m in self._voter_specs],
            "l2_voters":         [{"provider": p, "model": m} for p, m in self._l2_specs],
            "l3_voter":          {"provider": self._l3_spec[0], "model": self._l3_spec[1]},
        }

    # Public API

    def process(
        self,
        sentences: list[dict],
        pmcid: str,
        cache: PipelineCache | None = None,
        collector=None,  # TraceCollector | None
        start_chunk: int = 0,
        limit_chunks: int | None = None,
        usage_collector=None,  # costing.UsageCollector | None
        cascade_decision_log: CascadeDecisionLog | None = None,
        run_id: str | None = None,
    ) -> list[AuditableSummary]:
        """
        Map all sentences for one paper, returning one AuditableSummary per chunk.

        Chunks that hit the cache are returned immediately.  Remaining chunks
        run through the ABC cascade.

        Parameters
        ----------
        start_chunk:
            Zero-based index of the first chunk to process.  Chunks before
            this index are skipped entirely (not cached, not counted).
        limit_chunks:
            Maximum number of chunks to process starting from *start_chunk*.
            None means process all remaining chunks.
        """

        with self._escalation_lock:
            self._last_escalation_counts = {
                "total": 0, "l1_kept": 0, "l2_escalated": 0, "l2_kept": 0,
                "l3_escalated": 0, "l3_kept": 0, "finalized": 0, "dropped": 0,
                # Chunks served from PipelineCache; never reach _process_chunk
                # and therefore never bump the per-tier counters. Tracked
                # separately so the cascade report can distinguish API work
                # from cache work without conflating cost numbers.
                "cache_hits": 0,
            }
        # Reset invocation-usage records for this paper so the runner reads
        # only the current MAP run's records.
        with self._invocation_usage_lock:
            self._invocation_usage_records = []
        # Reset per-voter output buffer for this paper. Same lifecycle as
        # invocation_usage_records — populated during _cascade, read by
        # the runner after MAP completes, cleared at start of the next paper.
        with self._voter_output_lock:
            self._voter_output_records = []

        # Thread the usage collector onto the instance for the duration of
        # this paper. Reset in a finally so an exception cannot leave a stale
        # collector reference visible to a future caller.
        self._usage_collector = usage_collector
        self._usage_pmcid = pmcid

        # Per-paper cascade decision log (one JSONL row per L1/L2/L3 decision).
        # Lifetime matches usage_collector — reset in finally so a later caller
        # does not accidentally write into the previous paper's file.
        self._cascade_decision_log = cascade_decision_log
        self._cascade_run_id = run_id or "unknown"

        # Per-paper voter cache: dedups identical (provider, model, temp, text)
        # voter calls within one paper so e.g. Haiku at L1 and Haiku at L2 on
        # the same chunk only hit the API once. Reset per-paper because a stale
        # entry from a different paper would never key-match anyway (text hash
        # differs), but freeing the dict keeps memory bounded. The chunk-level
        # PipelineCache is unaffected; this layer sits underneath it.
        self._voter_cache: dict[tuple, AuditableSummary] = {}
        self._voter_cache_lock = threading.Lock()
        self._voter_cache_hits = 0

        try:
            return self._process_inner(
                sentences, pmcid, cache, collector, start_chunk, limit_chunks,
                usage_collector,
            )
        finally:
            self._usage_collector = None
            self._usage_pmcid = None
            self._voter_cache = {}
            self._voter_cache_hits = 0
            self._cascade_decision_log = None
            self._cascade_run_id = "unknown"

    def _process_inner(
        self,
        sentences,
        pmcid,
        cache,
        collector,
        start_chunk,
        limit_chunks,
        usage_collector,
    ):
        from ..observability.models import ChunkTrace
        all_chunks = self._make_chunks(sentences)
        total_paper_chunks = len(all_chunks)
        if start_chunk or limit_chunks is not None:
            end = (start_chunk + limit_chunks) if limit_chunks is not None else None
            chunks = all_chunks[start_chunk:end]
            logger.info(
                "[%s] MAP limited: processing %d chunk(s) starting at C%d (paper has %d total)",
                pmcid, len(chunks), start_chunk + 1, total_paper_chunks,
            )
        else:
            chunks = all_chunks
        results: list[AuditableSummary | None] = []
        uncached: list[tuple[int, list[dict]]] = []
        cache_hit_indices: list[int] = []

        lookup_md = self.cascade_lookup_metadata()
        for idx, chunk in enumerate(chunks):
            abs_idx = start_chunk + idx  # absolute position in the full paper
            if cache:
                hit = cache.get_map(chunk, lookup_md)
                if hit:
                    # Always assign chunk_id from current position — cached
                    # chunk_id may be stale (e.g. from a different run or paper).
                    cached_chunk_id = f"C{abs_idx + 1}"
                    hit = hit.model_copy(update={"chunk_id": cached_chunk_id})
                    results.append(hit)
                    cache_hit_indices.append(idx)
                    if usage_collector is not None:
                        usage_collector.record_cache_hit(
                            stage="MAP", role="voter",
                            substage="cache_hit",
                            item_id=cached_chunk_id,
                        )
                    # Cache hits skip _process_chunk where the per-tier
                    # escalation counters are incremented. Bump total /
                    # finalized / cache_hits here so the cascade report
                    # reflects all chunks the paper went through, not just
                    # the cache-miss ones. Tier counters (l1_kept etc.)
                    # stay zero — we don't know which tier originally
                    # produced the cached entry.
                    with self._escalation_lock:
                        self._last_escalation_counts["total"] += 1
                        self._last_escalation_counts["finalized"] += 1
                        self._last_escalation_counts["cache_hits"] += 1
                    # Replay per-voter outputs from the cache if the entry
                    # carries them so sum_map_voter_outputs gets populated
                    # even on cache hits. Older entries lack the field and
                    # silently contribute nothing — degraded but not broken.
                    cached_voter_outputs = cache.get_voter_outputs(chunk, lookup_md)
                    if cached_voter_outputs:
                        # Restamp chunk_id in case the cache was written
                        # under a different per-paper chunk numbering.
                        with self._voter_output_lock:
                            for r in cached_voter_outputs:
                                row = dict(r)
                                row["chunk_id"] = cached_chunk_id
                                self._voter_output_records.append(row)
                    continue
            results.append(None)
            uncached.append((idx, chunk))

        if cache_hit_indices:
            logger.info(
                "[%s] MAP cache: %d/%d chunks hit, %d to process",
                pmcid, len(cache_hit_indices), len(chunks), len(uncached),
            )

        # Record cache-hit chunk traces (lightweight — no voter/agreement data)
        if collector is not None:
            for idx in cache_hit_indices:
                chunk = chunks[idx]
                te_ids = sorted({item.get("text_element_id", 0) for item in chunk})
                text = _format_sentences(chunk)
                collector.add_chunk_trace(ChunkTrace(
                    chunk_id=f"C{start_chunk + idx + 1}",
                    run_id=collector.run_id,
                    pmcid=pmcid,
                    te_ids=te_ids,
                    sentence_count=len(chunk),
                    text_preview=text[:200],
                    cache_hit=True,
                    voters=[],
                    agreement=None,
                    selected_voter_index=None,
                    escalated=False,
                ))

        total_chunks = len(chunks)
        n_cached = len(cache_hit_indices)

        def _process_chunk(args: tuple):
            idx, chunk = args
            chunk_id = f"C{start_chunk + idx + 1}"
            t_chunk = time.monotonic()
            result, producer = self._cascade(chunk, pmcid, chunk_id, collector=collector)
            elapsed_ms = (time.monotonic() - t_chunk) * 1000
            return idx, result, producer, elapsed_ms

        with ThreadPoolExecutor(max_workers=self._chunk_workers) as pool:
            futures = {pool.submit(_process_chunk, (idx, chunk)): (pos, idx, chunk)
                       for pos, (idx, chunk) in enumerate(uncached, 1)}
            for future in as_completed(futures):
                pos, idx, chunk = futures[future]
                chunk_id = f"C{start_chunk + idx + 1}"
                idx_result, result, producer, elapsed_ms = future.result()
                n_findings = len(result.findings) if result else 0
                logger.info(
                    "[%s] MAP chunk %s/%s done (chunk_id=%s) — %d findings  [%.0fms]",
                    pmcid, n_cached + pos, total_chunks, chunk_id, n_findings, elapsed_ms,
                )
                results[idx_result] = result
                if cache and result is not None:
                    prov, model = producer or ("unknown", "unknown")
                    # Snapshot the voter-output rows for this chunk so the
                    # cache entry carries them. Future cache hits replay
                    # these into sum_map_voter_outputs without needing
                    # another live cascade.
                    chunk_voter_rows = [
                        dict(r) for r in self.voter_output_records()
                        if r.get("chunk_id") == chunk_id
                    ]
                    cache.set_map(
                        chunk, result, self._make_metadata(prov, model),
                        voter_outputs=chunk_voter_rows or None,
                    )

        live_results = [r for r in results if r is not None]

        # Record MAP-stage summary
        if collector is not None:
            n_cache = len(cache_hit_indices)
            n_miss = len(uncached)
            chunk_traces = collector.chunk_traces
            escalations = sum(1 for ct in chunk_traces if ct.escalated and not ct.cache_hit)
            l2_escalations = sum(
                1 for ct in chunk_traces
                if not ct.cache_hit and ct.escalation_level >= 2
            )
            keeps = sum(
                1 for ct in chunk_traces
                if not ct.cache_hit and not ct.escalated
            )
            # rejects = cache-missed chunks that aren't keeps or escalations
            # (edge case: all non-cache chunks are either kept or escalated in practice)
            rejects = n_miss - escalations - keeps
            collector.record_map_stage(
                total_chunks=len(chunks),
                cache_hits=n_cache,
                cache_misses=n_miss,
                escalations=escalations,
                l2_escalations=l2_escalations,
                keeps=keeps,
                rejects=max(rejects, 0),
                total_findings_out=sum(len(r.findings) for r in live_results),
            )

        return live_results  # type: ignore[return-value]

    # ABC cascade

    def _cascade(
        self,
        chunk: list[dict],
        pmcid: str,
        chunk_id: str,
        collector=None,  # TraceCollector | None
    ) -> tuple[AuditableSummary | None, tuple[str, str] | None]:
        """Run the cascade and return ``(result, producer_spec)``.

        ``producer_spec`` is a ``(provider, model)`` tuple identifying which
        voter / model produced the kept result, so MAP cache entries and run
        artifacts can carry accurate provenance. ``None`` when no result.

        The per-level KEEP/escalate decision is delegated to
        ``agreement.decision.evaluate_chunk`` — the same function the batch
        runner uses — so the sync and batch decision surfaces are identical
        by construction.
        """
        inp = {
            "pmcid": pmcid,
            "chunk_id": chunk_id,
            "text": _format_sentences(chunk),
        }

        # Level 1: all voter chains in parallel (one thread per chain).
        # _run_voters returns a length-N list aligned to voter_specs, with
        # None for any voter that failed its API call. The survivor list
        # below filters Nones but keeps a survivor → original-index map so
        # downstream producer attribution stays correct (B-041).
        voters_full, voter_timings = self._run_voters(inp)
        survivor_indices = [i for i, v in enumerate(voters_full) if v is not None]
        voters: list[AuditableSummary] = [voters_full[i] for i in survivor_indices]

        # Slot to hold the agreement bundle and grounding contexts for trace building
        _trace_bundle = None
        _escalated = False
        _escalation_level = 1  # 1=L1 kept, 2=L2 kept, 3=L3 used
        _voter_contexts = None  # list[VoterContext] | None — from router
        _producer: tuple[str, str] | None = None
        result: AuditableSummary | None = None

        # Level 1 decision (shared evaluate_chunk path)
        l1_outcome = evaluate_chunk(
            voters,
            chunk=chunk,
            pmcid=pmcid,
            source_text=inp["text"],
            agreement=self._agreement,
            router=self._router,
            voter_indices=survivor_indices,
        )
        _trace_bundle = l1_outcome.agreement_bundle
        if l1_outcome.routing_decision is not None:
            _voter_contexts = l1_outcome.routing_decision.voter_grounding_contexts
            logger.debug(
                "Chunk %s: router → %s (gate=%s reasons=%s)",
                chunk_id,
                l1_outcome.routing_decision.decision.value,
                l1_outcome.routing_decision.gate_origin.value,
                [r.value for r in l1_outcome.routing_decision.reason_codes],
            )
        self._record_cascade_decision(
            l1_outcome, pmcid=pmcid, chunk_id=chunk_id, level="l1",
            voter_count=len(voters), voter_specs=self._voter_specs,
        )

        l1_selected_global_idx = _selected_global_index(
            l1_outcome, kept=l1_outcome.keep,
        )
        self._buffer_voter_outputs(
            pmcid=pmcid, chunk_id=chunk_id, level="l1",
            voters_full=voters_full, voter_specs=self._voter_specs,
            voter_timings=voter_timings,
            selected_voter_index=l1_selected_global_idx,
        )

        if l1_outcome.keep:
            result = l1_outcome.best
            _producer = producer_from_outcome(l1_outcome, self._voter_specs)
        elif self._router is not None:
            # Router path: L1 → L3 directly (no L2 step)
            _escalated = True
            _escalation_level = 3
            decision = l1_outcome.routing_decision
            if decision is not None and decision.decision == ChunkDecision.REJECT:
                logger.info(
                    "Chunk %s rejected by router — escalating to strong model.",
                    chunk_id,
                )
            result = self._invoke_l3(inp, chunk_id)
            _producer = self._l3_spec
            # Router path skips L2; record L3 immediately so the buffer is
            # symmetric with the legacy L1→L2→L3 staircase below.
            self._buffer_voter_outputs(
                pmcid=pmcid, chunk_id=chunk_id, level="l3",
                voters_full=[result], voter_specs=[self._l3_spec],
                voter_timings=None, selected_voter_index=0,
            )
        elif l1_outcome.rejected:
            # Legacy drop-on-reject: deferral <= reject_theta at L1, so the chunk
            # does not escalate and emits nothing (result stays None → counted as
            # dropped). With reject_theta=0.0 (default) this branch is effectively
            # unreachable for non-degenerate chunks; raise reject_theta to enable.
            logger.info(
                "Chunk %s rejected at L1 (deferral <= reject_theta) — dropping, no escalation",
                chunk_id,
            )
        else:
            # Legacy path: L1 → L2 → L3 staircase (no router). Reuses
            # evaluate_chunk for the L2 decision so the sync/batch surface
            # stays identical at each level.
            logger.debug("Chunk %s: L1 escalating to L2 (no router)", chunk_id)
            l2_full, l2_timings = self._run_voters(
                inp, self._level2_voter_chains, level="L2",
            )
            l2_survivor_indices = [i for i, v in enumerate(l2_full) if v is not None]
            l2_voters: list[AuditableSummary] = [l2_full[i] for i in l2_survivor_indices]
            l2_outcome = evaluate_chunk(
                l2_voters,
                chunk=chunk,
                pmcid=pmcid,
                source_text=inp["text"],
                agreement=self._agreement,
                router=None,
                voter_indices=l2_survivor_indices,
            )
            _trace_bundle = l2_outcome.agreement_bundle
            voters = l2_voters
            voter_timings = l2_timings
            _escalation_level = 2
            self._record_cascade_decision(
                l2_outcome, pmcid=pmcid, chunk_id=chunk_id, level="l2",
                voter_count=len(l2_voters), voter_specs=self._l2_specs,
            )

            l2_selected_global_idx = _selected_global_index(
                l2_outcome, kept=l2_outcome.keep,
            )
            self._buffer_voter_outputs(
                pmcid=pmcid, chunk_id=chunk_id, level="l2",
                voters_full=l2_full, voter_specs=self._l2_specs,
                voter_timings=l2_timings,
                selected_voter_index=l2_selected_global_idx,
            )

            if l2_outcome.keep:
                result = l2_outcome.best
                _producer = producer_from_outcome(l2_outcome, self._l2_specs)
            elif l2_outcome.rejected:
                # Drop-on-reject at L2: no L3 escalation, emit nothing
                # (result stays None → counted as dropped).
                logger.info(
                    "Chunk %s rejected at L2 (deferral <= reject_theta) — dropping, no escalation",
                    chunk_id,
                )
            else:
                _escalated = True
                _escalation_level = 3
                logger.debug("Chunk %s: L2 escalating to L3", chunk_id)
                result = self._invoke_l3(inp, chunk_id)
                _producer = self._l3_spec
                # L3 is a single voter; record it with voter_index=0,
                # always selected (it's the final word at this chunk).
                self._buffer_voter_outputs(
                    pmcid=pmcid, chunk_id=chunk_id, level="l3",
                    voters_full=[result], voter_specs=[self._l3_spec],
                    voter_timings=None, selected_voter_index=0,
                )

        # Citation filter (B-080)
        # Drop findings whose citation fails structural validation against this
        # chunk's source index — the production check the dormant router-only
        # ProvenanceValidator never ran on the legacy path. Runs on the selected
        # result (any level), before runner._replace_verbatim_from_db, so the
        # optional verbatim check sees the cited sentence, not the DB paragraph.
        if self._citation_cfg.enabled and result is not None and result.findings:
            from nlp_histo.pipeline.stages.knowledge_extraction.provenance.citation_filter import filter_summary_by_citation  # noqa: PLC0415
            n_before = len(result.findings)
            result, dropped = filter_summary_by_citation(
                result, chunk, pmcid,
                check_verbatim=self._citation_cfg.check_verbatim,
                fabricated_threshold=self._citation_cfg.fabricated_threshold,
            )
            if dropped:
                logger.info(
                    "Chunk %s: citation filter dropped %d/%d finding(s) "
                    "(invalid sentence/te_id/pmcid citation).",
                    chunk_id, len(dropped), n_before,
                )

        # RoutingDataset audit row — emit on every router-path chunk (both
        # KEEP and escalate). Mirrors the original _cascade behaviour so
        # downstream calibration tools see a complete record set.
        if (self._router is not None
                and self._routing_collector is not None
                and result is not None
                and l1_outcome.routing_decision is not None):
            decision = l1_outcome.routing_decision
            best_eligible: AuditableSummary | None = None
            best_eligible_idx: int | None = None
            if decision.valid_voter_indices:
                valid = [voters[i] for i in decision.valid_voter_indices]
                best_eligible = self._agreement.best(
                    valid, bundle=decision.agreement_details
                )
                bundle = decision.agreement_details
                if (bundle is not None
                        and bundle.best_index is not None
                        and bundle.best_index < len(decision.valid_voter_indices)):
                    best_eligible_idx = decision.valid_voter_indices[bundle.best_index]
                else:
                    best_eligible_idx = decision.valid_voter_indices[0]
            self._routing_collector.append(
                RoutingRecord(
                    pmcid=pmcid,
                    chunk_id=chunk_id,
                    chunk_text=inp["text"],
                    n_voters=len(voters),
                    n_eligible=len(decision.valid_voter_indices or []),
                    gate_origin=decision.gate_origin.value,
                    decision=decision.decision.value,
                    reason_codes=[r.value for r in decision.reason_codes],
                    escalated=(decision.decision != ChunkDecision.KEEP),
                    used_agreement_gate=(
                        decision.gate_origin.value == "agreement_gate"
                    ),
                    deferral_score=(
                        decision.agreement_details.confidence
                        if decision.agreement_details
                        else None
                    ),
                    agreement_scorer=type(self._agreement._scorer).__name__,
                    output=result.model_dump(),
                    best_eligible_output=(
                        best_eligible.model_dump()
                        if best_eligible is not None
                        else None
                    ),
                    best_eligible_exists=(best_eligible is not None),
                    best_eligible_voter_index=best_eligible_idx,
                )
            )

        # Always record a synthetic L3 "decision" row when L3 ran, so
        # downstream aggregation can count per-level finalisations.
        if _escalation_level == 3 and result is not None:
            self._record_l3_decision(pmcid=pmcid, chunk_id=chunk_id)

        # Build chunk trace
        if collector is not None:
            self._record_chunk_trace(
                collector=collector,
                chunk=chunk,
                chunk_id=chunk_id,
                pmcid=pmcid,
                text=inp["text"],
                voters=voters,
                voter_timings=voter_timings,
                bundle=_trace_bundle,
                escalated=_escalated,
                voter_contexts=_voter_contexts,
                escalation_level=_escalation_level,
            )

        # Escalation counter (thread-safe)
        with self._escalation_lock:
            ec = self._last_escalation_counts
            ec["total"] += 1
            if result is not None:
                ec["finalized"] += 1
                if _escalation_level == 1:
                    ec["l1_kept"] += 1
                elif _escalation_level == 2:
                    ec["l2_escalated"] += 1
                    ec["l2_kept"] += 1
                elif _escalation_level == 3:
                    if self._router is None:
                        # Legacy cascade: went through L2 before L3
                        ec["l2_escalated"] += 1
                    ec["l3_escalated"] += 1
                    ec["l3_kept"] += 1
            else:
                ec["dropped"] += 1

        return result, _producer

    def _invoke_l3(self, inp: dict, chunk_id: str) -> AuditableSummary:
        """Invoke the L3 escalation chain, honouring the in-run voter cache.

        Used by both the router-escalation path and the legacy L2→L3 path so
        the L3 invocation surface stays identical across both branches.
        """
        p_l3, m_l3 = self._l3_spec
        l3_cache_key = self._voter_cache_key(p_l3, m_l3, self._l3_temp, inp)
        cached_l3 = self._voter_cache_get(l3_cache_key)
        if cached_l3 is not None:
            self._voter_cache_bump_hits()
            logger.debug(
                "Chunk %s: L3 — voter-cache HIT (%s/%s @T=%.2f)",
                chunk_id, p_l3, m_l3, self._l3_temp,
            )
            return cached_l3

        cfg_l3 = None
        if getattr(self, "_usage_collector", None) is not None:
            cfg_l3 = self._usage_collector.attach(
                stage="MAP", role="escalator", provider=p_l3, model=m_l3,
                cascade_level=3, substage="l3", item_id=chunk_id,
                was_escalation=True,
            )
        raw_result = (
            self._escalation_chain.invoke(inp, config=cfg_l3)
            if cfg_l3 else self._escalation_chain.invoke(inp)
        )
        from ..costing.invocation_usage import unwrap_structured_output  # noqa: PLC0415
        parsed, raw_msg, parsing_error = unwrap_structured_output(raw_result)
        if parsing_error is not None and parsed is None:
            log_bad_finding(
                raw=getattr(raw_msg, "content", str(raw_msg)) if raw_msg else None,
                error=str(parsing_error),
                context={
                    "stage": "AuditableSummary", "chunk_id": chunk_id,
                    "level": "L3", "provider": p_l3, "model": m_l3,
                },
            )
            # Match pre-include_raw behaviour: raise so the caller's retry /
            # cascade-handling sees a real exception rather than a None result.
            raise parsing_error if isinstance(parsing_error, BaseException) else RuntimeError(str(parsing_error))
        if parsed is not None and parsed.chunk_id != chunk_id:
            log_bad_finding(
                raw={"returned_chunk_id": parsed.chunk_id,
                     "expected_chunk_id": chunk_id},
                error=f"chunk_id mismatch: L3 returned "
                      f"{parsed.chunk_id!r} for expected {chunk_id!r}",
                context={
                    "stage": "AuditableSummary", "chunk_id": chunk_id,
                    "level": "L3", "provider": p_l3, "model": m_l3,
                },
            )
            # Repair in place rather than raising — L3 has no further
            # escalation tier, so dropping the result would lose the only
            # available finding for this chunk.
            parsed = parsed.model_copy(update={"chunk_id": chunk_id})
        self._record_invocation_usage(
            level="L3", provider=p_l3, model=m_l3, chunk_id=chunk_id, raw_msg=raw_msg,
        )
        self._voter_cache_set(l3_cache_key, parsed)
        return parsed

    def _record_cascade_decision(
        self,
        outcome,                                 # ChunkOutcome
        *,
        pmcid: str,
        chunk_id: str,
        level: str,
        voter_count: int,
        voter_specs: list[tuple[str, str]] | None,
    ) -> None:
        """Append one cascade-decision row to the per-paper JSONL log.

        No-op when no log is configured (the runner did not pass one in via
        ``process()``).  Failures during logging are swallowed so a flaky
        filesystem cannot kill an otherwise-successful MAP run.
        """
        # Voter-count regression guard: profile defines N voters; the cascade
        # should always run with exactly N at L1/L2 (L3 is single-voter by
        # design). A shortfall almost always means the router classified one
        # or more voters as UNUSABLE, a provider failed at submit time, or a
        # batch-job grouping bug stripped them silently. Emit a single WARNING
        # line so the next regression surfaces immediately instead of weeks
        # later. See B-019, B-020.
        expected: int | None = None
        if level == "l1":
            expected = len(self._voter_specs)
        elif level == "l2":
            expected = len(self._l2_specs)
        elif level == "l3":
            expected = 1
        if expected is not None and voter_count < expected:
            logger.warning(
                "[%s] cascade chunk %s %s ran with voter_count=%d expected=%d "
                "— %d voter(s) missing (UNUSABLE? failed batch job? router strip?)",
                pmcid, chunk_id, level.upper(),
                voter_count, expected, expected - voter_count,
            )
        log = getattr(self, "_cascade_decision_log", None)
        if log is None:
            return
        try:
            record = make_decision_record(
                outcome,
                run_id=getattr(self, "_cascade_run_id", "unknown"),
                pmcid=pmcid,
                chunk_id=chunk_id,
                level=level,
                voter_count=voter_count,
                cascade_signature=self._cascade_signature,
                cascade_profile=self._cascade_profile,
                voter_specs=voter_specs,
            )
            log.record(record)
        except Exception as exc:  # noqa: BLE001 — log-and-continue is intentional
            logger.warning(
                "Failed to record cascade decision for chunk %s level %s: %s",
                chunk_id, level, exc,
            )

    def _record_l3_decision(self, *, pmcid: str, chunk_id: str) -> None:
        """Append a synthetic L3 row so aggregation can count L3 finalisations.

        L3 is not a decision — it's the terminal escalation — but recording
        a row keeps per-level counts (l1 / l2 / l3) consistent.
        """
        log = getattr(self, "_cascade_decision_log", None)
        if log is None:
            return
        try:
            from ..agreement import CascadeDecisionRecord
            log.record(CascadeDecisionRecord(
                run_id=getattr(self, "_cascade_run_id", "unknown"),
                pmcid=pmcid,
                chunk_id=chunk_id,
                level="l3",
                voter_count=1,
                eligible_voter_count=None,
                decision="keep",
                gate_origin=None,
                reason_codes=[],
                deferral_score=None,
                best_voter_index=None,
                selected_provider=self._l3_spec[0],
                selected_model=self._l3_spec[1],
                cascade_signature=self._cascade_signature,
                cascade_profile=self._cascade_profile,
            ))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to record L3 cascade decision for chunk %s: %s",
                chunk_id, exc,
            )

    def _record_chunk_trace(
        self,
        collector,
        chunk: list[dict],
        chunk_id: str,
        pmcid: str,
        text: str,
        voters: list[AuditableSummary],
        voter_timings: dict[int, float | None],
        bundle,  # ScoreBundle | None
        escalated: bool,
        voter_contexts=None,  # list[VoterContext] | None — from router
        escalation_level: int = 1,
    ) -> None:
        from ..observability.models import (
            AgreementTrace,
            ChunkTrace,
            PairwiseScore,
            VoterTrace,
        )

        # Voter traces — use validated router grounding when available, else fallback.
        voter_traces = []
        grounding_source = "validated" if voter_contexts is not None else "fallback"
        for i, v in enumerate(voters):
            if voter_contexts is not None and i < len(voter_contexts):
                vc = voter_contexts[i]
                pf = vc.grounding_pass_fraction
                me = vc.mean_evidence_length
            else:
                pf, me = _voter_grounding(v)
            voter_traces.append(VoterTrace(
                voter_index=i,
                finding_count=len(v.findings),
                grounding_pass_fraction=round(pf, 4),
                mean_evidence_length=round(me, 4),
                latency_ms=round(voter_timings.get(i), 1) if voter_timings.get(i) is not None else None,
                grounding_source=grounding_source,
            ))

        # Agreement trace — built from ScoreBundle.score_details when available
        agreement_trace = None
        selected_voter_index: int | None = None

        if bundle is not None:
            sd = bundle.score_details or {}
            eligible = sd.get("eligible_voter_indices", list(range(len(voters))))
            avg_sim_raw = sd.get("avg_sim", [])
            # Build PairwiseScore with breakdown fields when present in score_details.
            _BREAKDOWN_KEYS = (
                "claim_count_a", "claim_count_b",
                "coverage_a_to_b", "coverage_b_to_a", "base",
                "count_factor", "reuse_factor",
                "polarity_contradiction_ratio", "numeric_contradiction_ratio",
                "contradiction_ratio", "contradiction_factor",
                "pre_grounding_score", "grounding_factor",
            )
            pairwise = [
                PairwiseScore(
                    voter_i=p["voter_i"],
                    voter_j=p["voter_j"],
                    score=round(p["score"], 6),
                    **{k: p[k] for k in _BREAKDOWN_KEYS if k in p},
                )
                for p in sd.get("pairwise_upper", [])
            ]

            score = bundle.confidence if bundle.confidence is not None else (
                bundle.embedding_agreement or 0.0
            )
            decision_val = bundle.decision.value if bundle.decision else "unknown"

            # Human-readable reason
            theta = self._agreement.theta
            reject_theta = self._agreement.reject_theta
            if not escalated:
                reason = (
                    f"Deferral score {score:.3f} ≥ theta {theta:.2f}; "
                    f"voter {bundle.best_index} selected"
                )
                selected_voter_index = bundle.best_index
            elif score <= reject_theta and score > 0.0:
                reason = (
                    f"Deferral score {score:.3f} ≤ reject_theta {reject_theta:.2f} — hard reject"
                )
            elif len(eligible) < 2:
                reason = (
                    f"Only {len(eligible)} non-empty voter(s) eligible — too few for agreement"
                )
            else:
                reason = (
                    f"Deferral score {score:.3f} in ({reject_theta:.2f}, {theta:.2f}) — escalated"
                )

            agreement_trace = AgreementTrace(
                eligible_voter_indices=eligible,
                avg_sim=[round(s, 4) for s in avg_sim_raw],
                pairwise_scores=pairwise,
                deferral_score=round(score, 4),
                theta=theta,
                reject_theta=reject_theta,
                decision=decision_val,
                reason=reason,
                selected_voter_index=selected_voter_index,
            )

        te_ids = sorted({item.get("text_element_id", 0) for item in chunk})
        collector.add_chunk_trace(ChunkTrace(
            chunk_id=chunk_id,
            run_id=collector.run_id,
            pmcid=pmcid,
            te_ids=te_ids,
            sentence_count=len(chunk),
            text_preview=text[:200],
            cache_hit=False,
            voters=voter_traces,
            agreement=agreement_trace,
            selected_voter_index=selected_voter_index,
            escalated=escalated,
            escalation_level=escalation_level,
        ))

    # Invocation-time usage capture (sole source of truth for cost)

    def _record_invocation_usage(
        self,
        *,
        level: str,
        provider: str,
        model: str,
        chunk_id: str | None,
        raw_msg,
    ) -> None:
        """Append one InvocationUsage record for a freshly-completed call.

        ``raw_msg`` is the AIMessage from ``build_map_chain``'s
        ``include_raw=True`` output (``result["raw"]``).  If ``raw_msg`` is
        ``None`` (e.g. structured-output dict had no raw, or the chain
        wasn't include_raw for any reason) the record is still appended
        with ``usage_source="missing"`` and a warning is logged — silent
        zeros corrupt cost reports far more than a noisy log line.
        """
        from ..costing.invocation_usage import extract_usage_from_message  # noqa: PLC0415
        if raw_msg is None:
            inp_tok, out_tok, src = 0, 0, "missing"
        else:
            inp_tok, out_tok, src = extract_usage_from_message(raw_msg)
        if src == "missing":
            logger.warning(
                "MAP invocation usage missing: level=%s provider=%s model=%s chunk=%s "
                "(raw_msg=%s) — cost report will under-count this call",
                level, provider, model, chunk_id,
                type(raw_msg).__name__ if raw_msg is not None else "None",
            )
        rec = self._InvocationUsage(
            level=level, provider=provider, model=model, chunk_id=chunk_id,
            input_tokens=inp_tok, output_tokens=out_tok, usage_source=src,
        )
        with self._invocation_usage_lock:
            self._invocation_usage_records.append(rec)

    def invocation_usage_records(self) -> list:
        """Return a snapshot of the current paper's invocation-usage records.

        Empty until ``process()`` is called.  Reset at the start of every
        ``process()`` invocation, so the caller should read this between
        papers (it carries the MAP usage for the most recent paper only).
        """
        with self._invocation_usage_lock:
            return list(self._invocation_usage_records)

    def _buffer_voter_outputs(
        self,
        *,
        pmcid: str,
        chunk_id: str,
        level: str,                                     # "l1" | "l2" | "l3"
        voters_full: list,                               # list[AuditableSummary | None]
        voter_specs: list[tuple[str, str]],
        voter_timings: dict[int, float | None] | None,
        selected_voter_index: int | None,
    ) -> None:
        """Buffer one row per voter for the (chunk, level) tuple.

        ``voters_full`` aligns positionally with ``voter_specs`` — both are
        length-N lists, ``None`` slots indicate voters whose API call
        failed (the run continues with the survivors, but the row is still
        emitted with ``failed=True`` and ``raw_output=None``).

        ``selected_voter_index`` is the global voter index whose
        AuditableSummary became the level's result, or None when the level
        escalated (no winner). Exactly one row per (chunk, level) has
        ``is_selected=True`` when a non-None value is passed.
        """
        from ..models import AuditableSummary  # noqa: PLC0415
        rows: list[dict] = []
        for idx, summary in enumerate(voters_full):
            provider, model = (
                voter_specs[idx] if idx < len(voter_specs) else ("unknown", "unknown")
            )
            latency = None
            if voter_timings is not None:
                latency = voter_timings.get(idx)
            failed = summary is None
            raw_output = None
            finding_count = 0
            if isinstance(summary, AuditableSummary):
                raw_output = summary.model_dump()
                finding_count = len(summary.findings or [])
            rows.append({
                "pmcid":          pmcid,
                "chunk_id":       chunk_id,
                "level":          level,
                "voter_index":    idx,
                "provider":       provider,
                "model":          model,
                "is_selected":    (selected_voter_index == idx),
                "failed":         failed,
                "error_message":  None,  # API errors are logged elsewhere; not threaded here yet
                "finding_count":  finding_count,
                "latency_ms":     latency,
                "raw_output":     raw_output,
            })
        with self._voter_output_lock:
            self._voter_output_records.extend(rows)

    def voter_output_records(self) -> list[dict]:
        """Return a snapshot of per-voter outputs collected during the
        current paper's MAP run.

        Empty until ``process()`` is called. Reset at the start of every
        ``process()`` invocation. The runner consumes this to persist
        ``sum_map_voter_outputs`` rows after the MAP stage finishes.
        """
        with self._voter_output_lock:
            return list(self._voter_output_records)

    @staticmethod
    def _voter_cache_key(provider: str, model: str, temperature: float, inp: dict) -> tuple:
        """Cache key for one voter call.

        Includes the chunk text hash so different chunks never collide and the
        cache works correctly even across the parallel chunk pool. Rounded
        temperature avoids float-identity false-misses.
        """
        text = inp.get("text", "") or ""
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
        return (provider, model, round(float(temperature), 3), text_hash)

    def _voter_cache_get(self, key: tuple) -> AuditableSummary | None:
        lock = getattr(self, "_voter_cache_lock", None)
        if lock is None:
            return None  # cache not initialised (called outside process())
        with lock:
            return self._voter_cache.get(key)

    def _voter_cache_set(self, key: tuple, value: AuditableSummary) -> None:
        lock = getattr(self, "_voter_cache_lock", None)
        if lock is None:
            return
        with lock:
            self._voter_cache[key] = value

    def _voter_cache_bump_hits(self) -> None:
        lock = getattr(self, "_voter_cache_lock", None)
        if lock is None:
            return
        with lock:
            self._voter_cache_hits += 1

    def _run_voters(
        self, inp: dict, chains: list | None = None, level: str = "L1"
    ) -> tuple[list[AuditableSummary | None], dict[int, float | None]]:
        """Invoke each voter chain concurrently; return per-voter results + latency.

        Returns a length-N list aligned to the input ``chains`` (original voter
        index): a successful call lands at its slot, a failed call leaves ``None``.
        Preserving the original-index mapping is load-bearing for downstream
        producer attribution — see B-041. The caller filters Nones when handing
        the list to agreement / router, but tracks the survivor → original
        mapping alongside.

        Parameters
        ----------
        chains:
            Voter chains to run.  Defaults to self._voter_chains (Level 1).
            Pass self._level2_voter_chains to run Level 2 voters.
        level:
            Label used in log messages, e.g. "L1", "L2".
        """
        target = chains if chains is not None else self._voter_chains
        chunk_id = inp.get("chunk_id", "?")
        pmcid = inp.get("pmcid", "?")
        results: list[AuditableSummary | None] = [None] * len(target)
        timings: dict[int, float | None] = {}

        # Match the chain list against our recorded specs so we can attribute
        # usage to (provider, model) per voter. _l3 only fires from _cascade.
        if chains is self._level2_voter_chains:
            specs = self._l2_specs
            temps = self._l2_temps
            cascade_level = 2
        else:
            specs = self._voter_specs
            temps = self._l1_temps
            cascade_level = 1
        collector = getattr(self, "_usage_collector", None)

        def _timed_invoke(chain, i: int):
            provider, model = (
                specs[i] if i < len(specs) else ("unknown", "unknown")
            )
            voter_temp = temps[i] if i < len(temps) else 0.0
            cache_key = self._voter_cache_key(provider, model, voter_temp, inp)
            cached = self._voter_cache_get(cache_key)
            if cached is not None:
                self._voter_cache_bump_hits()
                timings[i] = 0.0
                logger.debug(
                    "[%s] MAP chunk %s %s voter %d — voter-cache HIT (%s/%s @T=%.2f)",
                    pmcid, chunk_id, level, i, provider, model, voter_temp,
                )
                return cached

            attempt = 0
            t0 = time.monotonic()
            last_exc: BaseException | None = None
            while attempt < 2:
                attempt += 1
                if attempt > 1:
                    logger.warning(
                        "[%s] MAP chunk %s %s voter %d — retry attempt %d/2",
                        pmcid, chunk_id, level, i, attempt,
                    )
                try:
                    cfg = None
                    if collector is not None:
                        cfg = collector.attach(
                            stage="MAP", role="voter", provider=provider, model=model,
                            cascade_level=cascade_level, substage=level.lower(),
                            item_id=chunk_id,
                        )
                    raw_out = chain.invoke(inp, config=cfg) if cfg else chain.invoke(inp)
                    elapsed_ms = (time.monotonic() - t0) * 1000.0
                    from ..costing.invocation_usage import unwrap_structured_output  # noqa: PLC0415
                    parsed_out, raw_msg, parsing_error = unwrap_structured_output(raw_out)
                    if parsing_error is not None and parsed_out is None:
                        # Log structurally before raising so the bad payload
                        # survives in bad_findings.jsonl even if the retry
                        # succeeds (gives us frequency-per-voter data).
                        log_bad_finding(
                            raw=getattr(raw_msg, "content", str(raw_msg)) if raw_msg else None,
                            error=str(parsing_error),
                            context={
                                "stage": "AuditableSummary",
                                "pmcid": pmcid, "chunk_id": chunk_id, "level": level,
                                "voter_index": i, "provider": provider, "model": model,
                                "attempt": attempt,
                            },
                        )
                        # Surface the parsing error as an exception so the
                        # retry / failure path below handles it the same way
                        # it did before include_raw=True was introduced.
                        raise parsing_error if isinstance(parsing_error, BaseException) else RuntimeError(str(parsing_error))
                    # Chunk_id round-trip — fail-fast if the voter renamed the
                    # chunk. Triggers the existing 2-attempt retry; if retry
                    # also mismatches the voter is excluded from agreement.
                    if parsed_out is not None and parsed_out.chunk_id != chunk_id:
                        log_bad_finding(
                            raw={"returned_chunk_id": parsed_out.chunk_id,
                                 "expected_chunk_id": chunk_id},
                            error=f"chunk_id mismatch: voter returned "
                                  f"{parsed_out.chunk_id!r} for expected {chunk_id!r}",
                            context={
                                "stage": "AuditableSummary",
                                "pmcid": pmcid, "chunk_id": chunk_id, "level": level,
                                "voter_index": i, "provider": provider, "model": model,
                                "attempt": attempt,
                            },
                        )
                        raise ValueError(
                            f"chunk_id mismatch: voter returned "
                            f"{parsed_out.chunk_id!r}, expected {chunk_id!r}"
                        )
                    self._record_invocation_usage(
                        level=level, provider=provider, model=model,
                        chunk_id=chunk_id, raw_msg=raw_msg,
                    )
                    logger.debug(
                        "[%s] MAP chunk %s %s voter %d — OK  [%.0fms]",
                        pmcid, chunk_id, level, i, elapsed_ms,
                    )
                    timings[i] = elapsed_ms
                    self._voter_cache_set(cache_key, parsed_out)
                    return parsed_out
                except Exception as exc:
                    last_exc = exc
                    if _LengthFinishReasonError is not None and isinstance(exc, _LengthFinishReasonError):
                        logger.warning(
                            "[%s] MAP chunk %s %s voter %d — output truncated "
                            "(finish_reason=length); excluding voter",
                            pmcid, chunk_id, level, i,
                        )
                        break  # truncation is not fixed by retrying
                    non_retryable = is_non_retryable_llm_error(exc)
                    logger.warning(
                        "[%s] MAP chunk %s %s voter %d — attempt %d/%d failed "
                        "(%s): %s: %s",
                        pmcid, chunk_id, level, i, attempt, 2,
                        "non-retryable" if non_retryable else "retryable",
                        type(exc).__name__, exc,
                    )
                    if non_retryable:
                        break
            timings[i] = None
            raise last_exc  # type: ignore[misc]

        pool = ThreadPoolExecutor(max_workers=len(target))
        future_to_idx = {
            pool.submit(_timed_invoke, chain, i): i
            for i, chain in enumerate(target)
        }
        try:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.warning(
                        "[%s] MAP chunk %s %s voter %d — excluded from agreement: %s",
                        pmcid, chunk_id, level, idx, type(exc).__name__,
                    )
                    timings[idx] = None
        except KeyboardInterrupt:
            for f in future_to_idx:
                f.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)

        succeeded = sum(1 for r in results if r is not None)
        failed = len(target) - succeeded
        if failed:
            logger.warning(
                "[%s] MAP chunk %s %s — %d/%d voter(s) failed",
                pmcid, chunk_id, level, failed, len(target),
            )
        if not succeeded:
            logger.error(
                "[%s] MAP chunk %s %s — ALL voters failed; chunk will be escalated",
                pmcid, chunk_id, level,
            )

        return results, timings

    @property
    def last_escalation_counts(self) -> dict[str, int]:
        """Return escalation counters from the most recent process() call."""
        with self._escalation_lock:
            return dict(self._last_escalation_counts)

    # Helpers

    def _make_chunks(self, sentences: list[dict]) -> list[list[dict]]:
        """Split sentences into overlapping fixed-size chunks.

        With chunk_overlap=0 (default) behaviour is identical to the original
        disjoint split.  With overlap > 0, adjacent chunks share that many
        sentences so findings that span a boundary are seen in full context by
        at least one chunk.
        """
        if not sentences:
            return []
        stride = self.chunk_size - self.chunk_overlap
        chunks = [
            sentences[i : i + self.chunk_size]
            for i in range(0, len(sentences), stride)
        ]
        if self.chunk_overlap > 0:
            logger.debug(
                "_make_chunks: %d sentences → %d chunks "
                "(chunk_size=%d, overlap=%d, stride=%d)",
                len(sentences), len(chunks),
                self.chunk_size, self.chunk_overlap, stride,
            )
        return chunks
