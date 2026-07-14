"""Shared per-chunk cascade-decision function and decision log.

Used by both the sync ``MapStage`` and the batch ``BatchKnowledgeExtractionRunner``
so the cascade-decision surface is identical by construction. The previous
state had each runner re-implementing the decision branch (router vs. bare
agreement, KEEP/escalate) inline, which was a recurring source of drift
between sync and batch.

The decision is intentionally per-level:
  - sync runs it for L1 (and again for L2 in the legacy path)
  - batch runs it for L1 (and again for L2 in the legacy path)

The orchestration around it (which voters to call next, when to escalate to
L3, how to chain the batch phases) stays in each runner — only the
``given these voters, do we KEEP or escalate?`` decision is shared.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from nlp_histo.pipeline.stages.knowledge_extraction.interfaces.scoring import (
    ChunkDecision,
    ScoreBundle,
)
from nlp_histo.pipeline.stages.knowledge_extraction.models import AuditableSummary

logger = logging.getLogger(__name__)


@dataclass
class ChunkOutcome:
    """Result of one cascade-level evaluation for a single chunk."""

    keep: bool
    best: AuditableSummary | None
    agreement_bundle: ScoreBundle | None
    routing_decision: object | None = None  # RoutingDecision | None — avoid circular import
    valid_voter_indices: list[int] | None = None
    rejected: bool = False
    """True when this level's decision was ChunkDecision.REJECT (deferral score
    <= reject_theta). Distinguishes a *reject* from a plain *escalate* among the
    ``keep=False`` outcomes so the cascade can drop the chunk instead of
    escalating it (see ``MapConfig.reject_theta`` / the drop-on-reject policy).
    Always False on the empty-voter escalate path."""


def evaluate_chunk(
    voters: list[AuditableSummary],
    chunk: list[dict],
    pmcid: str,
    source_text: str,
    agreement,                          # AgreementChecker
    router=None,                        # MapOutputRouter | None
    voter_indices: list[int] | None = None,
) -> ChunkOutcome:
    """Decide whether one cascade level's voters agree, and select the best.

    ``voters`` is the survivor list (no None slots). ``voter_indices`` maps
    each survivor back to its original voter-spec index — `voter_indices[i]`
    is the spec index of `voters[i]`. When None, identity mapping is assumed
    (caller passed all voters in original order with no drops).

    ``ChunkOutcome.valid_voter_indices`` is always populated in
    ORIGINAL-index space (router maps its internal survivor-indices back
    through ``voter_indices`` before returning). This makes
    ``ChunkOutcome.valid_voter_indices`` the single source of truth for
    downstream producer attribution — see B-041.
    """
    if not voters:
        return ChunkOutcome(
            keep=False,
            best=None,
            agreement_bundle=None,
            routing_decision=None,
        )

    if voter_indices is None:
        voter_indices = list(range(len(voters)))
    elif len(voter_indices) != len(voters):
        raise ValueError(
            f"voter_indices length {len(voter_indices)} != voters length {len(voters)}"
        )

    if router is not None:
        decision = router.route(
            voters,
            chunk=chunk,
            pmcid=pmcid,
            source_text=source_text,
        )
        # decision.valid_voter_indices are SURVIVOR-list indices from the router's
        # perspective; map back to original-spec indices via voter_indices.
        router_survivor_indices = decision.valid_voter_indices or list(range(len(voters)))
        mapped_indices = [voter_indices[i] for i in router_survivor_indices]
        if decision.decision == ChunkDecision.KEEP:
            valid = [voters[i] for i in router_survivor_indices]
            best = agreement.best(valid, bundle=decision.agreement_details)
            return ChunkOutcome(
                keep=True,
                best=best,
                agreement_bundle=decision.agreement_details,
                routing_decision=decision,
                valid_voter_indices=mapped_indices,
            )
        return ChunkOutcome(
            keep=False,
            best=None,
            agreement_bundle=decision.agreement_details,
            routing_decision=decision,
            valid_voter_indices=mapped_indices,
            rejected=(decision.decision == ChunkDecision.REJECT),
        )

    bundle = agreement.compute(voters, source_text=source_text)
    if bundle.decision == ChunkDecision.KEEP:
        best = agreement.best(voters, bundle=bundle)
        return ChunkOutcome(
            keep=True,
            best=best,
            agreement_bundle=bundle,
            routing_decision=None,
            valid_voter_indices=list(voter_indices),
        )
    return ChunkOutcome(
        keep=False,
        best=None,
        agreement_bundle=bundle,
        routing_decision=None,
        valid_voter_indices=list(voter_indices),
        rejected=(bundle.decision == ChunkDecision.REJECT),
    )


# ── Per-chunk decision log ─────────────────────────────────────────────────


@dataclass
class CascadeDecisionRecord:
    """One row per cascade-level decision for offline analysis.

    Always emitted, regardless of trace_enabled, so cascade behaviour is
    inspectable even when the heavyweight TraceCollector is off.
    """

    run_id: str
    pmcid: str
    chunk_id: str
    level: str                              # "l1" | "l2" | "l3"
    voter_count: int                        # voters that survived API/parse
    eligible_voter_count: int | None        # post-router; None when router off
    decision: str                           # "keep" | "escalate" | "reject"
    gate_origin: str | None                 # "schema_gate" | "grounding_gate" | "agreement_gate" | None
    reason_codes: list[str] = field(default_factory=list)
    deferral_score: float | None = None
    best_voter_index: int | None = None     # index into the original voter list
    selected_provider: str | None = None
    selected_model: str | None = None
    cascade_signature: str | None = None
    cascade_profile: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class CascadeDecisionLog:
    """Append-only JSONL writer for cascade decisions. Thread-safe.

    Writes one line per ``record()`` call to ``output_path``. When
    ``output_path`` is None, all calls are no-ops — useful for tests and
    for explicitly disabling logging without code branches at call sites.
    """

    def __init__(self, output_path: Path | None) -> None:
        self._path: Path | None = (
            Path(output_path) if output_path is not None else None
        )
        self._lock = threading.Lock()
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path | None:
        return self._path

    def record(self, record: CascadeDecisionRecord) -> None:
        if self._path is None:
            return
        line = record.to_jsonl()
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def producer_from_outcome(
    outcome: ChunkOutcome,
    voter_specs: list[tuple[str, str]] | None,
) -> tuple[str, str] | None:
    """Resolve the (provider, model) of the voter selected by ``outcome``.

    Uses ``outcome.valid_voter_indices`` as the single source of truth — these
    are always ORIGINAL voter-spec indices (router maps its survivor-indices
    back inside ``evaluate_chunk``). ``bundle.best_index`` indexes into
    ``valid_voter_indices``; one lookup gives the original spec index.
    """
    if voter_specs is None:
        return None
    bundle = outcome.agreement_bundle
    best_idx = bundle.best_index if bundle is not None else None
    if best_idx is None:
        return None
    vvi = outcome.valid_voter_indices
    if vvi is None:
        # Legacy callers that didn't supply voter_indices to evaluate_chunk —
        # best_idx is then already in original-spec space.
        if 0 <= best_idx < len(voter_specs):
            return voter_specs[best_idx]
        return None
    if 0 <= best_idx < len(vvi):
        global_idx = vvi[best_idx]
        if 0 <= global_idx < len(voter_specs):
            return voter_specs[global_idx]
    return None


def make_decision_record(
    outcome: ChunkOutcome,
    *,
    run_id: str,
    pmcid: str,
    chunk_id: str,
    level: str,
    voter_count: int,
    cascade_signature: str | None,
    cascade_profile: str | None,
    voter_specs: list[tuple[str, str]] | None,
) -> CascadeDecisionRecord:
    """Build a ``CascadeDecisionRecord`` from a ``ChunkOutcome`` and context.

    Handles the router/no-router cases: when a routing decision is present
    the gate_origin and reason_codes come from it; otherwise they default
    to ``agreement_gate`` and the bundle's threshold path.
    """
    bundle = outcome.agreement_bundle
    rd = outcome.routing_decision

    if outcome.keep:
        decision_str = "keep"
    elif rd is not None and rd.decision == ChunkDecision.REJECT:
        decision_str = "reject"
    else:
        decision_str = "escalate"

    gate_origin = rd.gate_origin.value if rd is not None else "agreement_gate"
    reason_codes: list[str] = (
        [r.value for r in rd.reason_codes] if rd is not None else []
    )

    deferral_score: float | None = None
    if bundle is not None:
        if bundle.confidence is not None:
            deferral_score = float(bundle.confidence)
        elif bundle.embedding_agreement is not None:
            deferral_score = float(bundle.embedding_agreement)

    best_idx = bundle.best_index if bundle is not None else None
    selected_provider: str | None = None
    selected_model: str | None = None
    # Original-spec index that bundle.best_index resolves to. Used both for the
    # producer/model lookup and for the best_voter_index record field (which is
    # documented as "index into the original voter list").
    best_voter_index_original: int | None = None
    vvi = outcome.valid_voter_indices
    if best_idx is not None:
        if vvi is not None and 0 <= best_idx < len(vvi):
            best_voter_index_original = vvi[best_idx]
        elif vvi is None:
            # Legacy path: caller didn't supply voter_indices; best_idx is
            # already original-spec.
            best_voter_index_original = best_idx
    if outcome.keep and best_voter_index_original is not None and voter_specs is not None:
        if 0 <= best_voter_index_original < len(voter_specs):
            selected_provider, selected_model = voter_specs[best_voter_index_original]

    eligible_count: int | None = None
    if rd is not None and rd.valid_voter_indices is not None:
        eligible_count = len(rd.valid_voter_indices)

    return CascadeDecisionRecord(
        run_id=run_id,
        pmcid=pmcid,
        chunk_id=chunk_id,
        level=level,
        voter_count=voter_count,
        eligible_voter_count=eligible_count,
        decision=decision_str,
        gate_origin=gate_origin,
        reason_codes=reason_codes,
        deferral_score=deferral_score,
        best_voter_index=best_voter_index_original,
        selected_provider=selected_provider,
        selected_model=selected_model,
        cascade_signature=cascade_signature,
        cascade_profile=cascade_profile,
    )
