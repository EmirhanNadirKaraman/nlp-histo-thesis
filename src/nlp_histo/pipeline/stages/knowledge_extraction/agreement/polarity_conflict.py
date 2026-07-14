"""Polarity-conflict hard-fail detector for the MAP agreement gate (B-051).

Purpose
-------
The agreement scorer's `_polarity` heuristic applies a 20% multiplicative
penalty when voter claims appear to contradict. That is too soft: two
paraphrases with opposite directions and embedding similarity ≈ 1.0 still
produce a final score of 0.80, which passes `theta=0.7` and sits exactly on
the boundary at `theta=0.8`. The cascade can therefore KEEP a chunk where
voters directly contradict each other.

This module detects the hard-contradiction case structurally — by inspecting
`Finding.direction` directly rather than by token-spotting on `claim` text —
and returns a non-`None` payload that `AgreementChecker.compute` uses to
force `ChunkDecision.ESCALATE` regardless of the embedding score.

Policy v1 (intentionally conservative)
--------------------------------------
Hard-fail iff **two comparable findings** carry directions
`{positive, negative}` (exact pair, in any order).

**Comparability** = strict equality on four structural fields after
``.strip().casefold()`` on the two strings:
    subject_entity, outcome_entity, relation_type, category
Any ``None`` on either side disqualifies the pair (we cannot compare what
we cannot identify).

**Polarity values intentionally EXCLUDED from v1**:
    - ``absent``  — semantically overlaps with ``negative`` in
      biomarker-expression contexts ("marker absent" ≈ "marker not present").
      A clean rule needs calibration evidence — tracked by B-025.
    - ``partial`` — bundled with ``positive`` in
      ``relate_stage._POSITIVE_DIRECTIONS``. Making MAP stricter than RELATE
      on this would be inconsistent.
    - ``unclear`` / ``no_direction`` — no polarity signal to contradict.

**Scope fields intentionally EXCLUDED from v1 comparability**:
    ``FindingScope.{disease_subtype, tissue_site, treatment_context, ...}``
    are not part of the comparability key. Same biomarker / different
    cohort / opposite direction will therefore hard-fail today — a
    conservative *false-escalation* trade-off: an extra L3 call is cheap;
    a missed contradiction silently kept as L1 consensus is expensive.
    Scope-aware comparability is v2.

Return shape
------------
- ``None`` when no conflict found.
- ``dict`` with keys:
    ``voter_pairs``      list of per-pair conflict records
    ``count``            int, number of conflicting pairs
    ``policy_version``   str, ``MAP_AGREEMENT_POLICY_VERSION`` for audit

Each per-pair record has:
    ``voter_i``, ``voter_j``        original voter indices
    ``finding_i``, ``finding_j``    index within each voter's findings list
    ``direction_i``, ``direction_j`` string values for inspection
    ``comparability_key``           4-tuple (subject, outcome, relation, category)

Future v2 candidates
--------------------
- Evidence-disjoint hard-fail: gate via
  ``agreement/hybrid_structured.py:_evidence_jaccard`` once a non-trivial
  proximity policy exists. Today, two correct paraphrases citing different
  but valid sentences would false-escalate.
- Scope-aware comparability per category (prognostic → disease_subtype;
  IHC → tissue_site; treatment_response → treatment_context).
- Broaden polarity set to include ``absent`` and/or ``partial`` after
  B-025 calibration.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from nlp_histo.pipeline.stages.knowledge_extraction.models import (
    AuditableSummary,
    DirectionEnum,
    Finding,
    MAP_AGREEMENT_POLICY_VERSION,
)


# Exact polarity pair that hard-fails. Frozenset so order does not matter.
_HARD_POLARITY_PAIR: frozenset[DirectionEnum] = frozenset({
    DirectionEnum.positive,
    DirectionEnum.negative,
})


def _comparability_key(f: Finding) -> tuple[str, str, Any, Any] | None:
    """Return the 4-field comparability key, or None if any field is unusable.

    Mirrors ``group_stage._group_id`` semantics — the same definition of
    "this is the same claim" the codebase already uses post-B-022. Strings
    are normalised via ``.strip().casefold()``; enums compared by identity.
    """
    if f.subject_entity is None or f.outcome_entity is None:
        return None
    subj = f.subject_entity.strip().casefold()
    out = f.outcome_entity.strip().casefold()
    if not subj or not out:
        return None
    return (subj, out, f.relation_type, f.category)


def _stringify(x: Any) -> str | None:
    """JSON-safe stringifier for trace-payload values (any Enum → its value)."""
    if x is None:
        return None
    if isinstance(x, Enum):
        return x.value
    return str(x)


def detect_polarity_conflict(
    outputs: list[AuditableSummary],
) -> dict | None:
    """Return conflict-detail dict if any comparable voter-pair contradicts polarity.

    Pure function. Iterates voter pairs ``(i < j)`` × finding pairs. Records
    every conflicting pair (not just the first) so the trace can surface the
    full picture.
    """
    if not outputs or len(outputs) < 2:
        return None

    conflicts: list[dict[str, Any]] = []
    for i in range(len(outputs)):
        findings_i = outputs[i].findings or []
        for j in range(i + 1, len(outputs)):
            findings_j = outputs[j].findings or []
            for fi_idx, f_a in enumerate(findings_i):
                key_a = _comparability_key(f_a)
                if key_a is None:
                    continue
                if f_a.direction not in _HARD_POLARITY_PAIR:
                    continue
                for fj_idx, f_b in enumerate(findings_j):
                    if f_b.direction not in _HARD_POLARITY_PAIR:
                        continue
                    if f_a.direction == f_b.direction:
                        continue  # same polarity is not a conflict
                    key_b = _comparability_key(f_b)
                    if key_b != key_a:
                        continue
                    conflicts.append({
                        "voter_i": i,
                        "voter_j": j,
                        "finding_i": fi_idx,
                        "finding_j": fj_idx,
                        "direction_i": _stringify(f_a.direction),
                        "direction_j": _stringify(f_b.direction),
                        "comparability_key": [_stringify(v) for v in key_a],
                    })

    if not conflicts:
        return None
    return {
        "voter_pairs": conflicts,
        "count": len(conflicts),
        "policy_version": MAP_AGREEMENT_POLICY_VERSION,
    }
