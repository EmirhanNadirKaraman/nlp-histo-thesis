"""
CANONICALIZE stage: FindingGroup[] → CanonicalRule[]

For each FindingGroup:
  1. Pick a canonical predicate text — selects the finding with the highest
     mean_grounding_score (deterministic, no LLM call).
  2. Compute two scope fields: is_conflicted (bool) and study_coverage
     (single_study | multi_study | unknown) from direction counts and PMCID coverage.
  3. Groups with mixed directions (e.g. both "positive" and "negative") are
     split into separate CanonicalRules, one per majority direction.

No cross-paper merging is done here — that requires a multi-paper pool fed
into GROUP before CANONICALIZE runs.  With per-paper runs the groups will
typically have size=1; the stage still produces valid output in that case.
"""
from __future__ import annotations

import hashlib
import logging

from ..models import (
    CanonicalRule,
    DirectionEnum,
    FindingGroup,
    NormalFinding,
    POLARITY_BEARING_DIRS,
    direction_value,
)

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _canonical_rule_id(group_id: str, direction: str) -> str:
    return f"CR_{_sha8(group_id)}_{direction}"


def _study_coverage(member_nfs: list[NormalFinding]) -> str:
    """Return ``"single_study" | "multi_study" | "unknown"`` for a direction bin."""
    all_pmcids: set[str] = set()
    for nf in member_nfs:
        all_pmcids.update(nf.pmcids)
    if len(all_pmcids) >= 2:
        return "multi_study"
    if len(all_pmcids) == 1:
        return "single_study"
    return "unknown"


def _pick_best_predicate_deterministic(
    candidates: list[tuple[float, str]]
) -> str:
    """
    Fallback: return the predicate_text with the highest grounding score.
    candidates: list of (mean_grounding_score, predicate_text), possibly with None scores.
    """
    return max(candidates, key=lambda x: x[0] or 0.0)[1]


def _split_by_direction(
    member_nfs: list[NormalFinding],
) -> list[tuple[str, list[NormalFinding]]]:
    """Split member NormalFindings by direction — one bin per observed direction.

    B-049 redesign: no folding. Every direction observed in the group gets its
    own bin, including ``unclear`` and ``no_direction``. RELATE / corpus_relate
    skip pairs where either side's direction is non-polarity (`NON_POLARITY_DIRS`),
    so emitting these bins as full CanonicalRules is lossless but inert in the
    relation graph. Bin order is sorted for determinism — supersedes B-026.
    """
    bins: dict[str, list[NormalFinding]] = {}
    for nf in member_nfs:
        d = direction_value(nf.direction)  # handles None → "unclear"
        bins.setdefault(d, []).append(nf)
    return sorted(bins.items())


# ── Public API ─────────────────────────────────────────────────────────────────

class CanonicalizeStage:
    """CANONICALIZE stage: FindingGroup[] → CanonicalRule[]."""

    def __init__(self) -> None:
        pass

    def canonicalize(
        self,
        groups: list[FindingGroup],
        normal_findings_by_id: dict[str, NormalFinding],
        pmcid: str,
    ) -> list[CanonicalRule]:
        """
        Produce CanonicalRule objects from FindingGroups.

        Parameters
        ----------
        groups:
            Output of GroupStage.group().
        normal_findings_by_id:
            Dict mapping NormalFinding.normal_id → NormalFinding for all
            findings that were passed into GROUP.  Used to retrieve predicate
            texts and PMCID lists.
        pmcid:
            For logging.

        Returns
        -------
        List of CanonicalRule, possibly more than len(groups) when mixed
        directions cause a group to be split.
        """
        if not groups:
            return []

        canonical_rules: list[CanonicalRule] = []

        for group in groups:
            # Resolve member NormalFindings
            member_nfs: list[NormalFinding] = [
                normal_findings_by_id[mid]
                for mid in group.member_ids
                if mid in normal_findings_by_id
            ]
            if not member_nfs:
                logger.warning(
                    "[%s] CANONICALIZE: group %s has no resolvable members — skipping",
                    pmcid, group.group_id,
                )
                continue

            # Split by direction (B-049: one bin per observed direction, no folding)
            direction_bins = _split_by_direction(member_nfs)

            # Group-level is_conflicted (B-049): True iff this group emitted
            # ≥2 polarity-bearing direction bins. Computed once per group and
            # stamped on every CanonicalRule the group produces — this is NOT
            # a within-rule property (within a bin the direction is uniform
            # by construction). `partial` counts as polarity-bearing here per
            # the current `POLARITY_BEARING_DIRS`; semantic question on
            # `partial` is owned by B-025.
            polarity_count = sum(1 for (d, _) in direction_bins if d in POLARITY_BEARING_DIRS)
            group_is_conflicted = polarity_count >= 2

            for direction, bin_nfs in direction_bins:
                if not bin_nfs:
                    continue

                # Build candidate list (score, text) sorted by score desc
                candidates: list[tuple[float, str]] = sorted(
                    [(nf.mean_grounding_score or 0.0, nf.predicate_text) for nf in bin_nfs],
                    key=lambda x: x[0],
                    reverse=True,
                )

                predicate_text = self._select_predicate(
                    candidates, group, direction, pmcid
                )

                study_coverage = _study_coverage(bin_nfs)

                # Aggregate evidence and PMCID lists
                all_pmcids: list[str] = sorted(
                    {p for nf in bin_nfs for p in nf.pmcids}
                )
                all_member_ids = [nf.normal_id for nf in bin_nfs]

                # mean_grounding_score across the bin
                scores = [nf.mean_grounding_score for nf in bin_nfs if nf.mean_grounding_score is not None]
                mean_score = sum(scores) / len(scores) if scores else None

                # Representative scope + verbatim + paragraph pointer: all
                # taken from the highest-grounded NormalFinding in the bin so
                # they stay paired with the text the predicate_text actually
                # came from.  RelateConfig.{scope_aware_nli,
                # use_verbatim_for_nli} consume these downstream; the
                # representative_text_element_id is a hint for
                # provenance.paragraph_lookup.get_paragraph_for_rule (DB-backed,
                # paragraph itself is too large to bake into JSON).
                best_nf = max(
                    bin_nfs,
                    key=lambda n: (n.mean_grounding_score or 0.0),
                )
                representative_scope = best_nf.scope
                best_span = best_nf.evidence[0] if best_nf.evidence else None
                representative_verbatim = best_span.verbatim if best_span else None
                representative_text_element_id = (
                    best_span.text_element_id if best_span else None
                )

                rule = CanonicalRule(
                    canonical_id=_canonical_rule_id(group.group_id, direction),
                    group_id=group.group_id,
                    subject_entity=group.subject_entity,
                    outcome_entity=group.outcome_entity,
                    relation_type=group.relation_type,
                    direction=DirectionEnum(direction) if direction in DirectionEnum._value2member_map_ else None,
                    predicate_text=predicate_text,
                    is_conflicted=group_is_conflicted,
                    study_coverage=study_coverage,
                    category=group.category,
                    supporting_pmcids=all_pmcids,
                    member_normal_ids=all_member_ids,
                    mean_grounding_score=mean_score,
                    finding_count=len(bin_nfs),
                    scope=representative_scope,
                    representative_verbatim=representative_verbatim,
                    representative_text_element_id=representative_text_element_id,
                )
                canonical_rules.append(rule)

        logger.info(
            "[%s] CANONICALIZE: %d groups → %d CanonicalRules",
            pmcid, len(groups), len(canonical_rules),
        )
        return canonical_rules

    def _select_predicate(
        self,
        candidates: list[tuple[float, str]],
        group: FindingGroup,
        direction: str,
        pmcid: str,
    ) -> str:
        return _pick_best_predicate_deterministic(candidates)
