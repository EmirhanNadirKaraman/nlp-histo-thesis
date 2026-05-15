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
)

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _canonical_rule_id(group_id: str, direction: str) -> str:
    return f"CR_{_sha8(group_id)}_{direction}"


def _compute_scope_fields(
    member_nfs: list[NormalFinding],
) -> tuple[bool, str]:
    """
    Compute the two scope fields for a direction-bin.

    Returns
    -------
    (is_conflicted, study_coverage) where:
      is_conflicted  — True if the bin contains ≥2 distinct polarity-bearing
                       directions. 'unclear' (model couldn't decide) and
                       'no_direction' (polarity doesn't apply — demographic
                       facts, neutral counts) are both excluded, matching the
                       semantic split established in MAP enum coercion.
      study_coverage — "single_study" | "multi_study" | "unknown" based on PMCID coverage,
                       computed independently of is_conflicted so both signals are preserved.
    """
    bin_directions: set[str] = set()
    for nf in member_nfs:
        d = nf.direction.value if nf.direction is not None else "unclear"
        if d not in ("unclear", "no_direction"):
            bin_directions.add(d)
    is_conflicted = len(bin_directions) >= 2

    all_pmcids: set[str] = set()
    for nf in member_nfs:
        all_pmcids.update(nf.pmcids)

    if len(all_pmcids) >= 2:
        study_coverage = "multi_study"
    elif len(all_pmcids) == 1:
        study_coverage = "single_study"
    else:
        study_coverage = "unknown"

    return is_conflicted, study_coverage


def _pick_best_predicate_deterministic(
    candidates: list[tuple[float, str]]
) -> str:
    """
    Fallback: return the predicate_text with the highest grounding score.
    candidates: list of (mean_grounding_score, predicate_text), possibly with None scores.
    """
    return max(candidates, key=lambda x: x[0] or 0.0)[1]


def _split_by_direction(
    group: FindingGroup,
    member_nfs: list[NormalFinding],
) -> list[tuple[str, list[NormalFinding]]]:
    """
    Split member NormalFindings by direction.

    Polarity-bearing directions (`positive`, `negative`, `absent`, `partial`)
    each get their own bin. `unclear` (model couldn't decide) and
    `no_direction` (polarity doesn't apply) are both treated as non-polarity:
    if no polarity-bearing direction exists they collapse into a single
    `"unclear"` bin; if one polarity-bearing direction exists they join it;
    if several polarity-bearing directions exist they join the largest.
    """
    non_unclear = {
        d: [] for d, c in group.direction_counts.items()
        if d not in ("unclear", "no_direction") and c > 0
    }
    unclear_nfs: list[NormalFinding] = []

    for nf in member_nfs:
        d = nf.direction.value if nf.direction is not None else "unclear"
        if d in non_unclear:
            non_unclear[d].append(nf)
        else:
            unclear_nfs.append(nf)

    if not non_unclear:
        # All findings are 'unclear' — single bin
        return [("unclear", member_nfs)]

    if len(non_unclear) == 1:
        direction = next(iter(non_unclear))
        return [(direction, member_nfs)]  # keep unclear in same bucket

    # Mixed directions — assign unclear nfs to the largest direction bin
    largest_dir = max(non_unclear, key=lambda d: len(non_unclear[d]))
    non_unclear[largest_dir].extend(unclear_nfs)
    return list(non_unclear.items())


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

            # Split by direction (may yield multiple bins)
            direction_bins = _split_by_direction(group, member_nfs)

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

                is_conflicted, study_coverage = _compute_scope_fields(bin_nfs)

                # Aggregate evidence and PMCID lists
                all_pmcids: list[str] = sorted(
                    {p for nf in bin_nfs for p in nf.pmcids}
                )
                all_member_ids = [nf.normal_id for nf in bin_nfs]

                # mean_grounding_score across the bin
                scores = [nf.mean_grounding_score for nf in bin_nfs if nf.mean_grounding_score is not None]
                mean_score = sum(scores) / len(scores) if scores else None

                rule = CanonicalRule(
                    canonical_id=_canonical_rule_id(group.group_id, direction),
                    group_id=group.group_id,
                    subject_entity=group.subject_entity,
                    outcome_entity=group.outcome_entity,
                    relation_type=group.relation_type,
                    direction=DirectionEnum(direction) if direction in DirectionEnum._value2member_map_ else None,
                    predicate_text=predicate_text,
                    is_conflicted=is_conflicted,
                    study_coverage=study_coverage,
                    category=group.category,
                    supporting_pmcids=all_pmcids,
                    member_normal_ids=all_member_ids,
                    mean_grounding_score=mean_score,
                    finding_count=len(bin_nfs),
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
