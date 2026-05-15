"""
GROUP stage: NormalFinding[] → FindingGroup[]

Accepts only groupable NormalFindings (subject_entity and outcome_entity
non-None, relation_type not unclear).  Caller is responsible for filtering
before calling group(); this module does not silently drop or convert
ungroupable findings.

Groups by (subject_entity, outcome_entity, relation_type, category).
Direction is NOT part of the grouping key — a group may contain findings
with different directions.
Opposing directions on the same entity pair surface as CONTRADICT relations
in Phase 5 RELATE, not as separate groups.

All logic is deterministic.  No LLM, no embeddings.
"""
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict

from ..models import FindingGroup, NormalFinding, RelationTypeEnum

logger = logging.getLogger(__name__)

_SCOPE_FIELDS = (
    "disease_subtype",
    "cohort_n",
    "assay_method",
    "biomarker_cutoff",
    "tissue_site",
    "treatment_context",
    "endpoint",
    "study_design",
)


def is_groupable(nf: NormalFinding) -> bool:
    """
    A NormalFinding is groupable iff subject_entity and outcome_entity are
    non-None, and relation_type is a known type (not unclear).
    Direction is NOT part of the groupability invariant — a group may contain
    findings with different or absent directions.
    """
    return (
        nf.subject_entity is not None
        and nf.outcome_entity is not None
        and nf.relation_type is not RelationTypeEnum.unclear
    )


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _group_id(
    subject: str,
    outcome: str,
    relation_type: str,
    category: str = "",
    pmcid: str = "",
) -> str:
    # Keyed on the normalized entity STRING only — never the CUI. `NormalizeStage`
    # canonicalises subject/outcome to a stable string regardless of whether UMLS
    # linking produced a CUI; mixing CUI with string here split otherwise-identical
    # findings into separate buckets whenever CUI population was partial (B-022).
    return (
        f"GRP_{_sha8(pmcid)}_{_sha8(subject)}_{_sha8(outcome)}"
        f"_{relation_type}_{_sha8(category)}"
    )


def _scope_heterogeneity(members: list[NormalFinding]) -> float:
    """
    Fraction of the 8 scope fields that have more than one distinct non-None
    value across all members.

    0.0 — every extracted scope field agrees (or all None).
    1.0 — all 8 fields have conflicting values.

    None values are excluded from the distinct-value set; a field counts as
    varying only when at least 2 members have different non-None values for it.
    """
    if len(members) <= 1:
        return 0.0

    varying = 0
    for field in _SCOPE_FIELDS:
        values = {
            getattr(m.scope, field)
            for m in members
            if getattr(m.scope, field) is not None
        }
        if len(values) > 1:
            varying += 1

    return round(varying / len(_SCOPE_FIELDS), 4)


def _direction_counts(members: list[NormalFinding]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in members:
        key = m.direction.value if m.direction is not None else "unclear"
        counts[key] = counts.get(key, 0) + 1
    return counts


class GroupStage:
    """
    GROUP stage: deterministic bucketing of groupable NormalFindings.

    Caller must pre-filter findings with is_groupable() before passing them
    here.  Passing ungroupable findings raises ValueError.
    """

    def group(self, findings: list[NormalFinding], pmcid: str) -> list[FindingGroup]:
        """
        Group NormalFindings for one paper by (subject_entity, outcome_entity, relation_type, category).

        All findings must be groupable (subject_entity, outcome_entity non-None,
        relation_type not unclear).  Raises ValueError on the first violation.

        Returns one FindingGroup per distinct (subject, outcome, relation_type, category) tuple.
        """
        if not findings:
            return []

        buckets: dict[str, list[NormalFinding]] = defaultdict(list)

        for nf in findings:
            if not is_groupable(nf):
                raise ValueError(
                    f"Non-groupable NormalFinding passed to GroupStage: {nf.normal_id!r} "
                    f"(subject={nf.subject_entity!r}, outcome={nf.outcome_entity!r}, "
                    f"relation_type={nf.relation_type!r}). Filter with is_groupable() before calling group()."
                )
            key = _group_id(  # type: ignore[arg-type]
                nf.subject_entity, nf.outcome_entity, nf.relation_type.value, nf.category,
                pmcid=pmcid,
            )
            buckets[key].append(nf)

        groups = [
            FindingGroup(
                group_id=gid,
                subject_entity=members[0].subject_entity,  # type: ignore[arg-type]
                outcome_entity=members[0].outcome_entity,  # type: ignore[arg-type]
                relation_type=members[0].relation_type,
                category=members[0].category,
                member_ids=[m.normal_id for m in members],
                direction_counts=_direction_counts(members),
                scope_heterogeneity=_scope_heterogeneity(members),
            )
            for gid, members in buckets.items()
        ]

        logger.info(
            "[%s] GROUP: %d groupable NormalFindings → %d FindingGroups",
            pmcid, len(findings), len(groups),
        )
        return groups
