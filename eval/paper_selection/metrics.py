"""Pluggable metrics for paper selection.

Three concerns, three Protocols:

  - :class:`PairMetric`  scores a pair of papers (relatedness).
  - :class:`SetMetric`   scores a set of papers (diversity / aggregate
                          relatedness).
  - :class:`PaperMetric` scores a single paper (hardness).

All three are interchangeable as long as the type signature matches, so
:mod:`selectors` does not depend on a specific scoring formula.

Default implementations live here; weights are exposed via dataclasses so
callers can override without forking the implementation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Protocol

from .models import HardnessBreakdown, PaperFingerprint


# Protocols

class PairMetric(Protocol):
    """Score a pair of papers; higher = more related."""
    def score(self, a: PaperFingerprint, b: PaperFingerprint) -> float: ...


class SetMetric(Protocol):
    """Score a set of papers; higher = better according to the metric."""
    def score(self, papers: list[PaperFingerprint]) -> float: ...


class PaperMetric(Protocol):
    """Score a single paper."""
    def score(self, paper: PaperFingerprint) -> float: ...


# Helpers

def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


# Relatedness

@dataclass
class RelatednessWeights:
    """Bucket weights for :class:`Relatedness`. Sum is informational only."""
    disease:    float = 0.30
    biomarker:  float = 0.20
    gene:       float = 0.15
    cui:        float = 0.15
    tissue:     float = 0.10
    method:     float = 0.06
    outcome:    float = 0.04
    generic:    float = 0.02   # tiny — generic-entity overlap shouldn't dominate
    no_disease_overlap_penalty: float = 0.10


@dataclass
class Relatedness:
    """Weighted Jaccard relatedness across entity buckets.

    The score sits in roughly [0, 1] when the bucket weights sum to ~1; values
    outside that range are unusual but acceptable. The penalty is subtracted
    when the two papers share zero diseases — it's the strongest signal that
    two papers aren't comparable.
    """
    weights: RelatednessWeights = field(default_factory=RelatednessWeights)

    def score(self, a: PaperFingerprint, b: PaperFingerprint) -> float:
        w = self.weights
        s  = w.disease   * _jaccard(a.disease_entities,   b.disease_entities)
        s += w.biomarker * _jaccard(a.biomarker_entities, b.biomarker_entities)
        s += w.gene      * _jaccard(a.gene_entities,      b.gene_entities)
        s += w.cui       * _jaccard(a.umls_cuis,          b.umls_cuis)
        s += w.tissue    * _jaccard(a.tissue_entities,    b.tissue_entities)
        s += w.method    * _jaccard(a.method_entities,    b.method_entities)
        s += w.outcome   * _jaccard(a.outcome_entities,   b.outcome_entities)
        s += w.generic   * _jaccard(a.entities,           b.entities)
        if not (a.disease_entities & b.disease_entities):
            s -= w.no_disease_overlap_penalty
        return s


# Diversity

@dataclass
class DiversityWeights:
    coverage_disease:  float = 0.30
    coverage_method:   float = 0.20
    coverage_outcome:  float = 0.10
    coverage_tissue:   float = 0.10
    coverage_biomarker: float = 0.20
    pairwise_penalty:  float = 0.40   # subtracted: mean pairwise relatedness
    near_duplicate_threshold: float = 0.65
    near_duplicate_penalty: float = 0.20  # extra hit per near-duplicate pair


@dataclass
class Diversity:
    """Reward broad coverage; penalise pairwise relatedness + near-duplicates."""
    weights: DiversityWeights = field(default_factory=DiversityWeights)
    pair_metric: PairMetric = field(default_factory=Relatedness)

    def score(self, papers: list[PaperFingerprint]) -> float:
        if not papers:
            return 0.0
        w = self.weights

        diseases   = set().union(*(p.disease_entities   for p in papers))
        methods    = set().union(*(p.method_entities    for p in papers))
        outcomes   = set().union(*(p.outcome_entities   for p in papers))
        tissues    = set().union(*(p.tissue_entities    for p in papers))
        biomarkers = set().union(*(p.biomarker_entities for p in papers))

        # Coverage scaled to the size of the set: more papers → expect more.
        n = max(len(papers), 1)
        coverage = (
            w.coverage_disease   * len(diseases)   / max(3 * n, 1)
            + w.coverage_method  * len(methods)    / max(2 * n, 1)
            + w.coverage_outcome * len(outcomes)   / max(2 * n, 1)
            + w.coverage_tissue  * len(tissues)    / max(2 * n, 1)
            + w.coverage_biomarker * len(biomarkers) / max(3 * n, 1)
        )

        # Pairwise relatedness penalty
        if n < 2:
            return coverage
        pair_scores: list[float] = []
        near_dupes = 0
        for i in range(n):
            for j in range(i + 1, n):
                s = self.pair_metric.score(papers[i], papers[j])
                pair_scores.append(s)
                if s >= w.near_duplicate_threshold:
                    near_dupes += 1
        mean_pair = sum(pair_scores) / len(pair_scores)
        return coverage - w.pairwise_penalty * mean_pair - w.near_duplicate_penalty * near_dupes


# Hardness

@dataclass
class HardnessWeights:
    """Weights for the three hardness sub-scores.

    Each sub-score is normalised to roughly [0, 1] before weighting; final
    ``normalized_hardness`` lives in the same range.
    """
    content_complexity: float = 0.45
    layout_complexity:  float = 0.25
    relation_complexity: float = 0.30
    workload_log_base:   float = math.e
    # Optional layout-quality signal (None when not available).
    layout_noise_weight: float = 0.10


@dataclass
class HardnessConfig:
    weights: HardnessWeights = field(default_factory=HardnessWeights)
    # Reference values used to scale raw counts into [0, 1].
    ref_chunks: int = 30                 # ~ a 300-sentence paper
    ref_entity_density: float = 4.0      # entities per sentence
    ref_unique_density: float = 1.5      # unique entities per sentence
    ref_table_density: float = 0.05      # tables per text element
    ref_caption_density: float = 0.10    # captions per text element
    ref_short_paragraph_rate: float = 0.30
    ref_long_paragraph_rate: float = 0.10
    ref_abbrev_density: float = 0.5      # abbreviations per sentence
    ref_relation_anchor_count: float = 30  # gene + biomarker count


@dataclass
class Hardness:
    """Compute :class:`HardnessBreakdown` for one paper.

    The implementation is intentionally explicit: each sub-score has a clear
    formula and a reason string so the rationale block in the export can show
    why a paper is considered hard.
    """
    config: HardnessConfig = field(default_factory=HardnessConfig)

    def breakdown(self, paper: PaperFingerprint) -> HardnessBreakdown:
        cfg = self.config
        w = cfg.weights
        reasons: list[str] = []

        n_sent = max(paper.n_sentences, 1)
        n_te   = max(paper.n_text_elements, 1)

        # Content complexity
        ent_density        = len(paper.entities | paper.disease_entities
                                  | paper.biomarker_entities | paper.gene_entities) / n_sent
        unique_ent_density = len(paper.entity_counts) / n_sent
        abbr_density       = float(paper.source_stats.get("abbrev_count", 0)) / n_sent
        content = (
            min(ent_density / cfg.ref_entity_density, 1.0) * 0.4
            + min(unique_ent_density / cfg.ref_unique_density, 1.0) * 0.4
            + min(abbr_density / cfg.ref_abbrev_density, 1.0) * 0.2
        )
        if ent_density > cfg.ref_entity_density:
            reasons.append(f"high entity density ({ent_density:.2f}/sentence)")
        if abbr_density > cfg.ref_abbrev_density:
            reasons.append(f"high abbreviation density ({abbr_density:.2f}/sentence)")

        # Layout complexity
        table_density   = paper.n_tables   / n_te
        caption_density = paper.n_captions / n_te
        short_rate      = float(paper.source_stats.get("very_short_paragraphs", 0)) / n_te
        long_rate       = float(paper.source_stats.get("very_long_paragraphs",  0)) / n_te
        layout = (
            min(table_density / cfg.ref_table_density, 1.0) * 0.30
            + min(caption_density / cfg.ref_caption_density, 1.0) * 0.20
            + min(short_rate / cfg.ref_short_paragraph_rate, 1.0) * 0.25
            + min(long_rate / cfg.ref_long_paragraph_rate, 1.0) * 0.25
        )
        if table_density > cfg.ref_table_density:
            reasons.append(f"table-heavy ({paper.n_tables} tables / {n_te} text elements)")
        if short_rate > cfg.ref_short_paragraph_rate:
            reasons.append("paragraph fragmentation: many very short paragraphs")
        if long_rate > cfg.ref_long_paragraph_rate:
            reasons.append("very long paragraphs present")

        # Optional layout-noise signal
        layout_noise = paper.source_stats.get("layout_noise_score")
        if isinstance(layout_noise, (int, float)):
            layout = layout * (1 - w.layout_noise_weight) + float(layout_noise) * w.layout_noise_weight

        # Relation complexity
        anchor_count = (
            len(paper.biomarker_entities)
            + len(paper.gene_entities)
            + len(paper.disease_entities)
        )
        relation = min(anchor_count / max(cfg.ref_relation_anchor_count, 1), 1.0)
        if anchor_count > cfg.ref_relation_anchor_count:
            reasons.append(f"many relation anchors ({anchor_count} disease/biomarker/gene mentions)")

        normalized = (
            w.content_complexity * content
            + w.layout_complexity * layout
            + w.relation_complexity * relation
        )

        # Workload factor + absolute hardness
        workload = math.log1p(paper.n_chunks) / math.log1p(max(cfg.ref_chunks, 1))
        absolute = normalized * math.log1p(paper.n_chunks)

        if not reasons:
            reasons.append("no specific hardness signal — moderate baseline")

        return HardnessBreakdown(
            normalized_hardness=normalized,
            absolute_hardness=absolute,
            workload_factor=workload,
            content_complexity=content,
            layout_complexity=layout,
            relation_complexity=relation,
            reasons=reasons,
        )

    def score(self, paper: PaperFingerprint) -> float:
        """Convenience hook so :class:`Hardness` satisfies :class:`PaperMetric`."""
        return self.breakdown(paper).normalized_hardness
