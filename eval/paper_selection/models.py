"""Data models for paper fingerprinting and selection.

Pure dataclasses — no Pydantic, no DB, no LLM. Cheap to construct in tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PaperFingerprint:
    """Offline feature vector for one paper.

    All entity sets are lowercase, surface-form normalised. The ``*_entities``
    sub-sets are derived from ``entities`` via a combination of semantic-type
    filters and regex/keyword classifiers — see ``fingerprints.py``.
    """
    pmcid: str
    title: str | None = None

    # Workload counters
    n_text_elements: int = 0
    n_paragraphs:    int = 0
    n_sentences:     int = 0
    n_chunks:        int = 0
    n_tokens_est:    int | None = None

    # Layout counters
    n_tables:   int = 0
    n_figures:  int = 0
    n_captions: int = 0

    # Entity vocab
    entities:           set[str] = field(default_factory=set)
    disease_entities:   set[str] = field(default_factory=set)
    biomarker_entities: set[str] = field(default_factory=set)
    gene_entities:      set[str] = field(default_factory=set)
    tissue_entities:    set[str] = field(default_factory=set)
    method_entities:    set[str] = field(default_factory=set)
    outcome_entities:   set[str] = field(default_factory=set)

    # UMLS provenance
    umls_cuis:      set[str] = field(default_factory=set)
    semantic_types: set[str] = field(default_factory=set)

    # Frequency table for entity lookups
    entity_counts: dict[str, int] = field(default_factory=dict)

    # Free-form workload / quality stats (mean paragraph length, abbrev density …)
    source_stats: dict[str, float | int | str] = field(default_factory=dict)

    # Convenience accessors

    def is_empty(self) -> bool:
        """True when there is essentially no usable content."""
        return self.n_sentences == 0 and not self.entities

    def has_useful_entities(self) -> bool:
        """True when at least one structured entity bucket is populated."""
        return bool(
            self.disease_entities
            or self.biomarker_entities
            or self.gene_entities
        )


@dataclass
class HardnessBreakdown:
    """Decomposed hardness score for one paper.

    Sub-scores are roughly in [0, 1]. ``normalized_hardness`` is per-unit
    difficulty; ``absolute_hardness`` weights it by workload (number of
    chunks) so a giant paper at moderate density still ranks high.
    """
    normalized_hardness: float
    absolute_hardness:   float
    workload_factor:     float
    content_complexity:  float
    layout_complexity:   float
    relation_complexity: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class SelectionResult:
    """Container for the calibration set."""
    related: list[str] = field(default_factory=list)
    diverse: list[str] = field(default_factory=list)
    hard:    list[str] = field(default_factory=list)
    rationale: dict[str, dict] = field(default_factory=dict)

    def all_pmcids(self) -> list[str]:
        return [*self.related, *self.diverse, *self.hard]
