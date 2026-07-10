"""
Shared UMLS utilities for the knowledge_extraction pipeline.

Used by normalize_stage.py (entity normalization) and entity_linker.py (CUI enrichment).
Keeping junk-type filtering and threshold in one place prevents the two modules from drifting.
"""
from __future__ import annotations

import re

UMLS_THRESHOLD: float = 0.85

# UMLS semantic types that are never valid entity normalizations in a
# histopathology / clinical text mining context.  A concept whose *only*
# semantic types fall in this set is discarded regardless of linker score.
JUNK_SEMANTIC_TYPES: frozenset[str] = frozenset({
    "T001",  # Organism
    "T002",  # Plant
    "T004",  # Fungus
    "T005",  # Virus
    "T007",  # Bacterium
    "T008",  # Animal
    "T009",  # Invertebrate
    "T010",  # Vertebrate
    "T011",  # Amphibian
    "T012",  # Bird
    "T013",  # Fish
    "T014",  # Reptile
    "T015",  # Mammal  ← Cetacea lives here
    "T083",  # Geographic Area
    "T093",  # Health Care Related Organization
    "T097",  # Professional or Occupational Group
})

_ACRONYM_RE = re.compile(r'^[A-Z]{2,5}$')


def is_junk_umls(canonical_name: str, types: list[str] | None, entity_text: str) -> bool:
    """Return True if a UMLS hit should be discarded."""
    type_set = set(types or [])
    if type_set and type_set.issubset(JUNK_SEMANTIC_TYPES):
        return True
    if _ACRONYM_RE.match(entity_text.strip()) and type_set & JUNK_SEMANTIC_TYPES:
        return True
    return False


def best_cui(kb_ents_by_span: list[list[tuple[str, float]]], kb, entity_text: str) -> str | None:
    """
    Pick the best non-junk CUI from a list of (CUI, score) lists (one per span).

    kb_ents_by_span: e.g. [span._.kb_ents for span in ents]
    kb: the scispacy linker's knowledge base (linker.kb)
    entity_text: original surface string, used for acronym junk check.

    Returns the highest-scoring CUI whose semantic types are not all junk, or None.
    """
    best: str | None = None
    best_score: float = 0.0

    for kb_ents in kb_ents_by_span:
        for cui, score in kb_ents:
            if score <= best_score:
                continue
            concept = kb.cui_to_entity.get(cui)
            if concept is None:
                continue
            types = list(concept.types) if concept.types else []
            if is_junk_umls(concept.canonical_name, types, entity_text):
                continue
            best_score = score
            best = cui

    return best
