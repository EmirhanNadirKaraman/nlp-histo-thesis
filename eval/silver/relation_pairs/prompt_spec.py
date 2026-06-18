"""Single source of truth for the E13 relation claim-pair generation spec.

Both acquisition paths import these blocks so the label/field definitions can
never drift:
  * ``prepare_manual_batches`` — manual, per-batch JSONL prompts (legacy/fallback).
  * ``generate`` — Opus-4.7 batch-API generation with forced structured output.

Only the framing differs between the two (batch-of-30 vs one-shot-300, JSONL vs
forced tool_use); the definitions below are identical for both.
"""
from __future__ import annotations

PROMPT_SPEC_VERSION = "v1"

GOLD_LABELS = ("SUPPORTING", "CONTRADICTING", "UNRELATED")
DIFFICULTIES = ("easy", "medium", "hard")

# Controlled relation-subtype vocabulary, grouped by gold label.
RELATION_SUBTYPES_BY_LABEL = {
    "SUPPORTING": ("paraphrase", "entailment", "elaboration", "same_direction"),
    "CONTRADICTING": ("negation", "opposite_direction", "mutually_exclusive"),
    "UNRELATED": ("different_entity", "different_outcome", "orthogonal_topic"),
}
RELATION_SUBTYPES = tuple(s for subs in RELATION_SUBTYPES_BY_LABEL.values() for s in subs)

# ── Shared definition blocks (identical text for manual + API paths) ─────────────

LABEL_DEFINITIONS = """\
LABEL DEFINITIONS (the relation FROM claim_a TO claim_b):
  - SUPPORTING   : the two claims make compatible assertions about the same
                   disease/entity/context. They may be paraphrases, equivalent
                   statements, or cases where one claim clearly entails or
                   elaborates the other without changing the direction of the
                   finding. Do not make SUPPORTING pairs exact duplicates.
  - CONTRADICTING: the two claims directly conflict. They refer to the same
                   disease/entity/context and the same outcome or property, but
                   make opposite or mutually exclusive assertions (e.g. "X predicts
                   worse survival" vs "X predicts better survival", or "X is
                   associated with Y" vs "X is not associated with Y").
  - UNRELATED    : the claims neither support nor contradict each other. They may
                   involve different entities, different outcomes, different
                   contexts, or orthogonal findings. Near-domain unrelated pairs
                   are encouraged: the claims may look related because they share
                   an organ, biomarker, or disease family, but they still do not
                   assert the same relation or a direct contradiction."""

CLAIM_RULES = """\
EACH claim must be a single, self-contained finding sentence (8–30 words). Do NOT
quote any real paper; do NOT include citations, source sentences, or any
"verbatim" field. Write the claims as clean abstracted findings."""

DIFFICULTY_BLOCK = """\
DIFFICULTY: label each pair "easy", "medium", or "hard".
  - easy   : the relation is obvious (clear paraphrase / blatant opposite / clearly
             different topic).
  - medium : requires reading the claim carefully (subtle rewording, partial
             overlap, hedged direction).
  - hard   : near-miss cases (same entity but different outcome that looks related;
             same outcome but different entity; subtle negation or scope shift).
Aim for a spread of difficulties within each label."""

RELATION_SUBTYPE_BLOCK = """\
RELATION SUBTYPE: a short tag describing the mechanism, from this controlled
vocabulary (pick the closest):
  - SUPPORTING   : paraphrase | entailment | elaboration | same_direction
  - CONTRADICTING: negation | opposite_direction | mutually_exclusive
  - UNRELATED    : different_entity | different_outcome | orthogonal_topic"""

FIELD_DEFINITIONS = """\
FIELD DEFINITIONS:
  - disease_or_entity: a short disease/entity/context label for the pair. For
    SUPPORTING and CONTRADICTING pairs, this should usually be the shared disease,
    biomarker, lesion, or tissue context. For UNRELATED pairs, use the closest
    common context if one exists; otherwise use the main entity from claim_a."""


def definitions_block() -> str:
    """The five shared definition blocks, in canonical order, for either path."""
    return "\n\n".join([
        LABEL_DEFINITIONS,
        CLAIM_RULES,
        DIFFICULTY_BLOCK,
        RELATION_SUBTYPE_BLOCK,
        FIELD_DEFINITIONS,
    ])
