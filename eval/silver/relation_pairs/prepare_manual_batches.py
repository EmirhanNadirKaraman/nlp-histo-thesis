#!/usr/bin/env python3
"""E13 Part 1 — write the 10 manual generation-prompt files (no API).

The relation-classification evaluation set (thesis §3.x, RQ4 §9.5) is a synthetic
300 claim-pair dataset. We generate it **manually**: this script writes ten
per-batch prompt files; the user runs each prompt in **Opus 4.7 at temperature 0**
and saves the raw JSONL response. Nothing here calls an API or an LLM.

  python -m eval.silver.relation_pairs.prepare_manual_batches

writes ``eval/prompts/relation_pairs/batch_01_prompt.txt … batch_10_prompt.txt``.
Each prompt asks for **30 claim-pairs** (10 SUPPORTING / 10 CONTRADICTING /
10 UNRELATED) with IDs ``pair_{30k-29:04d}`` … ``pair_{30k:04d}`` (batch 10 →
pair_0271…pair_0300). The user pastes each model response into
``eval/data/relation_pairs/raw_batches/batch_0k.jsonl``; then
``validate_and_merge`` builds the final ``relation_claim_pairs_300.jsonl``.

The prompt template is committed (this file IS the source of truth for it); the
generated per-batch ``*.txt`` files are gitignored.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

N_BATCHES = 10
PAIRS_PER_BATCH = 30
PER_LABEL_PER_BATCH = 10  # SUPPORTING / CONTRADICTING / UNRELATED
TOTAL_PAIRS = N_BATCHES * PAIRS_PER_BATCH  # 300

PROMPTS_DIR = _REPO_ROOT / "eval" / "prompts" / "relation_pairs"

# ── Generation prompt template ──────────────────────────────────────────────────
# Placeholders: {BATCH_NUMBER} {START_ID} {END_ID} (zero-padded 4-digit pair ids).
# Run each filled prompt in Opus 4.7 @ temperature 0. NO source sentences / no
# representative_verbatim field — claims are abstract findings, not quoted text.
PROMPT_TEMPLATE = """\
You are constructing a labelled evaluation set for a histopathology
relation-classification system. Each item is a PAIR of short scientific claims
about histopathology / pathology / oncology, together with the INTENDED logical
relation between the two claims. This is batch {BATCH_NUMBER} of 10.

Produce EXACTLY 30 claim-pairs in this batch:
  - 10 pairs with gold_label = "SUPPORTING"
  - 10 pairs with gold_label = "CONTRADICTING"
  - 10 pairs with gold_label = "UNRELATED"

Assign ids sequentially from pair_{START_ID} to pair_{END_ID} (inclusive), in
output order. You may interleave the labels in any order within the batch.

LABEL DEFINITIONS (the relation FROM claim_a TO claim_b):
  - SUPPORTING   : the two claims agree / are mutually entailing. claim_b restates,
                   paraphrases, or is logically entailed by claim_a (and vice
                   versa). They make the SAME assertion about the SAME entity and
                   outcome, in the same direction.
  - CONTRADICTING: the two claims directly conflict. Same entity + same outcome,
                   but OPPOSITE direction / mutually exclusive assertions (e.g.
                   "X predicts worse survival" vs "X predicts better survival", or
                   "X is associated with Y" vs "X is not associated with Y").
  - UNRELATED    : the claims neither support nor contradict each other. Different
                   entity, different outcome, or simply orthogonal statements. A
                   one-directional/asymmetric relationship also counts as UNRELATED.

EACH claim must be a single, self-contained finding sentence (8–30 words). Do NOT
quote any real paper; do NOT include citations, source sentences, or any
"verbatim" field. Write the claims as clean abstracted findings.

Vary topics widely across the 300-pair set: biomarkers (e.g. Ki-67, p53, PD-L1,
HER2, ER/PR, EGFR, ALK, BRAF, MSI/MMR), histologic grade/stage, tumour subtypes,
staining patterns, prognosis/survival, treatment response, morphology. Each batch
should cover several distinct topics.

DIFFICULTY: label each pair "easy", "medium", or "hard".
  - easy   : the relation is obvious (clear paraphrase / blatant opposite / clearly
             different topic).
  - medium : requires reading the claim carefully (subtle rewording, partial
             overlap, hedged direction).
  - hard   : near-miss cases (same entity but different outcome that looks related;
             same outcome but different entity; subtle negation or scope shift).
Aim for a spread of difficulties within each label.

RELATION SUBTYPE: a short tag describing the mechanism, from this controlled
vocabulary (pick the closest):
  - SUPPORTING   : paraphrase | entailment | elaboration | same_direction
  - CONTRADICTING: negation | opposite_direction | mutually_exclusive
  - UNRELATED    : different_entity | different_outcome | orthogonal_topic

OUTPUT FORMAT: JSON Lines — one JSON object per line, no surrounding array, no
markdown fences, no commentary before or after. Each line has EXACTLY these keys:

{{"id": "pair_{START_ID}", "topic": "<short topic, e.g. 'Ki-67 prognosis'>", "disease_or_entity": "<the shared disease or biomarker entity, e.g. 'breast carcinoma' or 'Ki-67'>", "claim_a": "<first claim>", "claim_b": "<second claim>", "gold_label": "SUPPORTING|CONTRADICTING|UNRELATED", "difficulty": "easy|medium|hard", "relation_subtype": "<from the vocab above>", "rationale": "<one sentence: why this label>"}}

Constraints recap:
  - EXACTLY 30 lines, ids pair_{START_ID}..pair_{END_ID} in order.
  - EXACTLY 10 SUPPORTING, 10 CONTRADICTING, 10 UNRELATED.
  - Valid JSONL, one object per line, the 9 keys above and nothing else.
  - No real quotations, no "verbatim" field, no citations.
  - Run at temperature 0.
"""


def prompt_hash() -> str:
    """Stable hash of the template body — recorded in the dataset meta."""
    return hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:16]


def batch_id_range(batch_number: int) -> tuple[str, str]:
    """Return (start_id, end_id) zero-padded 4-digit for a 1-based batch number."""
    start = PAIRS_PER_BATCH * (batch_number - 1) + 1
    end = PAIRS_PER_BATCH * batch_number
    return f"{start:04d}", f"{end:04d}"


def render_batch(batch_number: int) -> str:
    start_id, end_id = batch_id_range(batch_number)
    return PROMPT_TEMPLATE.format(
        BATCH_NUMBER=batch_number, START_ID=start_id, END_ID=end_id,
    )


def main() -> None:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"E13 manual batch prompts → {PROMPTS_DIR}")
    print(f"prompt template hash: {prompt_hash()}\n")
    for k in range(1, N_BATCHES + 1):
        start_id, end_id = batch_id_range(k)
        out = PROMPTS_DIR / f"batch_{k:02d}_prompt.txt"
        out.write_text(render_batch(k), encoding="utf-8")
        print(f"  batch {k:2d}: pair_{start_id}–pair_{end_id}  →  {out.name}")
    print(
        f"\nWrote {N_BATCHES} prompt files ({PAIRS_PER_BATCH} pairs each, "
        f"{TOTAL_PAIRS} total).\n"
        "Run each prompt in Opus 4.7 @ temperature 0 and save the raw JSONL to:\n"
        f"  {_REPO_ROOT / 'eval' / 'data' / 'relation_pairs' / 'raw_batches'}/batch_0k.jsonl\n"
        "Then: python -m eval.silver.relation_pairs.validate_and_merge"
    )


if __name__ == "__main__":
    main()
