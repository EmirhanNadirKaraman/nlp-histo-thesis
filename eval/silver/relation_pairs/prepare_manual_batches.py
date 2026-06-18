#!/usr/bin/env python3
"""E13 (legacy/manual fallback) — write the 10 manual generation-prompt files.

The ACTIVE acquisition path is the Opus-4.7 batch API
(``prepare_api_batch_jsonl`` → ``submit_api_batch`` → ``collect_api_batch`` →
``validate_and_merge``). This module is kept as a no-API fallback: it writes ten
per-batch prompt files a human can run by hand and paste back. Both paths share
the SAME definition blocks (``prompt_spec``) and feed the SAME
``validate_and_merge`` gate, so the committed dataset is identical either way.

  python -m eval.silver.relation_pairs.prepare_manual_batches

writes ``eval/prompts/relation_pairs/batch_01_prompt.txt … batch_10_prompt.txt``.
Each prompt asks for **30 claim-pairs** (10/10/10) with IDs ``pair_{30k-29:04d}`` …
``pair_{30k:04d}`` (batch 10 → pair_0271…pair_0300). The user pastes each response
into ``eval/data/relation_pairs/raw_batches/batch_0k.jsonl``.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.silver.relation_pairs.prompt_spec import (  # noqa: E402
    CLAIM_RULES,
    DIFFICULTY_BLOCK,
    FIELD_DEFINITIONS,
    LABEL_DEFINITIONS,
    RELATION_SUBTYPE_BLOCK,
)

N_BATCHES = 10
PAIRS_PER_BATCH = 30
PER_LABEL_PER_BATCH = 10  # SUPPORTING / CONTRADICTING / UNRELATED
TOTAL_PAIRS = N_BATCHES * PAIRS_PER_BATCH  # 300

PROMPTS_DIR = _REPO_ROOT / "eval" / "prompts" / "relation_pairs"

# Single `.format()` template. The five definition blocks (which contain no
# braces) are injected as named fields; {BATCH_NUMBER}/{START_ID}/{END_ID} are the
# per-batch fields; literal JSON braces are doubled ({{ }}).
_TEMPLATE = """\
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

{label_definitions}

{claim_rules}

TOPIC DIVERSITY: within this batch, cover several distinct histopathology /
pathology areas (biomarkers e.g. Ki-67, p53, PD-L1, HER2, ER/PR, EGFR, ALK, BRAF,
MSI/MMR; histologic grade/stage; tumour subtypes; staining patterns;
prognosis/survival; treatment response; morphology). Avoid using the same disease,
biomarker, or organ system more than twice in one batch unless necessary.

{difficulty_block}

{relation_subtype_block}

{field_definitions}

OUTPUT FORMAT: JSON Lines — one JSON object per line, no surrounding array, no
markdown fences, no commentary before or after. Each line has EXACTLY these keys:

{{"id": "pair_{START_ID}", "topic": "<short topic, e.g. 'Ki-67 prognosis'>", "disease_or_entity": "<see FIELD DEFINITIONS, e.g. 'breast carcinoma' or 'Ki-67'>", "claim_a": "<first claim>", "claim_b": "<second claim>", "gold_label": "SUPPORTING|CONTRADICTING|UNRELATED", "difficulty": "easy|medium|hard", "relation_subtype": "<from the vocab above>", "rationale": "<one sentence: why this label>"}}

Constraints recap:
  - EXACTLY 30 lines, ids pair_{START_ID}..pair_{END_ID} in order.
  - EXACTLY 10 SUPPORTING, 10 CONTRADICTING, 10 UNRELATED.
  - Valid JSONL, one object per line, the 9 keys above and nothing else.
  - No real quotations, no "verbatim" field, no citations.
"""


def prompt_hash() -> str:
    """Stable hash of the template body — recorded in the dataset meta."""
    blob = "\n".join([_TEMPLATE, LABEL_DEFINITIONS, CLAIM_RULES,
                      DIFFICULTY_BLOCK, RELATION_SUBTYPE_BLOCK, FIELD_DEFINITIONS])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def batch_id_range(batch_number: int) -> tuple[str, str]:
    """Return (start_id, end_id) zero-padded 4-digit for a 1-based batch number."""
    start = PAIRS_PER_BATCH * (batch_number - 1) + 1
    end = PAIRS_PER_BATCH * batch_number
    return f"{start:04d}", f"{end:04d}"


def render_batch(batch_number: int) -> str:
    start_id, end_id = batch_id_range(batch_number)
    return _TEMPLATE.format(
        BATCH_NUMBER=batch_number, START_ID=start_id, END_ID=end_id,
        label_definitions=LABEL_DEFINITIONS, claim_rules=CLAIM_RULES,
        difficulty_block=DIFFICULTY_BLOCK, relation_subtype_block=RELATION_SUBTYPE_BLOCK,
        field_definitions=FIELD_DEFINITIONS,
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
        "Run each prompt in Opus 4.7 and save the raw JSONL to:\n"
        f"  {_REPO_ROOT / 'eval' / 'data' / 'relation_pairs' / 'raw_batches'}/batch_0k.jsonl\n"
        "Then: python -m eval.silver.relation_pairs.validate_and_merge\n"
        "(Or use the batch-API path: prepare_api_batch_jsonl → submit_api_batch → "
        "collect_api_batch → validate_and_merge.)"
    )


if __name__ == "__main__":
    main()
