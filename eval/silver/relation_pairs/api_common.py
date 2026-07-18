"""Shared layer for the E13 Opus-4.7 batch-API generation pipeline.

Holds the model/tool/prompt definitions and paths used by
``prepare_api_batch_jsonl`` → ``submit_api_batch`` → ``collect_api_batch``. The
prompt definitions come from ``prompt_spec`` (single source, shared with the
manual path). Structured output is enforced with a forced tool_use call.

TEMPERATURE: ``claude-opus-4-7`` REJECTS the ``temperature`` parameter
("`temperature` is deprecated for this model", B-072 — verified empirically). We
therefore send NO temperature; the model uses its default sampling and the forced
tool_use schema keeps generation variance low (same approach as silver-finding
generation, ``eval/silver/generation/generator.py``). The thesis text says "temperature
zero"; that is not literally achievable for this model and must be reconciled in
the write-up.
"""
from __future__ import annotations

import json
from pathlib import Path

from eval.silver.relation_pairs.prepare_manual_batches import (
    N_BATCHES,
    PAIRS_PER_BATCH,
    PER_LABEL_PER_BATCH,
    batch_id_range,
)
from eval.silver.relation_pairs.prompt_spec import (
    DIFFICULTIES,
    GOLD_LABELS,
    PROMPT_SPEC_VERSION,
    RELATION_SUBTYPES,
    SCHEMA_KEYS,
    definitions_block,
)

from eval.paths import REPO_ROOT as _REPO_ROOT  # noqa: E402

MODEL = "claude-opus-4-7"
MAX_TOKENS = 12000  # ~30 structured pairs ≈ 3–4k tokens; generous, no truncation
TOOL_NAME = "emit_relation_pairs"

# Paths
API_DIR = _REPO_ROOT / "eval" / "data" / "relation_pairs" / "api_requests"
REQUEST_JSONL = API_DIR / "relation_pairs_batch_requests.jsonl"
BATCH_STATES = API_DIR / "batch_states.json"           # list of submitted batches
RAW_RESULTS = API_DIR / "raw_results.jsonl"            # raw API results, one/line
GENERATION_META = API_DIR / "generation_meta.json"     # truthful provenance for the merge
FAILED_IDS = API_DIR / "failed_custom_ids.json"        # for per-custom_id retry
RAW_BATCHES_DIR = _REPO_ROOT / "eval" / "data" / "relation_pairs" / "raw_batches"

CUSTOM_ID_PREFIX = "relation_batch_"


def custom_id_for(batch_number: int) -> str:
    return f"{CUSTOM_ID_PREFIX}{batch_number:02d}"


def batch_number_of(custom_id: str) -> int:
    return int(custom_id.removeprefix(CUSTOM_ID_PREFIX))


# Structured-output tool
# Item fields are derived from SCHEMA_KEYS (single source) minus ``id``: ids are
# assigned deterministically per batch on collection, so they are always
# contiguous and correct regardless of model numbering.
_ITEM_KEYS = tuple(k for k in SCHEMA_KEYS if k != "id")
_ITEM_ENUMS = {
    "gold_label": list(GOLD_LABELS),
    "difficulty": list(DIFFICULTIES),
    "relation_subtype": list(RELATION_SUBTYPES),
}
_ITEM_PROPERTIES = {
    k: ({"type": "string", "enum": _ITEM_ENUMS[k]} if k in _ITEM_ENUMS else {"type": "string"})
    for k in _ITEM_KEYS
}
RELATION_PAIRS_TOOL = {
    "name": TOOL_NAME,
    "description": (
        "Return exactly 30 histopathology claim-pairs for this batch: 10 SUPPORTING, "
        "10 CONTRADICTING, 10 UNRELATED."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pairs": {
                "type": "array",
                "minItems": PAIRS_PER_BATCH,
                "maxItems": PAIRS_PER_BATCH,
                "items": {
                    "type": "object",
                    "properties": _ITEM_PROPERTIES,
                    "required": list(_ITEM_KEYS),
                },
            }
        },
        "required": ["pairs"],
    },
}

SYSTEM_PROMPT = (
    "You are an expert histopathologist building a labelled evaluation set for a "
    "relation-classification system. Each item is a PAIR of short, self-contained "
    "scientific claims about histopathology / pathology / oncology, with an INTENDED "
    "logical relation between them.\n\n" + definitions_block()
)


def build_user_prompt(batch_number: int) -> str:
    start_id, end_id = batch_id_range(batch_number)
    return (
        f"This is batch {batch_number} of {N_BATCHES}. Produce EXACTLY "
        f"{PAIRS_PER_BATCH} claim-pairs covering ids pair_{start_id}..pair_{end_id} "
        f"(conceptually; do not output id fields). The batch MUST contain exactly "
        f"{PER_LABEL_PER_BATCH} SUPPORTING, {PER_LABEL_PER_BATCH} CONTRADICTING, and "
        f"{PER_LABEL_PER_BATCH} UNRELATED pairs.\n\n"
        "TOPIC DIVERSITY: cover diverse pathology areas across the batch — include "
        "morphology, immunohistochemistry (IHC), molecular genetics, prognosis, "
        "treatment response, and grading/staging. Avoid using the same disease, "
        "biomarker, or organ system more than twice within one batch. Aim for a "
        "spread of easy/medium/hard difficulties within each label. Do NOT make "
        "SUPPORTING pairs exact duplicates.\n\n"
        f"Call the {TOOL_NAME} tool exactly once with all {PAIRS_PER_BATCH} pairs."
    )


def build_request(batch_number: int) -> dict:
    """One Message-Batches request object (custom_id + params). No temperature."""
    return {
        "custom_id": custom_id_for(batch_number),
        "params": {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "system": [{"type": "text", "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"}}],
            "tools": [RELATION_PAIRS_TOOL],
            "tool_choice": {"type": "tool", "name": TOOL_NAME},
            "messages": [{"role": "user", "content": build_user_prompt(batch_number)}],
        },
    }


def extract_pairs_from_content(content_blocks: list) -> list[dict]:
    """Pull the ``pairs`` array out of the forced tool_use block."""
    tool_use = next((b for b in content_blocks if getattr(b, "type", None) == "tool_use"), None)
    if tool_use is None:
        return []
    return list(tool_use.input.get("pairs", []))


def get_client():
    import os

    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")  # keys live in .env, not the shell env
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit(
            "ANTHROPIC_API_KEY not set (checked the shell env and .env). "
            "Add it to .env at the repo root."
        )
    import anthropic
    return anthropic.Anthropic(api_key=key)


def provenance_meta(batch_ids: list[str]) -> dict:
    """Truthful generation provenance written by collect, read by the merge."""
    return {
        "workflow": "api_batch",
        "generator_model": MODEL,
        "temperature": "unset — deprecated for this model (B-072); default sampling + forced tool_use",
        "max_tokens": MAX_TOKENS,
        "n_requests": N_BATCHES,
        "tool_name": TOOL_NAME,
        "prompt_spec_version": PROMPT_SPEC_VERSION,
        "batch_ids": batch_ids,
    }


def _load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default
