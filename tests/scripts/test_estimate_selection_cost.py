"""Regression tests for `scripts/estimate_selection_cost.py`.

Focused on the chunk input-token estimation formula (B-052). The worked
example pins the production defaults: chunk_size=10, chunk_overlap=2,
stride=8, n_sentences=100 → 13 chunks, 124 sentence-occurrences,
average 124/13 ≈ 9.54 sentences per chunk.
"""
from __future__ import annotations

import sys

import pytest

from tests.paths import SCRIPTS

# scripts/ is not a package, so it must be on sys.path to import the script by
# name. The repository root is already there (pytest inserts the rootdir).
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from estimate_selection_cost import (  # noqa: E402  (sys.path manipulation above)
    PROMPT_OVERHEAD_TOKENS,
    count_chunks,
    per_chunk_input_tokens,
)


# ── Worked example matches MapStage._make_chunks ─────────────────────────────

def test_per_chunk_input_tokens_matches_make_chunks_semantics():
    """100 sentences at default chunking → 13 chunks, total 124 occurrences."""
    chunk_size, chunk_overlap = 10, 2
    n_sentences = 100
    n_chunks = count_chunks(n_sentences, chunk_size, chunk_overlap)
    assert n_chunks == 13

    text_tokens = 10_000  # 100 tokens/sentence on average
    result = per_chunk_input_tokens(
        text_tokens, n_chunks, chunk_size, chunk_overlap, n_sentences,
    )

    # Expected = PROMPT_OVERHEAD + ceil(text_tokens * 124/13 / 100)
    #          = PROMPT_OVERHEAD + ceil(953.846...) = PROMPT_OVERHEAD + 954
    assert result == PROMPT_OVERHEAD_TOKENS + 954


def test_per_chunk_input_tokens_no_overlap():
    """chunk_overlap=0 → stride=chunk_size, no duplicated sentences."""
    result = per_chunk_input_tokens(
        text_tokens=1_000, n_chunks=10, chunk_size=10,
        chunk_overlap=0, n_sentences=100,
    )
    # 10 chunks × 10 sentences = 100 occurrences, average 10 per chunk.
    # chunk_text_tokens = 1000 * 10 / 100 = 100 → ceil = 100.
    assert result == PROMPT_OVERHEAD_TOKENS + 100


def test_per_chunk_input_tokens_single_chunk():
    """Fewer sentences than chunk_size → exactly one chunk holding all."""
    n_sentences = 7
    n_chunks = count_chunks(n_sentences, chunk_size=10, chunk_overlap=2)
    assert n_chunks == 1
    result = per_chunk_input_tokens(
        text_tokens=700, n_chunks=n_chunks, chunk_size=10,
        chunk_overlap=2, n_sentences=n_sentences,
    )
    # One chunk, 7 occurrences, average 7. chunk_text_tokens = 700.
    assert result == PROMPT_OVERHEAD_TOKENS + 700


def test_per_chunk_input_tokens_empty_input():
    """No sentences → just prompt overhead, no division by zero."""
    assert per_chunk_input_tokens(
        text_tokens=0, n_chunks=0, chunk_size=10,
        chunk_overlap=2, n_sentences=0,
    ) == PROMPT_OVERHEAD_TOKENS
    # Also covers zero n_chunks with non-zero stub inputs.
    assert per_chunk_input_tokens(
        text_tokens=1234, n_chunks=0, chunk_size=10,
        chunk_overlap=2, n_sentences=0,
    ) == PROMPT_OVERHEAD_TOKENS


def test_per_chunk_input_tokens_uses_ceil_not_round():
    """Conservative rounding: budget should never under-report tokens."""
    # Pick inputs where chunk_text_tokens is fractional and below .5 so the
    # behavioural difference between round() and ceil() shows up.
    # n_sentences=11, chunk_size=10, chunk_overlap=2, stride=8.
    # range(0, 11, 8) = [0, 8] → 2 chunks of sizes [10, 3]. Total = 13.
    # avg = 13/2 = 6.5. text_tokens=100 → 100*6.5/11 = 59.0909... → ceil = 60.
    # round() would have returned 59.
    result = per_chunk_input_tokens(
        text_tokens=100, n_chunks=2, chunk_size=10,
        chunk_overlap=2, n_sentences=11,
    )
    assert result == PROMPT_OVERHEAD_TOKENS + 60


def test_per_chunk_input_tokens_validates_overlap_negative():
    with pytest.raises(ValueError, match="chunk_overlap must satisfy"):
        per_chunk_input_tokens(
            text_tokens=100, n_chunks=1, chunk_size=10,
            chunk_overlap=-1, n_sentences=5,
        )


def test_per_chunk_input_tokens_validates_overlap_ge_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap must satisfy"):
        per_chunk_input_tokens(
            text_tokens=100, n_chunks=1, chunk_size=10,
            chunk_overlap=10, n_sentences=5,
        )
