"""Tests for MapStage._make_chunks overlapping chunking."""
from __future__ import annotations

import pytest

from nlp_histo.pipeline.stages.knowledge_extraction.stages.map_stage import MapStage


def _make_stage(chunk_size: int, chunk_overlap: int) -> MapStage:
    """Build a minimal MapStage with stub LLMs (no real API calls)."""
    stage = object.__new__(MapStage)
    stage.chunk_size = chunk_size
    stage.chunk_overlap = chunk_overlap
    return stage


def _sents(n: int) -> list[dict]:
    return [{"sentence": f"s{i}", "text_element_id": i} for i in range(n)]


# ── constructor validation ─────────────────────────────────────────────────────

def _real_stage(chunk_size: int, chunk_overlap: int) -> MapStage:
    """Instantiate MapStage with minimal stubs to exercise __init__ validation."""
    from unittest.mock import MagicMock
    stub_llm = MagicMock()
    return MapStage(
        voter_llms=[stub_llm],
        level2_voter_llms=[stub_llm],
        escalation_llm=stub_llm,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


class TestMapStageInit:
    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            _real_stage(chunk_size=10, chunk_overlap=-1)

    def test_overlap_equal_to_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size"):
            _real_stage(chunk_size=10, chunk_overlap=10)

    def test_overlap_greater_than_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size"):
            _real_stage(chunk_size=10, chunk_overlap=11)


# ── _make_chunks behaviour ─────────────────────────────────────────────────────

class TestMakeChunks:
    def test_empty_input_returns_empty(self):
        stage = _make_stage(chunk_size=10, chunk_overlap=0)
        assert stage._make_chunks([]) == []

    def test_zero_overlap_preserves_original_behaviour(self):
        """With overlap=0, output must be identical to the old disjoint split."""
        stage = _make_stage(chunk_size=10, chunk_overlap=0)
        sents = _sents(25)
        chunks = stage._make_chunks(sents)
        assert chunks == [sents[0:10], sents[10:20], sents[20:25]]

    def test_overlap_stride(self):
        """chunk_size=10, overlap=2 → stride=8 → windows at 0,8,16,…"""
        stage = _make_stage(chunk_size=10, chunk_overlap=2)
        sents = _sents(20)
        chunks = stage._make_chunks(sents)
        assert chunks[0] == sents[0:10]
        assert chunks[1] == sents[8:18]
        assert chunks[2] == sents[16:20]  # last chunk shorter than chunk_size

    def test_boundary_sentences_appear_in_two_chunks(self):
        """Sentences at chunk boundary appear in both the closing and opening chunk."""
        stage = _make_stage(chunk_size=10, chunk_overlap=2)
        sents = _sents(20)
        chunks = stage._make_chunks(sents)
        # sents[8] and sents[9] are the tail of chunk 0 and head of chunk 1
        assert sents[8] in chunks[0] and sents[8] in chunks[1]
        assert sents[9] in chunks[0] and sents[9] in chunks[1]

    def test_fewer_sents_than_chunk_size(self):
        stage = _make_stage(chunk_size=10, chunk_overlap=2)
        sents = _sents(5)
        chunks = stage._make_chunks(sents)
        assert len(chunks) == 1
        assert chunks[0] == sents

    def test_exact_chunk_size_no_remainder(self):
        stage = _make_stage(chunk_size=10, chunk_overlap=0)
        sents = _sents(20)
        chunks = stage._make_chunks(sents)
        assert len(chunks) == 2
        assert chunks[0] == sents[:10]
        assert chunks[1] == sents[10:]

    def test_chunk_count_increases_with_overlap(self):
        """Overlap > 0 produces strictly more chunks than the disjoint case."""
        sents = _sents(30)
        disjoint = _make_stage(10, 0)._make_chunks(sents)
        overlapping = _make_stage(10, 2)._make_chunks(sents)
        assert len(overlapping) > len(disjoint)

    def test_single_sentence(self):
        stage = _make_stage(chunk_size=10, chunk_overlap=3)
        sents = _sents(1)
        chunks = stage._make_chunks(sents)
        assert chunks == [sents]

    def test_overlap_one_less_than_chunk_size(self):
        """Maximum legal overlap (chunk_size - 1): stride = 1."""
        stage = _make_stage(chunk_size=3, chunk_overlap=2)
        sents = _sents(5)
        chunks = stage._make_chunks(sents)
        # stride=1 → windows at 0,1,2,3
        assert chunks[0] == sents[0:3]
        assert chunks[1] == sents[1:4]
        assert chunks[2] == sents[2:5]
        assert chunks[3] == sents[3:5]
