"""Tests for MapStage.process() start_chunk / limit_chunks slicing."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
from nlp_histo.pipeline.stages.knowledge_extraction.stages.map_stage import MapStage
from nlp_histo.pipeline.stages.knowledge_extraction.models import AuditMetadata, AuditableSummary


def _make_sentences(n: int) -> list[dict]:
    return [
        {"sentence": f"Sentence {i}.", "pmcid": "PMC1", "text_element_id": i}
        for i in range(n)
    ]


def _make_summary(chunk_id: str) -> AuditableSummary:
    meta = AuditMetadata(
        sentences_analyzed=1,
        sentences_cited=[],
        pmcids_referenced=[],
        uncited_sentences=[],
    )
    return AuditableSummary(
        chunk_id=chunk_id, findings=[], summary_text="ok", audit_metadata=meta
    )


def _make_stage() -> MapStage:
    """MapStage with mock LLMs — no real API calls."""
    dummy = MagicMock()
    dummy.with_structured_output.return_value = dummy
    return MapStage(
        voter_llms=[dummy],
        level2_voter_llms=[dummy],
        escalation_llm=dummy,
        chunk_size=5,
    )


@patch("nlp_histo.pipeline.stages.knowledge_extraction.stages.map_stage.MapStage._cascade")
def test_limit_chunks_processes_only_n(mock_cascade):
    stage = _make_stage()
    mock_cascade.side_effect = lambda chunk, pmcid, chunk_id, **_: (_make_summary(chunk_id), ("test", "mock"))
    sentences = _make_sentences(25)  # 5 chunks of 5

    results = stage.process(sentences, "PMC1", limit_chunks=2)

    assert len(results) == 2
    assert mock_cascade.call_count == 2


@patch("nlp_histo.pipeline.stages.knowledge_extraction.stages.map_stage.MapStage._cascade")
def test_start_chunk_skips_first_k(mock_cascade):
    stage = _make_stage()
    mock_cascade.side_effect = lambda chunk, pmcid, chunk_id, **_: (_make_summary(chunk_id), ("test", "mock"))
    sentences = _make_sentences(25)  # 5 chunks of 5

    stage.process(sentences, "PMC1", start_chunk=2)

    assert mock_cascade.call_count == 3  # chunks 2,3,4


@patch("nlp_histo.pipeline.stages.knowledge_extraction.stages.map_stage.MapStage._cascade")
def test_start_and_limit_combined(mock_cascade):
    stage = _make_stage()
    mock_cascade.side_effect = lambda chunk, pmcid, chunk_id, **_: (_make_summary(chunk_id), ("test", "mock"))
    sentences = _make_sentences(25)  # 5 chunks of 5

    stage.process(sentences, "PMC1", start_chunk=1, limit_chunks=2)

    assert mock_cascade.call_count == 2


@patch("nlp_histo.pipeline.stages.knowledge_extraction.stages.map_stage.MapStage._cascade")
def test_chunk_ids_are_absolute(mock_cascade):
    """chunk_id for chunk at absolute position k must be C{k+1}."""
    captured_ids = []

    def _side(chunk, pmcid, chunk_id, **_):
        captured_ids.append(chunk_id)
        return _make_summary(chunk_id), ("test", "mock")

    stage = _make_stage()
    mock_cascade.side_effect = _side
    sentences = _make_sentences(25)

    stage.process(sentences, "PMC1", start_chunk=2, limit_chunks=2)

    assert captured_ids == ["C3", "C4"]
