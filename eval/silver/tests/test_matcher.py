"""Unit tests for the silver-label matcher (no API calls)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from eval.silver.matcher import (
    _cosine,
    _field_mismatches,
    _normalize,
    compute_metrics,
    match_case,
)
from eval.silver.data.schemas import (
    PipelineCaseOutput,
    PipelineFinding,
    SilverCaseResult,
    SilverFinding,
    SilverFindingScope,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _silver(case_id="PMC1|1", claims=("claim A",), **kw) -> SilverCaseResult:
    findings = [
        SilverFinding(category="IHC", claim=c, confidence="high")
        for c in claims
    ]
    return SilverCaseResult(
        case_id=case_id, pmcid="PMC1", te_id=1,
        prompt_version="v1", model="claude-opus-4-5",
        findings=findings, **kw
    )


def _pipeline(case_id="PMC1|1", claims=("claim A",)) -> PipelineCaseOutput:
    findings = [
        PipelineFinding(
            pipeline_run_id=1, run_id="run1", pmcid="PMC1",
            chunk_id="C1", category="IHC", claim=c,
        )
        for c in claims
    ]
    return PipelineCaseOutput(
        case_id=case_id, pmcid="PMC1", te_id=1,
        run_id="run1", pipeline_run_id=1, findings=findings,
    )


def _unit_vec(dim: int, idx: int) -> list[float]:
    v = [0.0] * dim
    v[idx] = 1.0
    return v


# ── cosine ─────────────────────────────────────────────────────────────────────

def test_cosine_identical():
    v = [1.0, 0.0, 0.0]
    assert _cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _cosine(a, b) == pytest.approx(0.0)


def test_cosine_zero_vector():
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


# ── normalize ──────────────────────────────────────────────────────────────────

def test_normalize_lowercases():
    assert _normalize("ER+") == "er+"


def test_normalize_none():
    assert _normalize(None) == ""


# ── field mismatches ───────────────────────────────────────────────────────────

def test_no_mismatches_when_fields_match():
    sf = SilverFinding(category="IHC", claim="x",
                       relation_type="expression", direction="positive",
                       scope=SilverFindingScope(tissue_site="breast"))
    pf = PipelineFinding(
        pipeline_run_id=1, run_id="r", pmcid="PMC1",
        chunk_id="C1", category="IHC", claim="x",
        relation_type="expression", direction="positive",
        scope_tissue_site="breast",
    )
    assert _field_mismatches(sf, pf) == []


def test_mismatch_detected_for_direction():
    sf = SilverFinding(category="IHC", claim="x", direction="positive")
    pf = PipelineFinding(
        pipeline_run_id=1, run_id="r", pmcid="PMC1",
        chunk_id="C1", category="IHC", claim="x",
        direction="negative",
    )
    mismatches = _field_mismatches(sf, pf)
    assert any(m.field == "direction" for m in mismatches)


# ── match_case ─────────────────────────────────────────────────────────────────

def _mock_embedder(texts):
    """Deterministic embeddings: same text → same unit vector (hash-bucketed)."""
    import hashlib
    dim = 16
    result = []
    for text in texts:
        idx = int(hashlib.md5(text.encode()).hexdigest(), 16) % dim
        result.append(_unit_vec(dim, idx))
    return result


@patch("eval.silver.matcher.get_embeddings")
def test_perfect_match(mock_emb):
    # silver[0] and pipeline[0] should match (same position → cosine 1.0)
    mock_emb.side_effect = lambda texts, embedder, cache: _mock_embedder(texts)
    silver = _silver(claims=("claim A",))
    pipeline = _pipeline(claims=("claim A",))

    result = match_case(silver, pipeline, _mock_embedder)

    assert len(result.matched) == 1
    assert result.matched[0].similarity == pytest.approx(1.0)
    assert result.unmatched_silver == []
    assert result.unmatched_pipeline == []


@patch("eval.silver.matcher.get_embeddings")
def test_no_match_below_threshold(mock_emb):
    # silver[0] maps to unit vec 0, pipeline[0] to unit vec 1 → cosine 0.0
    mock_emb.side_effect = lambda texts, embedder, cache: _mock_embedder(texts)
    silver = _silver(claims=("claim A",))
    pipeline = _pipeline(claims=("claim B",))

    result = match_case(silver, pipeline, _mock_embedder)

    assert result.matched == []
    assert len(result.unmatched_silver) == 1
    assert len(result.unmatched_pipeline) == 1


@patch("eval.silver.matcher.get_embeddings")
def test_empty_silver(mock_emb):
    mock_emb.side_effect = lambda texts, embedder, cache: _mock_embedder(texts)
    silver = _silver(claims=())
    pipeline = _pipeline(claims=("claim A",))

    result = match_case(silver, pipeline, _mock_embedder)

    assert result.matched == []
    assert result.unmatched_silver == []
    assert len(result.unmatched_pipeline) == 1


@patch("eval.silver.matcher.get_embeddings")
def test_empty_pipeline(mock_emb):
    mock_emb.side_effect = lambda texts, embedder, cache: _mock_embedder(texts)
    silver = _silver(claims=("claim A",))
    pipeline = _pipeline(claims=())

    result = match_case(silver, pipeline, _mock_embedder)

    assert result.matched == []
    assert len(result.unmatched_silver) == 1
    assert result.unmatched_pipeline == []


# ── compute_metrics ────────────────────────────────────────────────────────────

@patch("eval.silver.matcher.get_embeddings")
def test_metrics_perfect(mock_emb):
    mock_emb.side_effect = lambda texts, embedder, cache: _mock_embedder(texts)
    silver = _silver(claims=("A",))
    pipeline = _pipeline(claims=("A",))
    result = match_case(silver, pipeline, _mock_embedder)

    metrics = compute_metrics([result], [silver], [pipeline], "v1", "opus")
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)


@patch("eval.silver.matcher.get_embeddings")
def test_metrics_zero_recall(mock_emb):
    mock_emb.side_effect = lambda texts, embedder, cache: _mock_embedder(texts)
    silver = _silver(claims=("A",))
    pipeline = _pipeline(claims=("B",))  # orthogonal → no match
    result = match_case(silver, pipeline, _mock_embedder)

    metrics = compute_metrics([result], [silver], [pipeline], "v1", "opus")
    assert metrics.recall == pytest.approx(0.0)
    assert metrics.f1 == pytest.approx(0.0)
