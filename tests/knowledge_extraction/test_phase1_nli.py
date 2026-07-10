"""
Phase 1 smoke test — real NLI model.

Downloads ~180 MB on first run (cross-encoder/nli-deberta-v3-base).
Run separately from the fast suite:

    pytest tests/knowledge_extraction/test_phase1_nli.py -v

Skip automatically if transformers is not installed:

    pytest tests/knowledge_extraction/test_phase1_nli.py -v -m nli
"""
import pytest

from pipeline.stages.knowledge_extraction.models import Finding

transformers = pytest.importorskip("transformers", reason="transformers not installed")


@pytest.mark.nli
def test_real_nli_entailed_claim():
    from pipeline.stages.knowledge_extraction.helpers.grounding_filter import GroundingFilter, score_findings

    gf = GroundingFilter(threshold=0.5)
    # Premise uses "predicts" — same verb as hypothesis — so the model correctly
    # scores entailment. "associated with" → "predicts" is a semantic leap the
    # cross-encoder correctly classifies as neutral, not entailment.
    f = Finding(
        category="IHC",
        claim="CD30 predicts worse overall survival",
        evidence=[],
        confidence="high",
        verbatim_support=(
            "CD30 positivity predicts worse overall survival in DLBCL patients."
        ),
    )
    score_findings([f], nli_pipe=gf._pipe)
    assert f.grounding_score is not None
    assert f.grounding_score >= 0.5, f"Expected entailed claim to score ≥ 0.5, got {f.grounding_score:.3f}"


@pytest.mark.nli
def test_real_nli_unentailed_claim():
    from pipeline.stages.knowledge_extraction.helpers.grounding_filter import GroundingFilter, score_findings

    gf = GroundingFilter(threshold=0.5)
    f = Finding(
        category="IHC",
        claim="CD30 predicts better overall survival",
        evidence=[],
        confidence="high",
        verbatim_support="Patient age and stage were recorded at diagnosis.",
    )
    score_findings([f], nli_pipe=gf._pipe)
    assert f.grounding_score is not None
    assert f.grounding_score < 0.5, f"Expected unentailed claim to score < 0.5, got {f.grounding_score:.3f}"
