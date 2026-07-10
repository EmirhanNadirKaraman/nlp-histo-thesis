"""Unit tests for the finding-level citation/provenance filter (B-080).

Covers the production check that the dormant router-only ProvenanceValidator
never ran on the legacy cascade: drop findings whose S{n}|PMCID|te_id citation
fails structural validation against the chunk source index.
"""
from __future__ import annotations

from pipeline.stages.knowledge_extraction.helpers.citation_filter import (
    citation_drop_indices,
    filter_summary_by_citation,
)
from pipeline.stages.knowledge_extraction.models import (
    AuditMetadata,
    AuditableSummary,
    Finding,
)

PMCID = "PMC10047158"
TE_ID = 42


def _finding(claim: str, evidence: list[str], verbatim: str = "CD31 was positive in all cases.") -> Finding:
    return Finding(
        category="IHC",
        claim=claim,
        evidence=evidence,
        confidence="high",
        verbatim_support=verbatim,
    )


def _summary(findings: list[Finding]) -> AuditableSummary:
    return AuditableSummary(
        chunk_id="C1",
        findings=findings,
        summary_text="Test summary.",
        audit_metadata=AuditMetadata(
            sentences_analyzed=1,
            sentences_cited=["1"],
            pmcids_referenced=[PMCID],
            uncited_sentences=[],
        ),
    )


def _chunk() -> list[dict]:
    return [{"pmcid": PMCID, "text_element_id": TE_ID, "sentence": "CD31 was positive in all cases."}]


# ── Valid citation is kept ──────────────────────────────────────────────────────

def test_valid_citation_kept():
    s = _summary([_finding("CD31 -> Positive", [f"S1|{PMCID}|{TE_ID}"])])
    s, dropped = filter_summary_by_citation(s, _chunk(), PMCID)
    assert dropped == []
    assert len(s.findings) == 1


# ── Each hard structural failure drops the finding ──────────────────────────────

def test_nonexistent_sentence_position_dropped():
    # S99 is out of range for a 1-sentence chunk.
    s = _summary([_finding("hallucinated", [f"S99|{PMCID}|{TE_ID}"])])
    s, dropped = filter_summary_by_citation(s, _chunk(), PMCID)
    assert len(dropped) == 1
    assert s.findings == []


def test_te_id_mismatch_dropped():
    # Cites the right sentence position but the wrong (hallucinated) te_id.
    s = _summary([_finding("wrong te_id", [f"S1|{PMCID}|999999"])])
    s, dropped = filter_summary_by_citation(s, _chunk(), PMCID)
    assert len(dropped) == 1
    assert s.findings == []


def test_cross_document_pmcid_dropped():
    s = _summary([_finding("wrong paper", [f"S1|PMC9999999|{TE_ID}"])])
    s, dropped = filter_summary_by_citation(s, _chunk(), PMCID)
    assert len(dropped) == 1
    assert s.findings == []


def test_unparseable_citation_dropped():
    s = _summary([_finding("garbage cite", ["not-a-citation"])])
    s, dropped = filter_summary_by_citation(s, _chunk(), PMCID)
    assert len(dropped) == 1
    assert s.findings == []


# ── Mixed batch: only the bad finding is removed ────────────────────────────────

def test_mixed_drops_only_invalid():
    good = _finding("good", [f"S1|{PMCID}|{TE_ID}"])
    bad = _finding("bad", [f"S1|{PMCID}|123456"])
    s = _summary([good, bad])
    drop_idx = citation_drop_indices(s, _chunk(), PMCID)
    assert drop_idx == {1}
    s, dropped = filter_summary_by_citation(s, _chunk(), PMCID)
    assert [f.claim for f in s.findings] == ["good"]
    assert [f.claim for f in dropped] == ["bad"]


# ── Verbatim fabrication is opt-in ──────────────────────────────────────────────

def test_fabricated_verbatim_kept_by_default():
    # Verbatim quote is unrecognisable in the cited sentence, but the citation
    # itself is structurally valid → kept when check_verbatim is off (default).
    s = _summary([_finding("ok cite", [f"S1|{PMCID}|{TE_ID}"], verbatim="totally unrelated fabricated text")])
    s, dropped = filter_summary_by_citation(s, _chunk(), PMCID)
    assert dropped == []
    assert len(s.findings) == 1


def test_fabricated_verbatim_dropped_when_enabled():
    # Quote with ~zero overlap with the cited sentence (SequenceMatcher ratio
    # 0.0 < 0.25 fabricated threshold) → dropped only when check_verbatim is on.
    s = _summary([_finding("ok cite", [f"S1|{PMCID}|{TE_ID}"], verbatim="zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")])
    s, dropped = filter_summary_by_citation(s, _chunk(), PMCID, check_verbatim=True)
    assert len(dropped) == 1
    assert s.findings == []


# ── B-082: bare-vs-suffixed PMC accession is the same paper ─────────────────────

def test_bare_accession_matches_suffixed_document():
    # Corpus document id is suffixed; the finding cites the bare canonical
    # accession (PMC10047158) — same paper, must NOT be flagged cross-document.
    doc = "PMC10047158_HIS-1-1"
    chunk = [{"pmcid": doc, "text_element_id": TE_ID, "sentence": "CD31 was positive in all cases."}]
    s = _summary([_finding("bare cite", [f"S1|PMC10047158|{TE_ID}"])])
    s, dropped = filter_summary_by_citation(s, chunk, doc)
    assert dropped == []
    assert len(s.findings) == 1


def test_different_base_accession_still_dropped():
    # A genuinely different paper (different base accession) is still caught.
    doc = "PMC10047158_HIS-1-1"
    chunk = [{"pmcid": doc, "text_element_id": TE_ID, "sentence": "CD31 was positive in all cases."}]
    s = _summary([_finding("foreign paper", [f"S1|PMC9999999|{TE_ID}"])])
    s, dropped = filter_summary_by_citation(s, chunk, doc)
    assert len(dropped) == 1
    assert s.findings == []


# ── Defensive: empty chunk never nukes the summary ──────────────────────────────

def test_empty_chunk_keeps_everything():
    s = _summary([_finding("a", [f"S1|{PMCID}|{TE_ID}"]), _finding("b", [f"S2|{PMCID}|{TE_ID}"])])
    s, dropped = filter_summary_by_citation(s, [], PMCID)
    assert dropped == []
    assert len(s.findings) == 2


def test_empty_findings_no_op():
    s = _summary([])
    s, dropped = filter_summary_by_citation(s, _chunk(), PMCID)
    assert dropped == []
    assert s.findings == []
