"""
Unit tests for the grounding-first MAP-stage routing layer.

Covered cases:
  1. Invalid schema (empty claim)          → REJECT, schema gate
  2. Non-existent source ID               → REJECT, grounding gate
  3. Fabricated verbatim quote            → REJECT, grounding gate
  4. Weak grounding (partial match)        → ESCALATE, grounding gate
  5. Valid grounding + high agreement     → KEEP,    agreement gate
  6. Valid grounding + low agreement      → ESCALATE, agreement gate (default)
  7. Valid grounding + very low agreement → REJECT,  agreement gate (opt-in)
"""
from __future__ import annotations

from unittest.mock import MagicMock


from pipeline.stages.knowledge_extraction.agreement.checker import AgreementChecker
from pipeline.stages.knowledge_extraction.interfaces.scoring import ChunkDecision, ScoreBundle
from pipeline.stages.knowledge_extraction.models import AuditMetadata, AuditableSummary, Finding
from pipeline.stages.knowledge_extraction.routing.models import GateOrigin, ReasonCode
from pipeline.stages.knowledge_extraction.routing.router import MapOutputRouter
from pipeline.stages.knowledge_extraction.routing.schema_validator import SchemaValidator


# ── Helpers ────────────────────────────────────────────────────────────────────

PMCID = "PMC10047158"


def _make_finding(
    claim: str = "CD31 -> Positive",
    category: str = "IHC",
    evidence: list[str] | None = None,
    verbatim: str = "CD31 was positive in all cases.",
) -> Finding:
    return Finding(
        category=category,
        claim=claim,
        evidence=evidence if evidence is not None else [f"S1|{PMCID}|42"],
        confidence="high",
        verbatim_support=verbatim,
    )


def _make_summary(findings: list[Finding], chunk_id: str = "C1") -> AuditableSummary:
    return AuditableSummary(
        chunk_id=chunk_id,
        findings=findings,
        summary_text="Test summary.",
        audit_metadata=AuditMetadata(
            sentences_analyzed=3,
            sentences_cited=["1"],
            pmcids_referenced=[PMCID],
            uncited_sentences=["2", "3"],
        ),
    )


def _make_chunk(
    sentence: str = "CD31 was positive in all cases.",
    te_id: int = 42,
) -> list[dict]:
    return [{"pmcid": PMCID, "text_element_id": te_id, "sentence": sentence}]


def _make_router(
    agreement_decision: ChunkDecision = ChunkDecision.KEEP,
    agreement_confidence: float = 0.9,
    reject_on_low_agreement: bool = False,
) -> MapOutputRouter:
    """Build a MapOutputRouter with a mocked AgreementChecker."""
    mock_checker = MagicMock(spec=AgreementChecker)
    mock_checker.compute.return_value = ScoreBundle(
        embedding_agreement=agreement_confidence,
        confidence=agreement_confidence,
        decision=agreement_decision,
    )
    mock_checker.best.side_effect = lambda outputs: outputs[0]
    # Ensure the scorer has no compute_scored so _agreement_gate uses compute().
    mock_checker._scorer = MagicMock(spec=[])
    return MapOutputRouter(
        agreement_checker=mock_checker,
        reject_on_low_agreement=reject_on_low_agreement,
    )


# ── SchemaValidator unit tests ─────────────────────────────────────────────────

class TestSchemaValidator:
    def setup_method(self):
        self.validator = SchemaValidator()

    def test_valid_finding_passes(self):
        summary = _make_summary([_make_finding()])
        results = self.validator.validate(summary)
        assert len(results) == 1
        assert results[0].passed
        assert results[0].reason_codes == []

    def test_empty_claim_fails(self):
        finding = Finding.model_construct(
            category="IHC",
            claim="",
            evidence=[f"S1|{PMCID}|42"],
            confidence="high",
            verbatim_support="CD31 was positive.",
        )
        summary = _make_summary([finding])
        results = self.validator.validate(summary)
        assert not results[0].passed
        assert ReasonCode.EMPTY_CLAIM in results[0].reason_codes

    def test_empty_evidence_fails(self):
        finding = Finding.model_construct(
            category="IHC",
            claim="CD31 -> Positive",
            evidence=[],
            confidence="high",
            verbatim_support="CD31 was positive.",
        )
        summary = _make_summary([finding])
        results = self.validator.validate(summary)
        assert not results[0].passed
        assert ReasonCode.EMPTY_EVIDENCE_CHAIN in results[0].reason_codes

    def test_malformed_citation_id_fails(self):
        finding = _make_finding(evidence=["S1-PMC123-42"])  # wrong delimiter
        summary = _make_summary([finding])
        results = self.validator.validate(summary)
        assert not results[0].passed
        assert ReasonCode.INVALID_SENTENCE_ID in results[0].reason_codes

    def test_empty_verbatim_fails(self):
        finding = Finding.model_construct(
            category="IHC",
            claim="CD31 -> Positive",
            evidence=[f"S1|{PMCID}|42"],
            confidence="high",
            verbatim_support="   ",
        )
        summary = _make_summary([finding])
        results = self.validator.validate(summary)
        assert not results[0].passed
        assert ReasonCode.MISSING_REQUIRED_FIELD in results[0].reason_codes

    def test_multiple_findings_independent(self):
        good = _make_finding()
        bad = Finding.model_construct(
            category="IHC",
            claim="",
            evidence=[],
            confidence="high",
            verbatim_support="",
        )
        summary = _make_summary([good, bad])
        results = self.validator.validate(summary)
        assert results[0].passed
        assert not results[1].passed

# ── MapOutputRouter integration tests ─────────────────────────────────────────

class TestMapOutputRouter:
    """
    These tests focus on routing decisions.  The AgreementChecker is mocked
    so tests run without any embedding API.
    """

    def _route(
        self,
        outputs: list[AuditableSummary],
        chunk: list[dict] | None = None,
        agreement_decision: ChunkDecision = ChunkDecision.KEEP,
        agreement_confidence: float = 0.9,
        reject_on_low_agreement: bool = False,
    ):
        if chunk is None:
            chunk = _make_chunk()
        router = _make_router(agreement_decision, agreement_confidence, reject_on_low_agreement)
        return router.route(outputs, chunk=chunk, pmcid=PMCID, source_text="test source")

    # 1. Invalid schema → REJECT, schema gate
    def test_invalid_schema_reject(self):
        bad_finding = Finding.model_construct(
            category="IHC",
            claim="",           # empty claim — hard schema failure
            evidence=[],        # empty evidence — hard schema failure
            confidence="high",
            verbatim_support="something",
        )
        outputs = [_make_summary([bad_finding]), _make_summary([bad_finding])]
        decision = self._route(outputs)
        assert decision.decision == ChunkDecision.REJECT
        assert decision.gate_origin == GateOrigin.SCHEMA_GATE
        assert ReasonCode.EMPTY_CLAIM in decision.reason_codes

    # 2. Non-existent source ID → REJECT, grounding gate
    def test_nonexistent_source_reject(self):
        # Citation references S99 but chunk only has 1 sentence.
        bad_finding = _make_finding(evidence=[f"S99|{PMCID}|42"])
        outputs = [_make_summary([bad_finding]), _make_summary([bad_finding])]
        decision = self._route(outputs)
        assert decision.decision == ChunkDecision.REJECT
        assert decision.gate_origin == GateOrigin.GROUNDING_GATE
        assert ReasonCode.NONEXISTENT_SOURCE in decision.reason_codes

    # 3. Fabricated verbatim quote → REJECT, grounding gate
    def test_fabricated_verbatim_reject(self):
        # Use a verbatim with SequenceMatcher ratio < 0.25 vs the source sentence.
        bad_finding = _make_finding(
            verbatim="tumor mutational burden high microsatellite instability"
        )
        outputs = [_make_summary([bad_finding]), _make_summary([bad_finding])]
        decision = self._route(outputs)
        assert decision.decision == ChunkDecision.REJECT
        assert decision.gate_origin == GateOrigin.GROUNDING_GATE
        assert ReasonCode.FABRICATED_VERBATIM_SUPPORT in decision.reason_codes

    # 4. Weak grounding → ESCALATE, grounding gate
    def test_weak_grounding_escalates(self):
        chunk = _make_chunk(sentence="CD31 expression confirmed immunohistochemically.")
        # Use very tight thresholds so partial quotes fall in the weak zone.
        mock_checker = MagicMock(spec=AgreementChecker)
        router = MapOutputRouter(
            agreement_checker=mock_checker,
            fabricated_threshold=0.01,
            weak_threshold=0.99,
        )
        finding = _make_finding(
            verbatim="CD31 positive staining",
            evidence=[f"S1|{PMCID}|42"],
        )
        outputs = [_make_summary([finding])]
        decision = router.route(outputs, chunk=chunk, pmcid=PMCID, source_text="test")
        assert decision.decision == ChunkDecision.ESCALATE
        assert decision.gate_origin == GateOrigin.GROUNDING_GATE
        assert ReasonCode.WEAK_GROUNDING in decision.reason_codes

    # 5. Valid grounding + high agreement → KEEP, agreement gate
    def test_valid_grounding_high_agreement_keep(self):
        chunk = _make_chunk(sentence="CD31 was positive in all cases.")
        finding = _make_finding(verbatim="CD31 was positive in all cases.")
        # Two eligible voters required to reach the agreement gate.
        outputs = [_make_summary([finding]), _make_summary([finding])]
        decision = self._route(
            outputs,
            chunk=chunk,
            agreement_decision=ChunkDecision.KEEP,
            agreement_confidence=0.9,
        )
        assert decision.decision == ChunkDecision.KEEP
        assert decision.gate_origin == GateOrigin.AGREEMENT_GATE
        assert ReasonCode.HIGH_AGREEMENT in decision.reason_codes

    # 6. Valid grounding + low agreement → ESCALATE by default
    def test_valid_grounding_low_agreement_escalates_by_default(self):
        chunk = _make_chunk(sentence="CD31 was positive in all cases.")
        finding = _make_finding(verbatim="CD31 was positive in all cases.")
        outputs = [_make_summary([finding]), _make_summary([finding])]
        decision = self._route(
            outputs,
            chunk=chunk,
            agreement_decision=ChunkDecision.ESCALATE,
            agreement_confidence=0.4,
            reject_on_low_agreement=False,
        )
        assert decision.decision == ChunkDecision.ESCALATE
        assert decision.gate_origin == GateOrigin.AGREEMENT_GATE
        assert ReasonCode.ESCALATED_DUE_TO_LOW_AGREEMENT in decision.reason_codes

    # 7. Valid grounding + very low agreement → REJECT only when opt-in enabled
    def test_very_low_agreement_rejects_when_enabled(self):
        chunk = _make_chunk(sentence="CD31 was positive in all cases.")
        finding = _make_finding(verbatim="CD31 was positive in all cases.")
        outputs = [_make_summary([finding]), _make_summary([finding])]
        decision = self._route(
            outputs,
            chunk=chunk,
            agreement_decision=ChunkDecision.REJECT,
            agreement_confidence=0.05,
            reject_on_low_agreement=True,
        )
        assert decision.decision == ChunkDecision.REJECT
        assert decision.gate_origin == GateOrigin.AGREEMENT_GATE
        assert ReasonCode.INSUFFICIENT_AGREEMENT in decision.reason_codes

    def test_very_low_agreement_escalates_when_disabled(self):
        # Same as above but with opt-in disabled — should ESCALATE not REJECT.
        chunk = _make_chunk(sentence="CD31 was positive in all cases.")
        finding = _make_finding(verbatim="CD31 was positive in all cases.")
        outputs = [_make_summary([finding]), _make_summary([finding])]
        decision = self._route(
            outputs,
            chunk=chunk,
            agreement_decision=ChunkDecision.REJECT,
            agreement_confidence=0.05,
            reject_on_low_agreement=False,
        )
        assert decision.decision == ChunkDecision.ESCALATE
        assert decision.gate_origin == GateOrigin.AGREEMENT_GATE

    # 8. Mixed voters: one hard failure, one pass → ESCALATE (not REJECT)
    def test_partial_voter_failure_escalates_not_rejects(self):
        chunk = _make_chunk(sentence="CD31 was positive in all cases.")
        good = _make_summary([_make_finding(verbatim="CD31 was positive in all cases.")])
        # Bad voter references a non-existent sentence
        bad_finding = _make_finding(evidence=[f"S99|{PMCID}|42"])
        bad = _make_summary([bad_finding])
        router = _make_router(agreement_decision=ChunkDecision.KEEP)
        decision = router.route([good, bad], chunk=chunk, pmcid=PMCID, source_text="test")
        assert decision.decision == ChunkDecision.ESCALATE
        assert decision.gate_origin == GateOrigin.GROUNDING_GATE

    # 9. Zero-findings voter → UNUSABLE with NO_FINDINGS_EXTRACTED
    def test_zero_findings_voter_classified_unusable(self):
        chunk = _make_chunk(sentence="CD31 was positive in all cases.")
        # One good voter (eligible), one zero-findings voter (unusable)
        good = _make_summary([_make_finding(verbatim="CD31 was positive in all cases.")])
        empty = _make_summary([])
        router = _make_router(agreement_decision=ChunkDecision.KEEP)
        decision = router.route([good, empty], chunk=chunk, pmcid=PMCID, source_text="test")
        # Only 1 eligible voter → ESCALATE (single_voter_policy defaults to "escalate")
        assert decision.decision == ChunkDecision.ESCALATE
        assert ReasonCode.ESCALATED_DUE_TO_LOW_AGREEMENT in decision.reason_codes

    # 10. All zero-findings voters → REJECT, schema gate
    def test_all_zero_findings_voters_reject(self):
        chunk = _make_chunk(sentence="CD31 was positive in all cases.")
        empty1 = _make_summary([])
        empty2 = _make_summary([])
        router = _make_router(agreement_decision=ChunkDecision.KEEP)
        decision = router.route([empty1, empty2], chunk=chunk, pmcid=PMCID, source_text="test")
        assert decision.decision == ChunkDecision.REJECT
        assert decision.gate_origin == GateOrigin.SCHEMA_GATE
        assert ReasonCode.NO_FINDINGS_EXTRACTED in decision.reason_codes
