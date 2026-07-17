# ── ProvenanceValidator unit tests ─────────────────────────────────────────────
from nlp_histo.pipeline.stages.knowledge_extraction.provenance.validator import ProvenanceValidator, SourceIndex
from tests.knowledge_extraction.routing.test_routing import (
    _make_chunk, _make_summary, _make_finding, PMCID
)
from nlp_histo.pipeline.stages.knowledge_extraction.routing.models import ReasonCode

class TestProvenanceValidator:
    def _make_validator(
        self,
        chunk: list[dict] | None = None,
        pmcid: str = PMCID,
        fabricated_threshold: float = 0.25,
        weak_threshold: float = 0.60,
    ) -> ProvenanceValidator:
        if chunk is None:
            chunk = _make_chunk()
        return ProvenanceValidator(
            source_index=SourceIndex(chunk),
            document_pmcid=pmcid,
            fabricated_threshold=fabricated_threshold,
            weak_threshold=weak_threshold,
        )

    def test_valid_finding_passes(self):
        validator = self._make_validator()
        summary = _make_summary([_make_finding(verbatim="CD31 was positive in all cases.")])
        results = validator.validate(summary)
        assert results[0].passed

    def test_nonexistent_source_id(self):
        # Chunk has only 1 sentence (position 1); citation references S5.
        finding = _make_finding(evidence=[f"S5|{PMCID}|42"])
        summary = _make_summary([finding])
        validator = self._make_validator()
        results = validator.validate(summary)
        assert not results[0].passed
        assert ReasonCode.NONEXISTENT_SOURCE in results[0].reason_codes

    def test_cross_document_pmcid(self):
        finding = _make_finding(evidence=["S1|PMC99999|42"])
        summary = _make_summary([finding])
        validator = self._make_validator()
        results = validator.validate(summary)
        assert not results[0].passed
        assert ReasonCode.CROSS_DOCUMENT_SOURCE_ERROR in results[0].reason_codes

    def test_text_element_id_mismatch(self):
        # Citation says te_id=99, but source index has te_id=42.
        finding = _make_finding(evidence=[f"S1|{PMCID}|99"])
        summary = _make_summary([finding])
        validator = self._make_validator()
        results = validator.validate(summary)
        assert not results[0].passed
        assert ReasonCode.INVALID_TEXT_ELEMENT_ID in results[0].reason_codes

    def test_fabricated_verbatim_rejected(self):
        # Source sentence is about CD31; verbatim is completely unrelated.
        # Ratio must be below the default fabricated_threshold=0.25.
        finding = _make_finding(
            verbatim="tumor mutational burden high microsatellite instability"
        )
        summary = _make_summary([finding])
        validator = self._make_validator()
        results = validator.validate(summary)
        assert not results[0].passed
        assert ReasonCode.FABRICATED_VERBATIM_SUPPORT in results[0].reason_codes

    def test_weak_grounding_escalates(self):
        # Verbatim is related but only loosely (no substring match, mid ratio).
        # Set thresholds so ratio falls between fabricated and weak.
        chunk = _make_chunk(sentence="CD31 expression was confirmed immunohistochemically.")
        finding = _make_finding(
            verbatim="CD31 positive staining observed",
            evidence=[f"S1|{PMCID}|42"],
        )
        summary = _make_summary([finding])
        validator = self._make_validator(chunk=chunk, fabricated_threshold=0.05, weak_threshold=0.99)
        results = validator.validate(summary)
        # Should not be a hard failure (above fabricated_threshold),
        # but below weak_threshold → WEAK_GROUNDING soft code.
        assert not results[0].passed
        hard_codes = {ReasonCode.FABRICATED_VERBATIM_SUPPORT, ReasonCode.NONEXISTENT_SOURCE,
                      ReasonCode.CROSS_DOCUMENT_SOURCE_ERROR, ReasonCode.INVALID_TEXT_ELEMENT_ID}
        assert not any(c in hard_codes for c in results[0].reason_codes)
        assert ReasonCode.WEAK_GROUNDING in results[0].reason_codes

    def test_direct_substring_verbatim_passes(self):
        # Verbatim is a direct substring of the source → ratio 1.0.
        chunk = _make_chunk(sentence="CD31 was positive in all cases examined.")
        finding = _make_finding(
            verbatim="CD31 was positive",
            evidence=[f"S1|{PMCID}|42"],
        )
        summary = _make_summary([finding])
        validator = self._make_validator(chunk=chunk)
        results = validator.validate(summary)
        assert results[0].passed