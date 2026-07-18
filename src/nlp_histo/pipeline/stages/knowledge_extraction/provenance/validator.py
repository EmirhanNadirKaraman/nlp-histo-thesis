"""
ProvenanceValidator — citation and verbatim provenance checks.

For each finding, verifies:
  1. Citation IDs parse as S{n}|{PMCID}|{te_id}.
  2. Sentence position S{n} exists in the chunk (source index).
  3. PMCID in citation matches the document PMCID.
  4. text_element_id in citation matches the source index entry.
  5. verbatim_support fuzzy-matches the cited source sentence.

Hard failures (REJECT when all voters fail):
  NONEXISTENT_SOURCE, CROSS_DOCUMENT_SOURCE_ERROR,
  INVALID_SENTENCE_ID, INVALID_TEXT_ELEMENT_ID, FABRICATED_VERBATIM_SUPPORT

Soft failures (ESCALATE):
  WEAK_GROUNDING  (verbatim exists but is only a partial/paraphrased match)
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from nlp_histo.pipeline.stages.knowledge_extraction.models import AuditableSummary, Finding

from nlp_histo.pipeline.stages.knowledge_extraction.validation.models import FindingValidation, ReasonCode

# PMC token accepts word-chars + hyphens so document-id suffixes such as
# `PMC10100421_HIS-82-393` or `PMC7150310_main` parse correctly.
_CITATION_RE = re.compile(r"^S(\d+)\|(PMC[\w\-]+)\|(\d+)$")

# B-082: the corpus uses suffixed document ids (e.g. `PMC4329418_his0066-0409`),
# but models frequently cite the bare canonical accession (`PMC4329418`) — the
# SAME paper. The cross-document check below compares the base `PMC<digits>`
# accession so bare-vs-suffixed is not a false cross-document error. A genuinely
# different paper still has a different base accession and is still caught.
_PMC_BASE_RE = re.compile(r"(PMC\d+)")


def _base_accession(pmcid: str) -> str:
    """Bare ``PMC<digits>`` accession, ignoring any ``_suffix`` (B-082)."""
    m = _PMC_BASE_RE.match(pmcid or "")
    return m.group(1) if m else (pmcid or "")

# Verbatim match thresholds.
# Below FABRICATED_THRESHOLD: the quote is unrecognisable in the source → hard REJECT.
# Below WEAK_THRESHOLD (but above fabricated): partial/paraphrased quote → soft ESCALATE.
_DEFAULT_FABRICATED_THRESHOLD = 0.25
_DEFAULT_WEAK_THRESHOLD = 0.60


class SourceIndex:
    """
    Maps 1-based sentence positions to their source metadata and text.

    Built from the raw chunk sentences that were passed into the MAP stage.
    Each entry is {"pmcid": str, "text_element_id": int, "sentence": str}.
    """

    def __init__(self, chunk: list[dict]) -> None:
        # Positions are 1-indexed to match the S{n} citation format.
        self._by_position: dict[int, dict] = {
            i + 1: item for i, item in enumerate(chunk)
        }

    def get(self, position: int) -> dict | None:
        """Return the source entry for a 1-based sentence position, or None."""
        return self._by_position.get(position)

    def __len__(self) -> int:
        return len(self._by_position)


class ProvenanceValidator:
    """
    Validates citation provenance for every Finding in an AuditableSummary.

    Parameters
    ----------
    source_index:
        SourceIndex built from the chunk sentences for this MAP call.
    document_pmcid:
        PMCID of the paper being processed (used for cross-document check).
    fabricated_threshold:
        SequenceMatcher ratio below which verbatim support is treated as
        fabricated (hard failure).  Default 0.25.
    weak_threshold:
        SequenceMatcher ratio below which verbatim support is treated as a
        weak/partial match (soft failure → ESCALATE).  Default 0.60.
    """

    def __init__(
        self,
        source_index: SourceIndex,
        document_pmcid: str,
        fabricated_threshold: float = _DEFAULT_FABRICATED_THRESHOLD,
        weak_threshold: float = _DEFAULT_WEAK_THRESHOLD,
    ) -> None:
        self._index = source_index
        self._pmcid = document_pmcid
        self._fabricated_threshold = fabricated_threshold
        self._weak_threshold = weak_threshold

    def validate(self, summary: AuditableSummary) -> list[FindingValidation]:
        """Return one FindingValidation per finding."""
        return [
            self._validate_finding(idx, finding)
            for idx, finding in enumerate(summary.findings)
        ]

    # Internal

    def _validate_finding(self, idx: int, finding: Finding) -> FindingValidation:
        codes: list[ReasonCode] = []
        explanations: list[str] = []
        has_hard_failure = False

        for cite in finding.evidence:
            m = _CITATION_RE.match(cite)
            if not m:
                # Already caught by SchemaValidator; record for completeness.
                codes.append(ReasonCode.INVALID_SENTENCE_ID)
                explanations.append(f"unparseable citation: {cite!r}")
                has_hard_failure = True
                continue

            pos = int(m.group(1))
            cited_pmcid = m.group(2)
            cited_te_id = int(m.group(3))

            # Cross-document PMCID — compare base accession (B-082), so citing the
            # bare `PMC4329418` matches the suffixed document `PMC4329418_his...`.
            if _base_accession(cited_pmcid) != _base_accession(self._pmcid):
                codes.append(ReasonCode.CROSS_DOCUMENT_SOURCE_ERROR)
                explanations.append(
                    f"citation PMCID {cited_pmcid!r} != document {self._pmcid!r}"
                )
                has_hard_failure = True
                continue

            # Sentence position must exist in the source index
            source = self._index.get(pos)
            if source is None:
                codes.append(ReasonCode.NONEXISTENT_SOURCE)
                explanations.append(
                    f"S{pos} is out of range "
                    f"(chunk has {len(self._index)} sentences)"
                )
                has_hard_failure = True
                continue

            # text_element_id must match
            if source["text_element_id"] != cited_te_id:
                codes.append(ReasonCode.INVALID_TEXT_ELEMENT_ID)
                explanations.append(
                    f"te_id mismatch for S{pos}: "
                    f"cited {cited_te_id} vs actual {source['text_element_id']}"
                )
                has_hard_failure = True
                continue

        # Verbatim check — only when citations themselves are valid
        if not has_hard_failure and finding.verbatim_support and finding.verbatim_support.strip():
            best_ratio = self._best_verbatim_ratio(finding)
            if best_ratio < self._fabricated_threshold:
                codes.append(ReasonCode.FABRICATED_VERBATIM_SUPPORT)
                explanations.append(
                    f"verbatim unrecognisable in any cited source "
                    f"(best ratio={best_ratio:.2f}, threshold={self._fabricated_threshold})"
                )
                has_hard_failure = True
            elif best_ratio < self._weak_threshold:
                codes.append(ReasonCode.WEAK_GROUNDING)
                explanations.append(
                    f"verbatim only partially matches cited source "
                    f"(best ratio={best_ratio:.2f}, threshold={self._weak_threshold})"
                )

        passed = not codes
        return FindingValidation(
            finding_index=idx,
            claim_preview=finding.claim[:80] if finding.claim else "",
            passed=passed,
            reason_codes=codes,
            explanation="; ".join(explanations) if explanations else "ok",
        )

    def _best_verbatim_ratio(self, finding: Finding) -> float:
        """
        Return the highest fuzzy-match ratio of verbatim_support against any
        cited source sentence.

        Strategy:
          1. If verbatim is a direct substring of the source → ratio 1.0.
          2. Otherwise use SequenceMatcher for partial / paraphrased quotes.
        """
        verbatim = finding.verbatim_support.lower().strip()
        best = 0.0

        for cite in finding.evidence:
            m = _CITATION_RE.match(cite)
            if not m:
                continue
            pos = int(m.group(1))
            source = self._index.get(pos)
            if source is None:
                continue

            source_text = source["sentence"].lower().strip()

            # Direct substring wins immediately
            if verbatim in source_text:
                return 1.0

            ratio = SequenceMatcher(None, verbatim, source_text).ratio()
            best = max(best, ratio)

        return best
