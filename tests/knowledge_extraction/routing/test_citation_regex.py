"""Regression test for B-019: citation regex must accept suffixed PMCIDs.

Pre-fix `_CITATION_RE = r"^S\\d+\\|PMC\\d+\\|\\d+$"` rejected suffixed
document IDs (`PMC10100421_HIS-82-393`, `PMC7150310_main`, etc.) used as
opaque doc keys. The router classified every voter as UNUSABLE, silently
reducing the cascade to single-voter at L1.

Both `routing/schema_validator.py` and `routing/provenance_validator.py`
hold their own copy of the regex; this test exercises both.
"""
from __future__ import annotations

import pytest

from pipeline.stages.knowledge_extraction.routing.schema_validator import _CITATION_RE as SCHEMA_RE
from pipeline.stages.knowledge_extraction.routing.provenance_validator import _CITATION_RE as PROV_RE


@pytest.mark.parametrize("cite", [
    "S1|PMC123|456",                       # plain digits-only pmcid
    "S1|PMC10100421_HIS-82-393|123",       # suffixed with hyphen-separated journal id
    "S1|PMC7150310_main|456",              # suffixed with underscore-only key
    "S1|PMC4329418_his0066-0409|7",        # mixed case + digits + hyphen suffix
    "S99|PMC100|999999",                   # large te_id, small pmcid
])
def test_valid_suffixed_pmcid_passes_both_regexes(cite):
    """Every shape produced by `_format_sentences()` for a real paper must
    pass both router-side and provenance-side validators."""
    assert SCHEMA_RE.match(cite), f"SchemaValidator rejected {cite!r}"
    assert PROV_RE.match(cite), f"ProvenanceValidator rejected {cite!r}"


@pytest.mark.parametrize("cite", [
    "S1|PMC|456",          # empty pmcid token
    "S1|PMC123|",          # missing te_id
    "|PMC123|456",         # missing sentence id
    "[S1|PMC123|456]",     # bracketed (Markdown form, must NOT match — Finding.evidence is bracket-stripped)
    "S1|123|456",          # PMC prefix missing
    "S1|PMC123|abc",       # non-numeric te_id
])
def test_malformed_citation_rejected_by_both_regexes(cite):
    """Relaxed PMC token must not widen the safety net on the rest of the
    citation shape — sentence-id and text_element_id remain digits-only."""
    assert not SCHEMA_RE.match(cite), f"SchemaValidator wrongly accepted {cite!r}"
    assert not PROV_RE.match(cite), f"ProvenanceValidator wrongly accepted {cite!r}"


def test_provenance_regex_captures_three_groups():
    """ProvenanceValidator uses the captured pmcid to enforce cross-document
    equality — the capture group must remain group(2) so callers like
    `provenance_validator._validate_finding` still work."""
    m = PROV_RE.match("S3|PMC10100421_HIS-82-393|789")
    assert m is not None
    assert m.group(1) == "3"
    assert m.group(2) == "PMC10100421_HIS-82-393"
    assert m.group(3) == "789"
