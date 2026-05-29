"""Tests for the scope-aware NLI input builder in the RELATE stage.

EXP B.2 / `docs/EXP_B2_RESULTS.md` highlighted that the pairwise NLI was
operating on impoverished input — `predicate_text` only, with scope
(disease_subtype, tissue_site, assay_method, …) stripped. That dropped
SCOPE_QUALIFY signal: a "TP53 absent in AciCCIS" vs "TP53 present in AciCC"
pair was getting CONTRADICT when the right label is SCOPE_QUALIFY (different
tumour subtypes).

Option 1 from the follow-up plan: prepend a scope-tag prefix to each rule's
NLI text. Default ON via `RelateConfig.scope_aware_nli=True`. These tests
lock the behaviour without touching the NLI model itself.
"""
from __future__ import annotations

import pytest

from pipeline.stages.summarization.current_stages.relate_stage import (
    _build_nli_text,
)
from pipeline.stages.summarization.models import (
    CanonicalRule,
    DirectionEnum,
    FindingScope,
    RelationTypeEnum,
)


def _rule(
    *,
    predicate: str,
    scope: FindingScope | None,
    subject: str = "TP53",
    outcome: str = "Mutation",
    direction: DirectionEnum | None = DirectionEnum.absent,
    verbatim: str | None = None,
    text_element_id: int | None = None,
) -> CanonicalRule:
    """Build a minimal CanonicalRule for testing the NLI text builder."""
    return CanonicalRule(
        canonical_id="CR_test",
        group_id="G_test",
        subject_entity=subject,
        outcome_entity=outcome,
        relation_type=RelationTypeEnum.expression,
        direction=direction,
        predicate_text=predicate,
        is_conflicted=False,
        study_coverage="single_study",
        category="molecular_genetics",
        supporting_pmcids=["PMC1"],
        member_normal_ids=["NF_1"],
        mean_grounding_score=0.9,
        finding_count=1,
        scope=scope,
        representative_verbatim=verbatim,
        representative_text_element_id=text_element_id,
    )


def _scope(**overrides) -> FindingScope:
    """Build a FindingScope with all fields nulled out unless overridden."""
    defaults = dict(
        disease_subtype=None,
        cohort_n=None,
        assay_method=None,
        biomarker_cutoff=None,
        tissue_site=None,
        treatment_context=None,
        endpoint=None,
        study_design=None,
        scope_parsed=True,
    )
    defaults.update(overrides)
    return FindingScope(**defaults)


# ── Legacy / opt-out path ────────────────────────────────────────────────────


def test_scope_aware_false_returns_bare_predicate():
    """`scope_aware=False` reverts to legacy: predicate_text only, no prefix."""
    rule = _rule(
        predicate="TP53 mutations absent in AciCCIS",
        scope=_scope(disease_subtype="AciCCIS", tissue_site="breast"),
    )
    text = _build_nli_text(rule, scope_aware=False)
    assert text == "TP53 mutations absent in AciCCIS"
    assert "[scope:" not in text


def test_none_scope_falls_back_to_bare_predicate():
    """Backward compat: rules from older artifacts (scope=None) skip the prefix."""
    rule = _rule(predicate="plain claim", scope=None)
    text = _build_nli_text(rule, scope_aware=True)
    assert text == "plain claim"


def test_empty_scope_fields_skip_prefix():
    """All-None scope → no prefix added (avoids empty `[scope: ]` noise)."""
    rule = _rule(predicate="plain claim", scope=_scope())  # all fields None
    text = _build_nli_text(rule, scope_aware=True)
    assert text == "plain claim"
    assert "[scope:" not in text


# ── Enriched path ────────────────────────────────────────────────────────────


def test_scope_aware_prepends_disease_subtype_tag():
    """The headline use case: disease_subtype lands in the NLI text."""
    rule = _rule(
        predicate="TP53 mutations absent",
        scope=_scope(disease_subtype="AciCCIS"),
    )
    text = _build_nli_text(rule, scope_aware=True)
    assert text.startswith("[scope: disease=AciCCIS]")
    assert text.endswith("TP53 mutations absent")


def test_scope_aware_includes_multiple_non_null_fields():
    """Multi-field scope produces a `|`-joined prefix in a stable order."""
    rule = _rule(
        predicate="TP53 mutations absent",
        scope=_scope(disease_subtype="AciCCIS", tissue_site="breast", assay_method="IHC"),
    )
    text = _build_nli_text(rule, scope_aware=True)
    # Pipe-joined, in the order the helper enumerates fields.
    assert text.startswith("[scope: disease=AciCCIS | tissue=breast | assay=IHC]")


def test_scope_aware_skips_null_fields_between_set_ones():
    """Null fields don't introduce trailing/empty separators in the prefix."""
    rule = _rule(
        predicate="claim",
        # Set disease + assay only — tissue stays None and must not appear.
        scope=_scope(disease_subtype="DLBCL", assay_method="flow_cytometry"),
    )
    text = _build_nli_text(rule, scope_aware=True)
    assert text == "[scope: disease=DLBCL | assay=flow_cytometry] claim"
    assert "tissue" not in text


def test_two_rules_differing_only_on_scope_produce_different_texts():
    """The B.2 motivating case: pairs that disagree only on disease subtype.

    Pre-fix, predicate_text would have been identical or near-identical and
    NLI saw effectively the same text on both sides → false CONTRADICT.
    With the scope prefix, the disease difference is now in the input.
    """
    rule_a = _rule(
        predicate="TP53 mutations not detected",
        scope=_scope(disease_subtype="AciCCIS"),
    )
    rule_b = _rule(
        predicate="TP53 mutations not detected",
        scope=_scope(disease_subtype="AciCC"),
    )
    text_a = _build_nli_text(rule_a, scope_aware=True)
    text_b = _build_nli_text(rule_b, scope_aware=True)
    assert text_a != text_b, (
        f"Scope-aware NLI must surface disease subtype divergence; "
        f"got identical texts:\n  a: {text_a!r}\n  b: {text_b!r}"
    )
    assert "disease=AciCCIS" in text_a
    assert "disease=AciCC" in text_b


def test_scope_aware_flag_toggles_off_at_runtime():
    """`RelateStage(scope_aware_nli=False)` must round-trip to `_build_nli_text`.

    Smoke check that the constructor stashes the flag for `relate()` to use.
    The real run plumbs `cfg.relate.scope_aware_nli` through here.
    """
    from pipeline.stages.summarization.current_stages.relate_stage import RelateStage
    s_on  = RelateStage(scope_aware_nli=True)
    s_off = RelateStage(scope_aware_nli=False)
    assert s_on._scope_aware_nli is True
    assert s_off._scope_aware_nli is False


# ── Verbatim path (use_verbatim_for_nli) ─────────────────────────────────────


def test_use_verbatim_substitutes_predicate_when_verbatim_present():
    """``use_verbatim=True`` swaps predicate_text for the verbatim sentence."""
    rule = _rule(
        predicate="TP53 absent in AciCCIS",  # the model's abstraction
        scope=None,
        verbatim="TP53 mutations were not detected in any of the 12 AciCCIS cases analysed.",
    )
    text = _build_nli_text(rule, scope_aware=False, use_verbatim=True)
    assert text.startswith("TP53 mutations were not detected")
    assert "[scope:" not in text


def test_use_verbatim_with_scope_aware_combines_both():
    """Both flags on: scope prefix + verbatim text (the richest input)."""
    rule = _rule(
        predicate="TP53 absent",
        scope=_scope(disease_subtype="AciCCIS", tissue_site="breast"),
        verbatim="TP53 mutations were not detected in 12 AciCCIS cases.",
    )
    text = _build_nli_text(rule, scope_aware=True, use_verbatim=True)
    assert text.startswith("[scope: disease=AciCCIS | tissue=breast]")
    assert text.endswith("TP53 mutations were not detected in 12 AciCCIS cases.")


def test_use_verbatim_falls_back_to_predicate_when_verbatim_missing():
    """``use_verbatim=True`` on a rule without verbatim → silent fallback."""
    rule = _rule(predicate="bare predicate only", scope=None, verbatim=None)
    text = _build_nli_text(rule, scope_aware=False, use_verbatim=True)
    assert text == "bare predicate only"


def test_use_verbatim_default_is_off_in_relate_stage():
    """The use_verbatim flag defaults False (backward-compat with measured B.2)."""
    from pipeline.stages.summarization.current_stages.relate_stage import RelateStage
    s = RelateStage()
    assert s._use_verbatim_for_nli is False


# ── Paragraph lookup helper ──────────────────────────────────────────────────


def test_get_paragraph_for_rule_returns_none_when_pointer_missing():
    """Rule without representative_text_element_id → no DB call, return None."""
    from pipeline.stages.summarization.helpers.paragraph_lookup import (
        get_paragraph_for_rule,
    )

    class _SessionThatShouldNotBeUsed:
        def execute(self, *_a, **_k):
            raise AssertionError("Session must not be queried when pointer is None")

    rule = _rule(predicate="x", scope=None, text_element_id=None)
    assert get_paragraph_for_rule(rule, _SessionThatShouldNotBeUsed()) is None


def test_get_paragraph_for_rule_returns_text_content_from_session(monkeypatch):
    """Pointer present + DB returns a row → returns the row's text_content."""
    from pipeline.stages.summarization.helpers.paragraph_lookup import (
        get_paragraph_for_rule,
    )

    class _FakeRow:
        text_content = "Full paragraph text from the source paper."

    class _FakeResult:
        def fetchone(self):
            return _FakeRow()

    class _FakeSession:
        def __init__(self):
            self.queries = []
        def execute(self, sql, params):
            self.queries.append((str(sql), params))
            return _FakeResult()

    rule = _rule(predicate="x", scope=None, text_element_id=42)
    session = _FakeSession()
    result = get_paragraph_for_rule(rule, session)
    assert result == "Full paragraph text from the source paper."
    # Confirm the SQL was parameterised, not concatenated.
    assert session.queries
    sql, params = session.queries[0]
    assert "text_elements" in sql
    assert params == {"te_id": 42}


def test_get_paragraph_for_rule_returns_none_on_lookup_exception(caplog):
    """DB exception → logged warning + return None (never raises)."""
    from pipeline.stages.summarization.helpers.paragraph_lookup import (
        get_paragraph_for_rule,
    )

    class _BrokenSession:
        def execute(self, *_a, **_k):
            raise RuntimeError("database not reachable")

    rule = _rule(predicate="x", scope=None, text_element_id=99)
    import logging as _logging
    with caplog.at_level(_logging.WARNING):
        result = get_paragraph_for_rule(rule, _BrokenSession())
    assert result is None
    assert any("text_element_id=99" in r.message for r in caplog.records)


def test_get_paragraphs_for_rules_batches_and_dedupes(monkeypatch):
    """Batch variant: one SQL call for N rules; shared text_element_id is reused."""
    from pipeline.stages.summarization.helpers.paragraph_lookup import (
        get_paragraphs_for_rules,
    )

    class _Row:
        def __init__(self, id_, content):
            self.id = id_
            self.text_content = content

    class _Result:
        def __init__(self, rows):
            self._rows = rows
        def fetchall(self):
            return self._rows

    captured: dict = {}

    class _FakeSession:
        def execute(self, sql, params):
            captured["sql"] = str(sql)
            captured["ids"] = sorted(params["ids"])
            # text_elements 100 and 200 exist; 999 does not.
            return _Result([
                _Row(100, "Paragraph for TE 100."),
                _Row(200, "Paragraph for TE 200."),
            ])

    rules = [
        _rule(predicate="p1", scope=None, text_element_id=100),
        _rule(predicate="p2", scope=None, text_element_id=100),  # dup pointer
        _rule(predicate="p3", scope=None, text_element_id=200),
        _rule(predicate="p4", scope=None, text_element_id=999),  # missing in DB
        _rule(predicate="p5", scope=None, text_element_id=None), # no pointer
    ]
    # Distinct canonical_ids so the map doesn't collapse.
    for i, r in enumerate(rules):
        r.canonical_id = f"CR_{i}"

    result = get_paragraphs_for_rules(rules, _FakeSession())

    # Dedup: 3 unique te_ids fetched (100, 200, 999), not 5.
    assert captured["ids"] == [100, 200, 999]
    assert result["CR_0"] == "Paragraph for TE 100."
    assert result["CR_1"] == "Paragraph for TE 100."   # same pointer → same text
    assert result["CR_2"] == "Paragraph for TE 200."
    assert result["CR_3"] is None                       # DB had no row
    assert result["CR_4"] is None                       # no pointer at all
