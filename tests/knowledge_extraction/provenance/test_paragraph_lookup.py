from __future__ import annotations

import logging

from nlp_histo.pipeline.stages.knowledge_extraction.models import (
    CanonicalRule,
    DirectionEnum,
    RelationTypeEnum,
)
from nlp_histo.pipeline.stages.knowledge_extraction.provenance.paragraph_lookup import (
    get_paragraph_for_rule,
    get_paragraphs_for_rules,
)


def _rule(
    *,
    text_element_id: int | None,
    canonical_id: str = "CR_test",
) -> CanonicalRule:
    """Build a minimal canonical rule for paragraph lookup tests."""
    return CanonicalRule(
        canonical_id=canonical_id,
        group_id="G_test",
        subject_entity="TP53",
        outcome_entity="Mutation",
        relation_type=RelationTypeEnum.expression,
        direction=DirectionEnum.absent,
        predicate_text="x",
        is_conflicted=False,
        study_coverage="single_study",
        category="molecular_genetics",
        supporting_pmcids=["PMC1"],
        member_normal_ids=["NF_1"],
        mean_grounding_score=0.9,
        finding_count=1,
        scope=None,
        representative_verbatim=None,
        representative_text_element_id=text_element_id,
    )


def test_get_paragraph_for_rule_returns_none_when_pointer_missing():
    """Rule without representative_text_element_id returns None without a DB call."""

    class _SessionThatShouldNotBeUsed:
        def execute(self, *_args, **_kwargs):
            raise AssertionError("Session must not be queried when pointer is None")

    rule = _rule(text_element_id=None)

    assert get_paragraph_for_rule(
        rule,
        _SessionThatShouldNotBeUsed(),
    ) is None


def test_get_paragraph_for_rule_returns_text_content_from_session():
    """Return text_content when the rule's source pointer resolves."""

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

    rule = _rule(text_element_id=42)
    session = _FakeSession()

    result = get_paragraph_for_rule(rule, session)

    assert result == "Full paragraph text from the source paper."
    assert session.queries

    sql, params = session.queries[0]
    assert "text_elements" in sql
    assert params == {"te_id": 42}


def test_get_paragraph_for_rule_returns_none_on_lookup_exception(caplog):
    """Return None and log a warning when the database lookup fails."""

    class _BrokenSession:
        def execute(self, *_args, **_kwargs):
            raise RuntimeError("database not reachable")

    rule = _rule(text_element_id=99)

    with caplog.at_level(logging.WARNING):
        result = get_paragraph_for_rule(rule, _BrokenSession())

    assert result is None
    assert any(
        "text_element_id=99" in record.message
        for record in caplog.records
    )


def test_get_paragraphs_for_rules_batches_and_dedupes():
    """Fetch unique text-element IDs in one query and map results to rules."""

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

            return _Result([
                _Row(100, "Paragraph for TE 100."),
                _Row(200, "Paragraph for TE 200."),
            ])

    rules = [
        _rule(text_element_id=100, canonical_id="CR_0"),
        _rule(text_element_id=100, canonical_id="CR_1"),
        _rule(text_element_id=200, canonical_id="CR_2"),
        _rule(text_element_id=999, canonical_id="CR_3"),
        _rule(text_element_id=None, canonical_id="CR_4"),
    ]

    result = get_paragraphs_for_rules(rules, _FakeSession())

    assert captured["ids"] == [100, 200, 999]
    assert result["CR_0"] == "Paragraph for TE 100."
    assert result["CR_1"] == "Paragraph for TE 100."
    assert result["CR_2"] == "Paragraph for TE 200."
    assert result["CR_3"] is None
    assert result["CR_4"] is None