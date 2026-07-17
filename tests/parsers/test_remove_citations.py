"""Regression tests for B-043 — `remove_citations` year-stripping.

Pre-fix: `\\d+` after period/comma stripped 4-digit publication years like
"Smith et al. 2020 reported X" → "Smith et al. reported X". The post-fix
regex caps the citation-index run at 1–3 digits so years survive.
"""
from __future__ import annotations

from nlp_histo.parsers.text_processing import remove_citations


def test_year_after_period_preserved():
    text = "Smith et al. 2020 reported a new mechanism."
    assert remove_citations(text) == text


def test_year_after_comma_preserved():
    text = "Earlier work, 2019 showed no effect."
    assert remove_citations(text) == text


def test_year_range_with_endash_preserved():
    text = "Over the period 1999–2003 the rate doubled."
    assert remove_citations(text) == text


def test_citation_after_period_still_stripped():
    assert remove_citations("The study found. 12 These results") == \
        "The study found. These results"


def test_citation_range_still_stripped():
    assert remove_citations("Multiple studies. 1-3 confirmed this") == \
        "Multiple studies. confirmed this"


def test_citation_list_still_stripped():
    assert remove_citations("Several papers. 1,2,5 agree") == \
        "Several papers. agree"


def test_bracket_citations_still_stripped():
    """Bracket-style citations keep the original \\d+ pattern (unchanged)."""
    # Trailing space is fine — `remove_citations` collapses spaces internally
    # but leaves any trailing whitespace; callers strip downstream.
    assert remove_citations("Confirmed [1,2] in later work [1023]").rstrip() == \
        "Confirmed in later work"


def test_standalone_citation_run_still_stripped():
    assert remove_citations("X 1,2,3 Y") == "X Y"


def test_standalone_year_not_stripped():
    """Bare 4-digit year (no separator, no period/comma prefix) survives."""
    assert remove_citations("In 2020 the trial closed.") == \
        "In 2020 the trial closed."
