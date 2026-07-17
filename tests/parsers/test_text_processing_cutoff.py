"""B-042 regression: `_is_cut_off` must stitch paragraphs ending in
mid-sentence abbreviations like "fig.", "et al.", "e.g.", "i.e.".

Pre-fix the period early-return shadowed the `_MID_SENTENCE_ABBREVS` check,
so every abbreviation in that frozenset (all period-terminal in the wild)
was dead code — paragraphs got fragmented at abbreviation boundaries.
"""
import pytest

from nlp_histo.parsers.text_processing import ContextAwareStitcher


@pytest.fixture
def stitcher():
    return ContextAwareStitcher()


@pytest.mark.parametrize(
    "text",
    [
        "As shown in fig.",
        "See figs.",
        "Smith et al.",
        "Tumor cells vs.",
        "approx.",
        "high expression, e.g.",
        "i.e.",
        "see ref.",
        "see refs.",
        "cf.",
        "in dept.",
        "panel no.",
        "panels nos.",
    ],
)
def test_b042_abbrev_terminal_text_is_cut_off(stitcher, text):
    assert stitcher._is_cut_off(text) is True, (
        f"Expected {text!r} to be treated as incomplete (abbreviation)"
    )


@pytest.mark.parametrize(
    "text",
    [
        "The patient survived.",
        "What is the diagnosis?",
        "Striking result!",
        "see (Smith 2020)",
        'as noted "below"',
        "ended in markup]",
    ],
)
def test_b042_real_sentence_endings_not_cut_off(stitcher, text):
    assert stitcher._is_cut_off(text) is False, (
        f"Expected {text!r} to be treated as sentence-final"
    )


def test_b042_connector_endings_still_trigger(stitcher):
    assert stitcher._is_cut_off("driven by the") is True
    assert stitcher._is_cut_off("associated with") is True


def test_b042_hyphen_and_comma_still_trigger(stitcher):
    assert stitcher._is_cut_off("immunohisto-") is True
    assert stitcher._is_cut_off("CD30 expression,") is True
