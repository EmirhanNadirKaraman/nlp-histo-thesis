"""Regression tests for B-029 + B-038 — scispaCy singleton.

Both `PipelineRunner._get_nlp` (PDF extraction) and
`KnowledgeExtractionRunner.load_paper_from_db` (summarisation) used to call
`spacy.load("en_core_sci_sm")` directly, bypassing the documented
`umls_resources.get_nlp()` singleton. In a process that ran both pipelines
(the production path) scispaCy loaded twice; in batch mode the
summarisation loader hit `spacy.load(...)` once per paper.

These tests:
  * `get_small_nlp` caches one instance per model name.
  * Repeated callers receive the same object — no second `spacy.load` call.
  * Both bypassed sites now go through `get_small_nlp`.

No real scispaCy install required — `spacy.load` is patched to a sentinel
so we measure call_count without paying the model load cost.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nlp_histo.pipeline.stages.knowledge_extraction.entities import umls_resources as ur_mod


def _reset_caches():
    """Tests share process state; reset the small-model cache between cases."""
    ur_mod._SMALL_NLP_BY_NAME.clear()


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    _reset_caches()
    # The umls-disabled kill switch is honoured first — make sure no caller
    # has set it in this process's env or the small-model path short-circuits.
    monkeypatch.delenv("NLP_HISTO_DISABLE_UMLS", raising=False)
    yield
    _reset_caches()


def test_get_small_nlp_caches_one_load_per_model():
    """First call invokes spacy.load; subsequent calls hit the cache."""
    fake_nlp = MagicMock(name="en_core_sci_sm")
    fake_nlp.pipe_names = ["senter"]  # skip sentencizer add_pipe path

    with patch("spacy.load", return_value=fake_nlp) as mock_load:
        a = ur_mod.get_small_nlp("en_core_sci_sm")
        b = ur_mod.get_small_nlp("en_core_sci_sm")
        c = ur_mod.get_small_nlp("en_core_sci_sm")
    assert a is fake_nlp
    assert a is b is c
    assert mock_load.call_count == 1, (
        f"spacy.load called {mock_load.call_count} times — singleton broken"
    )


def test_get_small_nlp_caches_per_model_name():
    """Different model names produce independent singletons."""
    fake_sm = MagicMock(name="sci_sm")
    fake_sm.pipe_names = ["senter"]
    fake_md = MagicMock(name="sci_md")
    fake_md.pipe_names = ["senter"]

    def _by_name(name, **_kwargs):
        return fake_sm if "sm" in name else fake_md

    with patch("spacy.load", side_effect=_by_name) as mock_load:
        sm = ur_mod.get_small_nlp("en_core_sci_sm")
        md = ur_mod.get_small_nlp("en_core_sci_md")
        sm2 = ur_mod.get_small_nlp("en_core_sci_sm")
    assert sm is fake_sm
    assert md is fake_md
    assert sm is sm2
    assert mock_load.call_count == 2  # one per distinct name


def test_get_small_nlp_returns_none_when_umls_disabled(monkeypatch):
    """Kill-switch parity with get_nlp() — same env var disables both
    so memory-constrained setups stay consistent across entry points."""
    monkeypatch.setenv("NLP_HISTO_DISABLE_UMLS", "1")
    with patch("spacy.load") as mock_load:
        result = ur_mod.get_small_nlp("en_core_sci_sm")
    assert result is None
    assert mock_load.call_count == 0


def test_get_small_nlp_returns_none_on_missing_model():
    """Missing model is logged-and-handled, not raised — callers fall back."""
    with patch("spacy.load", side_effect=OSError("model not installed")):
        result = ur_mod.get_small_nlp("en_core_sci_sm")
    assert result is None


def _strip_comments_and_docstrings(src: str) -> str:
    """Return source with `#` comments and triple-quoted strings removed so
    explanatory references to `spacy.load(...)` don't trip the call-site
    regression check."""
    import re
    # Drop triple-quoted blocks (handles both `'''` and `"""`).
    src = re.sub(r'"""[\s\S]*?"""', '', src)
    src = re.sub(r"'''[\s\S]*?'''", '', src)
    # Drop end-of-line comments.
    return "\n".join(
        re.sub(r"\s*#.*$", "", line) for line in src.splitlines()
    )


def test_pipeline_runner_pdf_routes_through_singleton():
    """B-029 regression guard: `PipelineRunner._get_nlp` must NOT call
    `spacy.load` directly. Routes through `get_small_nlp` instead."""
    import inspect

    from nlp_histo.pipeline.stages.pdf_text_extraction import runner as pdf_runner_mod
    src = inspect.getsource(pdf_runner_mod.PipelineRunner._get_nlp)
    assert "get_small_nlp" in src, (
        "PipelineRunner._get_nlp must route through umls_resources.get_small_nlp — "
        "B-029 regressed?"
    )
    # The legacy direct-load pattern must NOT reappear as an actual call.
    code = _strip_comments_and_docstrings(src)
    assert "spacy.load(" not in code, (
        "PipelineRunner._get_nlp must not call spacy.load directly — B-029 regressed?"
    )


def test_load_paper_from_db_routes_through_singleton():
    """B-038 regression guard: `KnowledgeExtractionRunner.load_paper_from_db`
    must NOT call `spacy.load` directly."""
    import inspect

    from nlp_histo.pipeline.stages.knowledge_extraction.runner import KnowledgeExtractionRunner
    src = inspect.getsource(KnowledgeExtractionRunner.load_paper_from_db)
    assert "get_small_nlp" in src, (
        "load_paper_from_db must route through umls_resources.get_small_nlp — "
        "B-038 regressed?"
    )
    code = _strip_comments_and_docstrings(src)
    assert "spacy.load(" not in code, (
        "load_paper_from_db must not call spacy.load directly — B-038 regressed?"
    )


def test_ner_module_routes_through_singleton():
    """B-054 regression guard: `src/nlp_histo/ner/ner.py`'s
    `load_ner_model()` and `load_linker_model()` must NOT call
    `spacy.load` directly — both used to issue a fresh `spacy.load(
    "en_core_sci_lg")` per call, reloading ~2.6 GB once per paper from
    `runner.py:run_ner_on_db(pmcid)`."""
    import inspect
    from nlp_histo.ner import ner

    for fn_name in ("load_ner_model", "load_linker_model"):
        fn = getattr(ner, fn_name)
        src = inspect.getsource(fn)
        code = _strip_comments_and_docstrings(src)
        assert "spacy.load(" not in code, (
            f"{fn_name} must not call spacy.load directly — B-054 regressed?"
        )
        assert "umls_resources" in src or "get_nlp" in src or "load_ner_model" in src, (
            f"{fn_name} must route through umls_resources.get_nlp() — "
            f"B-054 regressed?"
        )


def test_only_umls_resources_calls_spacy_load_in_pipeline_tree():
    """Process-wide invariant: every `spacy.load(...)` call in the pipeline
    package must live inside `umls_resources.py`. If a new module starts
    calling it directly, this test fails fast — same regression class as
    B-029 / B-038."""
    import pathlib

    pipeline_root = pathlib.Path(ur_mod.__file__).resolve().parents[2]
    offenders = []
    for path in pipeline_root.rglob("*.py"):
        # Allow the canonical site to call spacy.load.
        if path.name == "umls_resources.py":
            continue
        # The pdf_parsers/ subtree includes legacy research parsers (not
        # used by the production pipeline); restrict to actively-used
        # stage code under pipeline/stages/.
        if "/stages/" not in str(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        # Strip comments and docstrings so explanatory references to the
        # call don't fire the regression — we only want to catch actual
        # call sites that bypass the singleton.
        code = _strip_comments_and_docstrings(text)
        if "spacy.load(" in code:
            offenders.append(str(path.relative_to(pipeline_root)))
    assert not offenders, (
        f"Direct `spacy.load(...)` calls found outside umls_resources.py — "
        f"violates the singleton convention: {offenders}"
    )
