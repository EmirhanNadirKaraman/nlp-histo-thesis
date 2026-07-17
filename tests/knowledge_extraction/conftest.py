import pytest

from nlp_histo.pipeline.stages.knowledge_extraction.entities import umls_resources as _umls


def pytest_runtest_logstart(nodeid, location):
    """Print the test name to the terminal before each test runs."""
    print(f"\n>>> {nodeid}", flush=True)


@pytest.fixture(autouse=True)
def _unpoison_umls_singleton():
    """Stop UMLS-disabled tests from poisoning the process-wide ``get_nlp()``
    singleton for later tests.

    ``get_nlp()`` reads ``NLP_HISTO_DISABLE_UMLS`` only on its first call and caches
    the decision in ``_AVAILABLE``. A test that sets the kill-switch (via
    ``monkeypatch``) and then probes ``get_nlp()`` caches ``_AVAILABLE=False``; that
    sticks even after the env var reverts, so a later test that genuinely needs UMLS
    (e.g. ``test_normalize_stage_normalizes_entities``) silently gets ``None`` and
    fails. Deterministic file order happened to load the singleton un-disabled first;
    ``pytest-randomly`` exposes the coupling by reordering.

    After each test, clear a *disabled/failed* cache (``_AVAILABLE is False``) so the
    next real caller re-probes. A successful load (``_AVAILABLE is True``) is never
    touched — that is the expensive, legitimately-shared singleton. In production the
    env var is fixed at process start, so the cache is correct there; this is a
    test-isolation concern only.
    """
    yield
    if _umls._AVAILABLE is False:
        _umls._reset_for_tests()
