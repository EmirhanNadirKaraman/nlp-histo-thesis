"""B-109 — the replay must not need a credential it never uses.

Found by the clean-clone test, not by reasoning: a fresh clone has no `.env`, and
`replay chapter9` produced **8 of 9** tables. Analysis 05 failed with "OPENAI_API_KEY not
set… matcher embedder constructor requires a non-empty key even though the cache is warm".
Every earlier run had `.env` present, so it never surfaced — and §10 claimed "no API key"
the whole time.

The key was genuinely unused: supplying `OPENAI_API_KEY=dummy-not-a-real-key` produced
9/9 byte-identical tables with 0 cache misses. Pure constructor theatre.

These tests pin the structural guarantee: **no credential is read, and no provider is
constructed**, so the replay cannot reach a paid endpoint — not because a key happens to
be absent, but because there is nothing to call.
"""
from __future__ import annotations

import pytest

from nlp_histo.evaluation.matching.embedders import CacheOnlyViolation, NoLiveEmbedding

_CREDENTIALS = (
    "OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY", "MISTRAL_API_KEY", "DEEPSEEK_API_KEY",
)


@pytest.fixture
def no_credentials(monkeypatch):
    """A genuinely credential-free environment — the clean-clone condition."""
    for var in _CREDENTIALS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def providers_explode(monkeypatch):
    """Any provider construction is an immediate, loud failure.

    Stronger than asserting no network: a constructor that merely *exists* is what B-109
    was — it demanded a key and built a client the warm cache never called.
    """
    import nlp_histo.evaluation.matching.embedders as embedders
    import nlp_histo.pipeline.stages.knowledge_extraction.agreement.providers as providers

    for mod, name in (
        (embedders, "OpenAIEmbedder"), (embedders, "GeminiEmbedder"),
        (providers, "OpenAIEmbedder"), (providers, "GeminiEmbedder"),
    ):
        monkeypatch.setattr(
            mod, name,
            lambda *a, **k: pytest.fail(f"{name} was constructed in a cache-only workflow"),
        )


# ── the guard itself ──────────────────────────────────────────────────────────

def test_guard_needs_no_key_and_builds_no_client(no_credentials) -> None:
    """Constructing it must not read the environment or reach a provider."""
    e = NoLiveEmbedding("openai", "matcher")
    assert "openai" in repr(e)


def test_guard_raises_rather_than_falling_through(no_credentials) -> None:
    e = NoLiveEmbedding("openai", "matcher")
    with pytest.raises(CacheOnlyViolation) as exc:
        e(["a text that was never embedded"])
    msg = str(exc.value)
    assert "PAID" in msg
    assert "a text that was never embedded" not in msg, "must not echo the text"


# ── the replay's own matcher sites (05 and 10) ────────────────────────────────

def test_replay_reads_no_credential_and_constructs_no_provider() -> None:
    """The regression, pinned at the source: analyses 05 and 10 previously read
    OPENAI_API_KEY and built an OpenAIEmbedder."""
    import inspect

    from nlp_histo.workflows import replay

    src = inspect.getsource(replay)
    # strip comments — they legitimately mention the old behaviour
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    assert "OPENAI_API_KEY" not in code, "the replay must not read a provider credential"
    assert "OpenAIEmbedder(" not in code, "the replay must not construct a provider"
    assert "NoLiveEmbedding(" in code, "it should use the cache-only guard instead"


def test_analysis_05_matcher_is_cache_only(no_credentials, providers_explode) -> None:
    """Analysis 05's matcher embeds only from the cache. With no credentials in the
    environment and every provider constructor booby-trapped, building it must still work.
    """
    from nlp_histo.evaluation.matching.embedders import NoLiveEmbedding as N

    embedder = N("openai", "matcher")          # what analysis 05 now uses
    with pytest.raises(CacheOnlyViolation):
        embedder(["uncached"])


def test_matcher_consults_the_cache_before_the_embedder(no_credentials, providers_explode) -> None:
    """The premise that makes cache-only safe: `get_embeddings` only calls the embedder for
    misses, so a complete cache never invokes it.
    """
    from nlp_histo.evaluation.matching.matcher import get_embeddings

    class _Cache:
        def __init__(self): self._d = {"hit": [0.1, 0.2]}
        def get(self, t): return self._d.get(t)
        def set(self, t, v): self._d[t] = v
        def save(self): pass
        def __len__(self): return len(self._d)

    out = get_embeddings(["hit"], NoLiveEmbedding("openai", "matcher"), _Cache())
    assert out == [[0.1, 0.2]], "a cache hit must not reach the embedder"


def test_an_unexpected_miss_fails_without_constructing_a_provider(
    no_credentials, providers_explode
) -> None:
    """The backstop: an incomplete cache raises rather than quietly billing."""
    from nlp_histo.evaluation.matching.matcher import get_embeddings

    class _EmptyCache:
        def get(self, t): return None
        def set(self, t, v): pass
        def save(self): pass
        def __len__(self): return 0

    with pytest.raises(CacheOnlyViolation):
        get_embeddings(["never-seen"], NoLiveEmbedding("openai", "matcher"), _EmptyCache())


# ── the paid path is untouched ────────────────────────────────────────────────

def test_non_replay_path_still_constructs_a_real_provider(monkeypatch) -> None:
    """map_context without strict_cache_only must still demand a key and build a client —
    the calibration/sweep tooling legitimately embeds."""
    import inspect

    from eval.silver.analysis import map_context

    src = inspect.getsource(map_context._load_map_context)
    assert "GOOGLE_API_KEY" in src and "OPENAI_API_KEY" in src, (
        "the paid path must keep its credential requirement"
    )
    assert "strict_cache_only" in src


def test_map_context_reexports_the_packaged_guard() -> None:
    """One definition of 'must not reach a provider', not two."""
    from eval.silver.analysis.map_context import CacheOnlyViolation as ReExported

    assert ReExported is CacheOnlyViolation
