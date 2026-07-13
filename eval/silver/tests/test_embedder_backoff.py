"""GeminiEmbedder retry-with-backoff (429 / transient 5xx) — no network."""
from __future__ import annotations

import pytest

from eval.silver.matching.embedders import _is_retryable, _retry_with_backoff


def _no_sleep(_):  # never actually wait in tests
    return None


def test_is_retryable_classifies_codes_and_messages():
    assert _is_retryable(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert _is_retryable(RuntimeError("503 Service Unavailable"))
    assert not _is_retryable(RuntimeError("400 INVALID_ARGUMENT"))
    assert not _is_retryable(ValueError("bad input"))

    class _E(Exception):
        code = 429
    assert _is_retryable(_E("opaque"))


def test_retries_then_succeeds():
    calls = {"n": 0}
    def call():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return ["ok"]
    out = _retry_with_backoff(call, attempts=5, base=0.0, sleep=_no_sleep)
    assert out == ["ok"] and calls["n"] == 3


def test_non_retryable_reraises_immediately():
    calls = {"n": 0}
    def call():
        calls["n"] += 1
        raise RuntimeError("400 INVALID_ARGUMENT")
    with pytest.raises(RuntimeError, match="INVALID_ARGUMENT"):
        _retry_with_backoff(call, attempts=5, base=0.0, sleep=_no_sleep)
    assert calls["n"] == 1   # not retried


def test_gives_up_after_attempts_and_reraises():
    calls = {"n": 0}
    def call():
        calls["n"] += 1
        raise RuntimeError("429")
    with pytest.raises(RuntimeError, match="429"):
        _retry_with_backoff(call, attempts=3, base=0.0, sleep=_no_sleep)
    assert calls["n"] == 3   # exhausted all attempts
