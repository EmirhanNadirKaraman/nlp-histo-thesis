"""B-107 — a UMLS failure must stop the run, not shade the numbers.

The bug: with no network, scispaCy's linker load fails, ``get_nlp()`` swallows it and
returns ``None``, ``normalize_stage`` caches an empty CUI, and the chapter-9 replay
exits 0 having written plausible-but-wrong 06/12 tables. The only signal was one
WARNING line.

These tests reproduce the failure with the **real** exception a machine with no DNS
raises — a ``requests.exceptions.ConnectionError`` wrapping urllib3's
``NameResolutionError`` over a ``socket.gaierror`` — manufactured by pointing
``socket.getaddrinfo`` at a ``gaierror`` and letting ``requests`` fail for real. The
original investigation used a custom ``RuntimeError`` guard, which library fallback
paths (``except (ConnectionError, OSError)``) never catch, and which therefore
manufactured a failure that did not match reality. A RuntimeError-only mock would
re-introduce exactly that blind spot.
"""
from __future__ import annotations

import socket

import pytest

from nlp_histo.pipeline.stages.knowledge_extraction.entities import umls_resources
from nlp_histo.pipeline.stages.knowledge_extraction.entities.umls_resources import (
    UmlsUnavailableError,
    require_umls,
)

_S3_ARTIFACT = (
    "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/"
    "2023-04-23/umls/tfidf_vectors_sparse.npz"
)


def real_offline_connection_error() -> Exception:
    """The genuine exception chain scispaCy hits on a machine with no network.

    Not a hand-built stand-in: DNS is forced to fail the way a disconnected machine
    fails, and a real ``requests.head`` against scispaCy's actual S3 artifact URL is
    issued. Nothing leaves the machine — resolution dies first. Returns whatever
    requests raised, so the test binds to real library behaviour rather than a guess.
    """
    import requests

    real_getaddrinfo = socket.getaddrinfo

    def _no_dns(*_a, **_kw):
        raise socket.gaierror(8, "nodename nor servname provided, or not known")

    socket.getaddrinfo = _no_dns
    try:
        requests.head(_S3_ARTIFACT, allow_redirects=True)
    except Exception as exc:  # noqa: BLE001 — whatever requests raises IS the fixture
        return exc
    finally:
        socket.getaddrinfo = real_getaddrinfo
    raise AssertionError("expected requests.head to fail with DNS disabled")


@pytest.fixture(autouse=True)
def _reset_umls_singleton():
    umls_resources._reset_for_tests()
    yield
    umls_resources._reset_for_tests()


def test_the_offline_fixture_is_a_real_requests_error_not_a_runtimeerror() -> None:
    """Guard the guard: if this stops being the real chain, the tests below are theatre."""
    import requests

    exc = real_offline_connection_error()
    assert isinstance(exc, requests.exceptions.ConnectionError)
    assert not isinstance(exc, RuntimeError) or isinstance(exc, requests.exceptions.RequestException)
    assert "s3-us-west-2.amazonaws.com" in str(exc)


def _fail_load_offline(monkeypatch) -> None:
    """Make the linker load fail exactly where a disconnected machine fails.

    Faithful to the observed production path: ``spacy.load`` **succeeds** —
    ``en_core_sci_lg`` is an installed package needing no network — and the failure
    happens at ``add_pipe("scispacy_linker")``, where scispaCy fetches the UMLS KB and
    issues the ETag ``requests.head``. The real log shows exactly that ordering
    ("UMLS: loading en_core_sci_lg" immediately followed by "linker unavailable").

    Failing ``spacy.load`` instead would be a *different* bug: requests' ConnectionError
    subclasses OSError, so get_nlp()'s ``except OSError: "not installed, trying next"``
    would swallow it and report "No scispaCy model found", losing the real cause.
    """
    exc = real_offline_connection_error()

    class _StubNlp:
        def add_pipe(self, *_a, **_kw):
            raise exc

        def get_pipe(self, *_a, **_kw):  # pragma: no cover — add_pipe raises first
            raise AssertionError("get_pipe must not be reached once add_pipe fails")

    import spacy

    monkeypatch.setattr(spacy, "load", lambda *_a, **_kw: _StubNlp())


def test_get_nlp_still_returns_none_on_failure(monkeypatch) -> None:
    """The live pipeline's deliberate degradation contract is UNCHANGED.

    get_nlp() must keep swallowing and returning None — only require_umls() is strict.
    """
    _fail_load_offline(monkeypatch)
    assert umls_resources.get_nlp() is None
    assert umls_resources.is_available() is False
    assert "s3-us-west-2.amazonaws.com" in (umls_resources.failure_reason() or "")


def test_require_umls_raises_on_real_offline_failure(monkeypatch) -> None:
    _fail_load_offline(monkeypatch)
    with pytest.raises(UmlsUnavailableError) as exc:
        require_umls(context="The chapter-9 replay", affected_outputs=("06_x.csv",))
    assert "unavailable" in str(exc.value).lower()


def test_require_umls_error_is_actionable(monkeypatch) -> None:
    """The message must explain what broke, what it affects, and why a warm cache
    does not save you — that last point is the counter-intuitive one (B-107)."""
    _fail_load_offline(monkeypatch)
    with pytest.raises(UmlsUnavailableError) as exc:
        require_umls(
            context="The chapter-9 replay",
            affected_outputs=("06_exp_f_test_split.csv", "12_real_profile_grounding_polarity.csv"),
        )
    msg = str(exc.value)

    assert "06_exp_f_test_split.csv" in msg
    assert "12_real_profile_grounding_polarity.csv" in msg
    assert "s3-us-west-2.amazonaws.com" in msg          # the actual cause, replayed
    assert "etag" in msg.lower()                        # why a warm cache is not enough
    assert "Nothing has been written." in msg
    assert "not a paid" in msg.lower()                  # free fetch ≠ paid API call


def test_require_umls_is_quiet_when_available(monkeypatch) -> None:
    """The success path must not raise — a working install is unaffected."""
    monkeypatch.setattr(umls_resources, "_AVAILABLE", True)
    monkeypatch.setattr(umls_resources, "_NLP", object())
    require_umls(context="The chapter-9 replay", affected_outputs=("06_x.csv",))


def test_disabled_killswitch_reports_itself_distinctly(monkeypatch) -> None:
    """An explicit opt-out must not be reported as a mysterious network failure."""
    monkeypatch.setenv("NLP_HISTO_DISABLE_UMLS", "1")
    with pytest.raises(UmlsUnavailableError) as exc:
        require_umls(context="The chapter-9 replay")
    msg = str(exc.value)
    assert "NLP_HISTO_DISABLE_UMLS" in msg
    assert "etag" not in msg.lower(), "kill-switch is not a caching problem; don't say it is"
