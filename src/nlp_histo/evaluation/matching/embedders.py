"""Embedding providers for the silver eval harness."""
from __future__ import annotations

import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)

OPENAI_MODEL = "text-embedding-3-small"
GEMINI_MODEL = "gemini-embedding-001"

# HTTP statuses worth retrying: 429 rate-limit + 5xx transient server errors.
_RETRYABLE_CODES = {429, 500, 502, 503, 504}
_RETRYABLE_TOKENS = ("429", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL",
                     "500", "502", "503", "504")


def _is_retryable(exc: Exception) -> bool:
    """True for rate-limit (429 / RESOURCE_EXHAUSTED) and transient 5xx errors.

    Checks the error's status code (genai APIError carries `.code`/`.status_code`)
    and falls back to a substring match on the message, so it works regardless of
    the exact google.genai exception class.
    """
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and code in _RETRYABLE_CODES:
        return True
    msg = str(exc).upper()
    return any(tok in msg for tok in _RETRYABLE_TOKENS)


def _retry_with_backoff(
    call: Callable[[], list],
    *,
    attempts: int = 6,
    base: float = 2.0,
    cap: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    is_retryable: Callable[[Exception], bool] = _is_retryable,
) -> list:
    """Call ``call()``, retrying retryable errors with exponential backoff
    (base, 2·base, … capped at ``cap``). Re-raises non-retryable errors and the
    final failure. Default backoff (2→4→8→16→32→60, 6 attempts ≈ 2 min) is sized
    to ride out per-minute embedding rate-limit windows; a genuine daily-quota
    exhaustion will (correctly) exhaust the retries and re-raise.
    """
    delay = base
    for i in range(attempts):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 — retryability decided by _is_retryable
            if i == attempts - 1 or not is_retryable(exc):
                raise
            wait = min(delay, cap)
            logger.warning("embedding API %s — backing off %.1fs (attempt %d/%d)",
                           type(exc).__name__, wait, i + 1, attempts)
            sleep(wait)
            delay *= 2
    raise RuntimeError("unreachable")  # pragma: no cover


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = OPENAI_MODEL) -> None:
        self._api_key = api_key
        self.model = model

    def __call__(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI
        client = OpenAI(api_key=self._api_key)
        response = client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def __repr__(self) -> str:
        return f"OpenAIEmbedder(model={self.model!r})"


class GeminiEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str = GEMINI_MODEL,
        task_type: str = "SEMANTIC_SIMILARITY",
        max_retries: int = 6,
        base_delay: float = 2.0,
    ) -> None:
        self._api_key = api_key
        self.model = model
        self._task_type = task_type
        self._max_retries = max_retries
        self._base_delay = base_delay

    def __call__(self, texts: list[str]) -> list[list[float]]:
        import google.genai as genai

        def _call() -> list[list[float]]:
            client = genai.Client(api_key=self._api_key)
            result = client.models.embed_content(
                model=self.model,
                contents=texts,
                config={"task_type": self._task_type},
            )
            return [e.values for e in result.embeddings]

        # Retry 429 (RESOURCE_EXHAUSTED) + transient 5xx with exponential backoff,
        # so embedding-heavy re-runs ride out per-minute rate limits instead of crashing.
        return _retry_with_backoff(_call, attempts=self._max_retries, base=self._base_delay)

    def __repr__(self) -> str:
        return f"GeminiEmbedder(model={self.model!r})"


class CacheOnlyViolation(RuntimeError):
    """A cache-only context tried to embed text that was not in the frozen cache.

    Raised instead of calling a provider. Workflows that advertise themselves as API-free
    — the results replay, the frozen eval experiments — use :class:`NoLiveEmbedding` so
    an unexpected miss (a race, a malformed row, an incomplete preflight) fails loudly
    rather than silently spending money (B-112, B-109).
    """


class NoLiveEmbedding:
    """Stands in for an embedder where every embedding must come from the cache.

    Constructing it needs no API key and builds no client, so a workflow using it cannot
    reach a provider — the guarantee is structural rather than a matter of an unset
    environment variable or a credential that happens to be absent.

    Safe wherever the caller consults the cache first and only calls the embedder for
    misses (``matcher.embed_texts``, ``map_theta_sweep._make_cached_embed_fn``): with a
    complete cache it is never invoked, and with an incomplete one it raises instead of
    billing.
    """

    def __init__(self, embedder_kind: str, role: str) -> None:
        self._kind = embedder_kind
        self._role = role

    def __call__(self, texts):
        n = len(texts) if hasattr(texts, "__len__") else "?"
        raise CacheOnlyViolation(
            f"cache-only mode: the {self._role} embedder ({self._kind}) was asked to embed "
            f"{n} uncached text(s). Refusing — this workflow is documented as API-free, and "
            f"embedding them would issue PAID {self._kind} calls.\n"
            f"The frozen embedding cache is incomplete for these inputs; it must be "
            f"regenerated or supplied in full. No text is echoed here by design."
        )

    def __repr__(self) -> str:  # pragma: no cover — diagnostics only
        return f"NoLiveEmbedding(kind={self._kind!r}, role={self._role!r})"
