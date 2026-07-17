"""LLM error classification for the knowledge_extraction pipeline."""
from __future__ import annotations

_NON_RETRYABLE_EXCEPTION_NAMES = frozenset({
    "AuthenticationError",
    "NotFoundError",
    "InvalidRequestError",
    "BadRequestError",
    "PermissionDeniedError",
    "ContentFilterError",
    "ContentPolicyViolationError",
})

# HTTP status codes that are never worth retrying (auth / bad request / not found)
_NON_RETRYABLE_HTTP_CODES = frozenset({400, 401, 403, 404, 422})


def is_non_retryable_llm_error(exc: BaseException) -> bool:
    """Return True if *exc* should not be retried.

    Non-retryable: auth failures, invalid requests, resource not found,
    content policy violations, billing quota exhaustion.
    Retryable: transient 429 rate-limits, 5xx server errors, timeouts.
    """
    name = type(exc).__name__
    if name in _NON_RETRYABLE_EXCEPTION_NAMES:
        return True
    for attr in ("status_code", "http_status", "code"):
        code = getattr(exc, attr, None)
        if isinstance(code, int) and code in _NON_RETRYABLE_HTTP_CODES:
            return True
    # 429 is normally retryable (rate-limit), but quota-exhaustion is permanent.
    # OpenAI signals this via error.code == "insufficient_quota" in the body.
    msg = str(exc)
    if "insufficient_quota" in msg or "quota_exceeded" in msg:
        return True
    return False
