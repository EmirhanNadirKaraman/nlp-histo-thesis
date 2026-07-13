"""Tests for is_non_retryable_llm_error."""
import pytest
from pipeline.stages.knowledge_extraction.llm.llm_errors import is_non_retryable_llm_error


class _Exc(Exception):
    def __init__(self, name_override=None, status_code=None):
        self._name_override = name_override
        self.status_code = status_code

    @property
    def __class__(self):  # type: ignore[override]
        if self._name_override:
            class _Fake(Exception):
                pass
            _Fake.__name__ = self._name_override
            return _Fake
        return type(self)


def _exc_named(class_name: str, status_code=None):
    """Create an exception whose class name matches class_name."""
    klass = type(class_name, (Exception,), {})
    obj = klass()
    if status_code is not None:
        obj.status_code = status_code
    return obj


@pytest.mark.parametrize("class_name", [
    "AuthenticationError",
    "NotFoundError",
    "InvalidRequestError",
    "BadRequestError",
    "PermissionDeniedError",
    "ContentFilterError",
    "ContentPolicyViolationError",
])
def test_non_retryable_by_class_name(class_name):
    exc = _exc_named(class_name)
    assert is_non_retryable_llm_error(exc) is True


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_non_retryable_by_status_code(status_code):
    exc = _exc_named("SomeAPIError")
    exc.status_code = status_code
    assert is_non_retryable_llm_error(exc) is True


@pytest.mark.parametrize("status_code", [429, 500, 502, 503])
def test_retryable_status_codes(status_code):
    exc = _exc_named("RateLimitError")
    exc.status_code = status_code
    assert is_non_retryable_llm_error(exc) is False


def test_retryable_plain_exception():
    assert is_non_retryable_llm_error(TimeoutError("timed out")) is False
    assert is_non_retryable_llm_error(ConnectionError("reset")) is False
    assert is_non_retryable_llm_error(ValueError("bad value")) is False


def test_quota_exhaustion_is_non_retryable():
    # OpenAI returns 429 with code=insufficient_quota for billing issues
    exc = _exc_named("RateLimitError")
    exc.status_code = 429
    # Simulate OpenAI error body in str(exc)
    exc.args = ("Error code: 429 - {'error': {'code': 'insufficient_quota'}}",)
    assert is_non_retryable_llm_error(exc) is True


def test_transient_rate_limit_is_retryable():
    exc = _exc_named("RateLimitError")
    exc.status_code = 429
    exc.args = ("Rate limit exceeded. Please try again in 1s.",)
    assert is_non_retryable_llm_error(exc) is False
