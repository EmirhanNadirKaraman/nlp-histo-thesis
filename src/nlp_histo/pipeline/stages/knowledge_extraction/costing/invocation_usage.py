"""Invocation-time token-usage capture for the MAP cascade.

Records are appended directly after each ``chain.invoke()`` call inside
``MapStage``, by reading ``usage_metadata`` (LangChain ≥0.3 standard field)
off the raw ``AIMessage`` returned alongside the parsed structured output.

This bypasses the LangChain ``BaseCallbackHandler`` path, which is silently
stripped when ``ChatModel.with_structured_output(...)`` is used — the
historical reason the sync runner's cost report came out as ``$0.00000``
even on real calls.

Callbacks (``UsageCollector`` / ``_make_token_callback`` in
``scripts/run_paper.py``) are kept as a *fallback* signal and a side
channel for traces; the records here are the source of truth for cost.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class InvocationUsage:
    """One LLM invocation's token usage, captured at the call site.

    ``usage_source`` is a short tag describing where the counts came from
    so downstream reports can diagnose missing data without re-reading the
    raw payload:

    * ``"usage_metadata"``      — LangChain standardised AIMessage field.
    * ``"response_metadata"``   — provider-native ``token_usage`` /
      ``usage`` dict on ``AIMessage.response_metadata``.
    * ``"missing"``             — neither was populated; warning logged.
    """
    level: str                 # "L1" | "L2" | "L3"
    provider: str              # "anthropic" | "openai" | "gemini" | ...
    model: str                 # model id as configured in voter_configs
    chunk_id: str | None
    input_tokens: int
    output_tokens: int
    usage_source: str
    extra: dict[str, Any] = field(default_factory=dict)


def extract_usage_from_message(msg) -> tuple[int, int, str]:
    """Pull ``(input_tokens, output_tokens, source)`` from an AIMessage.

    Looks in this order:

    1. ``msg.usage_metadata`` — LangChain ≥0.3 unified shape:
       ``{"input_tokens": int, "output_tokens": int, "total_tokens": int}``.
       This is what ``langchain_anthropic``, ``langchain_openai`` and
       ``langchain_google_genai`` all emit on a successful call.
    2. ``msg.response_metadata["token_usage"]`` /
       ``msg.response_metadata["usage"]`` — provider-native shape that
       some older versions still surface (OpenAI legacy uses
       ``prompt_tokens``/``completion_tokens``).

    Returns ``(0, 0, "missing")`` if both are unavailable.  Caller is
    responsible for logging context (level, provider, model, chunk).
    """
    um = getattr(msg, "usage_metadata", None)
    if isinstance(um, dict):
        inp = int(um.get("input_tokens") or 0)
        out = int(um.get("output_tokens") or 0)
        if inp or out:
            return inp, out, "usage_metadata"

    rm = getattr(msg, "response_metadata", None) or {}
    if isinstance(rm, dict):
        tu = rm.get("token_usage") or rm.get("usage") or {}
        if isinstance(tu, dict):
            inp = int(tu.get("prompt_tokens") or tu.get("input_tokens") or 0)
            out = int(tu.get("completion_tokens") or tu.get("output_tokens") or 0)
            if inp or out:
                return inp, out, "response_metadata"

    return 0, 0, "missing"


def unwrap_structured_output(result):
    """Split a ``with_structured_output(include_raw=True)`` result.

    Returns ``(parsed, raw_message, parsing_error)``.

    Accepts either:

    * The dict shape ``{"raw": AIMessage, "parsed": Schema,
      "parsing_error": ...}`` produced by
      ``llm.with_structured_output(schema, include_raw=True)``.
    * A bare parsed schema (back-compat path: chain without ``include_raw``).
      ``raw_message`` will be ``None`` in that case and the caller will
      log a usage-missing warning.
    """
    if isinstance(result, dict) and ("parsed" in result or "raw" in result):
        return result.get("parsed"), result.get("raw"), result.get("parsing_error")
    return result, None, None
