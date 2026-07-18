"""
LLM provider factories for the knowledge_extraction pipeline.

Direct-API factories
--------------------
gemini_direct_chat    — Google Gemini API (GOOGLE_API_KEY)
anthropic_direct_chat — Anthropic API (ANTHROPIC_API_KEY)
openai_direct_chat    — OpenAI API (OPENAI_API_KEY)

Each returns a LangChain-compatible chat object that drops into
voter_llms / level2_voter_llms / escalation_llm without changing pipeline code.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI

from nlp_histo.pipeline.stages.knowledge_extraction.config import DEFAULT_MAX_TOKENS

if TYPE_CHECKING:
    from langchain_anthropic import ChatAnthropic


# Direct APIs

def gemini_direct_chat(
    model: str,
    *,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    request_timeout: int = 60,
) -> ChatOpenAI:
    """Google Gemini via its OpenAI-compatible endpoint (GOOGLE_API_KEY)."""
    key = api_key or os.environ["GOOGLE_API_KEY"]
    return ChatOpenAI(
        model=model,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=request_timeout,
    )


def anthropic_direct_chat(
    model: str,
    *,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    # Sonnet 4.6 on dense MAP chunks can take 60–120s, longer under
    # concurrent load (chunk_workers ≥ 10 funnels multiple L3 escalations
    # at the rate limiter). 180s gives headroom for retries to actually
    # succeed instead of double-timing-out and killing the paper.
    request_timeout: int = 180,
) -> "ChatAnthropic":
    """Direct Anthropic API via ChatAnthropic (ANTHROPIC_API_KEY)."""
    from langchain_anthropic import ChatAnthropic
    key = api_key or os.environ["ANTHROPIC_API_KEY"]
    return ChatAnthropic(
        model=model,
        api_key=key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=request_timeout,
    )


def openai_direct_chat(
    model: str,
    *,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    request_timeout: int = 60,
) -> ChatOpenAI:
    """Direct OpenAI API (OPENAI_API_KEY)."""
    key = api_key or os.environ["OPENAI_API_KEY"]
    return ChatOpenAI(
        model=model,
        api_key=key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=request_timeout,
    )
