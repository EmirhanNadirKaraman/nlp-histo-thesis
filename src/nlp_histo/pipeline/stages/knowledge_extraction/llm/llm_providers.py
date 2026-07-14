"""
LLM provider factories for the knowledge_extraction pipeline.

Direct-API factories (no Azure / Vertex required)
--------------------------------------------------
gemini_direct_chat   — Google Gemini API (GOOGLE_API_KEY)
anthropic_direct_chat — Anthropic API (ANTHROPIC_API_KEY)
openai_direct_chat   — OpenAI API (OPENAI_API_KEY)

Returns LangChain-compatible ChatOpenAI objects for both Azure AI Foundry and
Vertex AI, so any model can be dropped into voter_llms / level2_voter_llms /
escalation_llm without changing pipeline code.

Environment variables
---------------------
AZURE_FOUNDRY_ENDPOINT   Base URL of the Azure AI Foundry project endpoint,
                         e.g. https://my-project.eastus.models.ai.azure.com
AZURE_FOUNDRY_API_KEY    API key for the Azure AI Foundry project.
AZURE_FOUNDRY_API_VERSION  API version (default: 2024-05-01-preview).
VERTEX_PROJECT           Google Cloud project ID.
VERTEX_LOCATION          Vertex AI region for Gemini models (default: us-central1).
CLAUDE_VERTEX_LOCATION   Vertex AI region for Claude models (default: us-east5).

Authentication
--------------
Azure Foundry  : static API key from AZURE_FOUNDRY_API_KEY.
Vertex AI      : Google Application Default Credentials (ADC).
                 Run `gcloud auth application-default login` once before use,
                 or set GOOGLE_APPLICATION_CREDENTIALS to a service-account key.
                 Tokens are refreshed automatically on each call to
                 vertex_gemini_chat() — do not reuse the returned object across
                 long-running processes without refreshing.
"""
from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI

from nlp_histo.pipeline.stages.knowledge_extraction.config import DEFAULT_MAX_TOKENS

if TYPE_CHECKING:
    from langchain_anthropic import ChatAnthropic

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class _ThinkStrippingChatOpenAI(ChatOpenAI):
    """ChatOpenAI that strips <think>...</think> blocks from response content.

    Reasoning models (e.g. DeepSeek-V3.2-Speciale on Azure AI Foundry) prepend
    an internal chain-of-thought block before the actual answer.  Stripping it
    here — before LangChain's structured-output parser sees the message —
    ensures that ``with_structured_output`` can parse the JSON cleanly.
    """

    def invoke(self, input, config=None, **kwargs):  # type: ignore[override]
        result = super().invoke(input, config=config, **kwargs)
        if isinstance(result.content, str) and "<think>" in result.content:
            cleaned = _THINK_RE.sub("", result.content).strip()
            result = result.model_copy(update={"content": cleaned})
        return result

    async def ainvoke(self, input, config=None, **kwargs):  # type: ignore[override]
        result = await super().ainvoke(input, config=config, **kwargs)
        if isinstance(result.content, str) and "<think>" in result.content:
            cleaned = _THINK_RE.sub("", result.content).strip()
            result = result.model_copy(update={"content": cleaned})
        return result


# ── Azure AI Foundry ────────────────────────────────────────────────────────────

def azure_foundry_chat(
    model: str,
    *,
    endpoint: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    strip_thinking: bool = False,
) -> ChatOpenAI:
    """
    Return a LangChain ChatOpenAI pointed at an Azure AI Foundry endpoint.

    All Azure AI Foundry serverless-API deployments expose an OpenAI-compatible
    REST interface, so ChatOpenAI works for every model family (DeepSeek,
    Mistral, kimi, Claude, etc.).

    Parameters
    ----------
    model:
        Model name as it appears in the Azure Foundry deployment
        (e.g. "DeepSeek-V3.2-Speciale", "Mistral-Large-3", "Kimi-K2.5").
    endpoint:
        Override for AZURE_FOUNDRY_ENDPOINT.
    api_key:
        Override for AZURE_FOUNDRY_API_KEY.
    strip_thinking:
        Set True for reasoning models (e.g. DeepSeek-V3.2-Speciale) that
        prepend ``<think>...</think>`` blocks to their output.  The block is
        stripped from the AIMessage content before LangChain's
        structured-output parser runs.
    """
    ep = (endpoint or os.environ["AZURE_FOUNDRY_ENDPOINT"]).rstrip("/")
    key = api_key or os.environ["AZURE_FOUNDRY_API_KEY"]
    api_version = os.environ.get("AZURE_FOUNDRY_API_VERSION", "2024-05-01-preview")
    cls = _ThinkStrippingChatOpenAI if strip_thinking else ChatOpenAI
    return cls(
        model=model,
        base_url=f"{ep}/models",
        api_key=key,
        temperature=temperature,
        default_query={"api-version": api_version},
        default_headers={"api-key": key},
        model_kwargs={"max_tokens": max_tokens},
    )


# ── Vertex AI (Gemini) ──────────────────────────────────────────────────────────

def vertex_gemini_chat(
    model: str,
    *,
    project: str | None = None,
    location: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    request_timeout: int = 20,
) -> ChatOpenAI:
    """
    Return a LangChain ChatOpenAI pointed at the Vertex AI OpenAI-compatible
    endpoint for the given Gemini model.

    Vertex AI exposes an OpenAI-compatible API at:
        https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}
            /locations/{location}/endpoints/openapi

    Authentication uses Google ADC — the token is refreshed on each call to
    this factory.  For long-running batch jobs, call this function again if the
    token approaches expiry (typically 1 hour).

    Parameters
    ----------
    model:
        Vertex AI model name, e.g.
        "gemini-2.5-flash-lite"   (location="global")
        "gemini-2.5-flash"
    project:
        Override for VERTEX_PROJECT.
    location:
        Override for VERTEX_LOCATION (default: us-central1).
    """
    import google.auth  # google-auth is in requirements.txt
    import google.auth.transport.requests

    proj = project or os.environ["VERTEX_PROJECT"]
    loc = location or os.environ.get("VERTEX_LOCATION", "us-central1")

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(google.auth.transport.requests.Request())

    base_url = (
        f"https://{loc}-aiplatform.googleapis.com/v1beta1"
        f"/projects/{proj}/locations/{loc}/endpoints/openapi"
    )

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=credentials.token,  # short-lived bearer token
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=request_timeout,
    )


# ── Vertex AI (Claude) ──────────────────────────────────────────────────────────

def claude_vertex_chat(
    model: str,
    *,
    project: str | None = None,
    location: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    request_timeout: int = 20,
) -> ChatOpenAI:
    """
    Return a LangChain ChatOpenAI pointed at the Vertex AI OpenAI-compatible
    endpoint for the given Claude model.

    Claude on Vertex AI is accessible via the same OpenAI-compat REST API as
    Gemini models, so ChatOpenAI works here too — no ChatAnthropicVertex needed.

    Authentication uses Google ADC — same credentials as vertex_gemini_chat().

    Parameters
    ----------
    model:
        Vertex AI version name, e.g.
        "claude-haiku-4-5@20251001"
        "claude-sonnet-4-6@default"
    project:
        Override for VERTEX_PROJECT.
    location:
        Override for CLAUDE_VERTEX_LOCATION (default: us-east5).
    """
    import google.auth
    import google.auth.transport.requests

    proj = project or os.environ["VERTEX_PROJECT"]
    loc = location or os.environ.get("CLAUDE_VERTEX_LOCATION", "us-east5")

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(google.auth.transport.requests.Request())

    base_url = (
        f"https://{loc}-aiplatform.googleapis.com/v1beta1"
        f"/projects/{proj}/locations/{loc}/endpoints/openapi"
    )

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=credentials.token,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=request_timeout,
    )


# ── Direct APIs (no Azure / Vertex required) ────────────────────────────────────

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
