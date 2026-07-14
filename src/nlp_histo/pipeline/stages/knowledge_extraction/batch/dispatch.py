"""
Shared helpers: prompt formatting, tool schema, provider instantiation, result parsing.
"""
from __future__ import annotations

import json
import logging
import re

from langchain_core.prompts import ChatPromptTemplate

from ..models import AuditableSummary
from ..llm.prompts import _MAP_SYSTEM, _MAP_USER
from .models import BatchRequest, BatchResult, ProviderJob, VoterBatchConfig

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_MAP_PROMPT = ChatPromptTemplate([("system", _MAP_SYSTEM), ("user", _MAP_USER)])


def _add_additional_properties_false(schema: dict) -> None:
    """Recursively set ``additionalProperties: false`` on every object schema.

    Walks ``properties``, ``items``, ``anyOf``/``oneOf``/``allOf``, and
    ``$defs``/``definitions`` so nested object schemas are all covered.
    """
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object" or "properties" in schema:
        schema["additionalProperties"] = False
        for prop in schema.get("properties", {}).values():
            _add_additional_properties_false(prop)
    if "items" in schema:
        _add_additional_properties_false(schema["items"])
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(key, []):
            _add_additional_properties_false(sub)
    for defn in schema.get("$defs", {}).values():
        _add_additional_properties_false(defn)
    for defn in schema.get("definitions", {}).values():
        _add_additional_properties_false(defn)


_STRIP_KEYS = ("title", "default", "examples", "example")


def _strip_schema_keys(schema: dict) -> None:
    """Recursively drop keys that confuse OpenAI strict mode (titles/defaults/examples)."""
    if not isinstance(schema, dict):
        return
    for k in _STRIP_KEYS:
        schema.pop(k, None)
    for prop in schema.get("properties", {}).values():
        _strip_schema_keys(prop)
    if "items" in schema:
        _strip_schema_keys(schema["items"])
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(key, []):
            _strip_schema_keys(sub)
    for defn in schema.get("$defs", {}).values():
        _strip_schema_keys(defn)
    for defn in schema.get("definitions", {}).values():
        _strip_schema_keys(defn)


def _enforce_required_all_keys(schema: dict) -> None:
    """For every object schema, set ``required`` to every property key."""
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object" or "properties" in schema:
        props = schema.get("properties", {})
        if props:
            schema["required"] = list(props.keys())
        for prop in props.values():
            _enforce_required_all_keys(prop)
    if "items" in schema:
        _enforce_required_all_keys(schema["items"])
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(key, []):
            _enforce_required_all_keys(sub)
    for defn in schema.get("$defs", {}).values():
        _enforce_required_all_keys(defn)
    for defn in schema.get("definitions", {}).values():
        _enforce_required_all_keys(defn)


def validate_openai_strict_schema(schema: dict, path: str = "$") -> list[str]:
    """Recursively check OpenAI strict-mode invariants.

    Returns a list of human-readable violation messages. Empty list = valid.

    Invariants checked on every object schema:
      1. ``additionalProperties`` is explicitly ``False``.
      2. ``required`` exists and lists every key in ``properties``.
      3. ``required`` does not name any key absent from ``properties``.
    """
    errors: list[str] = []
    if not isinstance(schema, dict):
        return errors

    is_object = schema.get("type") == "object" or "properties" in schema
    if is_object:
        if schema.get("additionalProperties", None) is not False:
            errors.append(f"{path}: additionalProperties must be false")
        props = schema.get("properties", {}) or {}
        required = schema.get("required", [])
        if not isinstance(required, list):
            errors.append(f"{path}: required must be a list")
            required = []
        prop_keys = set(props.keys())
        req_keys = set(required)
        missing = prop_keys - req_keys
        if missing:
            errors.append(
                f"{path}: properties missing from required: {sorted(missing)}"
            )
        unknown = req_keys - prop_keys
        if unknown:
            errors.append(
                f"{path}: required references unknown properties: {sorted(unknown)}"
            )
        for k, prop in props.items():
            errors.extend(validate_openai_strict_schema(prop, f"{path}.{k}"))

    if "items" in schema:
        errors.extend(validate_openai_strict_schema(schema["items"], f"{path}[]"))
    for key in ("anyOf", "oneOf", "allOf"):
        for i, sub in enumerate(schema.get(key, [])):
            errors.extend(validate_openai_strict_schema(sub, f"{path}.{key}[{i}]"))
    for name, defn in schema.get("$defs", {}).items():
        errors.extend(validate_openai_strict_schema(defn, f"$defs.{name}"))
    for name, defn in schema.get("definitions", {}).items():
        errors.extend(validate_openai_strict_schema(defn, f"definitions.{name}"))
    return errors


def _build_openai_tool() -> dict:
    """Build the OpenAI strict tool schema for AuditableSummary.

    Raises ``ValueError`` locally if the produced schema violates OpenAI
    strict-mode invariants — preferable to a remote batch failure later.
    """
    from langchain_core.utils.function_calling import convert_to_openai_tool
    tool = convert_to_openai_tool(AuditableSummary)
    tool["function"]["strict"] = True
    params = tool["function"]["parameters"]
    _strip_schema_keys(params)
    _add_additional_properties_false(params)
    _enforce_required_all_keys(params)
    errors = validate_openai_strict_schema(params)
    if errors:
        raise ValueError(
            "OpenAI strict schema validation failed for AuditableSummary:\n  - "
            + "\n  - ".join(errors)
        )
    return tool


# Computed once at import time — shared across all batch providers
OPENAI_MAP_TOOL: dict = _build_openai_tool()


def format_messages(pmcid: str, chunk_id: str, text: str) -> list[dict]:
    """Render the MAP prompt into a list of OpenAI-style role/content dicts."""
    rendered = _MAP_PROMPT.invoke({"pmcid": pmcid, "chunk_id": chunk_id, "text": text})
    out = []
    for msg in rendered.messages:
        role = {"system": "system", "human": "user", "ai": "assistant"}.get(
            msg.type, msg.type
        )
        out.append({"role": role, "content": msg.content})
    return out


def build_requests(
    chunk_map: dict[str, list[dict]],
    pmcid: str,
    voter_configs: list[VoterBatchConfig],
    level: str,
    chunk_ids: list[str] | None = None,
    skip_voters: dict[str, set[int]] | None = None,
) -> list[BatchRequest]:
    """
    Create a BatchRequest for every (chunk, voter) combination.

    Parameters
    ----------
    chunk_map:   chunk_id → list of sentence dicts
    pmcid:       paper ID
    voter_configs: ordered list of voter configs for this level
    level:       "l1" | "l2" | "l3"
    chunk_ids:   subset of chunks to process; None = all chunks
    skip_voters: chunk_id → set of voter indices to SKIP submitting. Used by the
                 in-run voter dedup path when a voter slot has already been
                 satisfied from an earlier level's result (BatchHandle.synthetic_results).
    """
    from ..stages.map_stage import _format_sentences  # module-private helper

    targets = chunk_ids if chunk_ids is not None else list(chunk_map.keys())
    skip_voters = skip_voters or {}
    requests: list[BatchRequest] = []
    for chunk_id in targets:
        sentences = chunk_map[chunk_id]
        text = _format_sentences(sentences)
        messages = format_messages(pmcid, chunk_id, text)
        skip = skip_voters.get(chunk_id, set())
        for vi, cfg in enumerate(voter_configs):
            if vi in skip:
                continue
            requests.append(BatchRequest(
                custom_id=f"{pmcid}__{chunk_id}__{level}__{vi}",
                messages=messages,
                model=cfg.model,
                provider=cfg.provider,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            ))
    return requests


def parse_result(result: BatchResult, strip_thinking: bool = False) -> AuditableSummary | None:
    """Parse a BatchResult's content string into an AuditableSummary."""
    if result.error or not result.content:
        logger.warning("Batch result %s empty/error: %s", result.custom_id, result.error)
        return None

    content = result.content
    if strip_thinking:
        content = _THINK_RE.sub("", content).strip()

    # Strip markdown code fences if present
    if "```" in content:
        m = re.search(r"```(?:json)?\s*(\{.*?})\s*```", content, re.DOTALL)
        if m:
            content = m.group(1)

    try:
        return AuditableSummary.model_validate(json.loads(content))
    except Exception as exc:
        logger.warning("Failed to parse batch result %s: %s", result.custom_id, exc)
        # Surface the malformed payload to bad_findings.jsonl so we can audit
        # which voters / providers produce structurally bad output. Batch mode
        # can't retry mid-stream (would need a fresh batch submission), so
        # this is observability-only.
        from ..enum_logging import log_bad_finding  # noqa: PLC0415
        log_bad_finding(
            raw=content[:2000] if isinstance(content, str) else None,
            error=str(exc),
            context={
                "stage": "AuditableSummary",
                "custom_id": result.custom_id,
                "source": "batch.parse_result",
            },
        )
        return None


def build_providers(providers_needed: set[str]) -> dict:
    """Lazily instantiate only the required provider clients."""
    providers: dict = {}
    if "azure" in providers_needed:
        from .azure_batch import AzureBatchProvider
        providers["azure"] = AzureBatchProvider()
    if "claude" in providers_needed:
        from .claude_batch import AnthropicBatchProvider
        providers["claude"] = AnthropicBatchProvider()
    if "gemini" in providers_needed:
        from .gemini_batch import GeminiBatchProvider
        providers["gemini"] = GeminiBatchProvider()
    if "vertex_gemini" in providers_needed:
        from .vertex_batch import VertexGeminiBatchProvider
        providers["vertex_gemini"] = VertexGeminiBatchProvider()
    if "openai" in providers_needed:
        from .openai_batch import OpenAIBatchProvider
        providers["openai"] = OpenAIBatchProvider()
    return providers


def submit_level(
    chunk_map: dict[str, list[dict]],
    pmcid: str,
    voter_configs: list[VoterBatchConfig],
    level: str,
    chunk_ids: list[str] | None = None,
    skip_voters: dict[str, set[int]] | None = None,
) -> list[ProviderJob]:
    """Build requests, group by provider, submit, return list of ProviderJob.

    If ``skip_voters`` removes every voter for every chunk, no jobs are
    submitted and an empty list is returned. Callers must still advance the
    handle to the next phase so the synthetic results (BatchHandle.synthetic_results)
    are picked up by the next ``advance()`` call.
    """
    all_requests = build_requests(
        chunk_map, pmcid, voter_configs, level, chunk_ids, skip_voters
    )
    if not all_requests:
        return []
    # Group by (provider, model) — NOT just provider. OpenAI's Batch API
    # rejects any batch that mixes models ("mismatched_model"); the entire
    # batch then completes with output_file_id=None and every request is
    # discarded silently. Other providers don't share this constraint today,
    # but one-batch-per-model is uniformly safe and adds at most one extra
    # batch submission per profile. See B-020 for the historical incident.
    by_provider_model: dict[tuple[str, str], list[BatchRequest]] = {}
    for req in all_requests:
        by_provider_model.setdefault((req.provider, req.model), []).append(req)

    providers = build_providers({pname for (pname, _model) in by_provider_model})
    jobs: list[ProviderJob] = []
    for (pname, _model), reqs in by_provider_model.items():
        job = providers[pname].submit(reqs, OPENAI_MAP_TOOL)
        jobs.append(job)
    return jobs
