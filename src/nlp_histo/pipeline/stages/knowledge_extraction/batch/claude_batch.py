"""
Anthropic Messages Batches API client.

Uses the direct Anthropic API (api.anthropic.com) — not Vertex AI.
Requires ANTHROPIC_API_KEY in the environment.

The Claude models running on Vertex AI use Vertex-specific version strings
(e.g. ``claude-haiku-4-5@20251001``).  This module maps them to the
corresponding Anthropic API model IDs automatically.
"""
from __future__ import annotations

import json
import logging
import os

from .models import BatchRequest, BatchResult, ProviderJob

logger = logging.getLogger(__name__)

# Maps Vertex AI version strings to Anthropic direct-API IDs.
# Only needed if using provider="vertex_claude"; currently unused because
# all Claude voters use the direct Anthropic API (ANTHROPIC_API_KEY).
_VERTEX_TO_DIRECT: dict[str, str] = {
    "claude-haiku-4-5@20251001": "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6@default": "claude-sonnet-4-6",
}


def _resolve_model(model: str) -> str:
    return _VERTEX_TO_DIRECT.get(model, model)


class AnthropicBatchProvider:
    """Submit and retrieve batches via the Anthropic Messages Batches API."""

    def __init__(self) -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def submit(self, requests: list[BatchRequest], openai_tool: dict) -> ProviderJob:
        if not requests:
            raise ValueError("Cannot submit an empty batch.")

        # Convert OpenAI tool schema → Anthropic tool schema
        fn = openai_tool["function"]
        anthropic_tool = {
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn["parameters"],
        }

        batch_reqs = []
        for req in requests:
            system_content: str | None = None
            user_messages: list[dict] = []
            for msg in req.messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    user_messages.append({"role": msg["role"], "content": msg["content"]})

            params: dict = {
                "model": _resolve_model(req.model),
                "max_tokens": req.max_tokens,
                "tools": [anthropic_tool],
                "tool_choice": {"type": "tool", "name": anthropic_tool["name"]},
                "messages": user_messages,
            }
            if system_content:
                params["system"] = system_content

            batch_reqs.append({"custom_id": req.custom_id, "params": params})

        batch = self._client.messages.batches.create(requests=batch_reqs)
        logger.info("Anthropic batch submitted: %s  (%d requests)", batch.id, len(requests))
        return ProviderJob(
            provider="claude",
            job_id=batch.id,
            status="submitted",
            model=requests[0].model,
            request_count=len(requests),
            output_location=batch.id,
        )

    def check(self, job: ProviderJob) -> ProviderJob:
        batch = self._client.messages.batches.retrieve(job.job_id)
        # processing_status: "in_progress" | "canceling" | "ended"
        if batch.processing_status == "ended":
            job.status = "completed"
        elif batch.processing_status == "canceling":
            job.status = "failed"
        else:
            job.status = "in_progress"
        return job

    def retrieve(self, job: ProviderJob) -> list[BatchResult]:
        results: list[BatchResult] = []
        for item in self._client.messages.batches.results(job.job_id):
            custom_id = item.custom_id
            if item.result.type == "succeeded":
                msg = item.result.message
                tool_use = next(
                    (b for b in msg.content if b.type == "tool_use"), None
                )
                if tool_use:
                    content = json.dumps(tool_use.input)
                else:
                    content = next(
                        (b.text for b in msg.content if hasattr(b, "text")), ""
                    )
                usage = getattr(msg, "usage", None)
                results.append(BatchResult(
                    custom_id=custom_id,
                    content=content,
                    input_tokens=getattr(usage, "input_tokens", 0) or 0,
                    output_tokens=getattr(usage, "output_tokens", 0) or 0,
                ))
            else:
                err = getattr(item.result, "error", "unknown error")
                results.append(BatchResult(
                    custom_id=custom_id, content=None, error=str(err)
                ))
        return results
