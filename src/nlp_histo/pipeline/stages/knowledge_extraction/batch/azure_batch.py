"""
Azure AI Foundry batch API client.

Uses the OpenAI-compatible batch endpoint.  The same credentials and endpoint
URL as the synchronous provider are used (AZURE_FOUNDRY_ENDPOINT / _API_KEY).
"""
from __future__ import annotations

import json
import logging
import os

from .models import BatchRequest, BatchResult, ProviderJob

logger = logging.getLogger(__name__)


class AzureBatchProvider:
    """Submit and retrieve batches via Azure AI Foundry's OpenAI-compatible batch API."""

    def __init__(self) -> None:
        from openai import OpenAI
        endpoint = os.environ["AZURE_FOUNDRY_ENDPOINT"].rstrip("/")
        api_key = os.environ["AZURE_FOUNDRY_API_KEY"]
        api_version = os.environ.get("AZURE_FOUNDRY_API_VERSION", "2025-04-01-preview")
        self._client = OpenAI(
            base_url=f"{endpoint}/models",
            api_key=api_key,
            default_query={"api-version": api_version},
            default_headers={"api-key": api_key},
        )

    def submit(self, requests: list[BatchRequest], openai_tool: dict) -> ProviderJob:
        if not requests:
            raise ValueError("Cannot submit an empty batch.")

        lines = []
        for req in requests:
            body = {
                "model": req.model,
                "messages": req.messages,
                "max_tokens": req.max_tokens,
                "tools": [openai_tool],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": openai_tool["function"]["name"]},
                },
            }
            lines.append(json.dumps({
                "custom_id": req.custom_id,
                "method": "POST",
                "url": "/chat/completions",
                "body": body,
            }))

        jsonl_bytes = "\n".join(lines).encode()
        file_obj = self._client.files.create(
            file=("batch_input.jsonl", jsonl_bytes, "application/jsonl"),
            purpose="batch",
        )
        batch = self._client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        logger.info("Azure batch submitted: %s  (%d requests)", batch.id, len(requests))
        return ProviderJob(
            provider="azure",
            job_id=batch.id,
            status="submitted",
            model=requests[0].model,
            request_count=len(requests),
        )

    def check(self, job: ProviderJob) -> ProviderJob:
        batch = self._client.batches.retrieve(job.job_id)
        job.status = batch.status  # validating | in_progress | completed | failed | cancelled
        if batch.status == "completed" and batch.output_file_id:
            job.output_location = batch.output_file_id
        return job

    def retrieve(self, job: ProviderJob) -> list[BatchResult]:
        if not job.output_location:
            raise RuntimeError(f"Job {job.job_id} has no output file yet.")

        raw = self._client.files.content(job.output_location).text
        results: list[BatchResult] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            custom_id: str = data["custom_id"]

            if data.get("error"):
                results.append(BatchResult(
                    custom_id=custom_id, content=None, error=str(data["error"])
                ))
                continue

            body = data.get("response", {}).get("body", {})
            choices = body.get("choices", [])
            if not choices:
                results.append(BatchResult(
                    custom_id=custom_id, content=None, error="no choices in response"
                ))
                continue

            msg = choices[0].get("message", {})
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                content = tool_calls[0]["function"]["arguments"]
            else:
                content = msg.get("content") or ""

            usage = body.get("usage", {})
            results.append(BatchResult(
                custom_id=custom_id,
                content=content,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            ))

        return results
