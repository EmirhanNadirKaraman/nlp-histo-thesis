"""
OpenAI Messages Batches API client.

Uses the direct OpenAI API (api.openai.com) — requires OPENAI_API_KEY.
Submits requests as a JSONL batch file and retrieves results from the
output file once the batch completes (same format as Azure AI Foundry batch).
"""
from __future__ import annotations

import json
import logging
import os

from .models import BatchRequest, BatchResult, ProviderJob

logger = logging.getLogger(__name__)


class OpenAIBatchProvider:
    """Submit and retrieve batches via the OpenAI Batch API."""

    def __init__(self) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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
                "temperature": req.temperature,
            }
            lines.append(json.dumps({
                "custom_id": req.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
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
        logger.info("OpenAI batch submitted: %s  (%d requests)", batch.id, len(requests))
        return ProviderJob(
            provider="openai",
            job_id=batch.id,
            status="submitted",
            model=requests[0].model,
            request_count=len(requests),
        )

    def check(self, job: ProviderJob) -> ProviderJob:
        batch = self._client.batches.retrieve(job.job_id)
        if batch.status == "completed":
            if batch.output_file_id:
                job.status = "completed"
                job.output_location = batch.output_file_id
            else:
                # All requests failed — treat as failed so the runner doesn't hang
                job.status = "failed"
                logger.warning(
                    "OpenAI batch %s completed but output_file_id is None "
                    "(all %d requests likely failed); error_file_id=%s",
                    job.job_id, job.request_count, batch.error_file_id,
                )
        elif batch.status in ("failed", "cancelled", "expired"):
            job.status = "failed"
        else:
            job.status = "in_progress"
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
