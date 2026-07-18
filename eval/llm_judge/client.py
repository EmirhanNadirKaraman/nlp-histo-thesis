"""
eval.llm_judge.client — Anthropic API client for sync and batch modes.

Sync mode:  direct messages.create() with tool_use.
Batch mode: build Request objects → submit → poll → retrieve → parse tool_use.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import MODEL

logger = logging.getLogger(__name__)


@dataclass
class JudgeRequest:
    """One judge call — enough to build both sync and batch requests."""

    custom_id: str          # stable: "{task}_{cache_key}"
    cache_key: str
    task: str
    system: str
    user: str
    tool_name: str
    schema: dict
    metadata: dict = field(default_factory=dict)  # extra fields for output row


def _parse_tool_use(message: Any) -> dict:
    """Extract the tool_use input dict from a Messages API response."""
    for block in message.content:
        if block.type == "tool_use":
            return block.input  # type: ignore[return-value]
    raise ValueError(f"No tool_use block in response: {message.content}")


# Sync mode

def call_opus_sync(
    client: Any,
    request: JudgeRequest,
    model: str = MODEL,
) -> dict:
    """Single synchronous call → parsed tool_use dict."""
    msg = client.messages.create(
        model=model,
        max_tokens=2048,
        system=request.system,
        tools=[
            {
                "name": request.tool_name,
                "description": f"Return structured {request.tool_name} result",
                "input_schema": request.schema,
            }
        ],
        tool_choice={"type": "tool", "name": request.tool_name},
        messages=[{"role": "user", "content": request.user}],
    )
    return _parse_tool_use(msg)


# Batch mode

def build_batch_requests(
    requests: list[JudgeRequest],
    model: str = MODEL,
) -> list[dict]:
    """Convert JudgeRequests to Anthropic batch Request dicts."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    out = []
    for req in requests:
        out.append(
            Request(
                custom_id=req.custom_id,
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    max_tokens=2048,
                    system=req.system,
                    tools=[
                        {
                            "name": req.tool_name,
                            "description": f"Return structured {req.tool_name} result",
                            "input_schema": req.schema,
                        }
                    ],
                    tool_choice={"type": "tool", "name": req.tool_name},
                    messages=[{"role": "user", "content": req.user}],
                ),
            )
        )
    return out


def submit_batch(client: Any, batch_requests: list) -> str:
    """Submit a batch and return the batch ID."""
    batch = client.messages.batches.create(requests=batch_requests)
    logger.info("Batch submitted: id=%s  requests=%d", batch.id, len(batch_requests))
    return batch.id


def poll_batch(
    client: Any,
    batch_id: str,
    interval: int = 30,
) -> Any:
    """Poll until batch processing ends. Returns the final batch object."""
    logger.info("Polling batch %s every %ds …", batch_id, interval)
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        logger.info(
            "  %s — processing=%d succeeded=%d errored=%d expired=%d",
            batch.processing_status,
            counts.processing,
            counts.succeeded,
            counts.errored,
            counts.expired,
        )
        if batch.processing_status == "ended":
            return batch
        time.sleep(interval)


def retrieve_batch_results(client: Any, batch_id: str) -> dict[str, dict]:
    """Stream batch results and parse tool_use outputs.

    Returns {custom_id: tool_use_input_dict} for succeeded results.
    Logs errors/expired requests.
    """
    results: dict[str, dict] = {}
    errors: list[dict] = []

    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        if result.result.type == "succeeded":
            try:
                parsed = _parse_tool_use(result.result.message)
                results[cid] = parsed
            except ValueError as e:
                logger.error("Batch result %s: %s", cid, e)
                errors.append({"custom_id": cid, "error": str(e)})
        elif result.result.type == "errored":
            err_msg = str(result.result.error)
            logger.error("Batch result %s errored: %s", cid, err_msg)
            errors.append({"custom_id": cid, "error": err_msg})
        elif result.result.type == "expired":
            logger.warning("Batch result %s expired", cid)
            errors.append({"custom_id": cid, "error": "expired"})

    logger.info(
        "Batch %s: %d succeeded, %d failed/expired",
        batch_id, len(results), len(errors),
    )
    return results


def save_batch_meta(
    results_dir: Path,
    batch_id: str,
    request_count: int,
    custom_id_to_cache_key: dict[str, str],
    custom_id_to_request_meta: dict[str, dict] | None = None,
) -> None:
    """Persist batch metadata for resume support.

    custom_id_to_request_meta maps custom_id →
    {"task": ..., "cache_key": ..., "metadata": ...} so that _resume_batch
    can cache results with full request context rather than empty payloads.
    """
    meta = {
        "batch_id": batch_id,
        "request_count": request_count,
        "custom_id_to_cache_key": custom_id_to_cache_key,
        "custom_id_to_request_meta": custom_id_to_request_meta or {},
        "status": "in_progress",
    }
    path = results_dir / "batch_meta.json"
    path.write_text(json.dumps(meta, indent=2))
    logger.info("Batch metadata saved to %s", path)


def load_batch_meta(results_dir: Path) -> dict | None:
    """Load previously saved batch metadata."""
    path = results_dir / "batch_meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
