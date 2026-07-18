"""
Google Gemini Batch API client (direct API, GOOGLE_API_KEY).

Submits all requests as a single inline batch job via google.genai.
Uses response_mime_type='application/json' so the model returns JSON
that parse_result() can handle like any other batch provider.
"""
from __future__ import annotations

import logging
import os

from .models import BatchRequest, BatchResult, ProviderJob

logger = logging.getLogger(__name__)

_ID_SEP = "|||"  # separates job name from custom_id list in output_location


class GeminiBatchProvider:
    """Submit and retrieve batches via the Gemini Batch API (google.genai)."""

    def __init__(self) -> None:
        import google.genai as genai
        self._client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def submit(self, requests: list[BatchRequest], openai_tool: dict) -> ProviderJob:  # noqa: ARG002
        if not requests:
            raise ValueError("Cannot submit an empty batch.")

        models = {r.model for r in requests}
        if len(models) > 1:
            raise ValueError(
                f"GeminiBatchProvider requires all requests to use the same model; "
                f"got: {models}. Use separate VoterBatchConfig entries per model."
            )
        model = requests[0].model

        # Strict-mode parity note (Issue D audit, 2026-05-15):
        # OpenAI, Anthropic, Vertex Gemini, and Azure all pass `openai_tool`
        # to their submit endpoint so the API enforces the AuditableSummary
        # schema at generation time. The direct Gemini batch API
        # (google.genai) supports `response_schema` / `response_json_schema`
        # in GenerateContentConfig, but the OpenAI strict schema uses
        # `anyOf: [X, null]` for nullable fields which Gemini's stricter
        # validator historically rejects. We therefore submit with
        # `response_mime_type='application/json'` only — the model returns
        # JSON shape but format is not API-enforced. Downstream:
        # `batch.dispatch.parse_result` runs Pydantic validation and logs
        # any failure to `logs/bad_findings.jsonl` (per the 2026-05-15
        # AuditableSummary parse-error logging fix), so malformed Gemini
        # outputs ARE observable — they just don't get rejected at the
        # API boundary. If yield disparity vs OpenAI/Anthropic becomes
        # material, revisit by stripping `anyOf` from the schema before
        # passing it through.
        logger.info(
            "Gemini batch: submitting %d requests without API-side schema "
            "enforcement (model=%s) — relying on Pydantic validation in "
            "parse_result(). See Issue D audit for details.",
            len(requests), model,
        )

        inline_requests = []
        custom_ids = []
        for req in requests:
            system_content: str | None = None
            user_messages: list[dict] = []
            for msg in req.messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    user_messages.append(msg)

            config: dict = {
                "response_mime_type": "application/json",
                "temperature": req.temperature,
                "max_output_tokens": req.max_tokens,
            }
            if system_content:
                config["system_instruction"] = system_content

            gemini_req: dict = {
                "contents": [
                    {"role": "user", "parts": [{"text": m["content"]}]}
                    for m in user_messages
                ],
                "config": config,
                # B-081: tag every request with its custom_id so retrieve() can
                # match responses by KEY, not by list position. Gemini's
                # inlined_responses are not guaranteed to preserve src order or
                # to include a slot for every request, so positional recovery
                # silently grafts one paper's output onto another. InlinedRequest
                # /InlinedResponse both carry a `metadata` field that round-trips.
                "metadata": {"custom_id": req.custom_id},
            }

            inline_requests.append(gemini_req)
            custom_ids.append(req.custom_id)

        batch_job = self._client.batches.create(
            model=model,
            src=inline_requests,
            config={"display_name": f"nlp-histo-{model}"},
        )
        logger.info(
            "Gemini batch submitted: %s  (%d requests, model=%s)",
            batch_job.name, len(requests), model,
        )
        return ProviderJob(
            provider="gemini",
            job_id=batch_job.name,
            status="submitted",
            model=model,
            request_count=len(requests),
            output_location=f"{batch_job.name}{_ID_SEP}{','.join(custom_ids)}",
        )

    def check(self, job: ProviderJob) -> ProviderJob:
        batch_job = self._client.batches.get(name=job.job_id)
        state = batch_job.state.name
        if state == "JOB_STATE_SUCCEEDED":
            job.status = "completed"
        elif state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
            job.status = "failed"
        else:
            job.status = "in_progress"
        return job

    @staticmethod
    def _custom_id_from_metadata(inline_response) -> str | None:
        """Pull the round-tripped ``custom_id`` from an InlinedResponse's
        ``metadata`` (B-081). Returns None when absent so the caller can fall
        back to positional. Handles both dict and attr-style metadata."""
        md = getattr(inline_response, "metadata", None)
        if md is None:
            return None
        if isinstance(md, dict):
            cid = md.get("custom_id")
        else:
            cid = getattr(md, "custom_id", None)
            if cid is None and hasattr(md, "get"):
                cid = md.get("custom_id")
        return str(cid) if cid else None

    def retrieve(self, job: ProviderJob) -> list[BatchResult]:
        _, ids_str = job.output_location.split(_ID_SEP, 1)
        custom_ids = ids_str.split(",")

        batch_job = self._client.batches.get(name=job.job_id)
        inline_responses = batch_job.dest.inlined_responses or []

        # B-081 guard: a count mismatch GUARANTEES positional recovery is wrong
        # (the old code silently stamped `unknown_{i}` and grafted cross-paper
        # output). Fail loud — the caller's retry/finalize path treats this as a
        # provider error rather than persisting mislabelled results.
        if len(inline_responses) != len(custom_ids):
            raise RuntimeError(
                f"Gemini batch {job.job_id}: response/request count mismatch "
                f"({len(inline_responses)} responses vs {len(custom_ids)} "
                f"requests) — cannot safely map custom_ids (B-081)."
            )

        n_positional = 0
        results: list[BatchResult] = []
        for i, inline_response in enumerate(inline_responses):
            # Prefer the round-tripped metadata key; fall back to positional
            # Only when metadata is unavailable, and count it so the fallback is
            # never silent (the original B-081 failure mode).
            custom_id = self._custom_id_from_metadata(inline_response)
            if custom_id is None:
                custom_id = custom_ids[i] if i < len(custom_ids) else f"unknown_{i}"
                n_positional += 1
            if inline_response.error:
                results.append(BatchResult(
                    custom_id=custom_id, content=None, error=str(inline_response.error)
                ))
            elif inline_response.response:
                try:
                    content = inline_response.response.text
                    um = getattr(inline_response.response, "usage_metadata", None)
                    results.append(BatchResult(
                        custom_id=custom_id,
                        content=content,
                        input_tokens=getattr(um, "prompt_token_count", 0) or 0,
                        output_tokens=getattr(um, "candidates_token_count", 0) or 0,
                    ))
                except Exception as exc:
                    results.append(BatchResult(
                        custom_id=custom_id, content=None, error=str(exc)
                    ))
            else:
                results.append(BatchResult(
                    custom_id=custom_id, content=None, error="empty response"
                ))

        if n_positional:
            logger.warning(
                "Gemini batch %s: %d/%d results matched by POSITION, not metadata "
                "key — response metadata round-trip unavailable; results may be "
                "mislabelled (B-081). Verify the google-genai version echoes "
                "InlinedRequest.metadata onto InlinedResponse.metadata.",
                job.job_id, n_positional, len(inline_responses),
            )
        return results
