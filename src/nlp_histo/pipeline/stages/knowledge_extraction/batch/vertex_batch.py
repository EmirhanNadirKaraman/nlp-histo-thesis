"""
Vertex AI Gemini batch prediction client.

Requires a writable GCS bucket set via VERTEX_BATCH_GCS_BUCKET.
Input JSONL is uploaded to GCS before the job is created; output JSONL is
downloaded from GCS after the job completes.

Install: google-cloud-storage  (pip install google-cloud-storage)
"""
from __future__ import annotations

import json
import logging
import os
import uuid

from .models import BatchRequest, BatchResult, ProviderJob

logger = logging.getLogger(__name__)


class VertexGeminiBatchProvider:
    """Submit and retrieve Gemini batch prediction jobs via GCS."""

    def __init__(self, gcs_bucket: str | None = None) -> None:
        self._project = os.environ["VERTEX_PROJECT"]
        self._location = os.environ.get("VERTEX_LOCATION", "us-central1")
        raw = gcs_bucket or os.environ.get("VERTEX_BATCH_GCS_BUCKET", "")
        self._bucket = raw.removeprefix("gs://").rstrip("/")
        if not self._bucket:
            raise EnvironmentError(
                "VERTEX_BATCH_GCS_BUCKET must be set to use Vertex Gemini batch mode. "
                "Create a GCS bucket in the same project and set the env var to its name."
            )

    def submit(self, requests: list[BatchRequest], openai_tool: dict) -> ProviderJob:
        from google.cloud import storage
        from google import genai

        if not requests:
            raise ValueError("Cannot submit an empty batch.")

        run_id = uuid.uuid4().hex[:10]
        model = requests[0].model

        # Build Gemini function declaration from OpenAI tool schema
        fn = openai_tool["function"]
        gemini_tool = {
            "functionDeclarations": [{
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn["parameters"],
            }]
        }
        tool_config = {
            "functionCallingConfig": {
                "mode": "any",
                "allowedFunctionNames": [fn["name"]],
            }
        }

        # Build Vertex batch input JSONL
        lines = []
        for req in requests:
            system_text: str | None = None
            contents: list[dict] = []
            for msg in req.messages:
                if msg["role"] == "system":
                    system_text = msg["content"]
                else:
                    role = "user" if msg["role"] == "user" else "model"
                    contents.append({"role": role, "parts": [{"text": msg["content"]}]})

            request_body: dict = {
                "contents": contents,
                "tools": [gemini_tool],
                "toolConfig": tool_config,
            }
            if system_text:
                request_body["systemInstruction"] = {"parts": [{"text": system_text}]}

            lines.append(json.dumps({"custom_id": req.custom_id, "request": request_body}))

        # Upload input file to GCS
        input_blob = f"batch-inputs/{run_id}/input.jsonl"
        output_prefix = f"batch-outputs/{run_id}/"
        gcs_client = storage.Client(project=self._project)
        bucket = gcs_client.bucket(self._bucket)
        bucket.blob(input_blob).upload_from_string(
            "\n".join(lines).encode(), content_type="application/jsonl"
        )

        src_uri = f"gs://{self._bucket}/{input_blob}"
        dest_uri = f"gs://{self._bucket}/{output_prefix}"

        client = genai.Client(vertexai=True, project=self._project, location=self._location)
        batch_job = client.batches.create(
            model=model,
            src=src_uri,
            config={"dest": dest_uri},
        )
        logger.info("Vertex Gemini batch submitted: %s  (%d requests)", batch_job.name, len(requests))
        return ProviderJob(
            provider="vertex_gemini",
            job_id=batch_job.name,
            status="submitted",
            model=model,
            request_count=len(requests),
            output_location=dest_uri,
        )

    def check(self, job: ProviderJob) -> ProviderJob:
        from google import genai
        client = genai.Client(vertexai=True, project=self._project, location=self._location)
        batch = client.batches.get(name=job.job_id)
        state = str(batch.state)
        if "SUCCEEDED" in state:
            job.status = "completed"
        elif "FAILED" in state or "CANCELLED" in state:
            job.status = "failed"
        else:
            job.status = "in_progress"
        return job

    def retrieve(self, job: ProviderJob) -> list[BatchResult]:
        from google.cloud import storage

        out_uri = job.output_location or ""
        stripped = out_uri.removeprefix("gs://")
        bucket_name, prefix = stripped.split("/", 1)

        gcs_client = storage.Client(project=self._project)
        bucket = gcs_client.bucket(bucket_name)

        results: list[BatchResult] = []
        for blob in bucket.list_blobs(prefix=prefix):
            if not blob.name.endswith(".jsonl"):
                continue
            for line in blob.download_as_text().splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                custom_id = data.get("custom_id", "")
                response = data.get("response", {})
                candidates = response.get("candidates", [])
                if not candidates:
                    results.append(BatchResult(
                        custom_id=custom_id, content=None, error="no candidates"
                    ))
                    continue
                parts = candidates[0].get("content", {}).get("parts", [])
                fn_call = next(
                    (p["functionCall"] for p in parts if "functionCall" in p), None
                )
                if fn_call:
                    content = json.dumps(fn_call.get("args", {}))
                else:
                    content = next(
                        (p.get("text", "") for p in parts if "text" in p), ""
                    )
                results.append(BatchResult(custom_id=custom_id, content=content))

        return results
