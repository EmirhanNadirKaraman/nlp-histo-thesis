#!/usr/bin/env python3
"""E13 API step 2 — submit the request JSONL to the Anthropic Message Batches API.

Reads the request file written by ``prepare_api_batch_jsonl`` and submits it as one
Message Batch. Appends the batch metadata (batch_id + custom_ids + timestamp) to
``batch_states.json``. Does NOT do synchronous per-request generation.

  python -m eval.silver.relation_pairs.submit_api_batch            # submit all 10
  python -m eval.silver.relation_pairs.submit_api_batch --retry    # only failed custom_ids

``--retry`` reads ``failed_custom_ids.json`` (written by collect) and submits a new
batch containing ONLY those requests — retries are scoped per failed custom_id.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.silver.relation_pairs.api_common import (  # noqa: E402
    BATCH_STATES,
    FAILED_IDS,
    REQUEST_JSONL,
    _load_json,
    get_client,
)


def _read_requests() -> list[dict]:
    if not REQUEST_JSONL.exists():
        print(f"Request file not found: {REQUEST_JSONL}\nRun prepare_api_batch_jsonl first.")
        raise SystemExit(1)
    return [json.loads(ln) for ln in REQUEST_JSONL.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Submit E13 relation-pair requests to Message Batches (step 2).")
    ap.add_argument("--retry", action="store_true",
                    help="submit only the custom_ids in failed_custom_ids.json")
    args = ap.parse_args()

    requests = _read_requests()

    if args.retry:
        failed = set(_load_json(FAILED_IDS, []))
        if not failed:
            print("--retry: failed_custom_ids.json is empty or missing — nothing to retry.")
            return 0
        requests = [r for r in requests if r["custom_id"] in failed]
        if not requests:
            print(f"--retry: no requests in the JSONL match failed ids {sorted(failed)}.")
            return 1
        print(f"--retry: resubmitting {len(requests)} failed request(s): "
              f"{sorted(r['custom_id'] for r in requests)}")

    client = get_client()
    batch = client.beta.messages.batches.create(requests=requests)

    states = _load_json(BATCH_STATES, [])
    states.append({
        "batch_id": batch.id,
        "custom_ids": [r["custom_id"] for r in requests],
        "submitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "retry": bool(args.retry),
    })
    BATCH_STATES.parent.mkdir(parents=True, exist_ok=True)
    BATCH_STATES.write_text(json.dumps(states, indent=2))

    print(f"Submitted batch {batch.id} with {len(requests)} request(s). "
          f"State → {BATCH_STATES.relative_to(_REPO_ROOT)}")
    print("Next: python -m eval.silver.relation_pairs.collect_api_batch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
