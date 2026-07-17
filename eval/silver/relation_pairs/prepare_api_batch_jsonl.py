#!/usr/bin/env python3
"""E13 API step 1 — write the Message-Batches request JSONL (no network).

Writes ``eval/data/relation_pairs/api_requests/relation_pairs_batch_requests.jsonl``,
one line per batch request (10 lines):
  * custom_id = relation_batch_01 … relation_batch_10
  * params    = claude-opus-4-7, max_tokens=12000, forced tool_use (structured
                output), the batch's 30-pair prompt. NO temperature (deprecated
                for this model, B-072).

  python -m eval.silver.relation_pairs.prepare_api_batch_jsonl
"""
from __future__ import annotations

import hashlib
import json

from eval.paths import REPO_ROOT as _REPO_ROOT  # noqa: E402

from eval.silver.relation_pairs.api_common import (  # noqa: E402
    MAX_TOKENS,
    MODEL,
    N_BATCHES,
    REQUEST_JSONL,
    SYSTEM_PROMPT,
    TOOL_NAME,
    build_request,
    custom_id_for,
)


def main() -> None:
    REQUEST_JSONL.parent.mkdir(parents=True, exist_ok=True)
    requests = [build_request(k) for k in range(1, N_BATCHES + 1)]
    with open(REQUEST_JSONL, "w", encoding="utf-8") as fh:
        for req in requests:
            fh.write(json.dumps(req, ensure_ascii=False) + "\n")

    sys_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]
    print(f"E13 API request file → {REQUEST_JSONL.relative_to(_REPO_ROOT)}")
    print(f"  {len(requests)} requests  model={MODEL}  max_tokens={MAX_TOKENS}  "
          f"tool={TOOL_NAME}  (no temperature — deprecated for this model, B-072)")
    print(f"  custom_ids: {custom_id_for(1)} … {custom_id_for(N_BATCHES)}")
    print(f"  system-prompt hash: {sys_hash}")
    print("\nNext: python -m eval.silver.relation_pairs.submit_api_batch")


if __name__ == "__main__":
    main()
