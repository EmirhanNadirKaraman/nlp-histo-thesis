#!/usr/bin/env python3
"""E13 API step 3 — poll/collect the Message Batch(es) and extract per-batch JSONL.

Reads ``batch_states.json``, polls every submitted batch, and once all have ended:
  * saves the raw API results to ``raw_results.jsonl`` (one result/line),
  * maps results BY custom_id (latest-submitted batch wins, so a ``--retry`` batch
    overrides the original),
  * for each ``relation_batch_0k`` that succeeded and was not truncated, extracts
    the structured ``pairs`` from the forced tool_use call, assigns contiguous ids
    (pair_{30k-29}..pair_{30k}), and writes ``raw_batches/batch_0k.jsonl``,
  * re-validates each written batch (30 records, 10/10/10) and records any
    hard-failed / errored / truncated / missing custom_ids to
    ``failed_custom_ids.json`` for a per-custom_id ``submit_api_batch --retry``,
  * writes truthful ``generation_meta.json`` for the merge step.

  python -m eval.silver.relation_pairs.collect_api_batch

Re-run until all batches report ``ended``; truncation (stop_reason=max_tokens) is a
hard failure (never writes a partial set).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.silver.relation_pairs.api_common import (  # noqa: E402
    BATCH_STATES,
    FAILED_IDS,
    GENERATION_META,
    N_BATCHES,
    PAIRS_PER_BATCH,
    RAW_BATCHES_DIR,
    RAW_RESULTS,
    _load_json,
    custom_id_for,
    extract_pairs_from_content,
    get_client,
    provenance_meta,
)
from eval.silver.relation_pairs.validate_and_merge import load_batch  # noqa: E402


def _assign_ids(pairs: list[dict], batch_number: int) -> list[dict]:
    start = PAIRS_PER_BATCH * (batch_number - 1) + 1
    out = []
    for i, p in enumerate(pairs):
        rec = {"id": f"pair_{start + i:04d}"}
        rec.update(p)
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Poll + collect the E13 relation-pair Message Batches.")
    ap.add_argument("--poll-interval", type=int, default=15,
                    help="seconds between status checks while waiting (default 15)")
    ap.add_argument("--no-wait", action="store_true",
                    help="check once and exit if not all batches have ended (no polling loop)")
    args = ap.parse_args()

    states = _load_json(BATCH_STATES, [])
    if not states:
        print(f"No batch state at {BATCH_STATES}. Run submit_api_batch first.")
        return 1

    client = get_client()

    # ── poll all batches until every one has ended (every --poll-interval s) ───
    while True:
        all_ended = True
        for st in states:
            b = client.beta.messages.batches.retrieve(st["batch_id"])
            c = b.request_counts
            print(f"  batch {st['batch_id']}: {b.processing_status}  "
                  f"succeeded={c.succeeded} errored={c.errored} processing={c.processing} "
                  f"canceled={c.canceled} expired={getattr(c, 'expired', 0)}")
            if b.processing_status != "ended":
                all_ended = False
        if all_ended:
            break
        if args.no_wait:
            print("Not all batches have ended — re-run to check again (--no-wait set).")
            return 0
        print(f"  … not all ended; re-checking in {args.poll_interval}s (Ctrl-C to stop)\n")
        time.sleep(args.poll_interval)

    # ── gather results, latest-submitted batch wins per custom_id ─────────────
    merged: dict[str, object] = {}
    raw_dump: list[dict] = []
    for st in states:  # submission order; later overwrites earlier
        for item in client.beta.messages.batches.results(st["batch_id"]):
            merged[item.custom_id] = item
            try:
                raw_dump.append(item.model_dump(mode="json"))
            except Exception:
                raw_dump.append({"custom_id": item.custom_id, "type": item.result.type})

    RAW_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_RESULTS, "w", encoding="utf-8") as fh:
        for d in raw_dump:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    # ── extract per batch ─────────────────────────────────────────────────────
    RAW_BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    ok = 0
    for k in range(1, N_BATCHES + 1):
        cid = custom_id_for(k)
        item = merged.get(cid)
        if item is None:
            print(f"  {cid}: MISSING from results")
            failed.append(cid)
            continue
        if item.result.type != "succeeded":
            err = getattr(item.result, "error", "unknown")
            print(f"  {cid}: {item.result.type} ({err})")
            failed.append(cid)
            continue
        msg = item.result.message
        if msg.stop_reason == "max_tokens":
            print(f"  {cid}: TRUNCATED (stop_reason=max_tokens) — increase MAX_TOKENS")
            failed.append(cid)
            continue
        pairs = extract_pairs_from_content(msg.content)
        if not pairs:
            print(f"  {cid}: no tool_use pairs in response")
            failed.append(cid)
            continue
        out_path = RAW_BATCHES_DIR / f"batch_{k:02d}.jsonl"
        with open(out_path, "w", encoding="utf-8") as fh:
            for rec in _assign_ids(pairs, k):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _, errs = load_batch(out_path, k)
        if errs:
            print(f"  {cid}: wrote {len(pairs)} pairs but failed batch validation "
                  f"({len(errs)} issue(s)) — see validate_and_merge")
            failed.append(cid)
        else:
            print(f"  {cid}: OK → {out_path.relative_to(_REPO_ROOT)} ({len(pairs)} pairs)")
            ok += 1

    FAILED_IDS.write_text(json.dumps(sorted(set(failed)), indent=2))
    batch_ids = [st["batch_id"] for st in states]
    GENERATION_META.write_text(json.dumps(provenance_meta(batch_ids), indent=2))

    print(f"\nCollected: {ok}/{N_BATCHES} batches OK.  raw → {RAW_RESULTS.relative_to(_REPO_ROOT)}")
    if failed:
        print(f"Failed custom_ids ({len(failed)}): {sorted(set(failed))}")
        print("Retry just those: python -m eval.silver.relation_pairs.submit_api_batch --retry")
        return 1
    print("All batches collected. Next: python -m eval.silver.relation_pairs.validate_and_merge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
