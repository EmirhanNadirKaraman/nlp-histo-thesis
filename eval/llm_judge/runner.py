"""
eval.llm_judge.runner — Orchestrator: sampling → test dispatch → output.

Coordinates paper selection, request building, cache lookup, sync/batch
execution, result parsing, and output file writing.

Session discipline: DB queries (paper selection, request building) happen
inside a single session_scope that is closed before any cache operations.
This prevents nested session_scope calls, which would corrupt the shared
scoped_session on close.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cache import JudgeCache
from .client import (
    JudgeRequest, call_opus_sync,
    build_batch_requests, submit_batch, poll_batch,
    retrieve_batch_results, save_batch_meta, load_batch_meta,
)
from .sampling import select_papers
from .metrics import build_summary
from .tests.q1_precision import build_q1_requests, parse_q1_result
from .tests.q2_relations import build_q2_requests, parse_q2_result
from .tests.q3_recall import build_q3_requests, parse_q3_result
from .tests.q5_f1 import build_q5_requests, parse_q5_result

logger = logging.getLogger(__name__)

# Q4 (grounding threshold calibration via rejected findings) is intentionally
# deferred: SumRejectedFinding has no verbatim_support column, so there is no
# independent source evidence — evaluating a claim against itself is circular.
# Implement Q4 once SumRejectedFinding stores a proper verbatim_support reference.

_SYNC_MAX_RETRIES = 4
_SYNC_BASE_DELAY = 5.0  # seconds; doubles each attempt (5, 10, 20, 40)


@dataclass
class RunConfig:
    """All CLI flags packed into a single config."""
    mode: str = "sync"
    tests: set[str] = None  # type: ignore[assignment]
    n: int = 15
    seed: int = 42
    min_words: int = 1500
    max_words: int = 15000
    q1_findings: int = 20
    q2_relations: int = 10
    q3_with_extraction_paragraphs: int = 3
    q3_zero_extraction_paragraphs: int = 2
    q5_paragraphs: int = 5
    show_pipeline_label: bool = False
    force_refresh_cache: bool = False
    max_requests: int | None = None
    no_submit: bool = False
    dry_run: bool = False
    batch_id: str | None = None
    poll_interval: int = 30
    results_dir: Path = Path("eval/llm_judge_results")

    def __post_init__(self):
        if self.tests is None:
            self.tests = {"q1", "q2", "q3", "q5"}


def run(cfg: RunConfig) -> None:
    """Main entry point for the evaluation harness."""
    from nlp_histo.database import get_db_connection

    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    # Cache
    db = get_db_connection()
    cache = JudgeCache(db)
    if cfg.force_refresh_cache:
        cache.clear_version(tasks=cfg.tests)

    # Resume batch mode
    if cfg.batch_id:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        _resume_batch(client, cfg, cache)
        return

    # Phase 1: Paper selection + raw request building
    # All DB queries happen here. Session is closed before cache operations to
    # avoid nested session_scope calls (scoped_session shares one session per
    # thread — inner close() would corrupt the outer scope).
    with db.session_scope() as session:
        papers = select_papers(session, cfg.n, cfg.seed, cfg.min_words, cfg.max_words)
        if not papers:
            logger.error("No papers matched. Run the knowledge_extraction pipeline first.")
            sys.exit(1)

        sample_path = cfg.results_dir / "paper_sample.json"
        sample_path.write_text(json.dumps(
            [{"pmcid": p.pmcid, "run_id": p.run_id, "word_count": p.word_count}
             for p in papers],
            indent=2,
        ))
        logger.info("Paper sample saved to %s", sample_path)

        pre_cache_requests: list[JudgeRequest] = []
        all_skipped: list[dict] = []

        for paper in papers:
            pmcid, run_id = paper.pmcid, paper.run_id
            logger.info("── %s (run_id=%d) ──────────────────────", pmcid, run_id)

            if "q1" in cfg.tests:
                reqs, skipped = build_q1_requests(
                    session, pmcid, run_id, cfg.q1_findings, cfg.seed,
                )
                pre_cache_requests.extend(reqs)
                all_skipped.extend(skipped)

            if "q2" in cfg.tests:
                reqs, skipped = build_q2_requests(
                    session, pmcid, run_id, cfg.q2_relations, cfg.seed,
                    show_pipeline_label=cfg.show_pipeline_label,
                )
                pre_cache_requests.extend(reqs)
                all_skipped.extend(skipped)

            if "q3" in cfg.tests:
                reqs, skipped = build_q3_requests(
                    session, pmcid, run_id,
                    cfg.q3_with_extraction_paragraphs,
                    cfg.q3_zero_extraction_paragraphs,
                    cfg.seed,
                )
                pre_cache_requests.extend(reqs)
                all_skipped.extend(skipped)

            if "q5" in cfg.tests:
                reqs, skipped = build_q5_requests(
                    session, pmcid, run_id, cfg.q5_paragraphs, cfg.seed,
                )
                pre_cache_requests.extend(reqs)
                all_skipped.extend(skipped)

    # Session is now closed — safe to open cache sessions without nesting.

    # Phase 2: Cache filtering
    # Cached rows are passed through parse_*_result() so derived fields
    # (fields_changed, is_correct, F1 stats) are always computed from raw judgments.
    all_requests: list[JudgeRequest] = []
    all_cached_rows: dict[str, list[dict]] = {t: [] for t in ("q1", "q2", "q3", "q5")}

    for req in pre_cache_requests:
        raw_judgment = cache.get(req.cache_key)
        if raw_judgment is not None:
            parsed_row = _parse_result(req, raw_judgment)
            task_prefix = req.task.split("_")[0]
            if task_prefix in all_cached_rows:
                all_cached_rows[task_prefix].append(parsed_row)
        else:
            all_requests.append(req)

    # Apply max-requests cap
    if cfg.max_requests is not None and len(all_requests) > cfg.max_requests:
        logger.warning(
            "Capping %d uncached requests to --max-requests=%d",
            len(all_requests), cfg.max_requests,
        )
        all_requests = all_requests[: cfg.max_requests]

    logger.info(
        "Requests: %d uncached, %d cached, %d skipped",
        len(all_requests),
        sum(len(v) for v in all_cached_rows.values()),
        len(all_skipped),
    )

    # Dry-run: log what would be sent to Opus, then exit
    if cfg.dry_run:
        _log_dry_run(all_requests, cfg)
        cache.close()
        return

    # Execute
    if cfg.mode == "sync":
        fresh_results = _run_sync(all_requests, cache, cfg)
    elif cfg.mode == "batch":
        fresh_results = _run_batch(all_requests, cache, cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    # Parse + merge
    q1_rows = list(all_cached_rows["q1"])
    q2_rows = list(all_cached_rows["q2"])
    q3_rows = list(all_cached_rows["q3"])
    q5_rows = list(all_cached_rows["q5"])

    for req, judgment in fresh_results:
        row = _parse_result(req, judgment)
        if req.task == "q1_precision":
            q1_rows.append(row)
        elif req.task == "q2_relations":
            q2_rows.append(row)
        elif req.task == "q3_recall":
            q3_rows.append(row)
        elif req.task == "q5_f1":
            q5_rows.append(row)

    # Write outputs
    _write_jsonl(q1_rows, cfg.results_dir / "q1_precision.jsonl")
    _write_jsonl(q2_rows, cfg.results_dir / "q2_relations.jsonl")
    _write_jsonl(q3_rows, cfg.results_dir / "q3_recall.jsonl")
    _write_jsonl(q5_rows, cfg.results_dir / "q5_f1.jsonl")
    _write_jsonl(all_skipped, cfg.results_dir / "skipped_cases.jsonl")

    summary = build_summary(q1_rows, q2_rows, q3_rows, q5_rows)
    summary_path = cfg.results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n── SUMMARY ─────────────────────────────────────────")
    print(json.dumps(summary, indent=2))
    print(f"\nResults in: {cfg.results_dir}")

    cache.close()


# Result dispatcher

def _parse_result(req: JudgeRequest, judgment: dict) -> dict:
    """Route a (request, raw_judgment) pair through the appropriate parse function.

    Used for both fresh API responses and cache hits so derived fields are
    always computed, never stored stale.
    """
    if req.task == "q1_precision":
        return parse_q1_result(req, judgment)
    elif req.task == "q2_relations":
        return parse_q2_result(req, judgment)
    elif req.task == "q3_recall":
        return parse_q3_result(req, judgment)
    elif req.task == "q5_f1":
        return parse_q5_result(req, judgment)
    else:
        raise ValueError(f"Unknown task: {req.task}")


# Sync execution

def _run_sync(
    requests: list[JudgeRequest],
    cache: JudgeCache,
    cfg: RunConfig,
) -> list[tuple[JudgeRequest, dict]]:
    """Execute requests one at a time with exponential backoff retry."""
    import anthropic as _anthropic

    client = _anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    results: list[tuple[JudgeRequest, dict]] = []
    errors: list[dict] = []

    for i, req in enumerate(requests, 1):
        logger.info(
            "[%d/%d] %s %s …",
            i, len(requests), req.task, req.custom_id[:20],
        )
        last_exc: Exception | None = None
        for attempt in range(_SYNC_MAX_RETRIES):
            try:
                judgment = call_opus_sync(client, req)
                cache.set(req.cache_key, req.metadata, judgment, req.task)
                results.append((req, judgment))
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                if attempt < _SYNC_MAX_RETRIES - 1:
                    delay = _SYNC_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "[%d/%d] attempt %d/%d failed for %s — retrying in %.0fs: %s",
                        i, len(requests), attempt + 1, _SYNC_MAX_RETRIES,
                        req.custom_id[:20], delay, e,
                    )
                    time.sleep(delay)

        if last_exc is not None:
            logger.error(
                "Failed %s after %d attempts: %s",
                req.custom_id, _SYNC_MAX_RETRIES, last_exc,
            )
            errors.append({
                "custom_id": req.custom_id,
                "task": req.task,
                "error": str(last_exc),
                **{k: v for k, v in req.metadata.items() if k == "pmcid"},
            })

    if errors:
        _write_jsonl(errors, cfg.results_dir / "errors.jsonl")

    return results


# Batch execution

def _run_batch(
    requests: list[JudgeRequest],
    cache: JudgeCache,
    cfg: RunConfig,
) -> list[tuple[JudgeRequest, dict]]:
    """Submit to Anthropic Batch API, poll, retrieve."""
    import anthropic as _anthropic

    if not requests:
        logger.info("No uncached requests — nothing to submit.")
        return []

    client = _anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    req_map: dict[str, JudgeRequest] = {r.custom_id: r for r in requests}
    cid_to_ck = {r.custom_id: r.cache_key for r in requests}
    cid_to_request_meta = {
        r.custom_id: {"task": r.task, "cache_key": r.cache_key, "metadata": r.metadata}
        for r in requests
    }

    if cfg.no_submit:
        pending_path = cfg.results_dir / "pending_requests.jsonl"
        with open(pending_path, "w") as f:
            for req in requests:
                f.write(json.dumps({
                    "custom_id": req.custom_id,
                    "task": req.task,
                    "cache_key": req.cache_key,
                    "metadata": req.metadata,
                }) + "\n")
        logger.info("Wrote %d pending requests to %s (--no-submit)", len(requests), pending_path)
        return []

    batch_reqs = build_batch_requests(requests)
    batch_id = submit_batch(client, batch_reqs)
    save_batch_meta(cfg.results_dir, batch_id, len(requests), cid_to_ck, cid_to_request_meta)

    poll_batch(client, batch_id, cfg.poll_interval)

    return _collect_batch_results(client, batch_id, req_map, cache, cfg)


def _resume_batch(
    client: Any,
    cfg: RunConfig,
    cache: JudgeCache,
) -> None:
    """Resume a previously submitted batch."""
    batch_id = cfg.batch_id
    assert batch_id is not None

    meta = load_batch_meta(cfg.results_dir)
    cid_to_ck: dict[str, str] = {}
    cid_to_request_meta: dict[str, dict] = {}
    if meta and meta.get("batch_id") == batch_id:
        cid_to_ck = meta.get("custom_id_to_cache_key", {})
        cid_to_request_meta = meta.get("custom_id_to_request_meta", {})

    logger.info("Resuming batch %s", batch_id)
    poll_batch(client, batch_id, cfg.poll_interval)

    raw_results = retrieve_batch_results(client, batch_id)
    cached_count = 0
    skipped_count = 0

    for cid, judgment in raw_results.items():
        req_meta = cid_to_request_meta.get(cid, {})
        ck = req_meta.get("cache_key") or cid_to_ck.get(cid, cid)
        task = req_meta.get("task") or (cid.split("_")[0] if "_" in cid else "unknown")
        request_payload = req_meta.get("metadata", {})
        if request_payload:
            cache.set(ck, request_payload, judgment, task)
            cached_count += 1
        else:
            logger.warning("Resume: no request metadata for %s — result not cached", cid)
            skipped_count += 1

    logger.info(
        "Batch results cached: %d stored, %d skipped (no metadata). "
        "Re-run without --batch-id to generate output files.",
        cached_count, skipped_count,
    )


def _collect_batch_results(
    client: Any,
    batch_id: str,
    req_map: dict[str, JudgeRequest],
    cache: JudgeCache,
    cfg: RunConfig,
) -> list[tuple[JudgeRequest, dict]]:
    """Retrieve batch results, update cache, return (request, judgment) pairs."""
    raw_results = retrieve_batch_results(client, batch_id)
    results: list[tuple[JudgeRequest, dict]] = []
    errors: list[dict] = []

    for cid, judgment in raw_results.items():
        req = req_map.get(cid)
        if req is None:
            logger.warning("Unknown custom_id in batch results: %s", cid)
            continue
        cache.set(req.cache_key, req.metadata, judgment, req.task)
        results.append((req, judgment))

    for cid, req in req_map.items():
        if cid not in raw_results:
            errors.append({
                "custom_id": cid,
                "task": req.task,
                "error": "no_result_in_batch",
                **{k: v for k, v in req.metadata.items() if k == "pmcid"},
            })

    if errors:
        _write_jsonl(errors, cfg.results_dir / "errors.jsonl")

    return results


# Dry-run logger

def _log_dry_run(requests: list[JudgeRequest], cfg: RunConfig) -> None:
    """Write every pending Opus request to a JSONL file without calling the API."""
    out_path = cfg.results_dir / "dry_run_requests.jsonl"
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    by_task: dict[str, int] = {}
    with open(out_path, "w") as fh:
        for req in requests:
            by_task[req.task] = by_task.get(req.task, 0) + 1
            fh.write(json.dumps({
                "custom_id":  req.custom_id,
                "task":       req.task,
                "cache_key":  req.cache_key,
                "metadata":   req.metadata,
                "system":     req.system,
                "user":       req.user,
                "tool_name":  req.tool_name,
                "schema":     req.schema,
            }) + "\n")

    print(f"\n── DRY RUN — {len(requests)} requests (no API calls made) ──────────")
    for task, count in sorted(by_task.items()):
        print(f"  {task:<20} {count:>4} requests")
    print(f"\nFull prompts written to: {out_path}")
    print("Re-run without --dry-run to execute.")


# Helpers

def _write_jsonl(rows: list[dict], path: Path) -> None:
    if not rows:
        if path.exists():
            path.unlink()
        return
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    logger.info("Wrote %d rows → %s", len(rows), path)
