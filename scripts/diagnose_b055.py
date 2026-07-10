#!/usr/bin/env python3
"""B-055 diagnostic: replay on-disk batch handles through ``persist_map_findings``.

The B-055 failure is entirely in post-MAP persistence, so it reproduces from the
frozen ``finalized`` chunk summaries in ``out/summaries/batch_handles.prepatch/``
*without* spending any LLM batch dollars. For each handle this:

  1. rebuilds ``chunk_summaries`` exactly as ``finalize()`` does,
  2. creates a throwaway ``pipeline_runs`` row,
  3. calls ``persist_map_findings`` (which now re-raises on failure, B-055),
  4. reports PASS (n rows) or FAIL (exception type + message),
  5. deletes the throwaway run (FK CASCADE removes any inserted rows).

Usage (from repo root):

    python scripts/diagnose_b055.py                       # scan all handles
    python scripts/diagnose_b055.py PMC10047158 PMC...    # specific papers
    python scripts/diagnose_b055.py --keep                # leave scratch runs in DB
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

HANDLE_DIR = Path("out/summaries/batch_handles.prepatch")


def _iter_handle_paths(pmcids: list[str]) -> list[Path]:
    if pmcids:
        paths = []
        for pmcid in pmcids:
            matches = sorted(HANDLE_DIR.glob(f"{pmcid}*.batch.json"))
            if not matches:
                print(f"  ! no handle found for {pmcid} under {HANDLE_DIR}")
            paths.extend(matches)
        return paths
    return sorted(HANDLE_DIR.glob("*.batch.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description="B-055 persist_map_findings replay diagnostic")
    parser.add_argument("pmcids", nargs="*", help="Specific PMCIDs to test (default: all handles)")
    parser.add_argument("--keep", action="store_true", help="Do not delete scratch pipeline_runs")
    args = parser.parse_args()

    from database import get_db_connection, PipelineRun
    from pipeline.stages.knowledge_extraction.batch.models import BatchHandle
    from pipeline.stages.knowledge_extraction.models import AuditableSummary
    from pipeline.stages.knowledge_extraction.persistence import (
        create_pipeline_run,
        persist_map_findings,
    )

    db = get_db_connection()

    handle_paths = _iter_handle_paths(args.pmcids)
    if not handle_paths:
        print(f"No handles to test under {HANDLE_DIR}")
        return 1

    failures: list[tuple[str, str]] = []
    print(f"Replaying {len(handle_paths)} handle(s) through persist_map_findings\n")

    for path in handle_paths:
        handle = BatchHandle.load(path)
        pmcid = handle.pmcid
        chunk_summaries = [
            AuditableSummary.model_validate(v) for v in handle.finalized.values()
        ]
        n_findings = sum(len(cs.findings) for cs in chunk_summaries)
        run_id = f"b055_diag_{pmcid}_{int(time.time() * 1000)}"
        run_db_id = create_pipeline_run(db, pmcid, run_id, {"diagnostic": True})
        if run_db_id is None:
            print(f"  {pmcid:32s} SKIP — could not create scratch pipeline_run")
            continue
        try:
            persist_map_findings(db, run_db_id, pmcid, chunk_summaries)
            with db.session_scope() as session:
                from database.models import SumMapFinding  # noqa: PLC0415
                n_rows = (
                    session.query(SumMapFinding)
                    .filter_by(pipeline_run_id=run_db_id)
                    .count()
                )
            status = "PASS" if n_rows == n_findings else "MISMATCH"
            print(f"  {pmcid:32s} {status} — {n_rows}/{n_findings} rows ({len(chunk_summaries)} chunks)")
            if n_rows != n_findings:
                failures.append((pmcid, f"row/finding mismatch {n_rows}/{n_findings}"))
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            print(f"  {pmcid:32s} FAIL — {msg}")
            traceback.print_exc()
            failures.append((pmcid, msg))
        finally:
            if not args.keep:
                with db.session_scope() as session:
                    run = session.query(PipelineRun).filter_by(id=run_db_id).first()
                    if run is not None:
                        session.delete(run)  # CASCADE removes sum_map_findings rows

    print()
    if failures:
        print(f"=== {len(failures)} FAILURE(S) ===")
        for pmcid, msg in failures:
            print(f"  {pmcid}: {msg}")
        return 1
    print(f"All {len(handle_paths)} handle(s) persisted cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
