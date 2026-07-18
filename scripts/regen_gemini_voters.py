"""B-081: Gemini-only voter-cache regeneration (splice).

Regenerates only the two contaminated Gemini voter slots in the related15
voter_cache — L1 `gemini-2.5-flash-lite` (vi=0) and L2 `gemini-2.5-flash` (vi=0)
— with the FIXED batch mapper, and splices the fresh results back in, leaving the
cached OpenAI/Claude/Sonnet voters untouched.

Safe because the cache holds L1/L2/L3 for every chunk (no escalation-dependent
cache miss), so a clean-Gemini re-replay can't drop a chunk. Isolates the Gemini
fix exactly: only the 2 Gemini slots change, so the sweep delta vs 0.7135 is the
pure contamination-fix effect (no OpenAI/Claude re-run noise).

Resumable: submitted job ids + retrieved raw content are checkpointed to
`<cache>.regen_state.json`, so an interrupted run re-polls the existing jobs
instead of resubmitting (no double spend).

Run (related15, default):
  python scripts/regen_gemini_voters.py
Run (held-out 15):
  python scripts/regen_gemini_voters.py --primer-dir eval/data/map_primer_heldout15
After it writes the clean cache, score it (see each set's command at the end of
the run output).
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dotenv import load_dotenv
load_dotenv(str(_REPO / ".env"))

from nlp_histo.pipeline.stages.knowledge_extraction.batch.dispatch import (
    OPENAI_MAP_TOOL,
    build_providers,
    build_requests,
    parse_result,
)
from nlp_histo.pipeline.stages.knowledge_extraction.batch.models import BatchResult, ProviderJob
from nlp_histo.pipeline.stages.knowledge_extraction.batch.voter_configs import (
    make_l1_voters,
    make_l2_voters,
)

POLL_SECS = 30
MAX_WAIT_SECS = 90 * 60

# Gemini voter slots to regenerate (index 0 in both L1 and L2 by construction).
GEMINI_L1 = make_l1_voters()[0]
GEMINI_L2 = make_l2_voters()[0]
assert GEMINI_L1.provider == "gemini" and GEMINI_L2.provider == "gemini", \
    "voter index 0 is no longer Gemini — update this script"
STRIP = {"l1": GEMINI_L1.strip_thinking, "l2": GEMINI_L2.strip_thinking}


def _load_state(state_path: Path) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"jobs": [], "raw": {}, "retrieved": []}


def _save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state))


def _build_gemini_requests(cache: dict) -> list:
    reqs = []
    for safe_id, entry in cache.items():
        cmap = entry["chunk_map"]
        for voter, level in ((GEMINI_L1, "l1"), (GEMINI_L2, "l2")):
            reqs.extend(build_requests(cmap, safe_id, [voter], level=level))
    return reqs


def main() -> None:
    ap = argparse.ArgumentParser(description="Gemini-only voter-cache regeneration (B-081).")
    ap.add_argument("--primer-dir", default="eval/data/map_primer",
                    help="primer dir holding voter_cache.json "
                         "(use eval/data/map_primer_heldout15 for the held-out set)")
    args = ap.parse_args()

    primer_dir = Path(args.primer_dir)
    if not primer_dir.is_absolute():
        primer_dir = _REPO / primer_dir
    cache_path = primer_dir / "voter_cache.json"
    backup_path = primer_dir / "voter_cache.contaminated.json"
    state_path = primer_dir / "voter_cache.regen_state.json"
    if not cache_path.exists():
        raise SystemExit(f"no voter_cache.json under {primer_dir}")

    cache = json.loads(cache_path.read_text())

    # Back up the contaminated cache once (B-081 evidence + current baseline).
    if not backup_path.exists():
        backup_path.write_text(cache_path.read_text(), encoding="utf-8")
        print(f"backed up contaminated cache -> {backup_path}")

    provider = build_providers({"gemini"})["gemini"]
    state = _load_state(state_path)

    # Submit (only if not already submitted in a prior run)
    if not state["jobs"]:
        reqs = _build_gemini_requests(cache)
        by_model: dict[str, list] = defaultdict(list)
        for r in reqs:
            by_model[r.model].append(r)
        print(f"built {len(reqs)} Gemini requests:")
        for m, rs in by_model.items():
            print(f"  {m}: {len(rs)}")
        for m, rs in by_model.items():
            job = provider.submit(rs, OPENAI_MAP_TOOL)
            print(f"submitted {m}: {job.job_id} ({len(rs)} reqs)")
            state["jobs"].append(job.to_dict())
            _save_state(state_path, state)
    else:
        print(f"resuming: {len(state['jobs'])} job(s) already submitted, "
              f"{len(state['retrieved'])} retrieved, {len(state['raw'])} raw results so far")

    # Poll until all jobs done, retrieving as they complete
    waited = 0
    while True:
        jobs = [ProviderJob.from_dict(d) for d in state["jobs"]]
        pending = []
        for job in jobs:
            if job.job_id in state["retrieved"]:
                continue
            provider.check(job)
            if job.status == "completed":
                for res in provider.retrieve(job):
                    if res.content:
                        state["raw"][res.custom_id] = res.content
                state["retrieved"].append(job.job_id)
                print(f"  retrieved {job.job_id}")
            elif job.status == "failed":
                raise SystemExit(f"job {job.job_id} FAILED — aborting (no splice written)")
            else:
                pending.append(job)
        state["jobs"] = [j.to_dict() for j in jobs]
        _save_state(state_path, state)
        if not pending:
            break
        if waited >= MAX_WAIT_SECS:
            print(f"TIMEOUT after {waited}s with {len(pending)} job(s) pending — "
                  f"re-run to resume (jobs are still running server-side).")
            return
        time.sleep(POLL_SECS)
        waited += POLL_SECS
        print(f"  …{waited}s, {len(pending)} job(s) pending")

    # Parse + splice only the Gemini slots (vi=0, l1/l2)
    n_ok = n_fail = 0
    for custom_id, content in state["raw"].items():
        parts = custom_id.split("__")
        if len(parts) != 4:
            continue
        safe_id, chunk_id, level, vi_str = parts
        vi = int(vi_str) if vi_str.isdigit() else 0
        if safe_id not in cache or level not in ("l1", "l2") or vi != 0:
            print(f"  ! unexpected slot {custom_id} — skipping")
            continue
        parsed = parse_result(BatchResult(custom_id=custom_id, content=content),
                              strip_thinking=STRIP[level])
        slot = cache[safe_id][level].setdefault(chunk_id, [])
        while len(slot) <= vi:
            slot.append(None)
        if parsed is None:
            slot[vi] = None  # failed voter → absent (matches a fresh prime)
            n_fail += 1
        else:
            slot[vi] = parsed.model_dump()  # identical to _build_voter_cache
            n_ok += 1

    print(f"\nspliced {n_ok} Gemini slots ({n_fail} parse failures → set None)")

    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote clean cache -> {cache_path}")
    state_path.unlink(missing_ok=True)

    print("\nNext — score the clean cache:")
    if "heldout" in str(primer_dir):
        print("  python -m eval.silver.experiments.E14_heldout.heldout_eval")
    else:
        print("  python -m eval.silver.analysis.map_theta_sweep sweep --split all --embedder gemini \\")
        print("      --silver eval/data/silver_findings_related15.jsonl")
    print("Sanity-check contamination is gone:")
    print(f"  python legacy/scripts/probes/_citation_probe.py {cache_path}")


if __name__ == "__main__":
    main()
