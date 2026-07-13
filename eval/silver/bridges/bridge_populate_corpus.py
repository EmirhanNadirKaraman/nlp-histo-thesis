#!/usr/bin/env python3
"""Bridge: primer θ0.9 cascade MAP → per-paper JSONs for rebuild_from_cached_map.

Generalizes the validated one-case proof (eval/silver/bridges/bridge_one_case_proof.py) to
the whole corpus. For each paper it pools its primer CASES (per text element),
resolves every chunk at the frozen θ0.9 cascade from the cached l1/l2/l3 votes
(ZERO voter API — exactly the sweep's AgreementChecker), re-numbers chunk ids
uniquely per paper, and writes a per-paper JSON whose ``audit_trail.map_chunks``
is what ``scripts/rebuild_from_cached_map.py`` consumes to run downstream
(GROUNDING→…→RELATE→RESOLVE) and persist to DB + files — all locally, no API.

Default writes to a NON-DESTRUCTIVE staging dir. ``--install`` writes into
``out/summaries/summaries/`` (backing up any existing file first) so
rebuild_from_cached_map picks them up.

  # validate one paper (staging, zero API, checks output == sweep θ0.9):
  python -m eval.silver.bridges.bridge_populate_corpus --pmcids PMC7540531_HIS-77-460 --validate

  # all 15 papers, install into the summaries dir for the rebuild step:
  python -m eval.silver.bridges.bridge_populate_corpus --all --install
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from eval.paths import REPO_ROOT as _REPO_ROOT

load_dotenv(str(_REPO_ROOT / ".env"))

# Same hard spend guard as the proof: voters are replayed from the primer and must
# never be called; blank their keys so any stray call is an auth error, not a charge.
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

from .bridge_one_case_proof import (  # noqa: E402
    BEST_FORCE_ESCALATE_POLARITY,
    BEST_SINGLE_VOTER_POLICY,
    BEST_VOTER_SUBSET,
    REJECT,
    THETA,
    _claim_key,
    _filtered_voter_cache,
    _frozen_spec,
    _resolve_chunk,
    _sorted_cids,
)
from eval.silver.analysis.map_context import CACHE_PATH, _load_map_context  # noqa: E402
from eval.silver.analysis.map_theta_sweep import _build_scorer, _replay  # noqa: E402
from pipeline.stages.knowledge_extraction.agreement import AgreementChecker  # noqa: E402
from pipeline.stages.knowledge_extraction.models import AuditableSummary  # noqa: E402

STAGING_DIR = _REPO_ROOT / "out" / "summaries" / "bridge_staging"
SUMMARIES_DIR = _REPO_ROOT / "out" / "summaries" / "summaries"
BACKUP_DIR = _REPO_ROOT / "out" / "summaries" / "_pre_bridge_backup"


def _cases_by_pmcid(voter_cache: dict) -> dict[str, list[tuple[str, dict]]]:
    by: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for safe_id, entry in voter_cache.items():
        by[entry["pmcid"]].append((safe_id, entry))
    return by


def _build_paper_map_chunks(cases: list[tuple[str, dict]], checker) -> tuple[list[dict], int, int]:
    """Pool a paper's cases → ordered map_chunks (θ0.9 resolved), re-numbered C1..CN.

    Returns (map_chunks, n_chunks_seen, n_findings). Rejected/empty chunks are
    dropped (they contribute no findings — same as the live MAP stage).
    """
    map_chunks: list[dict] = []
    seen = 0
    # Deterministic paper order: by te_id then chunk id.
    for _safe_id, entry in sorted(cases, key=lambda kv: int(kv[1].get("te_id", 0))):
        for cid in _sorted_cids(entry.get("chunk_map", {})):
            seen += 1
            selected, _level = _resolve_chunk(entry, cid, checker)
            if not selected:
                continue
            chunk = dict(selected)
            chunk["chunk_id"] = f"C{len(map_chunks) + 1}"  # unique per paper
            map_chunks.append(chunk)
    n_findings = sum(len(c.get("findings") or []) for c in map_chunks)
    return map_chunks, seen, n_findings


def _write_paper_json(out_dir: Path, pmcid: str, map_chunks: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{pmcid}.json"
    payload = {
        "pmcid": pmcid,
        "cascade_profile": "real_5",  # 5-voter shipped config (drop_l2_2 applied during the cascade replay)
        "_bridge": {
            "source": "eval/data/map_primer/voter_cache.json",
            "theta": THETA, "reject_theta": REJECT,
            "regime": "per-case, chunk_overlap=0 (calibration regime)",
            "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "audit_trail": {"map_chunks": map_chunks},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _validate_paper(pmcid: str, cases: list[tuple[str, dict]], map_chunks: list[dict],
                    scorer) -> bool:
    """Phase-A validation (zero API): map_chunks reconstruct to AuditableSummary,
    and the pooled findings == the sweep's θ0.9 _replay output for this paper."""
    ok = True
    # (a) every chunk reconstructs (rebuild_from_cached_map will model_validate these)
    try:
        for c in map_chunks:
            AuditableSummary.model_validate(c)
    except Exception as exc:  # noqa: BLE001
        print(f"    ✗ map_chunk failed AuditableSummary.model_validate: {exc}")
        ok = False
    # (b) unique chunk ids
    cids = [c["chunk_id"] for c in map_chunks]
    if len(set(cids)) != len(cids):
        print("    ✗ duplicate chunk_id after pooling")
        ok = False
    # (c) pooled findings == sweep θ0.9 replay for this paper
    sub = {sid: e for sid, e in cases}
    replay_findings = [f for co in _replay(sub, scorer, THETA, REJECT,
                                           single_voter_policy=BEST_SINGLE_VOTER_POLICY,
                                           force_escalate_on_polarity_conflict=BEST_FORCE_ESCALATE_POLARITY)[0]
                       for f in co.findings]
    bridge_findings = [f for c in map_chunks for f in (c.get("findings") or [])]
    match = sorted(map(_claim_key, bridge_findings)) == sorted(map(_claim_key, replay_findings))
    print(f"    findings: bridge {len(bridge_findings)} vs sweep-replay {len(replay_findings)} "
          f"→ {'MATCH' if match else 'MISMATCH'}")
    return ok and match


def main() -> int:
    ap = argparse.ArgumentParser(description="Bridge primer θ0.9 MAP → per-paper JSONs (zero API).")
    ap.add_argument("--pmcids", nargs="*", default=None, help="restrict to these primer pmcids")
    ap.add_argument("--all", action="store_true", help="process every paper in the primer")
    ap.add_argument("--install", action="store_true",
                    help="write into out/summaries/summaries/ (backs up existing first); "
                         "default writes to the non-destructive staging dir")
    ap.add_argument("--validate", action="store_true",
                    help="Phase-A checks: map_chunks reconstruct + findings == sweep θ0.9 replay")
    ap.add_argument("--primer", default=None, metavar="PATH",
                    help="voter_cache.json to bridge (default: the related15 primer). "
                         "Point at eval/data/map_primer_heldout15/voter_cache.json for the "
                         "isolated held-out run.")
    ap.add_argument("--out-dir", default=None, metavar="PATH",
                    help="write per-paper JSONs here instead of the related15 staging/summaries "
                         "dir (e.g. out/summaries/heldout15/summaries). Overrides --install/staging "
                         "and skips the SUMMARIES_DIR backup — use for an isolated split.")
    args = ap.parse_args()

    if not args.all and not args.pmcids:
        ap.error("pass --all or --pmcids PMCID [PMCID ...]")

    print(f"Bridge: primer θ{THETA}/reject{REJECT} cascade MAP → per-paper JSONs (per-case, overlap 0)")
    primer_path = Path(args.primer) if args.primer else CACHE_PATH
    ctx = _load_map_context("gemini", embed_cache_path=None, cache_path=primer_path)
    voter_cache = _filtered_voter_cache(ctx.voter_cache, BEST_VOTER_SUBSET)  # shipped 5-voter selection
    print(f"  voter subset = {BEST_VOTER_SUBSET}, single_voter_policy = {BEST_SINGLE_VOTER_POLICY}")
    scorer = _build_scorer(_frozen_spec(), ctx.agreement_embed_fn)
    checker = AgreementChecker(scorer, theta=THETA, reject_theta=REJECT,
                               single_voter_policy=BEST_SINGLE_VOTER_POLICY,
                               force_escalate_on_polarity_conflict=BEST_FORCE_ESCALATE_POLARITY)

    by_pmcid = _cases_by_pmcid(voter_cache)
    targets = sorted(by_pmcid) if args.all else [p for p in args.pmcids]
    missing = [p for p in targets if p not in by_pmcid]
    if missing:
        print(f"  not in primer: {missing}\n  available e.g.: {sorted(by_pmcid)[:3]}")
        return 1

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = _REPO_ROOT / out_dir
    else:
        out_dir = SUMMARIES_DIR if args.install else STAGING_DIR
    print(f"  output dir: {out_dir.relative_to(_REPO_ROOT)}   papers: {len(targets)}\n")

    all_ok = True
    for pmcid in targets:
        cases = by_pmcid[pmcid]
        map_chunks, seen, n_find = _build_paper_map_chunks(cases, checker)
        # Back up the related15 corpus only when installing into the shared SUMMARIES_DIR;
        # an explicit --out-dir is an isolated split, nothing to protect.
        if args.install and not args.out_dir:
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            existing = SUMMARIES_DIR / f"{pmcid}.json"
            if existing.exists() and not (BACKUP_DIR / f"{pmcid}.json").exists():
                shutil.copy2(existing, BACKUP_DIR / f"{pmcid}.json")
        path = _write_paper_json(out_dir, pmcid, map_chunks)
        print(f"  {pmcid}: {len(cases)} cases, {seen} chunks → {len(map_chunks)} kept, "
              f"{n_find} findings  → {path.relative_to(_REPO_ROOT)}")
        if args.validate:
            if not _validate_paper(pmcid, cases, map_chunks, scorer):
                all_ok = False

    if args.validate:
        print(f"\n=== VALIDATION: {'PASS ✓' if all_ok else 'FAIL ✗'} ===")
        if all_ok:
            print("Next (downstream + DB + files, no API):\n"
                  "  python -m eval.silver.bridges.bridge_populate_corpus --all --install\n"
                  "  python scripts/rebuild_from_cached_map.py --profile real --health-check no")
        return 0 if all_ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
