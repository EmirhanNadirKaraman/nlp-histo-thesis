#!/usr/bin/env python3
"""
rebuild_from_cached_map.py — re-run downstream stages over cached MAP outputs.

Why this script exists
----------------------
``run_paper.py`` re-uses the cached final JSON at ``out/summaries/summaries/
<pmcid>.json`` via ``pipeline_config_hash``. Adding a new field to any
``KnowledgeExtractionConfig`` sub-config (even one whose default preserves prior
behaviour) flips that hash and forces a full pipeline re-run — including
paid L1/L2/L3 batch submissions.

But the raw MAP outputs are already on disk inside the cached JSONs
themselves: ``audit_trail.map_chunks`` holds the full
``AuditableSummary`` per chunk (chunk_id, findings, summary_text,
audit_metadata, verbatim per finding). The downstream stages
(GROUNDING → NORMALIZE → GROUP → CANONICALIZE → RELATE → RESOLVE) are
all local — NLI on HuggingFace, UMLS in-process, pure-Python scoring.

This script:

1. Walks ``out/summaries/summaries/*.json``
2. For each paper, reconstructs ``AuditableSummary`` objects from
   ``audit_trail.map_chunks`` and stuffs them into a fresh
   ``BatchHandle`` (``phase=COMPLETE``, ``cached_result_only=False``)
3. Calls ``runner.finalize(handle)`` directly — bypassing the
   ``submit()`` cache-check path entirely
4. The runner writes a fresh JSON over the old one with all current
   code paths applied (scope-aware NLI, scope/verbatim/text_element_id
   fields on CanonicalRule, etc.)
5. After every paper, fires the post-batch cross-paper relate step
   so ``out/summaries/corpus_relations.json`` is also refreshed

No API calls. ~10–20 min on 15 papers (NLI dominates).

Usage::

    python scripts/rebuild_from_cached_map.py \\
        --profile haiku_only \\
        --health-check no              # NLI/UMLS already loaded most likely

    # Restrict to specific papers
    python scripts/rebuild_from_cached_map.py \\
        --profile haiku_only --health-check no \\
        --pmcids PMC10100421 PMC4282326

    # Or read the PMCIDs from a paper-selection YAML
    python scripts/rebuild_from_cached_map.py \\
        --profile haiku_only --health-check no \\
        --from-selection configs/paper_selection/related15.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(str(_REPO_ROOT / ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rebuild_from_cached_map")

# Re-use the existing run_paper helpers so the rebuild path stays in lockstep
# with how the live runner builds its dependencies (config, db, embedder).
# ``scripts/`` has no ``__init__.py`` (project convention) so load by file
# path with importlib — same pattern as
# ``tests/eval/test_run_summarization_experiments.py``.
import importlib.util as _ilu  # noqa: E402
_RP_PATH = _REPO_ROOT / "scripts" / "run_paper.py"
_RP_SPEC = _ilu.spec_from_file_location("run_paper", _RP_PATH)
_RP_MOD = _ilu.module_from_spec(_RP_SPEC)
sys.modules["run_paper"] = _RP_MOD
_RP_SPEC.loader.exec_module(_RP_MOD)
_open_db_connection = _RP_MOD._open_db_connection
_run_corpus_relate  = _RP_MOD._run_corpus_relate
build_batch_runner  = _RP_MOD.build_batch_runner


_SUMMARIES_DIR     = _REPO_ROOT / "out" / "summaries" / "summaries"
_CORPUS_RELATIONS  = _REPO_ROOT / "out" / "summaries" / "corpus_relations.json"


def _load_pmcids_from_selection(path: Path) -> list[str]:
    """Pull the related/diverse/hard buckets out of a paper-selection YAML."""
    import yaml  # noqa: PLC0415
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise SystemExit(f"{path}: expected a top-level mapping")
    buckets = data[next(iter(data))] or {}
    out: list[str] = []
    seen: set[str] = set()
    for bucket in ("related", "diverse", "hard"):
        for pmcid in buckets.get(bucket) or []:
            if pmcid not in seen:
                out.append(pmcid)
                seen.add(pmcid)
    return out


def _match_summary_paths(pmcids: list[str], summaries_dir: Path = _SUMMARIES_DIR) -> list[Path]:
    """For each PMCID prefix, find its matching ``<pmcid>.json`` in summaries.

    The selection YAML uses bare PMC IDs (``PMC10100421``) but the on-disk
    JSON filenames include the journal suffix (``PMC10100421_HIS-82-393``).
    Greedy ``startswith`` match keeps a 1:1 mapping; ambiguous prefixes
    raise.
    """
    available = sorted(summaries_dir.glob("*.json"))
    available = [p for p in available if not p.name.startswith("_")]
    matched: list[Path] = []
    for pmcid in pmcids:
        candidates = [p for p in available if p.stem.startswith(pmcid)]
        if not candidates:
            logger.warning("No summary JSON for %s — skipping", pmcid)
            continue
        if len(candidates) > 1:
            logger.warning(
                "Multiple JSONs match prefix %s: %s — using the first",
                pmcid, [p.name for p in candidates],
            )
        matched.append(candidates[0])
    return matched


def _build_handle_from_cached_json(
    cached: dict,
    cascade_profile: str,
    cascade_signature: str,
    pipeline_run_db_id: int | None = None,
):
    """Reconstruct a BatchHandle whose ``finalized`` holds the raw MAP outputs
    from the cached JSON's ``audit_trail.map_chunks``.

    ``pipeline_run_db_id`` is the row id of an INSERT into ``pipeline_runs``
    created via ``runner._create_pipeline_run`` BEFORE calling this. Per-stage
    persisters (``persist_map_findings``, ``persist_normal_findings``, …) all
    no-op when this id is None — without it the DB tables stay empty even
    when ``finalize()`` finishes cleanly and writes the JSON.
    """
    from nlp_histo.pipeline.stages.knowledge_extraction.batch.models import (  # noqa: PLC0415
        BatchHandle, BatchPhase,
    )

    chunks = cached.get("audit_trail", {}).get("map_chunks", [])
    if not chunks:
        raise ValueError(
            f"cached JSON has no audit_trail.map_chunks — cannot rebuild "
            f"(pmcid={cached.get('pmcid')!r})"
        )

    # finalized = {chunk_id → AuditableSummary.model_dump()} per runner.finalize
    finalized: dict[str, dict] = {}
    for c in chunks:
        cid = c.get("chunk_id")
        if not cid:
            continue
        finalized[cid] = c

    handle = BatchHandle(
        pmcid=cached["pmcid"],
        phase=BatchPhase.COMPLETE,
        cached_result_only=False,          # IMPORTANT: force finalize() to actually re-run
        cascade_profile=cascade_profile,
        cascade_signature=cascade_signature,
        pipeline_run_db_id=pipeline_run_db_id,
    )
    handle.finalized = finalized
    return handle


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--profile", required=True, choices=["cheap", "real", "real_5", "haiku_only"],
        help="Cascade profile that the original run used; must match so the "
             "rebuilt cascade_signature in artifacts is consistent.",
    )
    parser.add_argument(
        "--health-check", default="no", choices=["yes", "no"],
        help="Run startup health checks. Default no since NLI/UMLS are "
             "already loaded by the time you reach this rebuild path.",
    )
    parser.add_argument(
        "--pmcids", nargs="*", default=None, metavar="PMCID",
        help="Restrict to specific PMCIDs (bare or with journal suffix). "
             "Default: every JSON in out/summaries/summaries/.",
    )
    parser.add_argument(
        "--from-selection", default=None, metavar="PATH",
        help="Load PMCIDs from a paper-selection YAML (e.g. "
             "configs/paper_selection/related15_full.yaml). Mutually "
             "exclusive with --pmcids.",
    )
    parser.add_argument(
        "--no-corpus-relate", action="store_true",
        help="Skip the post-rebuild cross-paper corpus_relate step.",
    )
    parser.add_argument(
        "--chunk-workers", type=int, default=None, metavar="N",
        help="Forwarded to build_batch_runner.",
    )
    parser.add_argument(
        "--config", default="configs/run.yaml", metavar="PATH",
        help="Summarisation YAML to load (default: configs/run.yaml).",
    )
    parser.add_argument(
        "--summaries-dir", default=None, metavar="PATH",
        help="Read per-paper MAP JSONs from this dir instead of "
             "out/summaries/summaries/ (corpus_relations.json is written to its "
             "parent). Use for an isolated split, e.g. "
             "out/summaries/heldout15/summaries.",
    )
    parser.add_argument(
        "--no-db", action="store_true",
        help="Files-only: never open a sum_pipeline_run row, so every per-stage "
             "DB persister no-ops (pipeline_run_db_id stays None). Keeps an "
             "isolated split (e.g. heldout15) out of the shared corpus tables. "
             "Funnel/rule/relation counts still come back in the result dict + JSON.",
    )
    args = parser.parse_args()

    if args.pmcids and args.from_selection:
        parser.error("--pmcids and --from-selection are mutually exclusive")

    # ── Resolve the (possibly isolated) summaries dir + corpus_relations target ──
    summaries_dir = Path(args.summaries_dir) if args.summaries_dir else _SUMMARIES_DIR
    corpus_relations = (
        summaries_dir.parent / "corpus_relations.json"
        if args.summaries_dir else _CORPUS_RELATIONS
    )

    # Make the profile resolvable to voter_configs.get_profile()
    os.environ["NLP_HISTO_PROFILE"] = args.profile

    # ── Resolve which papers to rebuild ─────────────────────────────────────
    if args.from_selection:
        pmcids = _load_pmcids_from_selection(Path(args.from_selection))
        paths = _match_summary_paths(pmcids, summaries_dir)
    elif args.pmcids:
        paths = _match_summary_paths(args.pmcids, summaries_dir)
    else:
        paths = sorted(
            p for p in summaries_dir.glob("*.json")
            if not p.name.startswith("_")
        )

    if not paths:
        logger.error("No matching JSONs found in %s", summaries_dir)
        return 2

    logger.info("Rebuilding downstream stages for %d papers from cached MAP", len(paths))

    # ── Build the same batch runner run_paper.py uses ───────────────────────
    logger.info("Building batch runner with profile=%s …", args.profile)
    runner = build_batch_runner(
        profile_name=args.profile,
        artifact_root=None,
        artifact_run_id=None,
        db=_open_db_connection("rebuild_from_cached_map"),
        chunk_workers=args.chunk_workers,
        config_path=args.config,
        # Isolate ALL runner outputs (per-paper JSON, cache, traces, cost) under the
        # split's dir so a held-out rebuild never writes into the related15 corpus.
        output_dir=summaries_dir.parent if args.summaries_dir else Path("out/summaries"),
    )

    # ── Rebuild each paper ──────────────────────────────────────────────────
    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []
    for i, path in enumerate(paths, 1):
        cached = json.loads(path.read_text(encoding="utf-8"))
        pmcid = cached.get("pmcid") or path.stem
        logger.info("─── [%d/%d] Rebuilding %s ───", i, len(paths), pmcid)
        try:
            # 1) Open a fresh sum_pipeline_run row so the per-stage persisters
            #    can attach to it (they no-op when db_id is None — that's
            #    what caused the post-rebuild "0 rows for pmcid=..." warnings
            #    in the first rebuild pass). With --no-db we deliberately leave
            #    it None so an isolated split never writes into the corpus tables.
            if args.no_db:
                pipeline_run_db_id = None
            else:
                run_id = runner._make_run_id(pmcid)
                pipeline_run_db_id = runner._create_pipeline_run(pmcid, run_id)

            handle = _build_handle_from_cached_json(
                cached,
                cascade_profile=runner._cascade_profile,
                cascade_signature=runner._cascade_signature,
                pipeline_run_db_id=pipeline_run_db_id,
            )
            result = runner.finalize(handle)
            n_can = len(result.get("canonical_rules", []) or [])
            n_rel = len(result.get("relations", []) or [])
            n_fin = len(result.get("final_rules", []) or [])
            logger.info(
                "    OK — %d canonical rules · %d intra-paper relations · "
                "%d final rules · pipeline_run_db_id=%s",
                n_can, n_rel, n_fin, pipeline_run_db_id,
            )
            succeeded.append(pmcid)
        except Exception as exc:  # noqa: BLE001 — log and continue
            logger.exception("    FAIL on %s: %s", pmcid, exc)
            failed.append((pmcid, repr(exc)))

    logger.info(
        "Rebuild complete: %d succeeded · %d failed", len(succeeded), len(failed),
    )
    if failed:
        for pmcid, err in failed:
            logger.warning("  FAILED  %s  %s", pmcid, err)

    # ── Post-rebuild cross-paper corpus_relate ──────────────────────────────
    # Forward the active relate config (scope_aware_nli, use_verbatim_for_nli,
    # NLI thresholds) so the corpus_relate NLI matches the within-paper one.
    _run_corpus_relate(
        summaries_dir=summaries_dir,
        output_path=corpus_relations,
        skip=args.no_corpus_relate,
        relate_config=getattr(runner, "_cfg_full", None) and runner._cfg_full.relate,
    )

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
