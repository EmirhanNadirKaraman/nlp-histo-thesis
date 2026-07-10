"""
Process one or more histopathology papers through the knowledge_extraction pipeline.

Usage
-----
Exactly one of --sync / --batch is required (no implicit default — prevents
accidentally launching the wrong mode).

python scripts/run_paper.py PMC1234567 --batch          # batch (async) mode
python scripts/run_paper.py PMC1234567 --sync           # sync (live) mode
python scripts/run_paper.py PMC1234567 --sync --trace   # enable JSONL traces (sync only)
python scripts/run_paper.py PMC1234567 --batch --dry-run  # print config and exit

# Omit PMCID to auto-sample from eval/data/source_cases_related15.jsonl:
python scripts/run_paper.py --batch                     # 2 random PMCIDs, seed=42
python scripts/run_paper.py --sync --sample 3 --seed 7  # 3 random PMCIDs, seed=7
python scripts/run_paper.py --batch --all               # all PMCIDs in source_cases_related15.jsonl

Requires: GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_paper")


# Single source of truth for the batch-poll interval default. Used by the
# argparse default and by both batch-runner function signatures so they
# cannot drift apart again (see B-028 / Tier 1 config audit, 2026-05-15).
DEFAULT_POLL_INTERVAL_SEC: int = 60


# ── Sync runner ────────────────────────────────────────────────────────────────


def _make_token_callback(model_id: str, store: dict, lock: threading.Lock):
    """Return a BaseCallbackHandler that accumulates per-model token counts.

    One callback per LLM; multiple callbacks share the same store dict and lock
    so totals accumulate safely across threads.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    class _CB(BaseCallbackHandler):
        raise_error = False

        def on_llm_end(self, response, **kwargs) -> None:  # type: ignore[override]
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    usage = getattr(msg, "usage_metadata", None) if msg is not None else None
                    if usage is not None:
                        inp = int(usage.get("input_tokens", 0) or 0)
                        out = int(usage.get("output_tokens", 0) or 0)
                    elif response.llm_output:
                        lu = response.llm_output.get("token_usage", {})
                        inp = int(lu.get("prompt_tokens", 0) or 0)
                        out = int(lu.get("completion_tokens", 0) or 0)
                    else:
                        continue
                    if inp or out:
                        with lock:
                            entry = store.setdefault(model_id, {"input": 0, "output": 0})
                            entry["input"]  += inp
                            entry["output"] += out

    return _CB()


def _open_db_connection(caller_label: str):
    """Return a DB connection or ``None`` if Postgres is unreachable.

    Both the sync runner and the batch runner need a live DB connection to
    persist per-paper ``sum_*`` rows AND to trigger
    ``_corpus_relate_incremental`` cross-paper RELATE.  Without it, runs
    succeed but write only to ``out/summaries/summaries/*.json`` — no DB
    rows, no cross-paper edges.  Failure is non-fatal so smoke tests still
    work on machines without a configured Postgres.
    """
    try:
        from database import get_db_connection  # noqa: PLC0415
        return get_db_connection()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "%s: DB connection unavailable (%s) — corpus_relate "
            "and per-paper sum_* persistence will be skipped.",
            caller_label, exc,
        )
        return None


DEFAULT_CONFIG_PATH = "configs/run.yaml"


def _load_summarization_config(config_path: str | Path | None):
    """Return a ``KnowledgeExtractionConfig`` populated from ``config_path``.

    Falls back to ``KnowledgeExtractionConfig()`` defaults when ``config_path`` is
    ``None`` / empty / missing on disk, or when the YAML has no
    ``knowledge_extraction`` block. Logs which path was taken so callers can see
    whether the YAML override actually applied.
    """
    from pipeline.stages.knowledge_extraction.config import KnowledgeExtractionConfig

    if not config_path:
        logger.info("Summarisation config: defaults (no --config path)")
        return KnowledgeExtractionConfig()

    path = Path(config_path)
    if not path.exists():
        logger.warning(
            "Summarisation config: %s not found — falling back to defaults", path,
        )
        return KnowledgeExtractionConfig()

    from pipeline.config_loader import load_config

    _pdf_cfg, sum_cfg = load_config(path)
    logger.info(
        "Summarisation config loaded from %s "
        "(grounding.threshold=%s, map.theta=%s, routing.enable_router=%s, "
        "relate.entailment_threshold=%s)",
        path, sum_cfg.grounding.threshold, sum_cfg.map.theta,
        sum_cfg.routing.enable_router, sum_cfg.relate.entailment_threshold,
    )
    return sum_cfg


def _make_embed_fn(embedder_name: str):
    """Build the agreement-scorer embedder from config: 'gemini' | 'openai'.

    Production historically hardcoded GeminiEmbedder; this routes the choice
    through cfg.agreement.embedder so the map_theta_sweep --embedder winner can
    be pinned in configs/run.yaml.
    """
    from pipeline.stages.knowledge_extraction.agreement.providers import (
        GeminiEmbedder,
        OpenAIEmbedder,
    )
    name = (embedder_name or "gemini").strip().lower()
    if name == "gemini":
        return GeminiEmbedder()
    if name == "openai":
        return OpenAIEmbedder()
    raise ValueError(
        f"agreement.embedder must be 'gemini' or 'openai', got {embedder_name!r}"
    )


def build_runner(
    trace: bool,
    *,
    profile_name: str | None = None,
    artifact_root: Path | None = None,
    artifact_run_id: str | None = None,
    chunk_workers: int | None = None,
    config_path: str | Path | None = DEFAULT_CONFIG_PATH,
):
    """Return (runner, token_usage) where token_usage is {level: {model: {input, output}}}.

    The cascade is resolved entirely from
    ``pipeline/stages/knowledge_extraction/batch/voter_configs.py`` via
    :func:`get_profile`. ``profile_name`` must be set (either directly or via
    ``$NLP_HISTO_PROFILE``); there is no implicit default — see
    :func:`get_profile`. ``artifact_root`` enables filesystem persistence
    (no-op when None). ``config_path`` selects the YAML used to override
    ``KnowledgeExtractionConfig`` defaults; pass ``None`` to skip YAML loading.
    """
    from pipeline.stages.knowledge_extraction import KnowledgeExtractionRunner
    from pipeline.stages.knowledge_extraction.batch.voter_configs import get_profile
    from dataclasses import replace as _dc_replace
    from pipeline.stages.knowledge_extraction.llm_providers import (
        anthropic_direct_chat,
        gemini_direct_chat,
        openai_direct_chat,
    )

    sum_cfg = _load_summarization_config(config_path)
    if chunk_workers is not None:
        sum_cfg = _dc_replace(sum_cfg, map=_dc_replace(sum_cfg.map, chunk_workers=chunk_workers))

    l1_store: dict = {}
    l2_store: dict = {}
    l3_store: dict = {}
    lock = threading.Lock()

    def with_cb(llm, model_id: str, store: dict):
        cb = _make_token_callback(model_id, store, lock)
        return llm.with_config(callbacks=[cb])

    _PROVIDER_FACTORY = {
        "claude":    anthropic_direct_chat,
        "anthropic": anthropic_direct_chat,
        "openai":    openai_direct_chat,
        "gemini":    gemini_direct_chat,
    }
    _PROVIDER_SPEC = {
        "claude":    "anthropic", "anthropic": "anthropic",
        "openai":    "openai",    "gemini":    "gemini",
    }

    prof = get_profile(profile_name)

    def _build(vc, store):
        factory = _PROVIDER_FACTORY[vc.provider]
        llm = factory(vc.model, temperature=vc.temperature)
        return with_cb(llm, vc.model, store)

    voter_llms          = [_build(v, l1_store) for v in prof.l1_voters]
    level2_voter_llms   = [_build(v, l2_store) for v in prof.l2_voters]
    escalation_llm      = _build(prof.l3_voter, l3_store)
    voter_specs         = [(_PROVIDER_SPEC[v.provider], v.model) for v in prof.l1_voters]
    level2_voter_specs  = [(_PROVIDER_SPEC[v.provider], v.model) for v in prof.l2_voters]
    escalation_spec     = (_PROVIDER_SPEC[prof.l3_voter.provider], prof.l3_voter.model)
    active_profile_name = prof.name

    token_usage = {"l1": l1_store, "l2": l2_store, "l3": l3_store}

    db_conn = _open_db_connection("build_runner")

    runner = KnowledgeExtractionRunner(
        voter_llms=voter_llms,
        level2_voter_llms=level2_voter_llms,
        escalation_llm=escalation_llm,
        embed_fn=_make_embed_fn(sum_cfg.agreement.embedder),
        config=sum_cfg,
        enable_router=sum_cfg.routing.enable_router,
        router_single_voter_policy=sum_cfg.routing.router_single_voter_policy,
        legacy_single_voter_policy=sum_cfg.routing.legacy_single_voter_policy,
        output_dir=Path("out/summaries"),
        trace_enabled=trace,
        voter_specs=voter_specs,
        level2_voter_specs=level2_voter_specs,
        escalation_spec=escalation_spec,
        cascade_profile=active_profile_name,
        artifact_root=artifact_root,
        artifact_run_id=artifact_run_id,
        db=db_conn,
    )
    return runner, token_usage


# ── Batch runner ───────────────────────────────────────────────────────────────

def build_batch_runner(
    profile_name: str | None = None,
    *,
    artifact_root: Path | None = None,
    artifact_run_id: str | None = None,
    run_reduce: bool = False,
    db=None,
    force_rerun: bool = False,
    run_ner: bool = False,
    chunk_workers: int | None = None,
    config_path: str | Path | None = DEFAULT_CONFIG_PATH,
    output_dir: Path = Path("out/summaries"),
):
    from dataclasses import replace as _dc_replace
    from pipeline.stages.knowledge_extraction.batch import BatchKnowledgeExtractionRunner
    from pipeline.stages.knowledge_extraction.batch.voter_configs import get_profile
    from pipeline.stages.knowledge_extraction.llm_providers import (
        anthropic_direct_chat,
    )

    profile = get_profile(profile_name)
    logger.info("Cascade profile: %s", profile.name)

    sum_cfg = _load_summarization_config(config_path)
    if chunk_workers is not None:
        sum_cfg = _dc_replace(sum_cfg, map=_dc_replace(sum_cfg.map, chunk_workers=chunk_workers))

    # REDUCE + RULES still run synchronously (one call per paper at the end).
    # The L3 voter model in the active profile is used so smoke profiles stay cheap.
    escalation_llm = anthropic_direct_chat(profile.l3_voter.model, temperature=0.0)

    return BatchKnowledgeExtractionRunner(
        l1_voters=profile.l1_voters,
        l2_voters=profile.l2_voters,
        l3_model=profile.l3_voter,
        escalation_llm=escalation_llm,
        config=sum_cfg,
        enable_router=sum_cfg.routing.enable_router,
        router_single_voter_policy=sum_cfg.routing.router_single_voter_policy,
        legacy_single_voter_policy=sum_cfg.routing.legacy_single_voter_policy,
        output_dir=output_dir,
        embed_fn=_make_embed_fn(sum_cfg.agreement.embedder),
        cascade_profile=profile.name,
        artifact_root=artifact_root,
        artifact_run_id=artifact_run_id,
        run_reduce=run_reduce,
        db=db,
        force_rerun=force_rerun,
        run_ner=run_ner,
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def _all_pmcids() -> list[str]:
    """Return all unique PMCIDs from eval/data/source_cases_related15.jsonl in file order."""
    import json
    source = Path(__file__).parent.parent / "eval" / "data" / "source_cases_related15.jsonl"
    if not source.exists():
        logger.error("source_cases_related15.jsonl not found at %s", source)
        sys.exit(1)
    pmcids: list[str] = []
    seen: set[str] = set()
    with source.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pmcid = json.loads(line).get("pmcid", "")
            if pmcid and pmcid not in seen:
                seen.add(pmcid)
                pmcids.append(pmcid)
    logger.info("Found %d unique PMCIDs in source_cases_related15.jsonl", len(pmcids))
    return pmcids


def _sample_pmcids(n: int, seed: int) -> list[str]:
    """Pick n unique PMCIDs from eval/data/source_cases_related15.jsonl using the given seed."""
    import random
    pmcids = _all_pmcids()
    rng = random.Random(seed)
    sample = rng.sample(pmcids, min(n, len(pmcids)))
    logger.info("Auto-sampled %d PMCIDs (seed=%d): %s", len(sample), seed, sample)
    return sample


def _load_selection_yaml(path: Path) -> list[str]:
    """Read a paper-selection YAML and return a de-duplicated PMCID list.

    Expected shape (produced by ``eval.paper_selection.export.write_calibration_set``)::

        version_name:
          related: [PMC..., ...]
          diverse: [PMC..., ...]
          hard:    [PMC..., ...]

    Buckets concatenate in related → diverse → hard order; duplicates are
    dropped while preserving first occurrence.
    """
    import yaml

    if not path.exists():
        logger.error("--from-selection path not found: %s", path)
        sys.exit(1)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        logger.error("%s: expected a top-level mapping with a version key", path)
        sys.exit(1)
    version_key = next(iter(data))
    buckets = data[version_key] or {}
    if not isinstance(buckets, dict):
        logger.error("%s: version %r is not a mapping", path, version_key)
        sys.exit(1)

    ordered: list[str] = []
    seen: set[str] = set()
    for bucket in ("related", "diverse", "hard"):
        for pmcid in buckets.get(bucket) or []:
            if pmcid not in seen:
                ordered.append(pmcid)
                seen.add(pmcid)
    if not ordered:
        logger.error("%s: no PMCIDs found under related/diverse/hard", path)
        sys.exit(1)
    return ordered


def main():
    parser = argparse.ArgumentParser(description="Run knowledge_extraction pipeline on one or more papers.")
    parser.add_argument("pmcid",          nargs="?", default=None,
                        help="PubMed Central ID, e.g. PMC1234567. "
                             "Omit to auto-sample from eval/data/source_cases_related15.jsonl.")
    parser.add_argument("--all",           action="store_true",
                        help="Run all PMCIDs from eval/data/source_cases_related15.jsonl")
    parser.add_argument("--from-selection", default=None, metavar="PATH",
                        help="Path to a paper-selection YAML "
                             "(e.g. configs/paper_selection/smoke_v2.yaml). "
                             "Loads all PMCIDs from related/diverse/hard. "
                             "Takes precedence over --all / --sample / positional pmcid.")
    parser.add_argument("--sample",       type=int, default=2, metavar="N",
                        help="Number of PMCIDs to sample when pmcid is omitted (default: 2)")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed for PMCID sampling (default: 42)")
    parser.add_argument("--trace",        action="store_true", help="Write JSONL traces (sync mode only)")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--sync",  action="store_true",
                            help="Use sync (live) mode. Required if --batch not given.")
    mode_group.add_argument("--batch", action="store_true",
                            help="Use batch (async) mode. Required if --sync not given.")
    parser.add_argument("--dry-run",       action="store_true", help="Print config and exit without API calls")
    parser.add_argument(
        "--chunk-workers", type=int, default=None, metavar="N",
        help="ThreadPoolExecutor size for MAP chunk-level parallelism. "
             "Each worker processes one MAP chunk and fires all L1 voters "
             "for it in parallel inside. Default 5 (from MapConfig). Raise to "
             "saturate provider rate limits on bigger Tier accounts; lower if "
             "you keep hitting 429s. Applies to both --sync and --batch.",
    )
    parser.add_argument("--limit-chunks",  type=int, default=None, metavar="N",
                        help="Process at most N MAP chunks (useful for cheap iteration)")
    parser.add_argument("--start-chunk",   type=int, default=0, metavar="K",
                        help="Start processing from chunk K (0-based, default 0)")
    parser.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL_SEC, metavar="S",
                        help=f"Seconds between batch status polls (default: {DEFAULT_POLL_INTERVAL_SEC})")
    parser.add_argument("--artifact-root", default=None, metavar="PATH",
                        help="Enable filesystem persistence under this directory "
                             "(sync mode only). Disabled when omitted.")
    parser.add_argument("--artifact-run-id", default=None, metavar="ID",
                        help="Reuse a single run_id across multiple papers so they "
                             "land in the same runs/{run_id}/ directory.")
    parser.add_argument(
        "--skip-umls-enrichment", action="store_true",
        help="Skip the post-CANONICALIZE UMLS enrichment step. NORMALIZE still "
             "uses UMLS unless --disable-umls is set."
    )
    parser.add_argument(
        "--disable-umls", action="store_true",
        help="Disable scispaCy/UMLS entirely. Both NORMALIZE and the enrichment "
             "step fall back to dict-only behavior. Avoids loading the multi-GB "
             "UMLS KB — useful on memory-constrained machines."
    )
    parser.add_argument(
        "--health-check", required=True, choices=["yes", "no"],
        metavar="yes|no",
        help="Run startup health checks before any LLM call: NLI model load, "
             "UMLS singleton, embedding provider, and a trivial DB query. "
             "Logs an aggregated OK/FAIL summary so silent fallbacks (Issue G, "
             "B-019/B-020-class) surface before paying for a cascade. Adds "
             "~90s on first run (UMLS + NLI model load); subsequent runs in "
             "the same process are fast. REQUIRED — no implicit default; "
             "pass 'yes' for safety on real runs, 'no' for fast dev iteration."
    )
    parser.add_argument(
        "--profile", required=True, metavar="NAME",
        choices=["cheap", "real", "real_5", "haiku_only"],
        help="Cascade profile (required — no implicit default to prevent "
             "accidental spend). Registered in "
             "pipeline/stages/knowledge_extraction/batch/voter_configs.py:\n"
             "  cheap      — Gemini + OpenAI only, no Claude. Smoke/dev runs.\n"
             "  real       — Gemini-Flash-Lite/GPT-4o-mini/GPT-4.1-nano at L1; "
             "Gemini-Flash + GPT-4.1-mini + Claude-Haiku at L2; Claude-Sonnet "
             "at L3. Production-quality eval runs.\n"
             "  haiku_only — Single-voter Claude-Haiku at L1, no escalation. "
             "Cost-effective production runs; ~$0.0036/chunk, ~4.2pp strict_f1 "
             "drop vs Sonnet-only per EXP B.2 (docs/EXP_B2_RESULTS.md).",
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH, metavar="PATH",
        help=f"YAML used to populate KnowledgeExtractionConfig (default: "
             f"{DEFAULT_CONFIG_PATH}). The knowledge_extraction block ("
             "knowledge_extraction.map, knowledge_extraction.grounding, "
             "knowledge_extraction.relate, knowledge_extraction.resolve, …) is read via "
             "pipeline.config_loader.load_config. Pass an empty string "
             "('--config \"\"') to skip YAML loading and use dataclass "
             "defaults — useful when scripting reproducible thesis runs.",
    )
    parser.add_argument(
        "--no-corpus-relate", action="store_true",
        help="Skip the post-batch cross-paper RelateStage. By default, after "
             "every paper finishes finalize(), `CorpusRelateStage.relate_from_dir` "
             "pools rules from `out/summaries/summaries/*.json` and writes "
             "`out/summaries/corpus_relations.json` (auto-detected by "
             "`scripts/inspect_pipeline_output.py`). Replay-only, local NLI, "
             "zero API calls — usually free; pass this flag to skip on very "
             "large corpora where pairwise NLI dominates runtime.",
    )
    args = parser.parse_args()

    # Make profile available to downstream builders without threading the arg
    # through every helper. Per voter_configs.get_profile() resolution rules.
    if args.profile:
        import os as _os
        _os.environ["NLP_HISTO_PROFILE"] = args.profile

    # UMLS kill-switches — read by umls_resources and entity_linker.
    import os as _os
    if args.disable_umls:
        _os.environ["NLP_HISTO_DISABLE_UMLS"] = "1"
        logger.info("UMLS disabled via --disable-umls")
    if args.skip_umls_enrichment:
        _os.environ["NLP_HISTO_SKIP_UMLS_ENRICHMENT"] = "1"
        logger.info("UMLS enrichment (post-CANONICALIZE) skipped via --skip-umls-enrichment")

    if args.health_check == "yes":
        # Run BEFORE any LLM call so operators see component-level health
        # before paying for a cascade. Default-on for safety; pass
        # `--health-check no` for fast single-paper dev iteration.
        from pipeline.stages.knowledge_extraction.health_checks import run_health_checks
        # DB probe needs a real connection; reuse the run-side helper.
        try:
            db_for_health = _open_db_connection("health_check") if "_open_db_connection" in globals() else None
        except Exception:  # noqa: BLE001
            db_for_health = None
        results = run_health_checks(db=db_for_health)
        if any(not r.ok for r in results):
            logger.warning(
                "Health checks reported %d failure(s) — proceeding anyway. "
                "Inspect the WARNING block above to decide whether to abort. "
                "Use --health-check no to skip this on future runs.",
                sum(1 for r in results if not r.ok),
            )
    else:
        logger.info("Skipping startup health checks (--health-check no)")

    if args.from_selection:
        pmcids = _load_selection_yaml(Path(args.from_selection))
        logger.info("Loaded %d PMCIDs from %s", len(pmcids), args.from_selection)
    elif args.all:
        pmcids = _all_pmcids()
    elif args.pmcid is None:
        pmcids = _sample_pmcids(args.sample, args.seed)
    else:
        pmcid = args.pmcid.strip()
        if not pmcid.startswith("PMC"):
            pmcid = "PMC" + pmcid
        pmcids = [pmcid]

    if args.dry_run:
        from pipeline.stages.knowledge_extraction.batch.voter_configs import get_profile
        profile = get_profile(args.profile)
        mode = "sync" if args.sync else "batch"
        print(f"PMCIDs:  {pmcids}")
        print(f"Mode:    {mode}")
        print(f"Profile: {profile.name}")
        print(f"Trace:   {args.trace} (sync only)")
        print("Models:")
        for v in profile.l1_voters:
            print(f"  L1  {v.model:<40} (t={v.temperature})  [{v.provider}]")
        for v in profile.l2_voters:
            print(f"  L2  {v.model:<40} (t={v.temperature})  [{v.provider}]")
        l3 = profile.l3_voter
        print(f"  L3  {l3.model:<40} (t={l3.temperature})  [{l3.provider}]")
        print("\nEnv vars required: GOOGLE_API_KEY, ANTHROPIC_API_KEY")
        return

    artifact_root_path = Path(args.artifact_root) if args.artifact_root else None

    if not args.sync and len(pmcids) > 1:
        _run_all_batch(
            pmcids, poll_interval=args.poll_interval,
            profile_name=args.profile,
            artifact_root=artifact_root_path,
            artifact_run_id=args.artifact_run_id,
            chunk_workers=args.chunk_workers,
            config_path=args.config,
            skip_corpus_relate=args.no_corpus_relate,
        )
    else:
        for pmcid in pmcids:
            logger.info("─── Processing %s (%d/%d) ───", pmcid, pmcids.index(pmcid) + 1, len(pmcids))
            if not args.sync:
                _run_batch(
                    pmcid, poll_interval=args.poll_interval,
                    profile_name=args.profile,
                    artifact_root=artifact_root_path,
                    artifact_run_id=args.artifact_run_id,
                    chunk_workers=args.chunk_workers,
                    config_path=args.config,
                    skip_corpus_relate=args.no_corpus_relate,
                )
            else:
                _run_sync(
                    pmcid, trace=args.trace,
                    start_chunk=args.start_chunk, limit_chunks=args.limit_chunks,
                    profile_name=args.profile,
                    artifact_root=artifact_root_path,
                    artifact_run_id=args.artifact_run_id,
                    chunk_workers=args.chunk_workers,
                    config_path=args.config,
                )


def _run_sync(pmcid: str, trace: bool,
              start_chunk: int = 0, limit_chunks: int | None = None,
              profile_name: str | None = None,
              artifact_root: Path | None = None,
              artifact_run_id: str | None = None,
              chunk_workers: int | None = None,
              config_path: str | Path | None = DEFAULT_CONFIG_PATH) -> None:
    logger.info("Building sync runner…")
    runner, token_usage = build_runner(
        trace=trace,
        profile_name=profile_name,
        artifact_root=artifact_root,
        artifact_run_id=artifact_run_id,
        chunk_workers=chunk_workers,
        config_path=config_path,
    )

    logger.info("Loading paper %s from database…", pmcid)
    try:
        file_data = runner.load_paper_from_db(pmcid)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    n = len(file_data["sentences_with_provenance"])
    logger.info("Loaded %d sentences — starting pipeline", n)

    result = runner.process(file_data, start_chunk=start_chunk, limit_chunks=limit_chunks)

    if result["status"] == "error":
        logger.error("Pipeline failed: %s", result["error"])
        sys.exit(1)

    final_rules = result.get("final_rules", [])
    canonical_rules = result.get("canonical_rules", [])
    relations = result.get("relations", [])
    logger.info("Done — %d final rules", len(final_rules))
    logger.info("Result saved to out/summaries/summaries/%s.json", pmcid)

    stats = _escalation_stats_sync(pmcid, token_usage, runner)
    _save_escalation_report([stats], Path("out/summaries/reports"))

    print(f"\n{'─' * 60}")
    print(f"Results for {pmcid}")
    print(f"{'─' * 60}")
    print(f"  {len(canonical_rules)} canonical rules")
    print(f"  {len(relations)} relations")
    print(f"  {len(final_rules)} final rules")


# Per-model batch pricing (USD per 1M tokens).
# Keys match model IDs in voter_configs.py.
_MODEL_PRICE: dict[str, tuple[float, float]] = {
    # model_id: (input_per_M, output_per_M) — batch / discounted rates
    "gemini-2.5-flash-lite":      (0.10,   0.40),
    "gpt-4o-mini":                (0.075,  0.30),
    "claude-haiku-4-5-20251001":  (0.80,   4.00),
    "gemini-2.5-flash":           (0.15,   0.60),
    "gpt-4.1-mini":               (0.20,   0.80),
    "claude-sonnet-4-6":          (3.00,  15.00),
}


def _level_cost(usage: dict, level: str) -> float:
    """Exact cost for one level using per-model pricing.

    usage format: {level: {model_id: {input, output}}}
    Falls back to 0 for unknown models (logs nothing; add to _MODEL_PRICE to fix).
    """
    total = 0.0
    for model_id, counts in usage.get(level, {}).items():
        inp = counts.get("input", 0)
        out = counts.get("output", 0)
        price = _MODEL_PRICE.get(model_id)
        if price:
            total += (inp * price[0] + out * price[1]) / 1_000_000
    return total


def _baseline_cost(usage: dict) -> float | None:
    """Counterfactual cost if all chunks had gone straight to L3 (no cascade).

    Estimate per-chunk tokens from L1 totals divided by voter count, then price
    at the L3 rate. Returns ``None`` when the baseline cannot be computed
    (no L1 usage seen, or no price configured for the L3 model). A missing
    baseline is reported as "unavailable" rather than silently returning 0,
    which would surface as misleading negative savings for profiles where
    every cascade level uses the same model (e.g. ``smoke_haiku``).
    """
    from pipeline.stages.knowledge_extraction.batch.voter_configs import make_l1_voters
    n_l1_voters = len(make_l1_voters())
    if not usage.get("l1"):
        return None
    l3_model = next(iter(usage.get("l3", {}).keys()), "claude-sonnet-4-6")
    l3_price = _MODEL_PRICE.get(l3_model)
    if l3_price is None:
        return None

    l1_total_inp = sum(c.get("input", 0) for c in usage.get("l1", {}).values())
    l1_total_out = sum(c.get("output", 0) for c in usage.get("l1", {}).values())
    if l1_total_inp == 0 and l1_total_out == 0:
        return None
    baseline_inp = l1_total_inp / n_l1_voters if n_l1_voters else 0
    baseline_out = l1_total_out / n_l1_voters if n_l1_voters else 0
    return (baseline_inp * l3_price[0] + baseline_out * l3_price[1]) / 1_000_000


def _escalation_stats(pmcid: str, handle) -> dict:
    total     = len(handle.chunk_map)
    l2_esc    = len(handle.l2_chunk_ids)
    l3_esc    = len(handle.l3_chunk_ids)
    l1_kept   = total - l2_esc
    l2_kept   = l2_esc - l3_esc
    l3_kept   = l3_esc
    finalized = len(handle.finalized)
    dropped   = total - finalized

    usage = handle.token_usage
    actual_cost = sum(_level_cost(usage, lvl) for lvl in ("l1", "l2", "l3"))
    baseline = _baseline_cost(usage)
    saved = (baseline - actual_cost) if baseline is not None else None

    return {
        "pmcid":            pmcid,
        "mode":             "batch",
        "total_chunks":     total,
        "l1_kept":          l1_kept,
        "l2_escalated":     l2_esc,
        "l2_kept":          l2_kept,
        "l3_escalated":     l3_esc,
        "l3_kept":          l3_kept,
        "finalized":        finalized,
        "dropped":          dropped,
        # Batch path: the handle does not track PipelineCache hits separately;
        # set to 0 so the column renders consistently with sync runs.
        "cache_hits":       0,
        "l1_pct":           round(100 * l1_kept / total, 1) if total else 0.0,
        "token_usage":      usage,
        "est_cost_usd":     round(actual_cost, 6),
        "est_saved_usd":    (round(saved, 6) if saved is not None else None),
    }


def _token_usage_from_records(records) -> dict:
    """Aggregate ``InvocationUsage`` records into the
    ``{level: {model: {input, output}}}`` shape consumed by ``_level_cost``.

    Levels are lowercased ("l1" / "l2" / "l3") to match the historical
    callback path's keys.  Empty input → empty dict.
    """
    usage: dict[str, dict[str, dict[str, int]]] = {"l1": {}, "l2": {}, "l3": {}}
    missing_count = 0
    for r in records or []:
        level = (r.level or "").lower()
        if level not in usage:
            usage.setdefault(level, {})
        entry = usage[level].setdefault(r.model, {"input": 0, "output": 0})
        entry["input"] += int(r.input_tokens or 0)
        entry["output"] += int(r.output_tokens or 0)
        if r.usage_source == "missing":
            missing_count += 1
    if missing_count:
        logger.warning(
            "Cost report: %d MAP invocations had no usage_metadata "
            "(usage_source='missing'). Cost is under-counted for those calls.",
            missing_count,
        )
    return usage


def _escalation_stats_sync(pmcid: str, token_usage: dict, runner) -> dict:
    """Compute cost stats for a sync (real-time) pipeline run.

    token_usage: {level: {model_id: {input, output}}} — same schema as batch.
        Kept as the fallback signal.  When the MAP stage emitted
        ``InvocationUsage`` records (the case in production since the
        ``include_raw=True`` change), those records are preferred because
        ``with_structured_output(...)`` strips LangChain callbacks and the
        callback-derived ``token_usage`` comes back empty.
    runner: KnowledgeExtractionRunner — used to read last_map_escalation_counts
        and last_map_invocation_usage_records.
    """
    ec = runner.last_map_escalation_counts
    total      = ec.get("total", 0)
    l1_kept    = ec.get("l1_kept", 0)
    l2_esc     = ec.get("l2_escalated", 0)
    l2_kept    = ec.get("l2_kept", 0)
    l3_esc     = ec.get("l3_escalated", 0)
    l3_kept    = ec.get("l3_kept", 0)
    finalized  = ec.get("finalized", 0)
    dropped    = ec.get("dropped", 0)
    cache_hits = ec.get("cache_hits", 0)

    # Records-first: read directly from MapStage. Callback-derived
    # token_usage is kept only as a fallback so older runners that don't
    # populate records still produce *something*.
    records = []
    try:
        records = runner.last_map_invocation_usage_records
    except AttributeError:
        records = []
    if records:
        token_usage = _token_usage_from_records(records)

    actual_cost = sum(_level_cost(token_usage, lvl) for lvl in ("l1", "l2", "l3"))
    baseline = _baseline_cost(token_usage)
    saved = (baseline - actual_cost) if baseline is not None else None

    return {
        "pmcid":            pmcid,
        "mode":             "sync",
        "total_chunks":     total,
        "l1_kept":          l1_kept,
        "l2_escalated":     l2_esc,
        "l2_kept":          l2_kept,
        "l3_escalated":     l3_esc,
        "l3_kept":          l3_kept,
        "finalized":        finalized,
        "dropped":          dropped,
        "cache_hits":       cache_hits,
        "l1_pct":           round(100 * l1_kept / total, 1) if total else 0.0,
        "token_usage":      token_usage,
        "est_cost_usd":     round(actual_cost, 6),
        "est_saved_usd":    (round(saved, 6) if saved is not None else None),
    }


def _save_escalation_report(stats: list[dict], output_dir: Path) -> None:
    # When any per-paper baseline is unavailable, the aggregate is too —
    # otherwise we'd be summing apples (real numbers) and oranges (zero
    # standing in for "unknown") and the user couldn't tell the difference.
    any_saved_missing = any(s.get("est_saved_usd") is None for s in stats)
    saved_total = (
        None if any_saved_missing
        else round(sum(s["est_saved_usd"] for s in stats), 6)
    )
    totals = {
        "pmcid":         "TOTAL",
        "mode":          "mixed" if len({s.get("mode") for s in stats}) > 1 else (stats[0].get("mode", "") if stats else ""),
        "total_chunks":  sum(s["total_chunks"]  for s in stats),
        "l1_kept":       sum(s["l1_kept"]       for s in stats),
        "l2_escalated":  sum(s["l2_escalated"]  for s in stats),
        "l2_kept":       sum(s["l2_kept"]        for s in stats),
        "l3_escalated":  sum(s["l3_escalated"]  for s in stats),
        "l3_kept":       sum(s["l3_kept"]        for s in stats),
        "finalized":     sum(s["finalized"]      for s in stats),
        "dropped":       sum(s["dropped"]        for s in stats),
        "cache_hits":    sum(s.get("cache_hits", 0) for s in stats),
        "est_cost_usd":  round(sum(s["est_cost_usd"]  for s in stats), 6),
        "est_saved_usd": saved_total,
    }
    t = totals["total_chunks"]
    totals["l1_pct"] = round(100 * totals["l1_kept"] / t, 1) if t else 0.0

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "est_saved_usd is vs an all-L3 baseline where each chunk is processed "
            "once by L3 (per-chunk tokens estimated as L1 total / n_l1_voters)."
        ),
        "papers": stats,
        "totals": totals,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    # ── JSON ─────────────────────────────────────────────────────────────────
    json_path = output_dir / f"escalation_report_{ts}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Escalation report (JSON) → %s", json_path)

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = output_dir / f"escalation_report_{ts}.csv"
    _csv_fields = [
        "pmcid", "mode", "total_chunks", "l1_kept", "l1_pct",
        "l2_escalated", "l2_kept", "l3_escalated", "l3_kept",
        "finalized", "dropped", "cache_hits", "est_cost_usd", "est_saved_usd",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_csv_fields, extrasaction="ignore")
        w.writeheader()
        for s in stats:
            w.writerow(s)
        w.writerow(totals)
    logger.info("Escalation report (CSV)  → %s", csv_path)

    # ── Console ───────────────────────────────────────────────────────────────
    def _fmt_money(v) -> str:
        return "    n/a " if v is None else f"{v:>8.5f}"

    print(f"\n{'─' * 90}")
    print("Cascade Escalation Report")
    print(f"{'─' * 90}")
    print(f"{'PMCID':<45} {'Md':>2} {'Total':>5} {'Cch':>4} {'L1%':>5} {'→L2':>4} {'→L3':>4} {'Drop':>4} {'Cost$':>8} {'Saved$':>8}")
    print(f"{'─' * 90}")
    for s in stats:
        print(
            f"{s['pmcid']:<45} {s.get('mode','?')[0]:>2} {s['total_chunks']:>5}"
            f" {s.get('cache_hits', 0):>4} {s['l1_pct']:>4.0f}%"
            f" {s['l2_escalated']:>4} {s['l3_escalated']:>4} {s['dropped']:>4}"
            f" {s['est_cost_usd']:>8.5f} {_fmt_money(s['est_saved_usd'])}"
        )
    print(f"{'─' * 90}")
    print(
        f"{'TOTAL':<45} {'':>2} {totals['total_chunks']:>5}"
        f" {totals.get('cache_hits', 0):>4} {totals['l1_pct']:>4.0f}%"
        f" {totals['l2_escalated']:>4} {totals['l3_escalated']:>4} {totals['dropped']:>4}"
        f" {totals['est_cost_usd']:>8.5f} {_fmt_money(totals['est_saved_usd'])}"
    )
    print(f"{'─' * 90}")
    print(f"\nActual cost:    ${totals['est_cost_usd']:.5f}")
    if totals["est_saved_usd"] is None:
        print("Saved vs L3-only baseline: unavailable (no L1 usage or L3 price missing)")
    elif totals["est_saved_usd"] < 0:
        print(
            f"Saved vs L3-only baseline: ${totals['est_saved_usd']:.5f}  "
            "(NEGATIVE — cascade was more expensive than L3-only; expected for "
            "single-model profiles like smoke_haiku)"
        )
    else:
        print(f"Saved vs L3-only baseline: ${totals['est_saved_usd']:.5f}")


def _run_corpus_relate(
    summaries_dir: Path,
    output_path: Path,
    *,
    skip: bool = False,
    relate_config=None,
) -> None:
    """Post-batch cross-paper RelateStage over the per-paper summary JSONs.

    Replay-only — NLI is local (HuggingFace). Zero API calls. Non-fatal:
    failures log a warning but do not crash the orchestrator.

    Skipped when:
      - ``skip=True`` (CLI ``--no-corpus-relate``)
      - the source dir is missing
      - fewer than two ``*.json`` files exist (no cross-paper pairs to relate)

    ``relate_config`` is a ``RelateConfig`` instance (typically
    ``runner._sum_cfg.relate`` from the active batch/sync runner). When
    provided, its ``scope_aware_nli`` and ``use_verbatim_for_nli`` knobs
    are forwarded to ``CorpusRelateStage``; without it the stage falls
    back to its own dataclass defaults. The forwarding closes a previously
    silent gap where ``configs/run.yaml`` flipped these flags for the
    within-paper RELATE stage but NOT for the post-batch corpus_relate
    pass — see the A/B-test trace 2026-05-27.

    Writes ``corpus_relations.json`` at ``output_path``. The inspector
    (`scripts/inspect_pipeline_output.py`) auto-detects this filename in the
    parent of the source dir, so the default layout
    (``out/summaries/summaries/*.json`` + ``out/summaries/corpus_relations.json``)
    is picked up by ``--batch-dir`` runs without extra flags.
    """
    if skip:
        logger.info("CORPUS RELATE: skipped via --no-corpus-relate")
        return
    if not summaries_dir.exists():
        logger.warning(
            "CORPUS RELATE: source dir not found: %s — skipping", summaries_dir,
        )
        return
    json_files = sorted(
        p for p in summaries_dir.glob("*.json")
        if not p.name.startswith("_")           # skip _archive_*, etc.
    )
    if len(json_files) < 2:
        logger.info(
            "CORPUS RELATE: %d JSON file(s) in %s — need ≥2 for cross-paper "
            "comparison; skipping.", len(json_files), summaries_dir,
        )
        return

    try:
        from pipeline.stages.knowledge_extraction.helpers.corpus_relate import (  # noqa: PLC0415
            CorpusRelateStage,
        )
        stage_kwargs: dict = {}
        if relate_config is not None:
            stage_kwargs["entailment_threshold"]    = relate_config.entailment_threshold
            stage_kwargs["contradiction_threshold"] = relate_config.contradiction_threshold
            stage_kwargs["scope_aware_nli"]         = relate_config.scope_aware_nli
            stage_kwargs["use_verbatim_for_nli"]    = relate_config.use_verbatim_for_nli
            logger.info(
                "CORPUS RELATE: pooling rules from %d papers in %s → %s "
                "(scope_aware_nli=%s, use_verbatim_for_nli=%s)",
                len(json_files), summaries_dir, output_path,
                relate_config.scope_aware_nli, relate_config.use_verbatim_for_nli,
            )
        else:
            logger.info(
                "CORPUS RELATE: pooling rules from %d papers in %s → %s "
                "(using CorpusRelateStage defaults — no run.yaml threading)",
                len(json_files), summaries_dir, output_path,
            )
        relations = CorpusRelateStage(**stage_kwargs).relate_from_dir(
            source_dir=summaries_dir,
            output_path=output_path,
        )
        logger.info(
            "CORPUS RELATE: wrote %d relations → %s",
            len(relations), output_path,
        )
    except Exception as exc:  # noqa: BLE001 — non-fatal post-step
        logger.warning(
            "CORPUS RELATE failed (non-fatal): %s — per-paper JSONs are still "
            "valid. Re-run manually with "
            "`python -c \"from pipeline.stages.knowledge_extraction.helpers.corpus_relate "
            "import CorpusRelateStage; CorpusRelateStage().relate_from_dir(...)\"`.",
            exc, exc_info=True,
        )


def _run_all_batch(
    pmcids: list[str],
    poll_interval: int = DEFAULT_POLL_INTERVAL_SEC,
    *,
    profile_name: str | None = None,
    artifact_root: Path | None = None,
    artifact_run_id: str | None = None,
    chunk_workers: int | None = None,
    config_path: str | Path | None = DEFAULT_CONFIG_PATH,
    skip_corpus_relate: bool = False,
) -> None:
    """Submit all papers simultaneously, poll together, finalize all."""
    from pipeline.stages.knowledge_extraction.batch import BatchPhase
    from pipeline.stages.knowledge_extraction.runner import KnowledgeExtractionRunner

    runner = build_batch_runner(
        profile_name=profile_name,
        artifact_root=artifact_root,
        artifact_run_id=artifact_run_id,
        db=_open_db_connection("build_batch_runner (multi)"),
        chunk_workers=chunk_workers,
        config_path=config_path,
    )

    # ── Phase 1: submit all papers in parallel ────────────────────────────────
    def _submit(pmcid: str):
        try:
            file_data = KnowledgeExtractionRunner.load_paper_from_db(pmcid)
        except ValueError as exc:
            logger.error("[%s] Load failed: %s", pmcid, exc)
            return pmcid, None
        handle = runner.load_or_submit(file_data)
        return pmcid, handle

    _MAX_WORKERS = 16

    handles: dict[str, object] = {}
    logger.info("Loading or submitting %d papers in parallel…", len(pmcids))
    with ThreadPoolExecutor(max_workers=min(len(pmcids), _MAX_WORKERS)) as ex:
        futures = {ex.submit(_submit, p): p for p in pmcids}
        for fut in as_completed(futures):
            pmcid, handle = fut.result()
            if handle is not None:
                handles[pmcid] = handle
                logger.info("[%s] Submitted (phase=%s)", pmcid, handle.phase.value)

    if not handles:
        logger.error("No papers submitted successfully.")
        return

    # ── Phase 2: poll all handles until every one is COMPLETE ─────────────────
    def _advance_one(pmcid: str) -> tuple[str, object]:
        return pmcid, runner.advance(handles[pmcid])

    while True:
        pending = [p for p, h in handles.items() if h.phase != BatchPhase.COMPLETE]
        if not pending:
            break
        logger.info("Advancing %d / %d pending papers…", len(pending), len(handles))
        with ThreadPoolExecutor(max_workers=min(len(pending), _MAX_WORKERS)) as ex:
            futures = {ex.submit(_advance_one, p): p for p in pending}
            for fut in as_completed(futures):
                pmcid, handle = fut.result()
                handles[pmcid] = handle
                logger.info("[%s] phase=%s", pmcid, handle.phase.value)
        pending = [p for p, h in handles.items() if h.phase != BatchPhase.COMPLETE]
        if not pending:
            break
        logger.info("%d / %d papers still pending — sleeping %ds…",
                    len(pending), len(handles), poll_interval)
        time.sleep(poll_interval)

    # ── Phase 3: finalize all ─────────────────────────────────────────────────
    # Sequential — finalize() is CPU/MPS-bound (NLI on a single device, UMLS
    # lookups, DB writes); zero remote LLM calls.  Parallelising it serialises
    # at the MPS device anyway while paying N× the model-load cost (each thread
    # races _get_nli_pipe before the cache is populated → N copies of the NLI
    # model load simultaneously, then N-1 are GC'd).  Phases 1 + 2 stay
    # parallel because they are IO-bound (Anthropic/OpenAI batch APIs).
    logger.info("All papers complete — finalising sequentially…")

    escalation_stats: list[dict] = []
    results: dict[str, dict] = {}
    for pmcid in handles:
        result = runner.finalize(handles[pmcid])
        escalation_stats.append(_escalation_stats(pmcid, handles[pmcid]))
        results[pmcid] = result

    for pmcid in handles:
        result = results.get(pmcid, {})
        final_rules = result.get("final_rules", [])
        canonical_rules = result.get("canonical_rules", [])
        relations = result.get("relations", [])
        print(f"\n{'─' * 60}")
        print(f"Results for {pmcid}")
        print(f"{'─' * 60}")
        print(f"  {len(canonical_rules)} canonical rules")
        print(f"  {len(relations)} relations")
        print(f"  {len(final_rules)} final rules")

    _save_escalation_report(escalation_stats, Path("out/summaries/reports"))

    # Post-batch cross-paper RelateStage over the freshly-finalised JSONs.
    # Runs *after* every paper has reached finalize() so the source dir holds
    # the full set. Replay-only (local NLI), zero API calls. Forwards the
    # active relate config so the corpus_relate NLI input matches the
    # within-paper one (scope_aware_nli / use_verbatim_for_nli).
    _run_corpus_relate(
        summaries_dir=Path("out/summaries/summaries"),
        output_path=Path("out/summaries/corpus_relations.json"),
        skip=skip_corpus_relate,
        relate_config=getattr(runner, "_cfg_full", None) and runner._cfg_full.relate,
    )


def _run_batch(
    pmcid: str,
    poll_interval: int = DEFAULT_POLL_INTERVAL_SEC,
    *,
    profile_name: str | None = None,
    artifact_root: Path | None = None,
    artifact_run_id: str | None = None,
    chunk_workers: int | None = None,
    config_path: str | Path | None = DEFAULT_CONFIG_PATH,
    skip_corpus_relate: bool = False,
) -> None:
    from pipeline.stages.knowledge_extraction.batch import BatchPhase
    from pipeline.stages.knowledge_extraction.runner import KnowledgeExtractionRunner

    logger.info("Building batch runner…")
    runner = build_batch_runner(
        profile_name=profile_name,
        artifact_root=artifact_root,
        artifact_run_id=artifact_run_id,
        db=_open_db_connection("build_batch_runner (single)"),
        chunk_workers=chunk_workers,
        config_path=config_path,
    )

    logger.info("Loading paper %s from database…", pmcid)
    try:
        file_data = KnowledgeExtractionRunner.load_paper_from_db(pmcid)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    handle = runner.load_or_submit(file_data)

    while handle.phase != BatchPhase.COMPLETE:
        handle = runner.advance(handle)
        if handle.phase == BatchPhase.COMPLETE:
            break
        n_pending = sum(1 for j in handle.jobs if j.status not in ("completed", "failed"))
        logger.info("[%s] Phase: %s  |  %d job(s) still running — sleeping %ds…",
                    pmcid, handle.phase.value, n_pending, poll_interval)
        time.sleep(poll_interval)

    logger.info("[%s] All batches complete — finalising…", pmcid)
    _save_escalation_report([_escalation_stats(pmcid, handle)], Path("out/summaries/reports"))
    result = runner.finalize(handle)
    final_rules = result.get("final_rules", [])
    canonical_rules = result.get("canonical_rules", [])
    relations = result.get("relations", [])
    print(f"\n{'─' * 60}")
    print(f"Results for {pmcid}")
    print(f"{'─' * 60}")
    print(f"  {len(canonical_rules)} canonical rules")
    print(f"  {len(relations)} relations")
    print(f"  {len(final_rules)} final rules")

    # Post-batch cross-paper RelateStage. Works on the full source dir, so a
    # single-paper run still relates against previously-summarised papers in
    # the same dir. Replay-only (local NLI), zero API calls. Forwards the
    # active relate config (see _run_all_batch comment).
    _run_corpus_relate(
        summaries_dir=Path("out/summaries/summaries"),
        output_path=Path("out/summaries/corpus_relations.json"),
        skip=skip_corpus_relate,
        relate_config=getattr(runner, "_cfg_full", None) and runner._cfg_full.relate,
    )


if __name__ == "__main__":
    main()
