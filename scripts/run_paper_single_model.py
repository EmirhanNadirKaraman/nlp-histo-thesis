"""
Run the full knowledge_extraction pipeline on one or more papers using a single LLM.

Bypasses ABC multi-model cascading by wiring the same model into all three
voter slots and setting theta=0.0, so Level-1 always "agrees" and escalation
never fires.  Useful for debugging Phases 1–6 without needing Azure or Claude
credentials.

Usage
-----
Single paper:
    PYTHONPATH=. python scripts/run_paper_single_model.py PMC10047158

Multiple papers:
    PYTHONPATH=. python scripts/run_paper_single_model.py PMC111 PMC222 PMC333

Random selection (same 5 papers every time):
    PYTHONPATH=. python scripts/run_paper_single_model.py --random 5
    PYTHONPATH=. python scripts/run_paper_single_model.py --random 5 --seed 99

From a paper-selection YAML:
    PYTHONPATH=. python scripts/run_paper_single_model.py \\
        --from-selection configs/paper_selection/smoke_v1.yaml \\
        --model haiku --trace

With options:
    PYTHONPATH=. python scripts/run_paper_single_model.py PMC10047158 --model haiku
    PYTHONPATH=. python scripts/run_paper_single_model.py PMC10047158 --trace
    PYTHONPATH=. python scripts/run_paper_single_model.py PMC10047158 --skip-nli
    PYTHONPATH=. python scripts/run_paper_single_model.py PMC10047158 --skip-ner
    PYTHONPATH=. python scripts/run_paper_single_model.py PMC10047158 --chunks 3
    PYTHONPATH=. python scripts/run_paper_single_model.py --list-only
    PYTHONPATH=. python scripts/run_paper_single_model.py PMC10047158 --force-rerun

Options
-------
--trace        Write JSONL traces to out/summaries/traces/.
--skip-nli     Disable NLI grounding filter and RELATE stage (faster, no GPU needed).
--chunks N     Limit to the first N×10 sentences (default: all).
--list-only    Print available PMCIDs from the database and exit.
--force-rerun  Ignore cached result JSON and reprocess from scratch.
--no-corpus    Skip corpus-level relation stage (only relevant when ≥2 papers are run).
--no-html      Skip HTML inspector generation.
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from pipeline.stages.knowledge_extraction import KnowledgeExtractionRunner

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_single_model")


# ── LLM factory ────────────────────────────────────────────────────────────────

MODEL_CHOICES = ("gemini", "haiku", "sonnet")


def build_llm(model: str = "gemini"):
    """Single LLM factory. ``model`` selects the underlying provider.

    - "gemini" — Gemini 2.5 Flash Lite via Vertex AI (cheap default).
    - "haiku"  — Claude Haiku 4.5 via Anthropic (needs ANTHROPIC_API_KEY).
    - "sonnet" — Claude Sonnet 4.6 via Anthropic (needs ANTHROPIC_API_KEY).
    """
    if model == "gemini":
        import os
        from langchain_google_vertexai import ChatVertexAI

        return ChatVertexAI(
            model="gemini-2.5-flash-lite",
            project=os.environ["VERTEX_PROJECT"],
            location="global",
            temperature=0.0,
            timeout=20,   # fail fast: normal responses are 5-10s
        )
    if model == "haiku":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            temperature=0.0,
            timeout=30,
        )
    if model == "sonnet":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-sonnet-4-6",
            temperature=0.0,
            timeout=60,
        )
    raise ValueError(f"unknown --model {model!r}; expected one of {MODEL_CHOICES}")


# ── Runner factories ───────────────────────────────────────────────────────────

def build_batch_runners(
    *,
    skip_nli: bool,
    skip_ner: bool,
):
    """
    Return (batch_runner, sync_runner).

    batch_runner  — BatchKnowledgeExtractionRunner using Claude Haiku 4.5 via the
                    Anthropic batch API (no Vertex rate limits, 50 % cheaper).
                    Handles the MAP phase for all papers simultaneously.

    sync_runner   — Standard KnowledgeExtractionRunner using Gemini Flash Lite for
                    NORMALIZE → CANONICALIZE (a handful of LLM calls per paper,
                    not subject to the per-minute rate limits that hurt MAP).
                    Its MAP cache is pre-populated with the batch results so MAP
                    is skipped entirely when process() is called.
    """
    from pipeline.stages.knowledge_extraction.batch.runner import BatchKnowledgeExtractionRunner
    from pipeline.stages.knowledge_extraction.batch.models import VoterBatchConfig
    from pipeline.stages.knowledge_extraction import KnowledgeExtractionRunner
    from pipeline.stages.knowledge_extraction.config import (
        KnowledgeExtractionConfig, MapConfig, GroundingConfig,
    )
    from database import get_db_connection

    haiku  = VoterBatchConfig(model="claude-haiku-4-5-20251001",  provider="claude")
    sonnet = VoterBatchConfig(model="claude-sonnet-4-6", provider="claude")
    sync_llm = build_llm()  # Gemini Flash Lite — used for NORMALIZE/CANONICALIZE

    # Single voter → always KEEP, no escalation.
    cfg = KnowledgeExtractionConfig(
        map=MapConfig(theta=0.0, reject_theta=-1.0),
        grounding=GroundingConfig(threshold=None if skip_nli else 0.3),
    )

    batch_runner = BatchKnowledgeExtractionRunner(
        l1_voters=[haiku],
        l2_voters=[haiku],
        l3_model=sonnet,
        escalation_llm=sync_llm,
        config=cfg,
        output_dir=Path("out/summaries"),
        db=get_db_connection(),
        run_ner=not skip_ner,
    )
    sync_runner = KnowledgeExtractionRunner(
        voter_llms=[sync_llm],
        level2_voter_llms=[sync_llm],
        escalation_llm=sync_llm,
        config=cfg,
        output_dir=Path("out/summaries"),
        db=get_db_connection(),
        run_ner=not skip_ner,
    )
    return batch_runner, sync_runner


def build_runner(
    *,
    trace: bool,
    skip_nli: bool,
    force_rerun: bool = False,
    skip_ner: bool = False,
    model: str = "gemini",
) -> "KnowledgeExtractionRunner":
    from pipeline.stages.knowledge_extraction import KnowledgeExtractionRunner
    from pipeline.stages.knowledge_extraction.config import (
        KnowledgeExtractionConfig, MapConfig, GroundingConfig,
    )
    from database import get_db_connection

    llm = build_llm(model)

    # theta=0.0 → first voter always "agrees" with itself; no cascade fires.
    # reject_theta=-1.0 ensures nothing is hard-rejected.
    # All three LLM slots receive the same model instance.
    cfg = KnowledgeExtractionConfig(
        map=MapConfig(theta=0.0, reject_theta=-1.0),
        grounding=GroundingConfig(threshold=0.3 if not skip_nli else None),
    )
    return KnowledgeExtractionRunner(
        voter_llms=[llm],
        level2_voter_llms=[llm],
        escalation_llm=llm,
        config=cfg,
        output_dir=Path("out/summaries"),
        trace_enabled=trace,
        db=get_db_connection(),
        force_rerun=force_rerun,
        run_ner=not skip_ner,
    )


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _fetch_all_pmcids() -> list[str]:
    from database import get_db_connection
    from database.models import Document

    db = get_db_connection()
    with db.session_scope() as session:
        return [
            row.pmcid
            for row in session.query(Document.pmcid).order_by(Document.pmcid).all()
        ]


def _load_selection_yaml(path: Path) -> list[str]:
    """Read a paper-selection YAML and return a de-duplicated list of PMCIDs.

    Expected shape (produced by ``eval.paper_selection.export.write_calibration_set``)::

        version_name:
          related: [PMC..., ...]
          diverse: [PMC..., ...]
          hard:    [PMC..., ...]

    Buckets concatenate in related → diverse → hard order; duplicates are
    dropped while preserving first occurrence.
    """
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path}: expected a top-level mapping with a version key")

    # The YAML wraps everything in a single version key; pick the first one.
    version_key = next(iter(data))
    buckets = data[version_key] or {}
    if not isinstance(buckets, dict):
        raise ValueError(f"{path}: version '{version_key}' is not a mapping")

    ordered: list[str] = []
    seen: set[str] = set()
    for bucket in ("related", "diverse", "hard"):
        for pmcid in buckets.get(bucket) or []:
            if pmcid not in seen:
                ordered.append(pmcid)
                seen.add(pmcid)
    if not ordered:
        raise ValueError(f"{path}: no PMCIDs found under related/diverse/hard")
    return ordered


def run_batch_mode(
    pmcids: list[str],
    args,
) -> list[dict]:
    """
    Submit MAP for all pmcids in parallel via the Anthropic batch API, poll
    until complete, then run NORMALIZE→RESOLVE synchronously for each paper.

    Handles are persisted to out/summaries/batch_handles/{pmcid}.batch.json so
    the process can be interrupted and resumed — restart with the same command
    and the script picks up from where it left off.
    """
    import time
    from pipeline.stages.knowledge_extraction.batch.models import BatchPhase
    from pipeline.stages.knowledge_extraction.models import AuditableSummary

    logger.info("Building batch runners (Claude Haiku 4.5 + Gemini Flash Lite)…")
    batch_runner, sync_runner = build_batch_runners(
        skip_nli=args.skip_nli,
        skip_ner=args.skip_ner,
    )

    # Load file data
    file_data_map: dict[str, dict] = {}
    for pmcid in pmcids:
        logger.info("Loading %s from database…", pmcid)
        try:
            fd = sync_runner.load_paper_from_db(pmcid)
            if args.chunks is not None:
                cap = args.chunks * 10
                fd = {**fd, "sentences_with_provenance": fd["sentences_with_provenance"][:cap]}
                logger.info("[%s] Capped to first %d sentences", pmcid, cap)
            file_data_map[pmcid] = fd
        except ValueError as exc:
            logger.error("%s", exc)

    if not file_data_map:
        logger.error("No papers loaded — aborting.")
        return []

    # Submit (or resume) all papers
    handles: dict[str, object] = {}
    for pmcid, fd in file_data_map.items():
        handle = batch_runner.load_or_submit(fd)
        handles[pmcid] = handle
        logger.info("[%s] Phase: %s", pmcid, handle.phase.value)

    # Poll until all COMPLETE
    poll_interval = args.poll_interval
    while True:
        pending = [p for p, h in handles.items() if h.phase != BatchPhase.COMPLETE]
        if not pending:
            logger.info("All batch MAP phases complete.")
            break
        logger.info(
            "%d paper(s) still in batch: %s — sleeping %ds "
            "(Ctrl+C safe; handles on disk, restart to resume)",
            len(pending), ", ".join(pending), poll_interval,
        )
        time.sleep(poll_interval)
        for pmcid in pending:
            handles[pmcid] = batch_runner.advance(handles[pmcid])
            logger.info("[%s] Phase after advance: %s", pmcid, handles[pmcid].phase.value)

    # Populate sync runner MAP cache and run NORMALIZE→RESOLVE for each paper
    results: list[dict] = []
    for pmcid, handle in handles.items():
        logger.info("[%s] Populating MAP cache from batch results…", pmcid)
        for chunk_id, sentences in handle.chunk_map.items():
            if chunk_id in handle.finalized:
                summary = AuditableSummary.model_validate(handle.finalized[chunk_id])
                sync_runner._cache.set_map(sentences, summary)
        sync_runner._cache.save()

        logger.info("[%s] Running NORMALIZE → RESOLVE (MAP from cache)…", pmcid)
        result = sync_runner.process(file_data_map[pmcid])
        if result["status"] == "error":
            logger.error("[%s] Pipeline failed: %s", pmcid, result.get("error"))
        else:
            _print_result(result)
        results.append(result)

    return results


def list_pmcids() -> None:
    pmcids = _fetch_all_pmcids()
    if not pmcids:
        print("No documents found in database.")
    else:
        print(f"{len(pmcids)} document(s) available:")
        for p in pmcids:
            print(f"  {p}")


def sample_pmcids(n: int, seed: int) -> list[str]:
    pmcids = _fetch_all_pmcids()
    if not pmcids:
        logger.error("No documents found in database.")
        sys.exit(1)
    if n > len(pmcids):
        logger.warning("Requested %d papers but only %d available; using all.", n, len(pmcids))
        n = len(pmcids)
    rng = random.Random(seed)
    chosen = rng.sample(pmcids, n)
    logger.info("Sampled %d papers (seed=%d): %s", n, seed, ", ".join(chosen))
    return chosen


# ── Result printer ─────────────────────────────────────────────────────────────

def _print_result(result: dict) -> None:
    pmcid = result["pmcid"]
    rules = result.get("rules", [])
    final_rules = result.get("final_rules", [])
    canonical_rules = result.get("canonical_rules", [])
    relations = result.get("relations", [])

    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  PMCID          : {pmcid}")
    print(f"  Rules (RULE)   : {len(rules)}")
    print(f"  Canonical rules: {len(canonical_rules)}")
    print(f"  Relations      : {len(relations)}")
    print(f"  Final rules    : {len(final_rules)}")
    print(sep)

    if result.get("summary"):
        print("\n── Narrative summary ──\n")
        print(result["summary"])

    if final_rules:
        print("\n── Top final rules (by score) ──\n")
        for fr in final_rules[:10]:
            flag = " ⚠ CONTRADICTED" if fr.get("is_contradicted") else ""
            print(
                f"  [{fr['final_score']:.3f}]{flag}  "
                f"{fr['subject_entity']} → {fr['outcome_entity']} "
                f"[{fr['relation_type']}] [{fr.get('direction') or '?'}]\n"
                f"    \"{fr['predicate_text'][:100]}\"\n"
                f"    conflicted={fr.get('is_conflicted')} coverage={fr.get('study_coverage')}  "
                f"pmcids={fr['supporting_pmcids']}\n"
            )

    if relations:
        print("\n── Relations ──\n")
        for rel in relations:
            print(
                f"  {rel['relation_type']:<15}  "
                f"{rel['rule_id_a']}  ↔  {rel['rule_id_b']}  "
                f"(A→B={rel['nli_score_a_to_b']:.2f}, B→A={rel['nli_score_b_to_a']:.2f})"
            )

    out_path = Path("out/summaries/summaries") / f"{pmcid}.json"
    print(f"\n  Full result → {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full pipeline with a single Gemini 2.5 Flash Lite model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pmcid", nargs="*", help="PubMed Central ID(s), e.g. PMC10047158 PMC222")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="gemini",
                        help="Which single LLM to wire into every voter slot "
                             "(default: gemini). Use 'haiku' for an Anthropic smoke run.")
    parser.add_argument("--from-selection", default=None, metavar="PATH",
                        help="Path to a paper-selection YAML "
                             "(e.g. configs/paper_selection/smoke_v1.yaml). "
                             "Loads all PMCIDs from related/diverse/hard. "
                             "Mutually exclusive with positional PMCIDs and --random.")
    parser.add_argument("--trace",       action="store_true", help="Write JSONL traces")
    parser.add_argument("--skip-nli",    action="store_true", help="Disable NLI (faster)")
    parser.add_argument("--chunks",      type=int, default=None,
                        help="Limit to first N chunks (each chunk = 10 sentences)")
    parser.add_argument("--list-only",   action="store_true", help="Print available PMCIDs and exit")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore cached result JSON and reprocess")
    parser.add_argument("--skip-ner",    action="store_true", help="Skip NER + UMLS linking")
    parser.add_argument("--no-corpus",   action="store_true",
                        help="Skip corpus-level relation stage (only relevant when ≥2 papers)")
    parser.add_argument("--no-html",     action="store_true", help="Skip HTML inspector generation")
    parser.add_argument("--random",      type=int, default=None, metavar="N",
                        help="Pick N random papers from the database (cannot be combined with pmcid args)")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Random seed for --random (default: 42)")
    parser.add_argument("--batch",       action="store_true",
                        help="Use Anthropic batch API for MAP (no rate limits, 50%% cheaper). "
                             "Requires ANTHROPIC_API_KEY. Polls until complete, then runs "
                             "NORMALIZE→RESOLVE synchronously.")
    parser.add_argument("--poll-interval", type=int, default=30, metavar="SECS",
                        help="Seconds between batch status checks (default: 30). "
                             "Only used with --batch.")
    args = parser.parse_args()

    if args.list_only:
        list_pmcids()
        return

    sources_set = sum(
        1 for v in (args.pmcid, args.random, args.from_selection) if v
    )
    if sources_set > 1:
        parser.error(
            "pass at most one of: positional PMCIDs, --random N, --from-selection PATH"
        )

    if args.from_selection is not None:
        sel_path = Path(args.from_selection)
        try:
            pmcids = _load_selection_yaml(sel_path)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(f"--from-selection: {exc}")
        logger.info("Loaded %d PMCIDs from %s", len(pmcids), sel_path)
    elif args.random is not None:
        pmcids = sample_pmcids(args.random, args.seed)
    elif args.pmcid:
        # Normalise explicit PMCIDs
        pmcids = [
            (p.strip() if p.strip().startswith("PMC") else "PMC" + p.strip())
            for p in args.pmcid
        ]
    else:
        parser.print_help()
        sys.exit(1)

    # ── Batch mode (Claude Haiku batch API for MAP) ────────────────────────────
    if args.batch:
        results = run_batch_mode(pmcids, args)
        n_ok = sum(1 for r in results if r["status"] in ("success", "skipped"))
    else:
        # ── Sync mode (one paper at a time, single LLM wired into every slot) ──
        logger.info("Building single-model runner (model=%s)…", args.model)
        runner = build_runner(
            trace=args.trace,
            skip_nli=args.skip_nli,
            force_rerun=args.force_rerun,
            skip_ner=args.skip_ner,
            model=args.model,
        )

        results = []
        for pmcid in pmcids:
            logger.info("Loading %s from database…", pmcid)
            try:
                file_data = runner.load_paper_from_db(pmcid)
            except ValueError as exc:
                logger.error("%s", exc)
                continue

            sentences = file_data["sentences_with_provenance"]
            if args.chunks is not None:
                cap = args.chunks * 10
                sentences = sentences[:cap]
                file_data = {**file_data, "sentences_with_provenance": sentences}
                logger.info("Capped to first %d sentences (%d chunks)", cap, args.chunks)

            logger.info("Starting pipeline on %d sentences…", len(sentences))
            result = runner.process(file_data)

            if result["status"] == "error":
                logger.error("Pipeline failed for %s: %s", pmcid, result.get("error"))
            else:
                _print_result(result)
            results.append(result)

    n_ok = sum(1 for r in results if r["status"] in ("success", "skipped"))
    if len(pmcids) > 1:
        logger.info("Batch complete: %d/%d succeeded", n_ok, len(pmcids))

    # ── Corpus-level relation stage (requires ≥2 successful papers) ───────────
    summaries_dir = Path("out/summaries/summaries")
    corpus_json   = Path("out/summaries/corpus_relations.json")
    ran_corpus    = False

    if not args.no_corpus and not args.skip_nli and n_ok >= 2:
        logger.info("Running corpus-level relation stage…")
        try:
            from pipeline.stages.knowledge_extraction.stages.corpus_relate import CorpusRelateStage
            CorpusRelateStage().relate_from_dir(summaries_dir, corpus_json)
            ran_corpus = True
        except Exception as exc:
            logger.warning("Corpus relate failed (non-fatal): %s", exc)
    elif n_ok >= 2 and args.skip_nli:
        logger.info("Skipping corpus relate (--skip-nli)")
    elif n_ok < 2:
        logger.info("Skipping corpus relate (need ≥2 successful papers; got %d)", n_ok)

    # ── HTML inspector ─────────────────────────────────────────────────────────
    if not args.no_html:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from inspect_pipeline_output import render, render_batch

            if len(pmcids) == 1 and n_ok == 1:
                # Single paper → per-paper inspector only
                pmcid = pmcids[0]
                json_path = summaries_dir / f"{pmcid}.json"
                html_path = Path("out/inspector") / f"{pmcid}.html"
                render(json_path, html_path)
                print(f"  Inspector → {html_path}")
            else:
                # Multiple papers → batch index + per-paper inspectors
                render_batch(
                    summaries_dir,
                    Path("out/inspector"),
                    corpus_relations_path=corpus_json if ran_corpus else None,
                )
                print("  Batch inspector → out/inspector/index.html")
        except Exception as exc:
            logger.warning("Inspector generation failed (non-fatal): %s", exc)


if __name__ == "__main__":
    main()
