"""
Diagnostic report for Phase 1–3 pipeline on a real paper.

Runs MAP → post-MAP scoring → NORMALIZE → GROUP and writes a structured
JSON report. Does NOT run RELATE or RESOLVE.

Usage:
    python scripts/inspect/inspect_phase123_pipeline.py PMC10047158
    python scripts/inspect/inspect_phase123_pipeline.py PMC10047158 --chunks 4
    python scripts/inspect/inspect_phase123_pipeline.py --list-only

Options:
    --chunks N      Process only the first N chunks (default: 3).
    --list-only     Print available PMCIDs and exit.

Output: out/inspect/phase123_<PMCID>_<timestamp>.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def list_pmcids() -> None:
    from database import get_db_connection
    from database.models import Document

    db = get_db_connection()
    with db.session_scope() as session:
        pmcids = [
            row.pmcid
            for row in session.query(Document.pmcid).order_by(Document.pmcid).all()
        ]
    if not pmcids:
        print("No documents found in database.")
    else:
        print(f"{len(pmcids)} document(s) in database:")
        for p in pmcids:
            print(f"  {p}")


def load_sentences(pmcid: str) -> list[dict]:
    import spacy
    from database import get_db_connection, Document, TextElement

    nlp = spacy.load("en_core_sci_sm")
    db = get_db_connection()

    sentences = []
    with db.session_scope() as session:
        doc = session.query(Document).filter_by(pmcid=pmcid).first()
        if doc is None:
            print(f"PMCID {pmcid!r} not found. Run --list-only to see available PMCIDs.")
            sys.exit(1)
        rows = (
            session.query(TextElement)
            .filter_by(document_id=doc.id)
            .order_by(TextElement.position_in_section)
            .all()
        )
        for te in rows:
            for sent in nlp(te.text_content).sents:
                text = sent.text.strip()
                if text:
                    sentences.append({
                        "pmcid": pmcid,
                        "text_element_id": te.id,
                        "sentence": text,
                    })
    logger.info("Loaded %d sentences for %s", len(sentences), pmcid)
    return sentences


def build_llm():
    from langchain_google_vertexai import ChatVertexAI
    import os

    return ChatVertexAI(
        model="gemini-2.5-flash-lite",
        project=os.environ["VERTEX_PROJECT"],
        location="global",
        temperature=0.0,
    )


def main() -> None:
    import time
    t0 = time.perf_counter()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pmcid", nargs="?")
    parser.add_argument("--chunks", type=int, default=3,
                        help="Number of chunks to process (default: 3)")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--skip-nli", action="store_true",
                        help="Skip NLI grounding scoring (faster, no grounding_score)")
    args = parser.parse_args()

    if args.list_only:
        list_pmcids()
        return

    if not args.pmcid:
        parser.print_help()
        sys.exit(1)

    pmcid = args.pmcid
    chunk_size = 10
    max_sentences = args.chunks * chunk_size

    from pipeline.stages.knowledge_extraction.cache import PipelineCache
    from pipeline.stages.knowledge_extraction.grounding_filter import GroundingFilter, score_findings
    from pipeline.stages.knowledge_extraction.group_stage import GroupStage, is_groupable
    from pipeline.stages.knowledge_extraction.map_stage import MapStage
    from pipeline.stages.knowledge_extraction.normalize_stage import NormalizeStage

    llm = build_llm()
    map_stage = MapStage(
        voter_llms=[llm],
        level2_voter_llms=[llm],
        escalation_llm=llm,
        theta=0.0,
        chunk_size=chunk_size,
    )
    cache = PipelineCache(pathlib.Path("out/inspect/map_cache.json"))

    # ── Phase 1: MAP ───────────────────────────────────────────────────────────
    t_map = time.perf_counter()
    sentences = load_sentences(pmcid)[:max_sentences]
    logger.info("Processing %d sentences (%d chunks)", len(sentences), args.chunks)

    chunk_summaries = map_stage.process(sentences, pmcid, cache=cache)
    cache.save()

    raw_findings = [f for cs in chunk_summaries for f in cs.findings]
    logger.info("MAP: %d raw findings  [%.1fs]", len(raw_findings), time.perf_counter() - t_map)

    # ── Phase 1b: NLI scoring (optional) ──────────────────────────────────────
    if not args.skip_nli:
        t_nli = time.perf_counter()
        gf = GroundingFilter(threshold=0.3)
        score_findings(raw_findings, nli_pipe=gf._pipe)
        logger.info(
            "NLI scoring: grounding_score written to %d findings  [%.1fs]",
            len(raw_findings), time.perf_counter() - t_nli,
        )
    else:
        logger.info("NLI scoring: skipped (--skip-nli)")

    # ── Phase 2: NORMALIZE ─────────────────────────────────────────────────────
    t_norm = time.perf_counter()
    normal_findings = NormalizeStage().normalize(raw_findings, pmcid)
    logger.info(
        "NORMALIZE: %d → %d NormalFindings  [%.1fs]",
        len(raw_findings), len(normal_findings), time.perf_counter() - t_norm,
    )

    # ── Phase 3: GROUP ─────────────────────────────────────────────────────────
    t_group = time.perf_counter()
    groupable = [nf for nf in normal_findings if is_groupable(nf)]
    non_groupable = [nf for nf in normal_findings if not is_groupable(nf)]
    groups = GroupStage().group(groupable, pmcid)
    logger.info(
        "GROUP: %d groupable, %d non-groupable → %d groups  [%.1fs]",
        len(groupable), len(non_groupable), len(groups), time.perf_counter() - t_group,
    )

    # ── Metrics ────────────────────────────────────────────────────────────────
    groupability_ratio = (
        round(len(groupable) / len(normal_findings) * 100, 1)
        if normal_findings else 0.0
    )
    avg_group_size = (
        round(len(groupable) / len(groups), 2) if groups else 0.0
    )

    direction_distribution: dict[str, int] = {
        "positive": 0, "negative": 0, "absent": 0,
        "partial": 0, "unclear": 0, "null": 0,
    }
    for nf in normal_findings:
        key = nf.direction.value if nf.direction is not None else "null"
        direction_distribution[key] = direction_distribution.get(key, 0) + 1

    top_groups = sorted(groups, key=lambda g: len(g.member_ids), reverse=True)[:10]

    sample_groupable = [
        {
            "subject_entity": nf.subject_entity,
            "outcome_entity": nf.outcome_entity,
            "relation_type": nf.relation_type.value,
            "direction": nf.direction.value if nf.direction else None,
            "category": nf.category,
            "grounding_score": round(nf.mean_grounding_score, 4)
            if nf.mean_grounding_score is not None else None,
        }
        for nf in groupable[:10]
    ]

    sample_non_groupable = [
        {
            "claim": nf.predicate_text,
            "subject_entity": nf.subject_entity,
            "outcome_entity": nf.outcome_entity,
            "relation_type": nf.relation_type.value,
            "direction": nf.direction.value if nf.direction else None,
            "category": nf.category,
        }
        for nf in non_groupable[:10]
    ]

    # ── Console summary ────────────────────────────────────────────────────────
    print()
    print(f"  PMCID              : {pmcid}")
    print(f"  Chunks processed   : {len(chunk_summaries)}")
    print(f"  Raw findings       : {len(raw_findings)}")
    print(f"  Normal findings    : {len(normal_findings)}")
    print(f"  Groupable          : {len(groupable)}  ({groupability_ratio}%)")
    print(f"  Non-groupable      : {len(non_groupable)}")
    print(f"  Groups             : {len(groups)}")
    print(f"  Avg group size     : {avg_group_size}")
    print()
    print("  Direction distribution:")
    for k, v in direction_distribution.items():
        if v > 0:
            print(f"    {k:<10} {v}")
    print()
    if top_groups:
        print("  Top groups:")
        for g in top_groups:
            dirs = ", ".join(f"{k}×{v}" for k, v in g.direction_counts.items())
            print(f"    [{len(g.member_ids):>2}] {g.subject_entity:<20} → {g.outcome_entity:<20} [{g.relation_type.value}]  {dirs}")
    print()

    # ── Write JSON ─────────────────────────────────────────────────────────────
    out_dir = pathlib.Path("out/inspect")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"phase123_{pmcid}_{ts}.json"

    report = {
        "pmcid": pmcid,
        "counts": {
            "raw_findings": len(raw_findings),
            "scored_findings": len(raw_findings),
            "normalized_findings": len(normal_findings),
            "groupable_findings": len(groupable),
            "non_groupable_findings": len(non_groupable),
            "groups": len(groups),
            "avg_group_size": avg_group_size,
        },
        "groupability_ratio": groupability_ratio,
        "direction_distribution": direction_distribution,
        "top_groups": [
            {
                "group_id": g.group_id,
                "subject_entity": g.subject_entity,
                "outcome_entity": g.outcome_entity,
                "category": g.category,
                "size": len(g.member_ids),
                "direction_counts": g.direction_counts,
                "scope_heterogeneity": g.scope_heterogeneity,
            }
            for g in top_groups
        ],
        "sample_groupable_findings": sample_groupable,
        "sample_non_groupable_findings": sample_non_groupable,
    }

    elapsed = time.perf_counter() - t0
    report["elapsed_seconds"] = round(elapsed, 1)

    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"  Total time         : {elapsed:.1f}s")
    print(f"  Report written to  : {out_path}")


if __name__ == "__main__":
    main()
