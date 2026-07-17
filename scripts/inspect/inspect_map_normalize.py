"""
Run real MAP → score_findings → NORMALIZE on one paper and write outputs.

Usage:
    python scripts/inspect/inspect_map_normalize.py PMC10047158
    python scripts/inspect/inspect_map_normalize.py PMC10047158 --chunks 3
    python scripts/inspect/inspect_map_normalize.py --list-only

Options:
    --chunks N      Process only the first N chunks (default: 2).
    --list-only     Print available PMCIDs and exit.

Output file: out/inspect/map_normalize_<PMCID>_<timestamp>.json

LLM used: single Vertex AI model (gemini-2.5-flash-lite) for all voters +
escalation (one API key, minimum calls).  Set VERTEX_PROJECT in .env before
running.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import pathlib
import sys
from pathlib import Path

# Bootstrap repo root onto sys.path so the lazy `from database …` / `from pipeline …`
# imports below resolve when this script is run directly
# (`python scripts/inspect/inspect_map_normalize.py PMC…`). `scripts/` has no
# `__init__.py` and the repo is not installed as a package, so Python would
# otherwise place only `scripts/inspect/` on sys.path (B-094).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def list_pmcids() -> None:
    from nlp_histo.database import get_db_connection
    from nlp_histo.database.models import Document

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
    from nlp_histo.database import get_db_connection, Document, TextElement

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
    import os
    from langchain_google_vertexai import ChatVertexAI

    return ChatVertexAI(
        model="gemini-2.5-flash-lite",
        project=os.environ["VERTEX_PROJECT"],
        location="global",
        temperature=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pmcid", nargs="?")
    parser.add_argument("--chunks", type=int, default=2,
                        help="Number of chunks to process (default: 2)")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    if args.list_only:
        list_pmcids()
        return

    if not args.pmcid:
        parser.print_help()
        sys.exit(1)

    pmcid = args.pmcid
    max_chunks = args.chunks

    from nlp_histo.pipeline.stages.knowledge_extraction.cache import PipelineCache
    from nlp_histo.pipeline.stages.knowledge_extraction.grounding.grounding_filter import GroundingFilter, score_findings
    from nlp_histo.pipeline.stages.knowledge_extraction.stages.map_stage import MapStage
    from nlp_histo.pipeline.stages.knowledge_extraction.stages.normalize_stage import NormalizeStage

    llm = build_llm()
    chunk_size = 10

    sentences = load_sentences(pmcid)
    # Truncate to keep cost small
    sentences = sentences[: max_chunks * chunk_size]
    logger.info("Processing %d sentences (%d chunks) for %s",
                len(sentences), max_chunks, pmcid)

    # ── MAP ────────────────────────────────────────────────────────────────────
    map_stage = MapStage(
        voter_llms=[llm],
        level2_voter_llms=[llm],
        escalation_llm=llm,
        theta=0.0,          # single voter → always "agree", no escalation
        chunk_size=chunk_size,
    )
    cache = PipelineCache(pathlib.Path("out/inspect/map_cache.json"))
    chunk_summaries = map_stage.process(sentences, pmcid, cache=cache)
    cache.save()

    raw_findings = [f for cs in chunk_summaries for f in cs.findings]
    logger.info("MAP produced %d raw findings across %d chunks",
                len(raw_findings), len(chunk_summaries))

    # ── SCORE (Phase 1 grounding) ──────────────────────────────────────────────
    gf = GroundingFilter(threshold=0.3)
    score_findings(raw_findings, nli_pipe=gf._pipe)
    logger.info("Grounding scores written to %d findings", len(raw_findings))

    # ── NORMALIZE (Phase 2) ────────────────────────────────────────────────────
    normal_findings = NormalizeStage().normalize(raw_findings, pmcid)
    logger.info("NORMALIZE: %d findings → %d NormalFindings",
                len(raw_findings), len(normal_findings))

    # ── Report ─────────────────────────────────────────────────────────────────
    print()
    print("RAW MAP FINDINGS")
    print("-" * 70)
    for f in raw_findings:
        score_str = f"{f.grounding_score:.3f}" if f.grounding_score is not None else "  n/a"
        subj = (f.subject_entity or "None")[:20]
        outc = (f.outcome_entity or "None")[:20]
        dirn = f.direction.value if f.direction else "None"
        print(f"  [{score_str}] {dirn:<10} {subj:<20} → {outc:<20}  {f.claim[:60]}")

    print()
    print("NORMAL FINDINGS")
    print("-" * 70)
    for nf in normal_findings:
        score_str = f"{nf.mean_grounding_score:.3f}" if nf.mean_grounding_score is not None else "  n/a"
        subj = (nf.subject_entity or "None")[:20]
        outc = (nf.outcome_entity or "None")[:20]
        dirn = nf.direction.value if nf.direction else "None"
        n_spans = len(nf.evidence)
        print(f"  [{score_str}] {dirn:<10} {subj:<20} → {outc:<20}  spans={n_spans}  {nf.predicate_text[:50]}")

    # ── Write output ───────────────────────────────────────────────────────────
    out_dir = pathlib.Path("out/inspect")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"map_normalize_{pmcid}_{ts}.json"

    output = {
        "pmcid": pmcid,
        "chunks_processed": len(chunk_summaries),
        "sentences_processed": len(sentences),
        "raw_findings": [f.model_dump() for f in raw_findings],
        "normal_findings": [nf.model_dump() for nf in normal_findings],
    }
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nOutput written to: {out_path}")


if __name__ == "__main__":
    main()
