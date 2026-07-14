"""
Inspect NORMALIZE and GROUP stage outputs on a real paper from the database.

Usage:
    python scripts/inspect/inspect_normalize_group.py PMC10047158
    python scripts/inspect/inspect_normalize_group.py PMC10047158 --list-only

--list-only prints available PMCIDs and exits.

Output file: out/inspect/normalize_group_<PMCID>_<timestamp>.json
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
# (`python scripts/inspect/inspect_normalize_group.py PMC…`). `scripts/` has no
# `__init__.py` and the repo is not installed as a package, so Python would
# otherwise place only `scripts/inspect/` on sys.path (B-094).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def list_pmcids() -> None:
    from nlp_histo.database import get_db_connection
    from nlp_histo.database.models import Document

    db = get_db_connection()
    with db.session_scope() as session:
        pmcids = [row.pmcid for row in session.query(Document.pmcid).order_by(Document.pmcid).all()]
    if not pmcids:
        print("No documents found in database.")
    else:
        print(f"{len(pmcids)} document(s) in database:")
        for p in pmcids:
            print(f"  {p}")


def load_findings_from_db(pmcid: str) -> list:
    """
    Load text elements, sentence-split them, and run MAP-equivalent extraction
    using the grounding filter to produce scored Finding objects.

    Because we are not running the full LLM MAP stage here, we instead
    construct minimal Finding objects directly from the DB text elements
    with subject_entity/outcome_entity/direction left as None.  This lets
    you inspect what NORMALIZE and GROUP produce on real text provenance
    even before a live MAP run.
    """
    import spacy
    from nlp_histo.database import get_db_connection, Document, TextElement
    from pipeline.stages.knowledge_extraction.models import Finding, FindingScope

    nlp = spacy.load("en_core_sci_sm")
    db = get_db_connection()

    findings = []
    with db.session_scope() as session:
        doc = session.query(Document).filter_by(pmcid=pmcid).first()
        if doc is None:
            print(f"PMCID {pmcid!r} not found in database. Run --list-only to see available PMCIDs.")
            sys.exit(1)

        rows = (
            session.query(TextElement)
            .filter_by(document_id=doc.id)
            .order_by(TextElement.position_in_section)
            .all()
        )

        for te in rows:
            for i, sent in enumerate(nlp(te.text_content).sents, 1):
                text = sent.text.strip()
                if not text:
                    continue
                findings.append(Finding(
                    category="morphology",   # placeholder — MAP LLM sets this
                    claim=text[:200],        # sentence itself as the claim proxy
                    evidence=[f"S{i}|{pmcid}|{te.id}"],
                    confidence="medium",
                    verbatim_support=text,
                    subject_entity=None,     # MAP LLM fills these
                    outcome_entity=None,
                    direction=None,
                    scope=FindingScope(),
                    grounding_score=None,
                ))

    logger.info("Loaded %d sentence-findings from %d text elements for %s",
                len(findings), len(rows), pmcid)
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pmcid", nargs="?", help="PubMed Central ID, e.g. PMC10047158")
    parser.add_argument("--list-only", action="store_true",
                        help="List available PMCIDs and exit")
    args = parser.parse_args()

    if args.list_only:
        list_pmcids()
        return

    if not args.pmcid:
        parser.print_help()
        sys.exit(1)

    pmcid = args.pmcid

    from pipeline.stages.knowledge_extraction.stages.normalize_stage import NormalizeStage
    from pipeline.stages.knowledge_extraction.stages.group_stage import GroupStage

    findings = load_findings_from_db(pmcid)

    # ── NORMALIZE ──────────────────────────────────────────────────────────────
    normal_findings = NormalizeStage().normalize(findings, pmcid)
    logger.info("NORMALIZE: %d findings → %d NormalFindings", len(findings), len(normal_findings))

    # ── GROUP ──────────────────────────────────────────────────────────────────
    groups = GroupStage().group(normal_findings, pmcid)
    logger.info("GROUP: %d NormalFindings → %d FindingGroups", len(normal_findings), len(groups))

    # ── Summary table ──────────────────────────────────────────────────────────
    print()
    print(f"{'group_id':<36} {'subject':<10} {'outcome':<20} {'members':>7}  {'directions'}")
    print("-" * 90)
    for g in groups:
        dirs = ", ".join(f"{k}×{v}" for k, v in g.direction_counts.items())
        subj = (g.subject_entity or "None")[:10]
        outc = (g.outcome_entity or "None")[:20]
        print(f"{g.group_id:<36} {subj:<10} {outc:<20} {len(g.member_ids):>7}   {dirs}")
    print()

    # ── Write output ───────────────────────────────────────────────────────────
    out_dir = pathlib.Path("out/inspect")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"normalize_group_{pmcid}_{ts}.json"

    output = {
        "pmcid": pmcid,
        "input_finding_count": len(findings),
        "normal_findings": [nf.model_dump() for nf in normal_findings],
        "finding_groups": [g.model_dump() for g in groups],
    }
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"Output written to: {out_path}")


if __name__ == "__main__":
    main()
