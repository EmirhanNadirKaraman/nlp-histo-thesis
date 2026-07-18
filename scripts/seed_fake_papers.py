"""Seed two synthetic papers (FAKE01_main, FAKE02_main) for end-to-end
validation of the knowledge_extraction pipeline's RELATE / corpus_relate stages.

Why this script exists
----------------------
Run A's real papers produced 196 canonical rules but zero relations because
``_should_compare`` requires (category, relation_type, subject, outcome) to
match exactly — 19,110 candidate pairs, all rejected at the gate. That's
expected behaviour, not a bug, but it means the pipeline can't be
*validated* against real corpora alone. These fakes inject paired claims
designed to survive the gate:

    FAKE01.claim_1  ↔  FAKE02.claim_1  → SUPPORT     (same wording, intensified)
    FAKE01.claim_2  ↔  FAKE02.claim_2  → CONTRADICT  (negation injected)
    FAKE01.claim_3                                   → UNRELATED to FAKE02

Claims share subject/outcome wording so NORMALIZE produces matching entities
and the RELATE gate lets the pair through to NLI.

Usage
-----
    python scripts/seed_fake_papers.py            # insert (idempotent — skips
                                                  # if pmcid already exists)
    python scripts/seed_fake_papers.py --reset    # delete + re-insert
    python scripts/seed_fake_papers.py --delete   # delete only

Then run the pipeline against them:

    python scripts/run_paper.py --from-selection \
        configs/paper_selection/fake.yaml --sync

Outputs of interest after the run:

* ``out/summaries/runs/<run_id>/relate/FAKE0{1,2}_main/relations.jsonl``
  — intra-paper relations.
* ``out/summaries/corpus_relations.json``
  — cross-paper relations.  FAKE01.claim_1 ↔ FAKE02.claim_1 should appear
  as SUPPORT; claim_2 pair as CONTRADICT.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

# Make the repo importable when running as `python scripts/seed_fake_papers.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlp_histo.database import get_db_connection, Document, TextElement  # noqa: E402

logger = logging.getLogger("seed_fake_papers")


# Synthetic content

# Paragraph design: keep each paragraph short so spaCy's sentence splitter
# produces clean per-claim sentences for MAP. Use almost-identical wording
# across the two papers so NORMALIZE collapses subject/outcome entities to
# the same canonical form and the strict 4-way RELATE gate lets the pair
# survive into NLI.

FAKE01_PARAGRAPHS: list[tuple[list[str], str]] = [
    # (path_list, text_content)
    (
        ["Abstract"],
        "We investigated three histopathological markers in glioblastoma and "
        "colorectal cancer cohorts.",
    ),
    (
        ["Findings", "Prognostic markers"],
        "EGFR overexpression is strongly associated with poor prognosis in "
        "glioblastoma patients. "
        "KRAS mutation predicts resistance to anti-EGFR therapy in "
        "colorectal cancer. "
        "HER2 amplification was observed in a subset of aggressive tumours.",
    ),
    (
        ["Methods", "Staining protocol"],
        "Hematoxylin and eosin staining was performed at four micrometres "
        "section thickness. "
        "All samples were fixed in formalin overnight before paraffin "
        "embedding.",
    ),
]

FAKE02_PARAGRAPHS: list[tuple[list[str], str]] = [
    (
        ["Abstract"],
        "We re-examined two of these markers in an independent histopathology "
        "cohort.",
    ),
    (
        ["Findings", "Replication"],
        # Designed to SUPPORT FAKE01 claim_1 — same subject + outcome, same direction.
        "EGFR overexpression is consistently linked to poor prognosis in "
        "glioblastoma patients in our independent cohort. "
        # Designed to CONTRADICT FAKE01 claim_2 — same subject + outcome, opposite direction.
        "KRAS mutation does not predict resistance to anti-EGFR therapy in "
        "colorectal cancer. "
        # UNRELATED filler (different subject + outcome from any FAKE01 claim).
        "Tumour budding was rare across both cohorts.",
    ),
    (
        ["Methods", "Imaging"],
        "Whole-slide images were scanned at forty times magnification. "
        "Digital pathology software was used for analysis.",
    ),
]


@dataclass(slots=True, frozen=True)
class FakePaper:
    pmcid: str
    title: str
    paragraphs: list[tuple[list[str], str]]


FAKE_PAPERS: list[FakePaper] = [
    FakePaper(
        pmcid="FAKE01_main",
        title="Synthetic paper FAKE01: EGFR / KRAS / HER2 prognostic markers",
        paragraphs=FAKE01_PARAGRAPHS,
    ),
    FakePaper(
        pmcid="FAKE02_main",
        title="Synthetic paper FAKE02: replication of EGFR + KRAS findings",
        paragraphs=FAKE02_PARAGRAPHS,
    ),
]


# DB operations

def _delete_paper(session, pmcid: str) -> bool:
    """Delete a Document by pmcid (cascades to TextElements). Returns True
    if anything was deleted."""
    doc = session.query(Document).filter_by(pmcid=pmcid).first()
    if doc is None:
        return False
    session.delete(doc)
    session.flush()
    return True


def _insert_paper(session, paper: FakePaper) -> None:
    """Insert one synthetic paper as Document + TextElement rows."""
    doc = Document(
        pmcid=paper.pmcid,
        filename=f"{paper.pmcid}.synthetic",
        file_path=f"<synthetic>/{paper.pmcid}",
        title=paper.title,
        journal="Synthetic Test Corpus",
        publication_year=2026,
        text_source="synthetic",
    )
    session.add(doc)
    session.flush()  # populate doc.id

    for position, (path_list, text) in enumerate(paper.paragraphs):
        path_string = " > ".join(path_list)
        unique_path = f"{paper.pmcid}/{path_string}/{position}"
        session.add(
            TextElement(
                document_id=doc.id,
                unique_path=unique_path,
                path_list=path_list,
                path_string=path_string,
                depth=len(path_list),
                position_in_section=position,
                text_content=text,
                word_count=len(text.split()),
                char_count=len(text),
            ),
        )

    session.flush()


def seed(reset: bool, delete_only: bool) -> None:
    db = get_db_connection()
    with db.session_scope() as session:
        for paper in FAKE_PAPERS:
            existed = session.query(Document).filter_by(pmcid=paper.pmcid).first() is not None
            if delete_only:
                deleted = _delete_paper(session, paper.pmcid)
                logger.info(
                    "%s: %s",
                    paper.pmcid,
                    "deleted" if deleted else "not present (nothing to delete)",
                )
                continue
            if reset and existed:
                _delete_paper(session, paper.pmcid)
                logger.info("%s: reset (deleted existing rows)", paper.pmcid)
                existed = False
            if existed:
                logger.info(
                    "%s: already present — skipping (use --reset to replace)",
                    paper.pmcid,
                )
                continue
            _insert_paper(session, paper)
            logger.info(
                "%s: inserted (%d paragraphs)",
                paper.pmcid, len(paper.paragraphs),
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete any existing FAKE0{1,2} rows before inserting.",
    )
    parser.add_argument(
        "--delete", action="store_true",
        help="Delete FAKE0{1,2} rows and exit without re-inserting.",
    )
    args = parser.parse_args()
    if args.reset and args.delete:
        parser.error("--reset and --delete are mutually exclusive")
    seed(reset=args.reset, delete_only=args.delete)


if __name__ == "__main__":
    main()
