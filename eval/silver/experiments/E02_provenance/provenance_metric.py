#!/usr/bin/env python3
"""E02 — corpus-level provenance-preservation metric (RQ1, §9.2/9.7).

Quantifies the pipeline's core provenance claim: every extracted artifact can be
traced back to its exact source location. Measured over the whole DB corpus:

  * TEXT provenance   — each TextElement carries a section-hierarchy address
                        (`unique_path` = PMCID/section/position, `path_list`).
  * MEDIA localization — Tables carry page + bbox (locatable in the PDF) + caption
                         + extracted content; Figures carry caption + section +
                         crop image (the schema stores no page/bbox for figures).
  * CROSS-REFERENCE    — in-text "Figure 3 / Table 1" mentions resolved to the
                         actual Figure/Table rows (junction tables).

Plus a qualitative example tracing one paragraph → its referenced table (caption,
page, bbox, crop). Read-only DB queries; no API.

Usage:
  python -m eval.silver.experiments.E02_provenance.provenance_metric
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(str(_REPO_ROOT / ".env"))

from sqlalchemy import distinct, func

from database import Document, Figure, Table, TextElement, get_db_connection
from database.models import TextElementFigureReference, TextElementTableReference

REPORTS_DIR = _REPO_ROOT / "eval" / "reports"


def _pct(n: int, d: int) -> str:
    return f"{n/d:6.1%}" if d else "   n/a"


def main() -> None:
    db = get_db_connection()
    rows: list[dict] = []

    def rec(metric: str, n: int, total: int):
        rows.append({"metric": metric, "n": n, "total": total,
                     "pct": round(n / total, 4) if total else None})
        print(f"  {metric:46} {n:6d} / {total:<6d} {_pct(n, total)}")

    with db.session_scope() as s:
        nd = s.query(func.count(Document.id)).scalar()
        nte = s.query(func.count(TextElement.id)).scalar()
        nfig = s.query(func.count(Figure.id)).scalar()
        ntab = s.query(func.count(Table.id)).scalar()
        print(f"E02 provenance — corpus: {nd} documents, {nte} text elements, "
              f"{nfig} figures, {ntab} tables\n")

        # ── TEXT provenance (one pass over the small unique_path/path_list cols) ──
        print("TEXT provenance (section-hierarchy address per paragraph):")
        te = s.query(TextElement.unique_path, TextElement.path_list).all()
        has_up = sum(1 for up, _ in te if up)
        wellformed = sum(1 for up, _ in te
                         if up and up.startswith("PMC") and up.rsplit("/", 1)[-1].isdigit())
        has_pl = sum(1 for _, pl in te if pl)
        has_text = s.query(func.count(TextElement.id)).filter(
            TextElement.text_content.isnot(None), TextElement.text_content != "").scalar()
        rec("text elements with unique_path", has_up, nte)
        rec("  unique_path well-formed (PMCID/…/pos)", wellformed, nte)
        rec("text elements with path_list (section array)", has_pl, nte)
        rec("text elements with non-empty text", has_text, nte)

        # ── TABLE provenance (caption + crop + page/bbox + cross-ref) ──
        # bbox is Docling coords (y=0 at page bottom, so y1>y2); validity = non-degenerate.
        print("\nTABLE provenance (caption + crop + page/bbox + cross-ref):")
        tab_page = s.query(func.count(Table.id)).filter(Table.page_number.isnot(None)).scalar()
        tab_bbox = s.query(func.count(Table.id)).filter(
            Table.bbox_x1.isnot(None), Table.bbox_y1.isnot(None),
            Table.bbox_x2.isnot(None), Table.bbox_y2.isnot(None),
            Table.bbox_x2 != Table.bbox_x1, Table.bbox_y2 != Table.bbox_y1).scalar()
        tab_cap = s.query(func.count(Table.id)).filter(
            Table.caption_text.isnot(None), Table.caption_text != "").scalar()
        tab_content = s.query(func.count(Table.id)).filter(
            Table.table_content.isnot(None), Table.table_content != "").scalar()
        tab_img = s.query(func.count(Table.id)).filter(
            Table.image_path.isnot(None), Table.image_path != "").scalar()
        tab_linked = s.query(func.count(distinct(TextElementTableReference.table_id))).scalar()
        rec("tables with caption", tab_cap, ntab)
        rec("tables with crop image", tab_img, ntab)
        rec("tables with page_number", tab_page, ntab)
        rec("tables with valid bbox", tab_bbox, ntab)
        rec("tables linked to ≥1 referencing paragraph", tab_linked, ntab)
        rec("tables with extracted content [unpopulated — no source in crop path]", tab_content, ntab)

        # ── FIGURE provenance (no page/bbox in schema) ──
        print("\nFIGURE provenance (caption + section + crop; no page/bbox stored):")
        fig_cap = s.query(func.count(Figure.id)).filter(
            Figure.caption_text.isnot(None), Figure.caption_text != "").scalar()
        fig_sec = s.query(func.count(Figure.id)).filter(
            Figure.section_context.isnot(None), Figure.section_context != "").scalar()
        fig_img = s.query(func.count(Figure.id)).filter(
            Figure.image_path.isnot(None), Figure.image_path != "").scalar()
        fig_linked = s.query(func.count(distinct(TextElementFigureReference.figure_id))).scalar()
        rec("figures with caption", fig_cap, nfig)
        rec("figures with section_context", fig_sec, nfig)
        rec("figures with crop image", fig_img, nfig)
        rec("figures linked to ≥1 referencing paragraph", fig_linked, nfig)

        # ── cross-reference resolution ──
        print("\nCROSS-REFERENCE resolution (in-text mentions → media rows):")
        n_figref = s.query(func.count(TextElementFigureReference.id)).scalar()
        n_tabref = s.query(func.count(TextElementTableReference.id)).scalar()
        te_fig = {x for (x,) in s.query(distinct(TextElementFigureReference.text_element_id))}
        te_tab = {x for (x,) in s.query(distinct(TextElementTableReference.text_element_id))}
        rec("text elements citing ≥1 figure/table", len(te_fig | te_tab), nte)
        print(f"  total figure-refs={n_figref}  total table-refs={n_tabref}")

        # ── qualitative example: paragraph → referenced table (caption + crop) ──
        print("\nQUALITATIVE example (provenance chain that the DB actually preserves):")
        link = (s.query(TextElementTableReference)
                .join(Table, Table.id == TextElementTableReference.table_id)
                .filter(Table.caption_text.isnot(None), Table.caption_text != "").first())
        if link:
            tx = s.get(TextElement, link.text_element_id)
            tb = s.get(Table, link.table_id)
            snippet = (tx.text_content or "")[:90].replace("\n", " ")
            print(f"  paragraph  {tx.unique_path}")
            print(f"             “{snippet}…”")
            print(f"  → cites    Table “{(tb.caption_text or '')[:70]}”")
            print(f"             crop {tb.image_path}")
            print("  → a rule extracted from this paragraph is traceable to its section address")
            print("    and to the cited table's caption + crop image (coordinate-level page/bbox")
            print("    is supported by the schema but not populated by the current ingest).")

    out_dir = REPORTS_DIR / "E02_provenance"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    csv_path = out_dir / f"provenance_{ts}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["metric", "n", "total", "pct"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV → {csv_path}")


if __name__ == "__main__":
    main()
