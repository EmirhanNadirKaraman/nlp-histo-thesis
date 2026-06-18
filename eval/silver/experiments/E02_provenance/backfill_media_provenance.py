#!/usr/bin/env python3
"""E02 backfill — populate Table.page_number + bbox from cached media JSON (B-075).

The doc-extraction cropper computes each table's page + bbox and writes them to
`out/json/<pmcid>_media.json`, but the DB ingester historically dropped them
(fixed forward in `db_ingester.py`; this script backfills the EXISTING corpus
without re-extraction or a DB drop).

For each `out/json/*_media.json`: find its `Document` by pmcid, match each JSON
table to its DB `Table` row by image filename, and set `page_number` + `bbox_*`
where currently NULL. Figures are skipped — the `Figure` model has no page/bbox
columns (schema gap, needs a migration). table_content/section_context have no
source in the media JSON.

Default is a DRY RUN (no writes). Pass --apply to commit.

Usage:
  python -m eval.silver.experiments.E02_provenance.backfill_media_provenance               # dry-run, all
  python -m eval.silver.experiments.E02_provenance.backfill_media_provenance --pmcid PMC…  # one paper
  python -m eval.silver.experiments.E02_provenance.backfill_media_provenance --apply        # commit
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(str(_REPO_ROOT / ".env"))

from database import Document, Table, get_db_connection

_JSON_DIR = _REPO_ROOT / "out" / "json"


def _bbox_ok(b) -> bool:
    return isinstance(b, dict) and all(b.get(k) is not None for k in ("x1", "y1", "x2", "y2"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill Table page/bbox from cached media JSON (B-075).")
    ap.add_argument("--pmcid", default=None, help="restrict to one document pmcid")
    ap.add_argument("--limit", type=int, default=None, help="process at most N media JSONs")
    ap.add_argument("--apply", action="store_true", help="commit changes (default: dry-run)")
    args = ap.parse_args()

    files = sorted(glob.glob(str(_JSON_DIR / "*_media.json")))
    n_papers = n_no_doc = n_json_tabs = n_matched = n_updated = n_already = n_unmatched = 0

    db = get_db_connection()
    with db.session_scope() as s:
        for f in files:
            d = json.loads(Path(f).read_text())
            pmcid = d.get("pmcid")
            if args.pmcid and pmcid != args.pmcid:
                continue
            tabs = d.get("tables") or []
            if not tabs:
                continue
            n_papers += 1
            doc = s.query(Document).filter_by(pmcid=pmcid).first()
            if not doc:
                n_no_doc += 1
                continue
            db_tabs = {t.image_filename: t for t in s.query(Table).filter_by(document_id=doc.id)
                       if t.image_filename}
            for jt in tabs:
                n_json_tabs += 1
                key = Path(jt["image_path"]).name if jt.get("image_path") else None
                row = db_tabs.get(key)
                if row is None:
                    n_unmatched += 1
                    continue
                n_matched += 1
                if row.page_number is not None:
                    n_already += 1
                    continue
                b = jt.get("bbox")
                if jt.get("page") is None or not _bbox_ok(b):
                    continue
                n_updated += 1
                if args.apply:   # dry-run never mutates the session
                    row.page_number = jt["page"]
                    row.bbox_x1, row.bbox_y1 = b["x1"], b["y1"]
                    row.bbox_x2, row.bbox_y2 = b["x2"], b["y2"]
            if args.limit and n_papers >= args.limit:
                break

        mode = "APPLY" if args.apply else "DRY-RUN"
        print(f"[{mode}] media JSONs with tables: {n_papers}  (no DB doc: {n_no_doc})")
        print(f"  JSON tables seen        : {n_json_tabs}")
        print(f"  matched to a DB Table   : {n_matched}  (unmatched: {n_unmatched})")
        print(f"  already had page_number : {n_already}")
        print(f"  WOULD update / updated  : {n_updated}")
        if args.apply:
            s.commit()
            print("  committed.")
        else:
            print("  dry-run — no writes (session never mutated). Re-run with --apply to commit.")


if __name__ == "__main__":
    main()
