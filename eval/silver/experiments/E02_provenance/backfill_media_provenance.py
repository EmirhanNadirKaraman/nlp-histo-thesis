#!/usr/bin/env python3
"""E02 backfill — populate Table/Figure page_number + bbox from cached media JSON (B-075).

The doc-extraction cropper computes each figure/table's page + bbox and writes them
to `out/json/<pmcid>_media.json`, but the DB ingester historically dropped them
(fixed forward in `db_ingester.py`; figure columns added in migration 0014). This
backfills the EXISTING corpus without re-extraction or a DB drop.

For each `out/json/*_media.json`: find its `Document` by pmcid, match each JSON
figure/table to its DB row by image filename, and set `page_number` + `bbox_*` where
currently NULL. `table_content`/`section_context` have no source in the media JSON.

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
from pathlib import Path

from eval.paths import REPO_ROOT as _REPO_ROOT  # noqa: E402

from dotenv import load_dotenv

load_dotenv(str(_REPO_ROOT / ".env"))

from nlp_histo.database import Document, Figure, Table, get_db_connection

_JSON_DIR = _REPO_ROOT / "out" / "json"
_KINDS = [("tables", Table), ("figures", Figure)]   # media-JSON key → ORM model


def _bbox_ok(b) -> bool:
    return isinstance(b, dict) and all(b.get(k) is not None for k in ("x1", "y1", "x2", "y2"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill Table/Figure page+bbox from cached media JSON (B-075).")
    ap.add_argument("--pmcid", default=None, help="restrict to one document pmcid")
    ap.add_argument("--limit", type=int, default=None, help="process at most N media JSONs")
    ap.add_argument("--apply", action="store_true", help="commit changes (default: dry-run)")
    args = ap.parse_args()

    files = sorted(glob.glob(str(_JSON_DIR / "*_media.json")))
    stat = {k: dict(seen=0, matched=0, updated=0, already=0, unmatched=0) for k, _ in _KINDS}
    n_papers = n_no_doc = 0

    db = get_db_connection()
    with db.session_scope() as s:
        for f in files:
            d = json.loads(Path(f).read_text())
            pmcid = d.get("pmcid")
            if args.pmcid and pmcid != args.pmcid:
                continue
            if not (d.get("tables") or d.get("figures")):
                continue
            n_papers += 1
            doc = s.query(Document).filter_by(pmcid=pmcid).first()
            if not doc:
                n_no_doc += 1
                continue
            for jkey, Model in _KINDS:
                items = d.get(jkey) or []
                if not items:
                    continue
                db_rows = {r.image_filename: r for r in s.query(Model).filter_by(document_id=doc.id)
                           if r.image_filename}
                st = stat[jkey]
                for it in items:
                    st["seen"] += 1
                    key = Path(it["image_path"]).name if it.get("image_path") else None
                    row = db_rows.get(key)
                    if row is None:
                        st["unmatched"] += 1
                        continue
                    st["matched"] += 1
                    if row.page_number is not None:
                        st["already"] += 1
                        continue
                    b = it.get("bbox")
                    if it.get("page") is None or not _bbox_ok(b):
                        continue
                    st["updated"] += 1
                    if args.apply:   # dry-run never mutates the session
                        row.page_number = it["page"]
                        row.bbox_x1, row.bbox_y1 = b["x1"], b["y1"]
                        row.bbox_x2, row.bbox_y2 = b["x2"], b["y2"]
            if args.limit and n_papers >= args.limit:
                break

        mode = "APPLY" if args.apply else "DRY-RUN"
        print(f"[{mode}] media JSONs processed: {n_papers}  (no DB doc: {n_no_doc})")
        for jkey, _ in _KINDS:
            st = stat[jkey]
            print(f"  {jkey:8}: seen {st['seen']:5}  matched {st['matched']:5}  "
                  f"already {st['already']:5}  updated {st['updated']:5}  unmatched {st['unmatched']}")
        if args.apply:
            s.commit()
            print("  committed.")
        else:
            print("  dry-run — no writes (session never mutated). Re-run with --apply to commit.")


if __name__ == "__main__":
    main()
