#!/usr/bin/env python3
"""
transfer_offtheshelf_labels.py — seed the off-the-shelf Docling variant's rubric
labels from an already-annotated source variant, but only for crops that are
genuinely identical.

A label is copied to an off-the-shelf crop only when a source-variant crop exists
with the same ``(pmcid, page, caption)`` and a bbox within ``--tol`` points. That
guarantees the transferred rubric judgement (which encodes caption/crop/footnote
correctness) actually applies — same region, same caption. Everything else is
left unlabelled for manual annotation via ``eval/annotate.py``.

This is the layout-correct replacement for ``eval/auto_annotate.py`` (which reads
the legacy flat ``annotations_<mode>.json`` under ``out/json/`` rather than the
per-variant ``eval/annotations/<variant>/<mode>.json`` the scorer now uses).

Usage::

    python scripts/eval/transfer_offtheshelf_labels.py            # source = 01_docling
    python scripts/eval/transfer_offtheshelf_labels.py --source 01_docling --tol 1.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEPS = _REPO_ROOT / "out" / "sweeps"
ANN = _REPO_ROOT / "eval" / "annotations"
TARGET = "00_docling_offtheshelf"
# (json key, annotation mode) — tables use the docling label set, figures shared.
KINDS = [("tables", "json_tables_docling"), ("figures", "json_figures")]


def _load_media(variant: str) -> dict[str, dict]:
    """pmcid -> media dict ({figures:[...], tables:[...]}) for a sweep variant."""
    out: dict[str, dict] = {}
    for f in sorted((SWEEPS / variant / "json").glob("*_media.json")):
        out[f.name.replace("_media.json", "")] = json.loads(f.read_text())
    return out


def _fname(item: dict) -> str | None:
    p = item.get("image_path")
    return Path(p).name if p else None


def _bbox_close(a: dict, b: dict, tol: float) -> bool:
    return all(abs(a[k] - b[k]) <= tol for k in ("x1", "y1", "x2", "y2"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="01_docling", help="annotated source variant")
    ap.add_argument("--tol", type=float, default=1.0, help="bbox match tolerance (points)")
    args = ap.parse_args(argv)

    tgt_media = _load_media(TARGET)
    src_media = _load_media(args.source)
    if not tgt_media:
        print(f"no media for {TARGET}; run baseline_offtheshelf_docling.py first", file=sys.stderr)
        return 1

    (ANN / TARGET).mkdir(parents=True, exist_ok=True)

    for key, mode in KINDS:
        src_labels_path = ANN / args.source / f"{mode}.json"
        src_labels = json.loads(src_labels_path.read_text()) if src_labels_path.exists() else {}

        transferred: dict[str, str] = {}
        n_tgt = n_match = 0
        for pmcid, tmedia in tgt_media.items():
            smedia = src_media.get(pmcid, {})
            src_items = smedia.get(key, []) or []
            for ti in tmedia.get(key, []) or []:
                tf = _fname(ti)
                if not tf:
                    continue
                n_tgt += 1
                for si in src_items:
                    if ti.get("page") != si.get("page"):
                        continue
                    if (ti.get("caption") or None) != (si.get("caption") or None):
                        continue
                    if not _bbox_close(ti["bbox"], si["bbox"], args.tol):
                        continue
                    sf = _fname(si)
                    if sf and sf in src_labels:
                        transferred[tf] = src_labels[sf]
                        n_match += 1
                    break

        out_path = ANN / TARGET / f"{mode}.json"
        out_path.write_text(json.dumps(transferred, indent=2, ensure_ascii=False))
        print(f"{key:8s} ({mode}): {n_tgt} crops, {n_match} labels transferred from "
              f"{args.source}, {n_tgt - n_match} left for manual annotation → {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
