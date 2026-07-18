#!/usr/bin/env python3
"""
baseline_offtheshelf_docling.py — a TRUE off-the-shelf Docling baseline for the
document-extraction rubric eval (E01).

Motivation
----------
The sweep's ``01_docling`` variant is *not* vanilla Docling: it runs the full
document-extraction pipeline (two-pass ghost-text filtering, region masking,
artifact filtering, caption *nearest-matching*, icon/size filtering, sub-figure
merging, 200-DPI cropping) with only the table-handling knobs disabled. So the
``01_docling`` rubric numbers credit pipeline post-processing (caption matching,
masking, cropping) to "Docling".

This script produces a clean baseline: stock Docling's *own* figure/table
detection and *own* native captions, with none of the pipeline scaffolding —
no masking, no two-pass, no ``nearest_caption``, no footnote-expand, no size/icon
filtering, no sub-figure merge. It crops each native ``PictureItem`` /
``TableItem`` bbox straight from the PDF and writes the same on-disk artifacts the
sweep scorer consumes, under a new variant ``00_docling_offtheshelf``.

The output plugs into the existing scorer unchanged:
``python scripts/eval/score_pdf_variants.py`` discovers ``out/sweeps/<variant>/``,
reads ``json/*_media.json`` for emitted crops, infers the DOCLING table-label mode
from the "docling" name, and reads labels from
``eval/annotations/00_docling_offtheshelf/<mode>.json``.

What this script does NOT do
----------------------------
It does not assign rubric labels — those are human judgements
(``eval/annotate.py``). After running this, transfer any exact-match labels from
existing variants, then hand-annotate the rest. Table recall is detector-
independent (``total_tables`` in ``ground_truth.csv``); figure recall reuses the
canonical ``missed_figures`` (a conservative approximation — off-the-shelf detects
a superset of pipeline figures, so it misses no more real figures).

Usage::

    python scripts/eval/baseline_offtheshelf_docling.py            # all 27 PDFs
    python scripts/eval/baseline_offtheshelf_docling.py --limit 1  # smoke test
    python scripts/eval/baseline_offtheshelf_docling.py --summary  # T1 recall only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nlp_histo.pipeline.stages.pdf_text_extraction.models.dto import BoundingBox, CroppedMedia  # noqa: E402
from nlp_histo.pipeline.stages.pdf_text_extraction.outputs.media_json_writer import MediaJsonWriter  # noqa: E402
from nlp_histo.parsers.layout_utils import FIG_NUM_RE, TAB_NUM_RE, parse_caption_num  # noqa: E402

VARIANT = "00_docling_offtheshelf"
PDF_DIR = _REPO_ROOT / "eval" / "pdfs"
OUT_DIR = _REPO_ROOT / "out" / "sweeps" / VARIANT
DPI = 200  # match CroppingConfig.dpi so crops render at the same scale as the sweep


def _bbox_from_prov(prov, page_height: float) -> BoundingBox:
    """Docling prov.bbox -> our BoundingBox (Docling bottom-left coords)."""
    bb = prov.bbox
    # Normalise to bottom-left origin (y1=top is the larger value), matching
    # BoundingBox / to_fitz_rect's assumption. Docling usually already is BL.
    try:
        from docling_core.types.doc import CoordOrigin  # type: ignore
        if getattr(bb, "coord_origin", None) == CoordOrigin.TOPLEFT:
            bb = bb.to_bottom_left_origin(page_height)
    except Exception:
        pass
    return BoundingBox(x1=bb.l, y1=bb.t, x2=bb.r, y2=bb.b, page=prov.page_no)


def _crop_to_png(fitz_doc, bbox: BoundingBox, page_height: float, out_path: Path) -> bool:
    """Render bbox to a PNG via PyMuPDF. Returns False if the rect is empty."""
    import fitz  # type: ignore
    rect = bbox.to_fitz_rect(page_height)
    if rect.is_empty:
        return False
    scale = DPI / 72.0
    page = fitz_doc[bbox.page - 1]
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out_path))
    return True


def _media_for(items, doc, kind: str, stem: str, fitz_doc, crops_dir: Path):
    """Build CroppedMedia for every native Docling picture/table item.

    No size/icon filtering, no merging, no nearest_caption — native bbox +
    native caption only.
    """
    pattern = FIG_NUM_RE if kind == "figure" else TAB_NUM_RE
    word = "Figure" if kind == "figure" else "Table"
    page_dims = {no: p.size.height for no, p in doc.pages.items()}
    media: list[CroppedMedia] = []
    running = 0
    for item in items:
        if not getattr(item, "prov", None):
            continue
        prov = item.prov[0]
        page_no = prov.page_no
        page_h = page_dims.get(page_no, 792.0)
        bbox = _bbox_from_prov(prov, page_h)

        caption = (item.caption_text(doc) or "").strip() or None
        num = parse_caption_num(caption, pattern) if caption else None
        if num is None:
            running += 1
            num = running
        label = f"{word} {num}"

        # Filename matching the sweep cropper: <stem>_<Label>_p<page>.png, with a
        # numeric suffix on collision.
        base = f"{stem}_{label.replace(' ', '_')}_p{page_no}"
        out_path = crops_dir / f"{base}.png"
        idx = 2
        while out_path.exists():
            out_path = crops_dir / f"{base}_{idx}.png"
            idx += 1

        if not _crop_to_png(fitz_doc, bbox, page_h, out_path):
            continue
        media.append(CroppedMedia(
            media_type=kind, label=label, number=num, caption=caption,
            image_path=out_path, bbox=bbox, page=page_no, source="docling_offtheshelf",
        ))
    return media


def process_pdf(pdf_path: Path, converter) -> tuple[int, int]:
    """Run stock Docling on one PDF; emit crops + media JSON. Returns (n_fig, n_tab)."""
    import fitz  # type: ignore
    stem = pdf_path.stem
    doc = converter.convert(str(pdf_path)).document
    fitz_doc = fitz.open(str(pdf_path))
    try:
        figures = _media_for(doc.pictures, doc, "figure", stem, fitz_doc, OUT_DIR / "figures")
        tables = _media_for(doc.tables, doc, "table", stem, fitz_doc, OUT_DIR / "tables")
    finally:
        fitz_doc.close()

    MediaJsonWriter(output_dir=OUT_DIR / "json").write(stem, [], figures, tables)
    return len(figures), len(tables)


def _load_total_tables() -> dict[str, int]:
    import csv
    gt = _REPO_ROOT / "eval" / "ground_truth.csv"
    out: dict[str, int] = {}
    if gt.exists():
        with gt.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[row["pmcid"]] = int(row.get("total_tables") or 0)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N PDFs (smoke test)")
    ap.add_argument("--summary", action="store_true",
                    help="after extraction, print a T1 detection-recall summary vs ground_truth totals")
    args = ap.parse_args(argv)

    # Restrict to the frozen 27-PDF rubric set = PDFs present in eval/pdfs AND in
    # ground_truth.csv. This excludes PDFs added after the rubric was frozen
    # (e.g. PMC11863705_main), keeping the variant comparable to the sweep.
    gt_pmcids = set(_load_total_tables().keys())
    pdfs = sorted(p for p in PDF_DIR.glob("*.pdf") if p.stem in gt_pmcids)
    skipped = sorted(p.stem for p in PDF_DIR.glob("*.pdf") if p.stem not in gt_pmcids)
    if skipped:
        print(f"skipping {len(skipped)} PDF(s) not in ground_truth.csv: {skipped}", file=sys.stderr)
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        print(f"no rubric PDFs in {PDF_DIR}", file=sys.stderr)
        return 1

    from docling.document_converter import DocumentConverter  # type: ignore
    converter = DocumentConverter()

    print(f"Off-the-shelf Docling baseline → {OUT_DIR}  ({len(pdfs)} PDFs)", file=sys.stderr)
    counts: dict[str, tuple[int, int]] = {}
    for i, pdf in enumerate(pdfs, 1):
        try:
            n_fig, n_tab = process_pdf(pdf, converter)
            counts[pdf.stem] = (n_fig, n_tab)
            print(f"  [{i}/{len(pdfs)}] {pdf.stem}: {n_fig} figures, {n_tab} tables", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 — keep going; record nothing for failures
            print(f"  [{i}/{len(pdfs)}] {pdf.stem}: FAILED — {exc}", file=sys.stderr)

    # Minimal manifest so eval/annotate.py (its [p] source+bbox overlay) and the
    # scorer's detector_modes can resolve the source PDFs + detector without a full
    # pipeline run. Without it, annotate.py's resolve_pdf_dir returns None and [p]
    # silently no-ops.
    manifest_dir = OUT_DIR / "run_metadata"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "run_offtheshelf.json").write_text(json.dumps({
        "input": {"pdf_dir": "eval/pdfs"},
        "config": {"table_detector": "DOCLING"},
        "note": "off-the-shelf Docling baseline (no pipeline scaffolding)",
    }, indent=2))

    if args.summary:
        totals = _load_total_tables()
        tab_emitted = sum(t for _, t in counts.values())
        tab_total = sum(totals.get(s, 0) for s in counts)
        print("\n=== T1 detection summary (off-the-shelf Docling) ===")
        print(f"figures emitted : {sum(f for f, _ in counts.values())}")
        print(f"tables emitted  : {tab_emitted}")
        print(f"tables in GT     : {tab_total}  (sum of total_tables over processed PDFs)")
        if tab_total:
            print(f"table emit/total : {tab_emitted / tab_total:.1%}  "
                  f"(crude — counts emitted vs real, pre-rubric)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
