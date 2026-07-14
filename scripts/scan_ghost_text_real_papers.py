"""
Scan a sample of real papers for ghost-text candidates.

For each sampled paper we:
  1. Load its cached Docling layout from out/docling_full/.
  2. Locate the source PDF in files/organized_pdfs/.
  3. Run PyMuPDFEvidenceGatherer.gather() on every TEXT-like element.
  4. Flag elements where visually_blank=True OR invisible_char_fraction > 0.5.
  5. Print a per-paper summary and a sample of suspicious elements.

Usage:
    python scripts/scan_ghost_text_real_papers.py [N]

N defaults to 8.  Sampling is deterministic (seeded sorted slice).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nlp_histo.pipeline.stages.pdf_text_extraction.components.evidence_gatherer import (  # noqa: E402
    PyMuPDFEvidenceGatherer,
)
from nlp_histo.pipeline.stages.pdf_text_extraction.config import TwoPassConfig  # noqa: E402
from nlp_histo.pipeline.stages.pdf_text_extraction.models.dto import (  # noqa: E402
    BoundingBox,
    LayoutElement,
)


PDF_DIR    = REPO_ROOT / "files" / "organized_pdfs"
LAYOUT_DIR = REPO_ROOT / "out" / "docling_full"

# Element types that carry rendered glyphs.  Match TEXT_ELEMENT_TYPES in
# parsers/layout_utils.py loosely (we want to evaluate anything text-bearing).
TEXT_TYPES = {
    "TEXT", "SECTION_HEADER", "CAPTION", "TITLE",
    "LIST_ITEM", "FOOTNOTE", "PAGE_HEADER", "PAGE_FOOTER",
}

INVISIBLE_CHAR_FRAC_THRESHOLD = 0.5  # > half of chars white-on-white → suspicious


def _pmcid_from_stem(stem: str) -> str:
    """`PMC123_NEUP-42-540_tbl1_ocr0_fp0_engna_sc2.0_layout` → `PMC123_NEUP-42-540`."""
    if "_tbl" in stem:
        return stem.split("_tbl", 1)[0]
    return stem


def _resolve_pdf(layout_path: Path) -> Optional[Path]:
    base = _pmcid_from_stem(layout_path.stem)
    candidate = PDF_DIR / f"{base}.pdf"
    return candidate if candidate.exists() else None


def _scan_paper(layout_path: Path, pdf_path: Path) -> dict:
    data = json.loads(layout_path.read_text())
    page_dims = {int(k): v for k, v in data["page_dims"].items()}
    elements_raw = data.get("elements", [])

    cfg = TwoPassConfig()
    total = 0
    blank_hits: list[dict] = []
    color_hits: list[dict] = []

    with PyMuPDFEvidenceGatherer(pdf_path, cfg) as gatherer:
        for raw in elements_raw:
            if raw.get("type") not in TEXT_TYPES:
                continue
            text = (raw.get("text") or "").strip()
            if len(text) < cfg.min_text_chars_for_word_check:
                continue
            page = raw.get("page", 1)
            ph = page_dims.get(page, {}).get("height")
            if ph is None:
                continue

            bbox = BoundingBox.from_dict(raw["bbox"], page=page)
            el = LayoutElement(type=raw["type"], page=page, bbox=bbox, text=text)
            ev = gatherer.gather(el, page_height=ph)
            total += 1

            if ev.render_skipped:
                continue

            if ev.visually_blank:
                blank_hits.append({
                    "page": page, "type": raw["type"],
                    "text_preview": text[:80].replace("\n", " "),
                    "brightness":   round(ev.pixel_brightness_mean or 0, 1),
                    "dark_frac":    round(ev.dark_pixel_fraction or 0, 4),
                    "inv_char_frac": round(ev.invisible_char_fraction or 0, 3)
                                     if ev.invisible_char_fraction is not None else None,
                    "bbox": [round(bbox.x1, 1), round(bbox.y1, 1),
                             round(bbox.x2, 1), round(bbox.y2, 1)],
                })
            if (ev.invisible_char_fraction or 0) > INVISIBLE_CHAR_FRAC_THRESHOLD \
               and not ev.visually_blank:
                color_hits.append({
                    "page": page, "type": raw["type"],
                    "text_preview": text[:80].replace("\n", " "),
                    "inv_char_frac": round(ev.invisible_char_fraction, 3),
                    "brightness":   round(ev.pixel_brightness_mean or 0, 1),
                    "dark_frac":    round(ev.dark_pixel_fraction or 0, 4),
                })

    return {
        "total_text_elements": total,
        "visually_blank_count": len(blank_hits),
        "white_on_white_count": len(color_hits),
        "blank_hits": blank_hits,
        "color_hits": color_hits,
    }


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8

    # Deterministic spread: take every Kth full-layout file across PMCID-sorted list.
    layouts = sorted(p for p in LAYOUT_DIR.glob("*_tbl1_ocr0_fp0_engna_sc2.0_layout.json")
                     if "twopass" not in p.name)
    if not layouts:
        print(f"No layouts in {LAYOUT_DIR}")
        return 1
    step = max(1, len(layouts) // n)
    sample = layouts[::step][:n]

    print(f"Scanning {len(sample)} papers (1 every {step} of {len(layouts)})\n")
    grand_total = 0
    grand_blank = 0
    grand_white = 0

    for lp in sample:
        pdf = _resolve_pdf(lp)
        if pdf is None:
            print(f"[skip] no PDF for {lp.name}")
            continue
        try:
            r = _scan_paper(lp, pdf)
        except Exception as exc:  # noqa: BLE001
            print(f"[err]  {lp.stem}: {exc}")
            continue

        grand_total += r["total_text_elements"]
        grand_blank += r["visually_blank_count"]
        grand_white += r["white_on_white_count"]

        print(
            f"{_pmcid_from_stem(lp.stem):<55} "
            f"text_els={r['total_text_elements']:>4}  "
            f"visually_blank={r['visually_blank_count']:>3}  "
            f"white_on_white={r['white_on_white_count']:>3}"
        )

        for h in r["blank_hits"][:3]:
            print(
                f"    BLANK  p{h['page']} {h['type']:<14} "
                f"bright={h['brightness']:<6} dark={h['dark_frac']:<7} "
                f"inv_frac={h['inv_char_frac']}\n"
                f"           text: {h['text_preview']!r}"
            )
        if len(r["blank_hits"]) > 3:
            print(f"    ... +{len(r['blank_hits']) - 3} more BLANK hits")

        for h in r["color_hits"][:3]:
            print(
                f"    WHITE  p{h['page']} {h['type']:<14} "
                f"inv_frac={h['inv_char_frac']} bright={h['brightness']} "
                f"dark={h['dark_frac']}\n"
                f"           text: {h['text_preview']!r}"
            )
        if len(r["color_hits"]) > 3:
            print(f"    ... +{len(r['color_hits']) - 3} more WHITE hits")

    print()
    print(
        f"TOTAL: text_elements={grand_total}  "
        f"visually_blank={grand_blank} ({100*grand_blank/max(grand_total,1):.2f}%)  "
        f"white_on_white={grand_white} ({100*grand_white/max(grand_total,1):.2f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
