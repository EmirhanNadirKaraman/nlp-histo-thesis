"""
Empirical check: does the existing pixel-render path in PyMuPDFEvidenceGatherer
catch PDF text rendered with Tr=3 (invisible / "use as clipping path") and with
fill_opacity=0?

We construct a minimal one-page PDF with three text rows:
  A. visible baseline                   (Tr=0)
  B. invisible via render-mode 3        (Tr=3)
  C. invisible via fill_opacity=0       (ExtGState ca=0)

Then we run PyMuPDFEvidenceGatherer.gather() on each row's bbox and report
visually_blank / dark_pixel_fraction / pixel_brightness_mean / invisible_char_fraction.

Expected:
  - Row A: visually_blank=False, dark_pixel_fraction high enough to look like real text.
  - Rows B and C: visually_blank=True, dark_pixel_fraction ~0.

Run:  python scripts/verify_ghost_text_detection.py
Exit code is 0 on pass, 1 on fail.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make repo modules importable when run as a script
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


PAGE_W, PAGE_H = 612.0, 792.0  # US Letter, points


def _build_pdf(path: Path) -> dict[str, tuple[float, float, float, float]]:
    """
    Hand-craft a single-page PDF with three text rows. Returns row → bbox in
    Docling PDF coords (y=0 at bottom; y1=top, y2=bottom).
    """
    # Row positions: text baseline y (PDF coords, y=0 bottom). Font size 24.
    rows = {
        "visible":     720.0,
        "tr3":         640.0,
        "opacity_0":   560.0,
    }
    font_size = 24.0
    text_h    = 28.0          # generous bbox height
    x0, x1    = 72.0, 540.0   # bbox x range

    bboxes: dict[str, tuple[float, float, float, float]] = {}
    for name, baseline_y in rows.items():
        # Docling-style bbox: y1=top (larger), y2=bottom (smaller)
        bboxes[name] = (x0, baseline_y + text_h * 0.2, x1, baseline_y - text_h * 0.8)

    # Content stream with three text objects.
    # /GS1 is an ExtGState resource with fill alpha 0.
    content = (
        "BT /F1 {fs} Tf {x} {y_vis} Td (VISIBLE BASELINE) Tj ET\n"
        "BT /F1 {fs} Tf {x} {y_tr3} Td 3 Tr (GHOST RENDER MODE 3) Tj 0 Tr ET\n"
        "q /GS1 gs BT /F1 {fs} Tf {x} {y_op} Td (GHOST OPACITY ZERO) Tj ET Q\n"
    ).format(
        fs=font_size, x=x0,
        y_vis=rows["visible"], y_tr3=rows["tr3"], y_op=rows["opacity_0"],
    ).encode("ascii")

    # Object 1: Catalog. Object 2: Pages. Object 3: Page. Object 4: Contents.
    # Object 5: Font. Object 6: ExtGState (ca = 0 → fully transparent fill).
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R "
            b"/MediaBox [0 0 " + f"{PAGE_W} {PAGE_H}".encode() + b"] "
            b"/Resources << /Font << /F1 5 0 R >> "
            b"/ExtGState << /GS1 6 0 R >> >> "
            b"/Contents 4 0 R >>"
        ),
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /ExtGState /ca 0 /CA 0 >>",
    ]

    # Assemble PDF with computed byte offsets for xref.
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode()
        out += body
        out += b"\nendobj\n"

    xref_start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()

    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n"
    ).encode()

    path.write_bytes(bytes(out))
    return bboxes


def _make_element(name: str, bbox_tuple: tuple[float, float, float, float]) -> LayoutElement:
    x1, y1, x2, y2 = bbox_tuple
    return LayoutElement(
        type="TEXT",
        page=1,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, page=1),
        text=name,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "ghost_text_probe.pdf"
        bboxes = _build_pdf(pdf_path)

        # Sanity: PyMuPDF can open it
        import fitz
        with fitz.open(str(pdf_path)) as doc:
            page_h = doc[0].rect.height
        assert abs(page_h - PAGE_H) < 1e-3, f"page height mismatch: {page_h}"

        cfg = TwoPassConfig()
        results: dict[str, dict] = {}
        with PyMuPDFEvidenceGatherer(pdf_path, cfg) as gatherer:
            for name, bbox_tuple in bboxes.items():
                el = _make_element(name, bbox_tuple)
                ev = gatherer.gather(el, page_height=PAGE_H)
                results[name] = {
                    "visually_blank":          ev.visually_blank,
                    "pixel_brightness_mean":   round(ev.pixel_brightness_mean, 2)
                                                  if ev.pixel_brightness_mean is not None else None,
                    "dark_pixel_fraction":     round(ev.dark_pixel_fraction, 4)
                                                  if ev.dark_pixel_fraction is not None else None,
                    "invisible_char_fraction": round(ev.invisible_char_fraction, 4)
                                                  if ev.invisible_char_fraction is not None else None,
                    "render_skipped":          ev.render_skipped,
                }

    # Report
    print(f"{'row':<14} {'visually_blank':<15} {'brightness':<12} {'dark_frac':<10} {'inv_char_frac':<14} render_skipped")
    for name, r in results.items():
        print(
            f"{name:<14} "
            f"{str(r['visually_blank']):<15} "
            f"{str(r['pixel_brightness_mean']):<12} "
            f"{str(r['dark_pixel_fraction']):<10} "
            f"{str(r['invisible_char_fraction']):<14} "
            f"{r['render_skipped']}"
        )

    # Assertions
    failures: list[str] = []

    vis = results["visible"]
    if vis["render_skipped"]:
        failures.append("visible row: render_skipped=True (numpy missing?)")
    elif vis["visually_blank"]:
        failures.append("visible row was flagged visually_blank — pixel check is over-eager")
    elif vis["dark_pixel_fraction"] is None or vis["dark_pixel_fraction"] < 0.02:
        failures.append(
            f"visible row dark_pixel_fraction={vis['dark_pixel_fraction']} "
            "— text did not render as expected"
        )

    for ghost in ("tr3", "opacity_0"):
        g = results[ghost]
        if g["render_skipped"]:
            failures.append(f"{ghost}: render_skipped=True (numpy missing?)")
            continue
        if not g["visually_blank"]:
            failures.append(
                f"{ghost}: pixel check did NOT flag visually_blank "
                f"(brightness={g['pixel_brightness_mean']}, "
                f"dark_frac={g['dark_pixel_fraction']})"
            )

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nPASS: pixel-render path catches both Tr=3 and fill_opacity=0 text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
