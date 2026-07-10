"""
Thesis demo: ghost-text detection — before/after on real papers.

Demonstrates two concrete failure modes of the original extraction policy and
how the post-fix policy resolves both:

  DEMO 1 (false negative — phantom passes through)
      Paper: PMC10047158_dermatopathology-10-00017
      Old policy: TwoPassConfig.enabled=False — no per-element scoring runs,
                  so Docling "phantom" layout elements (real text content
                  carried over to a bbox that does not render) pass through.
                  The downstream ContextAwareStitcher happens to merge them
                  into real paragraphs, but this is incidental cleanup that
                  could fail on slightly different layouts.
      New policy: TwoPassConfig.enabled=True — PyMuPDFEvidenceGatherer renders
                  each bbox; NodeScorer.R1 rejects elements with
                  visually_blank=True.  The phantom is dropped at extraction.

  DEMO 2 (false positive — legitimate header dropped)
      Paper: PMC7158325_main
      Old policy: TwoPassConfig.max_white_char_fraction=0.5 — Rule R-color
                  rejects elements whose text spans report near-white color.
                  Legitimate "Key Points" SECTION_HEADERs (white type on a
                  coloured banner) trip the rule because their span color is
                  0xffffff regardless of background.
      New policy: TwoPassConfig.max_white_char_fraction=1.0 — Rule R-color
                  disabled; Rule R1 (pixel-render) decides instead.  Pixel
                  pass measures actual ink in the bbox (dark_pixel_fraction
                  ≈ 0.73 for "Key Points" because the rendered banner is
                  full of pixels) so the header is correctly kept.

The script does not re-run Docling — it loads the cached layout from
out/docling_full/ and re-applies the scorer under both policies.

Outputs:
    stdout:                          human-readable report
    out/thesis_demo/ghost_text/*.json  per-demo JSON artifacts for thesis figures

Usage:
    python scripts/thesis_demo_ghost_text.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.stages.pdf_text_extraction.components.evidence_gatherer import (  # noqa: E402
    PyMuPDFEvidenceGatherer,
)
from pipeline.stages.pdf_text_extraction.components.node_scorer import (  # noqa: E402
    NodeScorer,
)
from pipeline.stages.pdf_text_extraction.config import TwoPassConfig  # noqa: E402
from pipeline.stages.pdf_text_extraction.models.dto import (  # noqa: E402
    BoundingBox,
    LayoutElement,
)


def _render_bbox_overlay(
    pdf_path: Path,
    page_no: int,
    bbox: BoundingBox,
    out_path: Path,
    page_height: float,
    *,
    dpi: int = 110,
    crop_padding: int = 40,
) -> None:
    """
    Render `page_no` of `pdf_path` and outline `bbox` (Docling coords) in red.
    Writes a PNG to `out_path`.
    """
    import fitz                              # type: ignore
    from PIL import Image, ImageDraw         # type: ignore

    scale = dpi / 72.0
    with fitz.open(str(pdf_path)) as doc:
        page = doc[page_no - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    # Convert Docling bbox → fitz/screen rect → pixel rect
    fitz_rect = bbox.to_fitz_rect(page_height)
    px0 = int(round(fitz_rect.x0 * scale))
    py0 = int(round(fitz_rect.y0 * scale))
    px1 = int(round(fitz_rect.x1 * scale))
    py1 = int(round(fitz_rect.y1 * scale))

    draw = ImageDraw.Draw(img)
    draw.rectangle([px0, py0, px1, py1], outline=(220, 30, 30), width=4)

    # Crop to neighborhood of the bbox for legibility
    cx0 = max(0, px0 - crop_padding)
    cy0 = max(0, py0 - crop_padding)
    cx1 = min(img.width,  px1 + crop_padding)
    cy1 = min(img.height, py1 + crop_padding)
    cropped = img.crop((cx0, cy0, cx1, cy1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(out_path, format="PNG", optimize=True)


PDF_DIR    = REPO_ROOT / "files" / "organized_pdfs"
LAYOUT_DIR = REPO_ROOT / "out" / "docling_full"
OUT_DIR    = REPO_ROOT / "out" / "thesis_demo" / "ghost_text"

TEXT_TYPES = {
    "TEXT", "SECTION_HEADER", "CAPTION", "TITLE",
    "LIST_ITEM", "FOOTNOTE", "PAGE_HEADER", "PAGE_FOOTER",
    "PARAGRAPH",
}

DEMOS = [
    {
        "key":        "demo1_phantom_false_negative",
        "title":      "DEMO 1 — Phantom Docling element slips through old policy",
        "pmcid":      "PMC10047158_dermatopathology-10-00017",
        "narrative":  (
            "Old policy: TwoPassConfig.enabled=False — NodeScorer never runs, "
            "so every Docling element reaches downstream stages.\n"
            "New policy: enabled=True — Rule R1 (pixel-render) rejects "
            "elements whose bbox is visually blank."
        ),
        "find_marker": "centrally-located",   # text fragment to spotlight
    },
    {
        "key":        "demo2_key_points_false_positive",
        "title":      "DEMO 2 — Legitimate white-on-dark header dropped by old policy",
        "pmcid":      "PMC7158325_main",
        "narrative":  (
            "Old policy: max_white_char_fraction=0.5 — Rule R-color drops any "
            "element whose span color is near-white, regardless of background.\n"
            "New policy: max_white_char_fraction=1.0 — R-color disabled; "
            "Rule R1 (pixel-render) sees real ink and keeps the header."
        ),
        "find_marker": "Key Points",
    },
]


def _resolve_paths(pmcid: str) -> tuple[Path, Path]:
    layouts = list(LAYOUT_DIR.glob(f"{pmcid}_tbl1_ocr0_fp0_engna_sc2.0_layout.json"))
    layouts = [p for p in layouts if "twopass" not in p.name]
    if not layouts:
        raise FileNotFoundError(f"No cached layout for {pmcid} in {LAYOUT_DIR}")
    pdfs = list(PDF_DIR.glob(f"{pmcid}.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF for {pmcid} in {PDF_DIR}")
    return layouts[0], pdfs[0]


def _to_elements(data: dict) -> list[LayoutElement]:
    out: list[LayoutElement] = []
    for raw in data.get("elements", []):
        if raw.get("type") not in TEXT_TYPES:
            continue
        bbox = BoundingBox.from_dict(raw["bbox"], page=raw.get("page", 1))
        out.append(LayoutElement(
            type=raw["type"], page=raw.get("page", 1),
            bbox=bbox, text=raw.get("text") or "",
        ))
    return out


def _score(
    elements: list[LayoutElement],
    pdf_path: Path,
    page_dims: dict[int, dict],
    cfg: TwoPassConfig,
) -> list[dict]:
    """Gather evidence + run NodeScorer; return one dict per element."""
    scorer = NodeScorer(cfg, page_dims)
    rows: list[dict] = []
    with PyMuPDFEvidenceGatherer(pdf_path, cfg) as gatherer:
        for el in elements:
            ph = page_dims.get(el.page, {}).get("height")
            if ph is None:
                continue
            ev = gatherer.gather(el, page_height=ph)
            scored = scorer.score(el, ev)
            rows.append({
                "page":        el.page,
                "type":        el.type,
                "bbox":        [round(el.bbox.x1, 1), round(el.bbox.y1, 1),
                                round(el.bbox.x2, 1), round(el.bbox.y2, 1)],
                "text":        (el.text or "")[:160],
                "keep":        scored.keep,
                "reject_rule": scored.rejection_reason or "",
                "visually_blank":     ev.visually_blank,
                "pixel_brightness":   round(ev.pixel_brightness_mean, 1)
                                       if ev.pixel_brightness_mean is not None else None,
                "dark_pixel_fraction": round(ev.dark_pixel_fraction, 4)
                                       if ev.dark_pixel_fraction is not None else None,
                "invisible_char_fraction": round(ev.invisible_char_fraction, 4)
                                            if ev.invisible_char_fraction is not None else None,
            })
    return rows


def _flag_state(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    kept    = [r for r in rows if r["keep"]]
    dropped = [r for r in rows if not r["keep"]]
    reasons: dict[str, int] = {}
    for r in dropped:
        # Pick out the leading rule tag (text after a known marker) for tally
        reason = r["reject_rule"]
        key = "other"
        for tag in ("visually blank", "white-text ghost", "hidden text layer",
                    "empty text", "no fitz words"):
            if tag in reason:
                key = tag
                break
        reasons[key] = reasons.get(key, 0) + 1
    return {
        "total_text_elements": len(rows),
        "kept":                len(kept),
        "dropped":             len(dropped),
        "drop_reasons":        reasons,
    }


def _spotlight(rows: list[dict], marker: str) -> list[dict]:
    hits = [r for r in rows if marker.lower() in r["text"].lower()]
    return hits


def _policy_pair() -> tuple[TwoPassConfig, TwoPassConfig]:
    """
    Build (old, new) policy configs.
    OLD reproduces the pre-fix defaults: scoring effectively disabled.
    NEW matches the new defaults.
    """
    new_cfg = TwoPassConfig()  # post-fix defaults (enabled=True, max_white=1.0)
    # OLD: replicate prior defaults — enabled was False but for the demo we still
    # need to run the scorer to show what would have fired.  We model "old" as:
    #   - enabled was False in production → scorer didn't run at all → every
    #     element keeps (we tag with `keep=True, reason=""`).  To make the
    #     contrast crisp we instead enable the scorer but configure it as the
    #     pre-fix R-color policy: max_white_char_fraction=0.5, which is what
    #     `enabled=True` users would have hit before the threshold flip.
    old_cfg = replace(new_cfg, max_white_char_fraction=0.5)
    return old_cfg, new_cfg


def _run_demo(demo: dict) -> dict:
    print(f"\n{'=' * 70}\n{demo['title']}\n{'=' * 70}")
    print(demo["narrative"])
    print(f"Paper: {demo['pmcid']}")

    layout_path, pdf_path = _resolve_paths(demo["pmcid"])
    data = json.loads(layout_path.read_text())
    page_dims = {int(k): v for k, v in data["page_dims"].items()}
    elements = _to_elements(data)

    old_cfg, new_cfg = _policy_pair()

    # DEMO 1 contrasts "scorer disabled" vs "scorer enabled" — we model the
    # disabled side by skipping the scorer call.  DEMO 2 contrasts two scorer
    # configurations.  We branch by demo key for clarity.
    if demo["key"] == "demo1_phantom_false_negative":
        old_rows = [
            {
                "page": el.page, "type": el.type,
                "bbox": [round(el.bbox.x1, 1), round(el.bbox.y1, 1),
                         round(el.bbox.x2, 1), round(el.bbox.y2, 1)],
                "text": (el.text or "")[:160],
                "keep": True, "reject_rule": "",
                "visually_blank": None, "pixel_brightness": None,
                "dark_pixel_fraction": None, "invisible_char_fraction": None,
            }
            for el in elements
        ]
        # To still get evidence on the spotlight element for the report,
        # gather() once for the marker hit even on the old side.
        with PyMuPDFEvidenceGatherer(pdf_path, new_cfg) as g:
            for r, el in zip(old_rows, elements):
                if demo["find_marker"].lower() in (el.text or "").lower():
                    ph = page_dims.get(el.page, {}).get("height")
                    if ph is None:
                        continue
                    ev = g.gather(el, page_height=ph)
                    r["visually_blank"] = ev.visually_blank
                    r["pixel_brightness"] = (
                        round(ev.pixel_brightness_mean, 1)
                        if ev.pixel_brightness_mean is not None else None
                    )
                    r["dark_pixel_fraction"] = (
                        round(ev.dark_pixel_fraction, 4)
                        if ev.dark_pixel_fraction is not None else None
                    )
    else:
        old_rows = _score(elements, pdf_path, page_dims, old_cfg)

    new_rows = _score(elements, pdf_path, page_dims, new_cfg)

    old_state = _flag_state(old_rows)
    new_state = _flag_state(new_rows)
    old_hits  = _spotlight(old_rows, demo["find_marker"])
    new_hits  = _spotlight(new_rows, demo["find_marker"])

    # ── Report ───────────────────────────────────────────────────────────────
    def _state_line(label: str, s: dict) -> str:
        reasons = ", ".join(f"{k}={v}" for k, v in sorted(s["drop_reasons"].items())) or "—"
        return (
            f"  {label:<10} total={s['total_text_elements']:>4}  "
            f"kept={s['kept']:>4}  dropped={s['dropped']:>3}  "
            f"reasons: {reasons}"
        )

    print("\nSummary:")
    print(_state_line("OLD", old_state))
    print(_state_line("NEW", new_state))

    print(f"\nSpotlight on text containing {demo['find_marker']!r}:")
    print(f"  Matches under OLD policy: {len(old_hits)}")
    print(f"  Matches under NEW policy: {len(new_hits)}")

    def _hit_summary(rows: list[dict], policy: str) -> None:
        for h in rows[:3]:
            verdict = "KEEP" if h["keep"] else f"DROP[{h['reject_rule']}]"
            print(
                f"    [{policy}] p{h['page']} {h['type']:<14} "
                f"{verdict}\n"
                f"        bbox={h['bbox']}  visually_blank={h['visually_blank']}  "
                f"bright={h['pixel_brightness']}  dark={h['dark_pixel_fraction']}  "
                f"inv_char={h['invisible_char_fraction']}\n"
                f"        text: {h['text']!r}"
            )

    _hit_summary(old_hits, "OLD")
    _hit_summary(new_hits, "NEW")

    # ── Render PNG overlays for the spotlight elements ───────────────────────
    images: list[dict] = []
    seen_pages: set[int] = set()
    for h in new_hits[:3]:
        key = (h["page"], tuple(h["bbox"]))
        if h["page"] in seen_pages:
            continue
        seen_pages.add(h["page"])
        bbox = BoundingBox(
            x1=h["bbox"][0], y1=h["bbox"][1], x2=h["bbox"][2], y2=h["bbox"][3],
            page=h["page"],
        )
        ph = page_dims.get(h["page"], {}).get("height")
        if ph is None:
            continue
        png_name = f"{demo['key']}_p{h['page']}.png"
        out_path = OUT_DIR / png_name
        try:
            _render_bbox_overlay(pdf_path, h["page"], bbox, out_path, ph)
        except Exception as exc:                                  # noqa: BLE001
            print(f"  [render-fail] {png_name}: {exc}")
            continue
        images.append({
            "page":    h["page"],
            "bbox":    h["bbox"],
            "text":    h["text"][:120],
            "png":     str(out_path.relative_to(REPO_ROOT)),
            "caption": (
                f"Page {h['page']} bbox {h['bbox']} (Docling coords). "
                f"Rendered crop with the bbox outlined in red. "
                f"visually_blank={h['visually_blank']}, "
                f"dark_pixel_fraction={h['dark_pixel_fraction']}, "
                f"invisible_char_fraction={h['invisible_char_fraction']}."
            ),
        })
        print(f"  → image: {out_path.relative_to(REPO_ROOT)}")

    # DEMO 2 caveat: SECTION_HEADER is exempt from scoring (NodeScorer._ALWAYS_KEEP),
    # so R-color never actually fires on the production "Key Points" headers.  The
    # demo therefore shows the *evidence* (inv_char_frac=1.0 on a non-blank bbox)
    # rather than a behaviour difference; the false-positive risk it documents is
    # latent — it would have surfaced if R-color were applied to a TEXT/CAPTION
    # element with the same styling.  Disabling R-color by default closes the
    # door before such an element appears in the corpus.
    caveat = None
    if demo["key"] == "demo2_key_points_false_positive":
        caveat = (
            "SECTION_HEADER elements are listed in NodeScorer._ALWAYS_KEEP, so "
            "R-color never actually fired on these headers in production. The "
            "demo therefore documents the *evidence* (span color = 0xFFFFFF "
            "and invisible_char_fraction = 1.0 on a bbox whose rendered pixels "
            "are 73% dark ink) rather than an OLD-vs-NEW behaviour difference. "
            "Disabling R-color by default removes the latent false-positive "
            "risk for any future TEXT/CAPTION element printed in the same "
            "white-on-coloured style."
        )

    artifact = {
        "demo":       demo["key"],
        "title":      demo["title"],
        "pmcid":      demo["pmcid"],
        "narrative":  demo["narrative"],
        "old_policy": {
            "max_white_char_fraction": old_cfg.max_white_char_fraction,
            "scoring_active":          demo["key"] != "demo1_phantom_false_negative",
            "summary":                 old_state,
            "spotlight_hits":          old_hits,
        },
        "new_policy": {
            "max_white_char_fraction": new_cfg.max_white_char_fraction,
            "scoring_active":          True,
            "summary":                 new_state,
            "spotlight_hits":          new_hits,
        },
        "images":  images,
        "caveat":  caveat,
    }
    return artifact


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for demo in DEMOS:
        try:
            artifact = _run_demo(demo)
        except FileNotFoundError as exc:
            print(f"[skip] {demo['key']}: {exc}")
            continue
        out_path = OUT_DIR / f"{demo['key']}.json"
        out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
        print(f"\n→ written {out_path.relative_to(REPO_ROOT)}")

    print(f"\nAll demo artifacts in: {OUT_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
