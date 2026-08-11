#!/usr/bin/env python3
"""Assets for the mechanisms that had no example (TODO §G).

G1  header / footer / sidebar masking      -> fm_g1_header_footer_sidebar.png
G3c top-strip table drop (masthead)        -> fm_g3c_masthead_as_table.png
E14 three suppressed detector FPs          -> fm_e14_false_positives.png
E16 text-assembly cleanup                  -> fm_e16_text_cleanup.png
"""
import json
import glob
import re
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from nlp_histo.pipeline.stages.pdf_text_extraction.config import PipelineConfig
from nlp_histo.pipeline.stages.pdf_text_extraction.components.node_scorer import NodeScorer
from nlp_histo.pipeline.stages.pdf_text_extraction.components.evidence_gatherer import (
    PyMuPDFEvidenceGatherer)
from nlp_histo.pipeline.stages.pdf_text_extraction.models.dto import LayoutElement, BoundingBox
from nlp_histo.parsers.layout_utils import filter_artifacts

R = Path("/Users/emir/Documents/GitHub/nlp-histo")
OUT = R / "presentation/assets/fault_modes"
V18 = "18_hybrid_best_family_fixes_footnote_expand_1_2"
BLUE, ORANGE, GREY, PURPLE = "#005293", "#E37222", "#595959", "#9B4F96"
GREEN, RED = "#6B8E00", "#C4161C"
cfg = PipelineConfig()


def layout(pm):
    lay = [p for p in glob.glob(f"out/sweeps/{V18}/docling_full/{pm}_tbl*_layout.json")
           if "twopass" not in p][0]
    L = json.load(open(lay))
    dims = {int(k): v for k, v in (L.get("page_dims") or {}).items()}
    els = [LayoutElement(type=e["type"], page=e["page"],
           bbox=BoundingBox(page=e["page"], x1=e["bbox"]["x1"], y1=e["bbox"]["y1"],
                            x2=e["bbox"]["x2"], y2=e["bbox"]["y2"]),
           text=e.get("text") or "", level=e.get("level"))
           for e in L["elements"] if e.get("bbox")]
    return dims, els


def scored_for(pm):
    dims, els = layout(pm)
    g = PyMuPDFEvidenceGatherer(R / "eval/pdfs" / f"{pm}.pdf", cfg.two_pass)
    ev = [g.gather(e, (dims.get(e.page) or {}).get("height", 842)) for e in els]
    g.close()
    return dims, els, NodeScorer(cfg.two_pass, dims).score_all(els, ev)


def to_fitz(bb, H):
    xs, ys = sorted([bb.x1, bb.x2]), sorted([bb.y1, bb.y2])
    return fitz.Rect(xs[0], H - ys[1], xs[1], H - ys[0])


def caption_page(png, title, sub, fw=7.4):
    """Wrap a rendered page PNG with a caption strip."""
    img = mpimg.imread(png)
    h, w = img.shape[0], img.shape[1]
    ih = fw * h / w
    fig = plt.figure(figsize=(fw, ih + 0.66), dpi=150, facecolor="white")
    ax = fig.add_axes([0, 0, 1, ih / (ih + 0.66)])
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#CCCCCC")
    fig.text(.012, .985, title, fontsize=11.5, fontweight="bold", va="top")
    fig.text(.012, .950, sub, fontsize=8.6, color=GREY, va="top")
    return fig


# ===================================================== G1 header/footer/sidebar
# Faithful before/after: the pipeline already writes the redacted PDF, so we
# render the real thing rather than re-implementing the masking geometry.
print("[G1] header / footer / sidebar masking (real redacted PDF)")

pm = "PMC11755463_dermatopathology-12-00002"
orig_p = R / "eval/pdfs" / f"{pm}.pdf"
mask_p = next(iter(glob.glob(str(R / f"out/sweeps/{V18}/masked_pdfs/{pm}*.pdf"))))
o, m = fitz.open(orig_p), fitz.open(mask_p)
PG = 1
ot = [ln.strip() for ln in o[PG - 1].get_text().splitlines() if ln.strip()]
mt = {ln.strip() for ln in m[PG - 1].get_text().splitlines() if ln.strip()}
gone = [ln for ln in ot if ln not in mt]
print(f"   page {PG}: {len(gone)} text lines removed by redaction")

fig = plt.figure(figsize=(10.6, 5.4), dpi=170, facecolor="white")
fig.text(.012, .972, "Header, footer and sidebar are redacted before the second Docling pass",
         fontsize=13.5, fontweight="bold", va="top")
fig.text(.012, .925,
         f"{pm} p{PG}  ·  original vs the pipeline's own *_twopass_masked.pdf"
         f"  ·  {len(gone)} text lines removed",
         fontsize=9.5, color=GREY, va="top")
for i, (doc, ttl, col) in enumerate(((o, "Original PDF", "#B0B0B0"),
                                     (m, "After two-pass redaction", BLUE))):
    pix = doc[PG - 1].get_pixmap(dpi=150, alpha=False)
    tmpf = OUT / f"_g1_{i}.png"
    pix.save(tmpf)
    ax = fig.add_axes([.012 + i * .245, .20, .225, .70])
    ax.imshow(mpimg.imread(tmpf))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(col)
        sp.set_linewidth(1.8)
    ax.set_title(ttl, fontsize=11.5, color=col, pad=5)
    tmpf.unlink()

fig.text(.515, .862, "Text present in the original and gone after redaction",
         fontsize=11.5, fontweight="bold", color=ORANGE, va="top")
y = .800
for line in gone[:14]:
    fig.text(.515, y, line[:62], family="monospace", fontsize=8.4, color=ORANGE, va="top")
    y -= .042
if len(gone) > 14:
    fig.text(.515, y, f"... and {len(gone) - 14} more", family="monospace",
             fontsize=8.4, color="#AAAAAA", va="top")
fig.text(.012, .085,
         "MDPI keeps the editorial sidebar — Academic Editor, submission dates, the citation block and\n"
         "the licence notice — in the left margin of page 1. None of it is body text, yet all of it would\n"
         "otherwise be chunked and sent to a language model as if it were content.",
         fontsize=9.4, color=GREY, va="top")
fig.savefig(OUT / "fm_g1_header_footer_sidebar.png", facecolor="white")
plt.close(fig)
o.close()
m.close()
print("   -> fm_g1_header_footer_sidebar.png")

# ============================================== G3c masthead detected as a table
print("[G3c] running header detected as a table")
pm3 = "PMC4945808_dpa-0003-0055"
regs = [t for t in json.load(open(R / f"out/sweeps/07_hybrid_099/json/{pm3}_media.json"))["tables"]]
doc = fitz.open(R / "eval/pdfs" / f"{pm3}.pdf")
H3 = doc[0].rect.height
page = doc[0]
rr = tuple(int(RED[i:i + 2], 16) / 255 for i in (1, 3, 5))
n = 0
for t in regs:
    if t["page"] != 1:
        continue
    b = t["bbox"]
    xs, ys = sorted([b["x1"], b["x2"]]), sorted([b["y1"], b["y2"]])
    r = fitz.Rect(xs[0], H3 - ys[1], xs[1], H3 - ys[0])
    if r.y0 > 60:
        continue
    page.draw_rect(r, color=rr, width=2.0, dashes="[8 3 2 3] 0")
    page.insert_text(fitz.Point(r.x0, r.y1 + 11),
                     f"detected as a table — {r.y0:.0f} pt from page top",
                     fontsize=8.5, color=rr, fontname="hebo")
    n += 1
page.draw_line(fitz.Point(0, 50), fitz.Point(page.rect.width, 50),
               color=(0.45, 0.45, 0.45), width=0.9, dashes="[3 3] 0")
page.insert_text(fitz.Point(page.rect.width - 150, 46), "drop_tables_in_top_pts = 50",
                 fontsize=8, color=(0.45, 0.45, 0.45), fontname="hebo")
tmp = OUT / "_g3c.png"
page.get_pixmap(dpi=230, alpha=False).save(tmp)
doc.close()
fig = caption_page(tmp, "The journal running header, detected as a table on six pages",
                   f"{pm3} p1  ·  present with drop_tables_in_top_pts = 0, absent at 50  ·  "
                   f"666 such regions dropped corpus-wide")
fig.savefig(OUT / "fm_g3c_masthead_as_table.png", facecolor="white")
plt.close(fig)
tmp.unlink()
print(f"   -> fm_g3c_masthead_as_table.png ({n} region(s) boxed on p1)")

# ================================================ E14 three suppressed FPs
print("[E14] three suppressed detector false positives")
panels = [
    (R / "out/sweeps/07_hybrid_099/tables/PMC4945808_dpa-0003-0055_Table_1_p1.png",
     "Running header", "detected as a table on 6 pages\ndropped by drop_tables_in_top_pts"),
    (R / "out/sweeps/00_docling_offtheshelf/tables/PMC8420544_CYT-33-93_Table_1_p1.png",
     "Author byline", "rubric: “should be masked,\nnot a table in the scientific sense”"),
    (R / "out/sweeps/01_docling/tables/PMC11791726_HIS-86-485_Table_3_p9.png",
     "Table inside a figure", "rubric: table_in_figure\ndropped by drop_tables_inside_figures"),
]
fig = plt.figure(figsize=(10.2, 2.95), dpi=180, facecolor="white")
for i, (p, ttl, sub) in enumerate(panels):
    ax = fig.add_axes([0.035 + i * 0.324, 0.33, 0.29, 0.44])
    if p.exists():
        ax.imshow(mpimg.imread(p))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(RED)
        s.set_linewidth(1.8)
    ax.set_title(ttl, fontsize=12.5, color=RED, fontweight="bold", pad=6)
    fig.text(0.035 + i * 0.324, 0.265, sub, fontsize=9.5, color=GREY, va="top")
fig.text(.035, .965, "All three were emitted as tables by a detector and suppressed by the pipeline",
         fontsize=12.5, fontweight="bold", va="top")
fig.text(.035, .895, "none is a table in the scientific sense", fontsize=9.5, color=GREY, va="top")
fig.savefig(OUT / "fm_e14_false_positives.png", facecolor="white")
plt.close(fig)
print("   -> fm_e14_false_positives.png")

# ================================================ E16 text-assembly cleanup
print("[E16] text-assembly cleanup")
pm4 = "PMC5803687_dpa-0004-0013"
raw = (R / f"out/text_raw/{pm4}_raw.txt").read_text(encoding="utf-8", errors="replace")
asm = (R / f"out/text/{pm4}_text.txt").read_text(encoding="utf-8", errors="replace")
cit_r = len(re.findall(r"\[\d+(?:[,\-–]\s*\d+)*\]", raw))
cit_a = len(re.findall(r"\[\d+(?:[,\-–]\s*\d+)*\]", asm))

# filter_artifacts on a real layout
dropped = []
for pmx in ("PMC10297671_dermatopathology-10-00024", "PMC11755463_dermatopathology-12-00002"):
    try:
        _, els = layout(pmx)
    except Exception:
        continue
    dicts = [dict(type=e.type, page=e.page, text=e.text,
                  bbox=dict(x1=e.bbox.x1, y1=e.bbox.y1, x2=e.bbox.x2, y2=e.bbox.y2))
             for e in els]
    kept = filter_artifacts(list(dicts))
    keptset = {(d["page"], d.get("text")) for d in kept}
    for d in dicts:
        if (d["page"], d.get("text")) not in keptset and (d.get("text") or "").strip():
            dropped.append((pmx, d["page"], (d["text"] or "").strip()))
    if len(dropped) >= 6:
        break
print(f"   filter_artifacts dropped {len(dropped)} elements in the sampled documents")

fig = plt.figure(figsize=(10.6, 4.2), dpi=175, facecolor="white")
fig.text(.012, .965, "Text assembly also strips citations and layout artifacts",
         fontsize=13, fontweight="bold", va="top")
fig.text(.012, .905, f"{pm4} and two MDPI documents  ·  out/text_raw vs out/text",
         fontsize=9.5, color=GREY, va="top")
fig.text(.012, .82, "Citation stripping", fontsize=11.5, fontweight="bold", color=BLUE, va="top")
for i, (lab, a, b) in enumerate((("bracketed citation markers", cit_r, cit_a),
                                 ("double spaces", raw.count("  "), asm.count("  ")),
                                 ("lines", len(raw.splitlines()), len(asm.splitlines())))):
    fig.text(.012, .755 - i * .058, f"{lab:32s} {a:>6}  →  {b}", va="top",
             family="monospace", fontsize=9.6,
             color=BLUE if i == 0 else "#333333", fontweight="bold" if i == 0 else "normal")
fig.text(.52, .82, "filter_artifacts — elements removed", fontsize=11.5, fontweight="bold",
         color=BLUE, va="top")
y = .755
for pmx, pgx, t in dropped[:6]:
    fig.text(.52, y, f"p{pgx}  {t[:52]!r}", va="top", family="monospace", fontsize=8.6,
             color=ORANGE)
    y -= .055
fig.text(.012, .085,
         "filter_artifacts drops: elements with no alphabetic characters · short sidebar metadata "
         "lines (Received/Revised/Accepted/Published) ·\nsingle-line TEXT outside the anchor "
         "y-range · NER-scored singletons on anchorless pages.",
         fontsize=9.2, color=GREY, va="top")
fig.savefig(OUT / "fm_e16_text_cleanup.png", facecolor="white")
plt.close(fig)
print("   -> fm_e16_text_cleanup.png")


# ============================================ E15 missed tables + captions
print("[E15] tables and captions the baseline did not produce")
V01, OTS = "01_docling", "00_docling_offtheshelf"
cells = [
    (None,
     R / f"out/sweeps/{V18}/tables/PMC8420544_CYT-33-93_Table_1_p3.png",
     "Table never emitted", "PMC8420544 p3 · recovered by TATR\nrubric: correct"),
    (R / f"out/sweeps/{OTS}/tables/PMC11863984_main_Table_1_p3.png",
     R / f"out/sweeps/{V18}/tables/PMC11863984_main_Table_1_p3.png",
     "Table caption not attached", "PMC11863984 p3 · caption '' →\n'Table 1 Overall characteristics…'"),
    (R / f"out/sweeps/{OTS}/figures/PMC10297671_dermatopathology-10-00024_Figure_7_p2.png",
     R / f"out/sweeps/{V18}/figures/PMC10297671_dermatopathology-10-00024_Figure_1_p2.png",
     "Figure caption not attached", "PMC10297671 p2 · caption '' →\n'Figure 1. p16 Ink4a staining…'"),
]
fig = plt.figure(figsize=(10.4, 4.5), dpi=175, facecolor="white")
fig.text(.032, .965, "Three artifacts the baseline failed to produce correctly",
         fontsize=12.5, fontweight="bold", va="top")
fig.text(.032, .905, "left of each pair: Docling baseline   ·   right: proposed pipeline",
         fontsize=9.5, color=GREY, va="top")
for i, (bp, vp, ttl, sub) in enumerate(cells):
    x0 = .032 + i * .325
    fig.text(x0, .845, ttl, fontsize=11.5, fontweight="bold", color=BLUE, va="top")
    for j, (pp, col) in enumerate(((bp, "#B0B0B0"), (vp, BLUE))):
        ax = fig.add_axes([x0 + j * .145, .34, .132, .44])
        if pp is not None and Path(pp).exists():
            ax.imshow(mpimg.imread(pp))
        else:
            ax.set_facecolor("#F4F4F4")
            ax.text(.5, .5, "not\nemitted", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9.5, color="#9A9A9A", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(col)
            sp.set_linewidth(1.5)
            if pp is None:
                sp.set_linestyle((0, (4, 3)))
    fig.text(x0, .275, sub, fontsize=9.2, color=GREY, va="top")
fig.text(.032, .075,
         "Captions are not cosmetic: a caption is the only human-readable description of what a "
         "table or figure shows,\nand it is the first thing an auditor reads when checking a rule "
         "that cites it.", fontsize=9.4, color=GREY, va="top")
fig.savefig(OUT / "fm_e15_missed_and_captions.png", facecolor="white")
plt.close(fig)
print("   -> fm_e15_missed_and_captions.png")
