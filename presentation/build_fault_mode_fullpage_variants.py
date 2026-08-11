#!/usr/bin/env python3
"""Full-page variants for the fault modes where *position on the page* is the
message rather than pixel detail.

E3  caption sits to the right of the figure -> fm_e3_side_caption_page.png
E14 three detector false positives          -> fm_e14_false_positives_page.png
E15 a missed table and two unattached caps  -> fm_e15_missed_and_captions_page.png
"""
import json
import glob
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

R = Path("/Users/emir/Documents/GitHub/nlp-histo")
OUT = R / "presentation/assets/fault_modes"
V18 = "18_hybrid_best_family_fixes_footnote_expand_1_2"
OTS, V01, V07 = "00_docling_offtheshelf", "01_docling", "07_hybrid_099"
BLUE, ORANGE, GREY, RED, GREEN = "#005293", "#E37222", "#595959", "#C4161C", "#6B8E00"


def rgb(h):
    """'#RRGGBB' -> the 0-1 float triple fitz wants."""
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


def media(v, pm, kind="tables"):
    p = R/"out/sweeps"/v/"json"/f"{pm}_media.json"
    return json.load(open(p)).get(kind) or [] if p.exists() else []


def layout_elements(pm):
    lay = [p for p in glob.glob(str(R/f"out/sweeps/{V18}/docling_full/{pm}_tbl*_layout.json"))
           if "twopass" not in p][0]
    return json.load(open(lay))


def to_fitz(b, H):
    xs, ys = sorted([b["x1"], b["x2"]]), sorted([b["y1"], b["y2"]])
    return fitz.Rect(xs[0], H - ys[1], xs[1], H - ys[0])


def render_page(pm, page_no, boxes, dpi=200):
    """boxes: list of (rect, colour, width, dashes, label)"""
    doc = fitz.open(R/"eval/pdfs"/f"{pm}.pdf")
    pg = doc[page_no - 1]
    for rect, col, w, dash, lab in boxes:
        pg.draw_rect(rect, color=rgb(col), width=w, dashes=dash)
        if lab:
            pg.insert_text(fitz.Point(rect.x0, max(rect.y0 - 5, 10)), lab,
                           fontsize=9, color=rgb(col), fontname="hebo")
    tmp = OUT/f"_tmp_{pm}_{page_no}.png"
    pg.get_pixmap(dpi=dpi, alpha=False).save(tmp)
    doc.close()
    return tmp


# =============================================================== E3 side caption
print("[E3] caption sits to the right of the figure — full page")
pm = "PMC12129649_NEUP-45-177"
PG = 4
L = layout_elements(pm)
H = (L.get("page_dims") or {}).get(str(PG), {}).get("height", 782)
fig_rec = next(f for f in media(V18, pm, "figures")
               if f["page"] == PG and "Figure_2" in Path(f["image_path"]).name)
fr = to_fitz(fig_rec["bbox"], H)
# the caption element that the matcher attached
caps = [e for e in L["elements"] if e.get("type") == "CAPTION" and e.get("page") == PG]
cap = min(caps, key=lambda c: abs(c["bbox"]["x1"] - fig_rec["bbox"]["x2"])) if caps else None
boxes = [(fr, BLUE, 2.4, None, "figure region")]
if cap:
    cr = to_fitz(cap["bbox"], H)
    boxes.append((cr, ORANGE, 2.4, "[6 4] 0", None))
    _lbl = ("caption sits HERE — right of the figure, not below", cr)
doc = fitz.open(R / "eval/pdfs" / f"{pm}.pdf")
pg_ = doc[PG - 1]
for rect, col, w, dash, lab in boxes:
    pg_.draw_rect(rect, color=rgb(col), width=w, dashes=dash)
    if lab:
        pg_.insert_text(fitz.Point(rect.x0, max(rect.y0-5, 10)), lab, fontsize=9,
                        color=rgb(col), fontname="hebo")
if cap:
    _t, cr = _lbl
    pg_.insert_text(fitz.Point(max(cr.x0-6, 6), min(cr.y1+12, pg_.rect.height-6)), _t,
                    fontsize=9, color=rgb(ORANGE), fontname="hebo")
tmp = OUT / f"_tmp_{pm}_{PG}.png"
pg_.get_pixmap(dpi=200, alpha=False).save(tmp)
doc.close()

img = mpimg.imread(tmp)
h, w = img.shape[0], img.shape[1]
fig = plt.figure(figsize=(8.3, 4.0), dpi=180, facecolor="white")
ih = 3.35
iw = ih * w / h
ax = fig.add_axes([0.035, 0.09, iw/8.3, ih/4.0])
ax.imshow(img)
ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_edgecolor("#CCCCCC")
x0 = 0.035 + iw/8.3 + 0.05
fig.text(x0, .90, "Why the default matcher fails here", fontsize=12.5, fontweight="bold",
         color=BLUE, va="top")
fig.text(x0, .82, "Docling's matcher looks below a figure for its\ncaption. On this page the caption "
         "sits in the\nadjacent column, to the right.", fontsize=11, va="top")
fig.text(x0, .62, "Docling baseline", fontsize=11, fontweight="bold", color="#8C8C8C", va="top")
fig.text(x0, .565, "caption:  '(empty)'", fontsize=10.5, family="monospace",
         color="#8C8C8C", va="top")
fig.text(x0, .45, "Proposed pipeline", fontsize=11, fontweight="bold", color=BLUE, va="top")
fig.text(x0, .395, "caption:  'Fig 2 Schematic diagram\n           of iron kinetics in the CNS.'",
         fontsize=10.5, family="monospace", color=BLUE, va="top", fontweight="bold")
fig.text(x0, .245, "The crop is byte-identical on both sides (IoU 1.0)\n— only the attached string changes.\n"
         "5 of the 16 caption fixes are side captions.", fontsize=9.8, color=GREY, va="top")
fig.savefig(OUT / "fm_e3_side_caption_page.png", facecolor="white")
plt.close(fig)
tmp.unlink()
print("   -> fm_e3_side_caption_page.png")


# ========================================================= E14 three FPs, full page
print("[E14] three detector false positives — full pages")
cases = []
# masthead
pm1 = "PMC4945808_dpa-0003-0055"
L1 = layout_elements(pm1)
H1 = (L1.get("page_dims") or {}).get("1", {}).get("height", 782)
mast = [t for t in media(V07, pm1) if t["page"] == 1
        and (H1 - max(t["bbox"]["y1"], t["bbox"]["y2"])) < 60]
cases.append((pm1, 1, [(to_fitz(t["bbox"], H1), RED, 2.2, "[8 3 2 3] 0", None) for t in mast],
              "Running header", "detected as a table on 6 pages\ndropped by drop_tables_in_top_pts"))
# author byline
pm2 = "PMC8420544_CYT-33-93"
L2 = layout_elements(pm2)
H2 = (L2.get("page_dims") or {}).get("1", {}).get("height", 782)
by = [t for t in media(OTS, pm2) if t["page"] == 1]
cases.append((pm2, 1, [(to_fitz(t["bbox"], H2), RED, 2.2, "[8 3 2 3] 0", None) for t in by],
              "Author byline", "rubric: “should be masked, not a\ntable in the scientific sense”"))
# table inside a figure
pm3 = "PMC11791726_HIS-86-485"
L3 = layout_elements(pm3)
H3 = (L3.get("page_dims") or {}).get("9", {}).get("height", 782)
tif = [t for t in media(V01, pm3) if t["page"] == 9]
cases.append((pm3, 9, [(to_fitz(t["bbox"], H3), RED, 2.2, "[8 3 2 3] 0", None) for t in tif],
              "Table inside a figure", "rubric: table_in_figure\ndropped by drop_tables_inside_figures"))

fig = plt.figure(figsize=(10.0, 4.35), dpi=180, facecolor="white")
fig.text(.032, .975, "All three were emitted as tables by a detector — none is a table",
         fontsize=12.5, fontweight="bold", va="top")
fig.text(.032, .922, "shown in page context, because *where* each sits is what makes it wrong",
         fontsize=9.6, color=GREY, va="top")
for i, (pm_, pg_, boxes_, ttl, sub) in enumerate(cases):
    tmp = render_page(pm_, pg_, boxes_)
    img = mpimg.imread(tmp)
    h, w = img.shape[0], img.shape[1]
    ih = 2.85
    iw = ih * w / h
    x0 = .045 + i * .318
    ax = fig.add_axes([x0, .175, iw/10.0, ih/4.35])
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#CCCCCC")
    fig.text(x0, .885, ttl, fontsize=11.5, fontweight="bold", color=RED, va="top")
    fig.text(x0, .135, sub, fontsize=9.3, color=GREY, va="top")
    tmp.unlink()
fig.savefig(OUT / "fm_e14_false_positives_page.png", facecolor="white")
plt.close(fig)
print("   -> fm_e14_false_positives_page.png")


# ================================================== E15 missed table + captions, full page
print("[E15] missed table and unattached captions — full pages")
items = []
# missed table
pmA = "PMC8420544_CYT-33-93"
LA = layout_elements(pmA)
HA = (LA.get("page_dims") or {}).get("3", {}).get("height", 782)
recA = next(t for t in media(V18, pmA) if t["page"] == 3 and "Table_1_p3" in Path(t["image_path"]).name)
items.append((pmA, 3, [(to_fitz(recA["bbox"], HA), GREEN, 2.4, None, None)],
              "Table never emitted", "PMC8420544 p3 · Docling proposed\nnothing here; TATR recovered it"))
# table caption
pmB = "PMC11863984_main"
LB = layout_elements(pmB)
HB = (LB.get("page_dims") or {}).get("3", {}).get("height", 794)
recB = next(t for t in media(V18, pmB) if t["page"] == 3 and "Table_1_p3" in Path(t["image_path"]).name)
capB = [e for e in LB["elements"] if e.get("type") == "CAPTION" and e.get("page") == 3]
bx = [(to_fitz(recB["bbox"], HB), BLUE, 2.4, None, None)]
if capB:
    bx.append((to_fitz(capB[0]["bbox"], HB), ORANGE, 2.2, "[6 4] 0", None))
items.append((pmB, 3, bx, "Table caption not attached",
              "PMC11863984 p3 · caption '' →\n'Table 1 Overall characteristics…'"))
# figure caption
pmC = "PMC10297671_dermatopathology-10-00024"
LC = layout_elements(pmC)
HC = (LC.get("page_dims") or {}).get("2", {}).get("height", 842)
recC = next(f for f in media(V18, pmC, "figures") if f["page"] == 2)
capC = [e for e in LC["elements"] if e.get("type") == "CAPTION" and e.get("page") == 2]
bx = [(to_fitz(recC["bbox"], HC), BLUE, 2.4, None, None)]
if capC:
    bx.append((to_fitz(capC[0]["bbox"], HC), ORANGE, 2.2, "[6 4] 0", None))
items.append((pmC, 2, bx, "Figure caption not attached",
              "PMC10297671 p2 · caption '' →\n'Figure 1. p16 Ink4a staining…'"))

fig = plt.figure(figsize=(10.0, 4.7), dpi=180, facecolor="white")
fig.text(.032, .975, "Three artifacts the baseline failed to produce correctly",
         fontsize=12.5, fontweight="bold", va="top")
fig.text(.032, .922, "blue = the region the pipeline emitted   ·   orange dashed = the caption it "
         "attached   ·   green = recovered by TATR", fontsize=9.6, color=GREY, va="top")
for i, (pm_, pg_, boxes_, ttl, sub) in enumerate(items):
    tmp = render_page(pm_, pg_, boxes_)
    img = mpimg.imread(tmp)
    h, w = img.shape[0], img.shape[1]
    ih = 2.95
    iw = ih * w / h
    x0 = .045 + i * .318
    ax = fig.add_axes([x0, .215, iw/10.0, ih/4.7])
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#CCCCCC")
    fig.text(x0, .885, ttl, fontsize=11.5, fontweight="bold", color=BLUE, va="top")
    fig.text(x0, .175, sub, fontsize=9.3, color=GREY, va="top")
    tmp.unlink()
fig.text(.032, .055, "For the two caption cases the crop is byte-identical on both sides — only the "
         "attached string changes.", fontsize=9.8, color=GREY, va="top")
fig.savefig(OUT / "fm_e15_missed_and_captions_page.png", facecolor="white")
plt.close(fig)
print("   -> fm_e15_missed_and_captions_page.png")
