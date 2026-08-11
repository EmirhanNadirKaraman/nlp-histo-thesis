#!/usr/bin/env python3
"""Render screenshot-ready evidence for every fault mode.

Outputs
-------
presentation/assets/fault_modes/pages/   annotated source-PDF pages, one PNG per
                                         fault mode/page, region outlined with a
                                         role-specific border style + legend
presentation/assets/fault_modes/text/    before/after panels for the two text
                                         fault modes (ghost text, paragraph merging)
presentation/assets/fault_modes/PAGES.md the PDF + page + bbox index

Border styles (distinct on purpose, so a screenshot stays readable):
  grey  dashed    what the Docling baseline emitted
  blue  solid     what the proposed pipeline emitted
  green solid     recovered — the baseline emitted nothing here
  red   dash-dot  false positive suppressed by the pipeline
  purple dotted   ghost-text node removed by the two-pass scorer
"""
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nlp_histo.pipeline.stages.pdf_text_extraction.config import PipelineConfig
from nlp_histo.pipeline.stages.pdf_text_extraction.components.node_scorer import NodeScorer
from nlp_histo.pipeline.stages.pdf_text_extraction.components.evidence_gatherer import (
    PyMuPDFEvidenceGatherer)
from nlp_histo.pipeline.stages.pdf_text_extraction.models.dto import LayoutElement, BoundingBox

R = Path("/Users/emir/Documents/GitHub/nlp-histo")
BASE = R / "presentation/assets/fault_modes"
PAGES = BASE / "pages"
ZOOM = BASE / "zooms"
FULL = BASE / "fullpages"
TEXT = BASE / "text"
for d in (PAGES, ZOOM, FULL, TEXT):
    d.mkdir(parents=True, exist_ok=True)

OTS = "00_docling_offtheshelf"
V01 = "01_docling"
V18 = "18_hybrid_best_family_fixes_footnote_expand_1_2"
DPI = 200

# role -> (rgb 0-1, width pt, dash spec or None, legend label)
STYLE = {
    "baseline":  ((0.42, 0.42, 0.42), 1.6, "[6 4] 0", "Docling baseline emitted this"),
    "pipeline":  ((0.00, 0.32, 0.58), 2.6, None,      "Proposed pipeline emitted this"),
    "recovered": ((0.40, 0.53, 0.00), 2.6, None,      "Recovered — baseline emitted nothing"),
    "suppressed":((0.77, 0.09, 0.11), 2.2, "[8 3 2 3] 0", "False positive suppressed"),
    "ghost":     ((0.61, 0.31, 0.59), 2.2, "[1 3] 0", "Ghost-text node removed (two-pass)"),
    "delta":     ((0.40, 0.53, 0.00), 1.2, None,      "Region the pipeline added"),
    "frag1":     ((0.89, 0.45, 0.13), 2.0, "[6 4] 0", "Raw fragment 1"),
    "frag2":     ((0.00, 0.32, 0.58), 2.0, "[6 4] 0", "Raw fragment 2 (merged into 1)"),
}


def media(variant, pmcid):
    p = R / "out/sweeps" / variant / "json" / f"{pmcid}_media.json"
    if not p.exists():
        return []
    d = json.load(open(p))
    out = []
    for kind in ("tables", "figures"):
        for it in d.get(kind) or []:
            out.append(dict(kind=kind[:-1], page=it.get("page"), bbox=it.get("bbox"),
                            fname=Path(it["image_path"]).name,
                            label=it.get("label"), source=it.get("source")))
    return out


def to_fitz(bbox, page_h):
    """docling bbox (y=0 at bottom, y1=top) -> fitz rect (y=0 at top)."""
    xs = sorted([bbox["x1"], bbox["x2"]])
    ys = sorted([bbox["y1"], bbox["y2"]])
    return fitz.Rect(xs[0], page_h - ys[1], xs[1], page_h - ys[0])


def annotate(out_name, pmcid, page_no, boxes, caption, sub=""):
    """boxes: list of (role, fitz.Rect, tag)"""
    pdf = R / "eval/pdfs" / f"{pmcid}.pdf"
    doc = fitz.open(pdf)
    page = doc[page_no - 1]
    norm = [(b + ("above",))[:4] for b in boxes]
    for role, rect, tag, pos in norm:
        col, w, dash, _ = STYLE[role]
        if role == "delta":
            page.draw_rect(rect, color=col, fill=col, fill_opacity=0.18, width=1.2)
        else:
            page.draw_rect(rect, color=col, width=w, dashes=dash)
        if tag:
            y = max(rect.y0 - 5, 10) if pos == "above" else min(rect.y1 + 11, page.rect.height - 6)
            page.insert_text(fitz.Point(rect.x0, y), tag, fontsize=8.5, color=col, fontname="hebo")
    # clean full page: boxes only, no legend, no caption strip
    page.get_pixmap(dpi=300, alpha=False).save(FULL / f"{out_name}_full.png")

    # legend, only for the roles actually used
    used, seen = [], set()
    for role, _, _, _ in norm:
        if role not in seen:
            seen.add(role)
            used.append(role)
    lx, ly = 14, page.rect.height - 14 - 13 * len(used)
    page.draw_rect(fitz.Rect(lx - 5, ly - 12, lx + 250, ly + 13 * len(used) - 2),
                   color=(0.75, 0.75, 0.75), fill=(1, 1, 1), width=0.7)
    for i, role in enumerate(used):
        col, w, dash, lab = STYLE[role]
        y = ly + 13 * i
        page.draw_line(fitz.Point(lx, y), fitz.Point(lx + 22, y), color=col, width=w, dashes=dash)
        page.insert_text(fitz.Point(lx + 28, y + 3), lab, fontsize=7.5, color=(0.15, 0.15, 0.15))
    pix = page.get_pixmap(dpi=DPI, alpha=False)
    tmp = PAGES / f"_{out_name}.png"
    pix.save(tmp)
    # tight zoom on the union of the annotated regions, for close-up screenshots
    ux0 = min(b[1].x0 for b in norm)
    uy0 = min(b[1].y0 for b in norm)
    ux1 = max(b[1].x1 for b in norm)
    uy1 = max(b[1].y1 for b in norm)
    padx, pady = 26, 30
    clip = fitz.Rect(max(ux0 - padx, 0), max(uy0 - pady, 0),
                     min(ux1 + padx, page.rect.width), min(uy1 + pady, page.rect.height))
    page.get_pixmap(dpi=320, clip=clip, alpha=False).save(ZOOM / f"{out_name}_zoom.png")
    doc.close()

    # add a caption strip above the page image
    import matplotlib.image as mpimg
    img = mpimg.imread(tmp)
    h, w = img.shape[0], img.shape[1]
    fw = 7.2
    fig = plt.figure(figsize=(fw, fw * h / w + 0.62), dpi=150, facecolor="white")
    ax = fig.add_axes([0, 0, 1, h / (h + 0.62 * 150 * w / (fw * 150)) if False else
                       (fw * h / w) / (fw * h / w + 0.62)])
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#CCCCCC")
    fig.text(0.012, 0.982, caption, fontsize=11, fontweight="bold", va="top")
    if sub:
        fig.text(0.012, 0.949, sub, fontsize=8.5, color="#595959", va="top")
    fig.savefig(PAGES / f"{out_name}.png", facecolor="white")
    plt.close(fig)
    tmp.unlink()
    return f"{out_name}.png"


IDX = []


def add(key, title, pmcid, page_no, boxes, sub):
    fn = annotate(f"{key}_p{page_no:02d}", pmcid, page_no, boxes, title, sub)
    IDX.append(dict(key=key, title=title, pdf=f"eval/pdfs/{pmcid}.pdf", page=page_no,
                    png=f"presentation/assets/fault_modes/pages/{fn}",
                    zoom=f"presentation/assets/fault_modes/zooms/{fn[:-4]}_zoom.png",
                    fullpage=f"presentation/assets/fault_modes/fullpages/{fn[:-4]}_full.png",
                    boxes=[(b[0], [round(v, 1) for v in (b[1].x0, b[1].y0, b[1].x1, b[1].y1)], b[2])
                           for b in boxes]))
    print(f"  {fn}")


def find(recs, fname):
    for r in recs:
        if r["fname"] == fname:
            return r
    return None


print("rendering annotated pages")

# ---------------- 01 footnotes ----------------
pm = "PMC11649516_HIS-86-236"
doc = fitz.open(R / "eval/pdfs" / f"{pm}.pdf")
H = doc[4].rect.height
doc.close()
b = find(media(OTS, pm), f"{pm}_Table_1_p5.png")
v = find(media(V18, pm), f"{pm}_Table_1_p5.png")
rb, rv = to_fitz(b["bbox"], H), to_fitz(v["bbox"], H)
band = fitz.Rect(rv.x0, rb.y1, rv.x1, rv.y1)          # the strip the pipeline added
add("01_footnotes", "Fault mode 1 — footnotes cut off below the table crop", pm, 5,
    [("delta", band, "", "above"),
     ("baseline", rb, "baseline crop ends here", "below"),
     ("pipeline", rv, "pipeline crop — footnotes absorbed", "below")],
    "footnote expansion x1.2  ·  rubric 'missing footnotes' -> 'correct'  ·  crop 1320 -> 1468 px")

# ---------------- 02 split table ----------------
pm = "PMC11674653_dermatopathology-11-00035"
doc = fitz.open(R / "eval/pdfs" / f"{pm}.pdf")
H = doc[2].rect.height
doc.close()
for pg, fn in ((3, f"{pm}_Table_1_p3.png"), (4, f"{pm}_Table_1_p4.png")):
    v = find(media(V18, pm), fn)
    add("02_split_table", f"Fault mode 2 — table split across pages 3-4, never emitted (page {pg})",
        pm, pg, [("recovered", to_fitz(v["bbox"], H), "recovered by TATR")],
        "hybrid Docling + TATR  ·  Docling emitted nothing on this page  ·  rubric 'correct'")

# ---------------- 03 missed table ----------------
pm = "PMC8420544_CYT-33-93"
doc = fitz.open(R / "eval/pdfs" / f"{pm}.pdf")
H = doc[2].rect.height
doc.close()
v = find(media(V18, pm), f"{pm}_Table_1_p3.png")
add("03_missed_table", "Fault mode 3 — single table never emitted by the baseline", pm, 3,
    [("recovered", to_fitz(v["bbox"], H), "recovered by TATR")],
    "hybrid Docling + TATR  ·  rubric 'correct'")

# ---------------- 04 table caption ----------------
pm = "PMC11863984_main"
doc = fitz.open(R / "eval/pdfs" / f"{pm}.pdf")
H = doc[2].rect.height
doc.close()
v = find(media(V18, pm), f"{pm}_Table_1_p3.png")
add("04_table_caption", "Fault mode 4 — table caption not attached", pm, 3,
    [("pipeline", to_fitz(v["bbox"], H), "caption now attached")],
    "geometric caption matcher  ·  caption '' -> 'Table 1 Overall characteristics of the patient cohort...'")

# ---------------- 05 spurious table ----------------
pm = "PMC8420544_CYT-33-93"
doc = fitz.open(R / "eval/pdfs" / f"{pm}.pdf")
H = doc[0].rect.height
doc.close()
b = find(media(OTS, pm), f"{pm}_Table_1_p1.png")
add("05_spurious_table", "Fault mode 5 — author byline emitted as a table", pm, 1,
    [("suppressed", to_fitz(b["bbox"], H), "emitted by Docling — suppressed by the pipeline")],
    "region handling  ·  rubric 'should be masked, not a table in the scientific sense'")

# ---------------- 06 table in figure ----------------
pm = "PMC11791726_HIS-86-485"
doc = fitz.open(R / "eval/pdfs" / f"{pm}.pdf")
H = doc[8].rect.height
doc.close()
b = find(media(V01, pm), f"{pm}_Table_3_p9.png")
add("06_table_in_figure", "Fault mode 6 — a table detected inside a figure", pm, 9,
    [("suppressed", to_fitz(b["bbox"], H), "nested table — suppressed")],
    "drop_tables_inside_figures  ·  rubric 'table_in_figure'")

# ---------------- 07 icons ----------------
pm = "PMC10297671_dermatopathology-10-00024"
doc = fitz.open(R / "eval/pdfs" / f"{pm}.pdf")
H = doc[0].rect.height
doc.close()
recs = [r for r in media(OTS, pm) if r["page"] == 1 and r["kind"] == "figure"]
boxes = []
for r in recs:
    rect = to_fitz(r["bbox"], H)
    boxes.append(("suppressed", rect, f"{rect.width:.0f}x{rect.height:.0f} pt"))
add("07_icons", "Fault mode 7 — decorative icons emitted as figures", pm, 1, boxes,
    "min_figure_pts = 50 (width AND height)  ·  all rated 'icon' by the annotator, all dropped")

# ---------------- 08 figure caption ----------------
doc = fitz.open(R / "eval/pdfs" / f"{pm}.pdf")
H = doc[1].rect.height
doc.close()
v = find(media(V18, pm), f"{pm}_Figure_1_p2.png")
add("08_figure_caption", "Fault mode 8 — figure caption not attached", pm, 2,
    [("pipeline", to_fitz(v["bbox"], H), "caption now attached")],
    "custom caption matcher  ·  caption '' -> 'Figure 1. p16 Ink4a staining in the epidermis...'")

# ---------------- 09 side caption ----------------
pm = "PMC12129649_NEUP-45-177"
doc = fitz.open(R / "eval/pdfs" / f"{pm}.pdf")
H = doc[3].rect.height
doc.close()
v = find(media(V18, pm), f"{pm}_Figure_2_p4.png")
add("09_side_caption", "Fault mode 9 — caption sits to the right of the figure", pm, 4,
    [("pipeline", to_fitz(v["bbox"], H), "caption now attached")],
    "custom caption matcher  ·  caption '' -> 'Fig 2 Schematic diagram of iron kinetics in the CNS.'")

# ---------------- 10 ghost text ----------------
# Box ONLY nodes the two-pass scorer actually rejected, tagged with its real
# reason string.  (Diffing pass-1 vs pass-2 layouts would also catch elements
# removed by figure/table region masking, which are not ghost text.)
pm = "PMC10296831_dermatopathology-10-00026"
lay = [p_ for p_ in (R / "out/sweeps" / V18 / "docling_full").glob(f"{pm}_tbl*_layout.json")
       if "twopass" not in p_.name][0]
L1 = json.load(open(lay))
dims = {int(k): v for k, v in (L1.get("page_dims") or {}).items()}
els = [LayoutElement(type=e["type"], page=e["page"],
                     bbox=BoundingBox(page=e["page"], x1=e["bbox"]["x1"], y1=e["bbox"]["y1"],
                                      x2=e["bbox"]["x2"], y2=e["bbox"]["y2"]),
                     text=e.get("text") or "", level=e.get("level"))
       for e in L1["elements"] if e.get("bbox")]
cfg = PipelineConfig()
g = PyMuPDFEvidenceGatherer(R / "eval/pdfs" / f"{pm}.pdf", cfg.two_pass)
ev = [g.gather(e, (dims.get(e.page) or {}).get("height", 842)) for e in els]
g.close()
rejected = [sn for sn in NodeScorer(cfg.two_pass, dims).score_all(els, ev) if not sn.keep]

by_page = {}
for sn in rejected:
    by_page.setdefault(sn.element.page, []).append(sn)
for pg in sorted(by_page, key=lambda k: -len(by_page[k]))[:1]:
    H = dims.get(pg, {}).get("height", 842)
    boxes = []
    for sn in by_page[pg][:6]:
        e = sn.element
        rect = to_fitz(dict(x1=e.bbox.x1, y1=e.bbox.y1, x2=e.bbox.x2, y2=e.bbox.y2), H)
        short = sn.rejection_reason.split(" in header")[0][:44]
        boxes.append(("ghost", rect, short, "above" if len(boxes) % 2 == 0 else "below"))
    add("10_ghost_text",
        f"Fault mode 10 — invisible / hidden text nodes rejected (page {pg})", pm, pg, boxes,
        "two-pass NodeScorer  ·  only nodes the scorer itself rejected are boxed, "
        "tagged with its reason")

# a second document: the pure R3 hidden-duplicate case
pm2 = "PMC11755463_dermatopathology-12-00002"
lay2 = [p_ for p_ in (R / "out/sweeps" / V18 / "docling_full").glob(f"{pm2}_tbl*_layout.json")
        if "twopass" not in p_.name][0]
L2 = json.load(open(lay2))
dims2 = {int(k): v for k, v in (L2.get("page_dims") or {}).items()}
els2 = [LayoutElement(type=e["type"], page=e["page"],
                      bbox=BoundingBox(page=e["page"], x1=e["bbox"]["x1"], y1=e["bbox"]["y1"],
                                       x2=e["bbox"]["x2"], y2=e["bbox"]["y2"]),
                      text=e.get("text") or "", level=e.get("level"))
        for e in L2["elements"] if e.get("bbox")]
g = PyMuPDFEvidenceGatherer(R / "eval/pdfs" / f"{pm2}.pdf", cfg.two_pass)
ev2 = [g.gather(e, (dims2.get(e.page) or {}).get("height", 842)) for e in els2]
g.close()
rej2 = [sn for sn in NodeScorer(cfg.two_pass, dims2).score_all(els2, ev2) if not sn.keep]
bp2 = {}
for sn in rej2:
    bp2.setdefault(sn.element.page, []).append(sn)
for pg in sorted(bp2, key=lambda k: -len(bp2[k]))[:1]:
    H = dims2.get(pg, {}).get("height", 842)
    boxes = []
    for sn in bp2[pg][:8]:
        e = sn.element
        rect = to_fitz(dict(x1=e.bbox.x1, y1=e.bbox.y1, x2=e.bbox.x2, y2=e.bbox.y2), H)
        boxes.append(("ghost", rect, (e.text or "")[:26].strip(),
                      "above" if len(boxes) % 2 == 0 else "below"))
    add("10b_ghost_text_header",
        f"Fault mode 10b — invisible running header read as body text (page {pg})", pm2, pg, boxes,
        "two-pass NodeScorer, rule R1_blank_pixels  ·  the MDPI running header sits in an "
        "invisible layer")

# ---------------- 11 text merge ----------------
def merge_page(pmcid, frag1_tail, frag2_head, title, sub, key_):
    pdf = R / "eval/pdfs" / f"{pmcid}.pdf"
    doc = fitz.open(pdf)
    for i, page in enumerate(doc):
        r1 = page.search_for(frag1_tail)
        r2 = page.search_for(frag2_head)
        if r1 and r2:
            doc.close()
            add(key_, title, pmcid, i + 1,
                [("frag1", r1[0], "raw fragment 1 ends", "above"),
                 ("frag2", r2[0], "raw fragment 2 begins", "below")],
                sub)
            return True
    doc.close()
    return False


merge_page("PMC11791726_HIS-86-485",
           "increases from LNM to ENE to TD", "increasingly complex shapes",
           "Fault mode 11 — paragraph split into two fragments, correctly rejoined",
           "ContextAwareStitcher  ·  trailing connective 'Furthermore,' triggers the join",
           "11_text_merge")
merge_page("PMC12129649_NEUP-45-177",
           "prevent iron-dependent proliferation of", "2025 The Author(s). Neuropathology",
           "Merge ERROR — body text glued to the copyright footer",
           "ContextAwareStitcher  ·  purely syntactic rule; a trailing 'of' is enough. Thesis limitation.",
           "12_text_merge_error")

json.dump(IDX, open(BASE / "pages_index.json", "w"), indent=2)
print(f"\n{len(IDX)} annotated pages -> {PAGES}")
