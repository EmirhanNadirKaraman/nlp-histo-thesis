#!/usr/bin/env python3
"""Verify the on-disk path of every fault-mode example, then render side-by-side
gallery PNGs into presentation/assets/fault_modes/."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

R = Path("/Users/emir/Documents/GitHub/nlp-histo")
OUT = R / "presentation/assets/fault_modes"
OUT.mkdir(parents=True, exist_ok=True)
OTS = "00_docling_offtheshelf"
V01 = "01_docling"
V18 = "18_hybrid_best_family_fixes_footnote_expand_1_2"
BLUE, ORANGE, GREY = "#005293", "#E37222", "#595959"


def crop(variant, fname):
    for sub in ("tables", "figures"):
        p = R / "out/sweeps" / variant / sub / fname
        if p.exists():
            return p
    return None


# fault mode -> (baseline variant, baseline file | None, v18 files, note)
CASES = [
    ("01_footnotes", "Footnotes cut off below the crop", "footnote expansion x1.2",
     OTS, ["PMC11649516_HIS-86-236_Table_1_p5.png"],
     ["PMC11649516_HIS-86-236_Table_1_p5.png"], "missing footnotes -> correct"),
    ("02_split_table", "Table split across a page boundary", "hybrid Docling + TATR",
     OTS, [], ["PMC11674653_dermatopathology-11-00035_Table_1_p3.png",
               "PMC11674653_dermatopathology-11-00035_Table_1_p4.png"],
     "not emitted -> correct x2"),
    ("03_missed_table", "Single table never emitted", "hybrid Docling + TATR",
     OTS, [], ["PMC8420544_CYT-33-93_Table_1_p3.png"], "not emitted -> correct"),
    ("04_table_caption", "Table caption not attached", "geometric caption matcher",
     OTS, ["PMC11863984_main_Table_1_p3.png"], ["PMC11863984_main_Table_1_p3.png"],
     "no caption -> correct"),
    ("05_spurious_table", "Non-table region emitted as a table", "region handling",
     OTS, ["PMC8420544_CYT-33-93_Table_1_p1.png"], [],
     "'should be masked, not a table' -> removed"),
    ("06_table_in_figure", "Table detected inside a figure", "drop_tables_inside_figures",
     V01, ["PMC11791726_HIS-86-485_Table_3_p9.png"], [], "table_in_figure -> removed"),
    ("07_icons", "Decorative icons emitted as figures", "min_figure_pts = 50",
     OTS, ["PMC10297671_dermatopathology-10-00024_Figure_1_p1.png",
           "PMC10297671_dermatopathology-10-00024_Figure_2_p1.png",
           "PMC10297671_dermatopathology-10-00024_Figure_3_p1.png",
           "PMC10297671_dermatopathology-10-00024_Figure_4_p1.png",
           "PMC10297671_dermatopathology-10-00024_Figure_5_p1.png"], [],
     "icon x5 -> all removed"),
    ("08_figure_caption", "Figure caption not attached", "custom caption matcher",
     OTS, ["PMC10297671_dermatopathology-10-00024_Figure_7_p2.png"],
     ["PMC10297671_dermatopathology-10-00024_Figure_1_p2.png"],
     "'correct figure, no caption' -> correct"),
    ("09_side_caption", "Caption sits to the right of the figure", "custom caption matcher",
     OTS, ["PMC12129649_NEUP-45-177_Figure_3_p4.png"],
     ["PMC12129649_NEUP-45-177_Figure_2_p4.png"],
     "'caption is to the right' -> correct"),
]

print("=" * 104)
print("PATH MANIFEST — every file verified to exist on disk")
print("=" * 104)
manifest = []
for key, title, mech, bvar, bfiles, vfiles, note in CASES:
    print(f"\n[{key}]  {title}")
    print(f"    mechanism : {mech}")
    print(f"    rubric    : {note}")
    rec = dict(key=key, title=title, mechanism=mech, note=note, baseline=[], pipeline=[])
    if bfiles:
        for f in bfiles:
            p = crop(bvar, f)
            print(f"    BASELINE  : {'OK ' if p else 'MISSING'} {p or (bvar + '/' + f)}")
            if p:
                rec["baseline"].append(str(p))
    else:
        print("    BASELINE  : (nothing emitted — that is the fault)")
    if vfiles:
        for f in vfiles:
            p = crop(V18, f)
            print(f"    PIPELINE  : {'OK ' if p else 'MISSING'} {p or (V18 + '/' + f)}")
            if p:
                rec["pipeline"].append(str(p))
    else:
        print("    PIPELINE  : (nothing emitted — the false positive is gone)")
    manifest.append(rec)

# ---------------------------------------------------------------- text cases
print("\n[10_ghost_text]  Invisible / duplicated text layers")
print("    mechanism : two-pass NodeScorer (R1_blank_pixels, R3_dense_text)")
for p in [R/"out/sweeps"/V18/"docling_full"/"PMC10296831_dermatopathology-10-00026_tbl1_ocr0_fp0_engna_sc2.0_layout.json",
          R/"out/sweeps"/V18/"docling_full"/"PMC10296831_dermatopathology-10-00026_twopass_masked_tbl1_ocr0_fp0_engna_sc2.0_layout.json",
          R/"eval/pdfs/PMC10296831_dermatopathology-10-00026.pdf",
          R/"out/run_metadata/PMC10296831_dermatopathology-10-00026_stats.json"]:
    print(f"    {'OK ' if p.exists() else 'MISSING'} {p}")

print("\n[11_text_merge]  Paragraph fragmentation + citation markers")
for p in [R/"out/text_raw/PMC10092270_HIS-82-324_raw.txt",
          R/"out/text/PMC10092270_HIS-82-324_text.txt",
          R/"out/text_raw/PMC5803687_dpa-0004-0013_raw.txt",
          R/"out/text/PMC5803687_dpa-0004-0013_text.txt"]:
    print(f"    {'OK ' if p.exists() else 'MISSING'} {p}")

json.dump(manifest, open(OUT / "manifest.json", "w"), indent=2)

# ================================================================ rendering
def panel(ax, img_path, border, label, sub=None):
    if img_path is None:
        ax.set_facecolor("#F4F4F4")
        for s in ax.spines.values():
            s.set_edgecolor("#C0C0C0")
            s.set_linewidth(1.4)
            s.set_linestyle((0, (5, 4)))
        ax.text(.5, .56, "No crop emitted", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#8C8C8C", fontweight="bold")
        if sub:
            ax.text(.5, .38, sub, ha="center", va="center", transform=ax.transAxes,
                    fontsize=9.5, color="#ABABAB")
    else:
        ax.imshow(mpimg.imread(img_path))
        for s in ax.spines.values():
            s.set_edgecolor(border)
            s.set_linewidth(1.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label, fontsize=11, color=border, pad=5)


def two_panel(key, title, mech, left, right, lnote, rnote, note):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.0), dpi=170)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=.975)
    fig.text(.5, .905, f"{mech}   ·   rubric: {note}", ha="center", fontsize=10, color=GREY)
    panel(axes[0], left, "#B0B0B0", f"Docling baseline\n{lnote}")
    panel(axes[1], right, BLUE, f"Proposed pipeline\n{rnote}")
    fig.tight_layout(rect=[0, 0, 1, .875])
    fig.savefig(OUT / f"{key}.png", facecolor="white")
    plt.close(fig)


print("\n" + "=" * 104)
print("RENDERING gallery images")
print("=" * 104)

# 1 footnote pair
b = crop(OTS, "PMC11649516_HIS-86-236_Table_1_p5.png")
v = crop(V18, "PMC11649516_HIS-86-236_Table_1_p5.png")
two_panel("01_footnotes", "Fault mode: footnotes cut off below the table crop",
          "footnote expansion ×1.2", b, v,
          "1349 × 1320 px — footnote band missing", "1349 × 1468 px (+11%) — footnotes restored",
          "'missing footnotes' → 'correct'   ·   PMC11649516_HIS-86-236 p5")

# 2 split table
fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.8), dpi=170)
fig.suptitle("Fault mode: table split across a page boundary, never emitted",
             fontsize=14, fontweight="bold", y=.975)
fig.text(.5, .905, "hybrid Docling + TATR   ·   rubric: not emitted → 'correct' ×2   ·   "
         "PMC11674653 p3+p4", ha="center", fontsize=10, color=GREY)
panel(axes[0], None, "#B0B0B0", "Docling baseline\npage 3 — nothing", "region not recovered")
panel(axes[1], crop(V18, "PMC11674653_dermatopathology-11-00035_Table_1_p3.png"), BLUE,
      "Proposed pipeline\npage 3 — 'Table 1'")
panel(axes[2], crop(V18, "PMC11674653_dermatopathology-11-00035_Table_1_p4.png"), BLUE,
      "Proposed pipeline\npage 4 — 'Table 1. Cont.'")
fig.tight_layout(rect=[0, 0, 1, .875])
fig.savefig(OUT / "02_split_table.png", facecolor="white")
plt.close(fig)

# 3 missed single table
two_panel("03_missed_table", "Fault mode: single table never emitted",
          "hybrid Docling + TATR", None,
          crop(V18, "PMC8420544_CYT-33-93_Table_1_p3.png"),
          "nothing emitted", "emitted, rated 'correct'",
          "not emitted → 'correct'   ·   PMC8420544_CYT-33-93 p3")

# 5 spurious table
two_panel("05_spurious_table", "Fault mode: non-table region emitted as a table",
          "region handling", crop(OTS, "PMC8420544_CYT-33-93_Table_1_p1.png"), None,
          "emitted — 'should be masked,\nnot a table in the scientific sense'",
          "correctly suppressed",
          "false positive → removed   ·   PMC8420544_CYT-33-93 p1")

# 6 table in figure
two_panel("06_table_in_figure", "Fault mode: a table detected inside a figure",
          "drop_tables_inside_figures", crop(V01, "PMC11791726_HIS-86-485_Table_3_p9.png"), None,
          "emitted — rubric 'table_in_figure'", "correctly suppressed",
          "false positive → removed   ·   PMC11791726_HIS-86-485 p9")

# 4 table caption
two_panel("04_table_caption", "Fault mode: table caption not attached",
          "geometric caption matcher",
          crop(OTS, "PMC11863984_main_Table_1_p3.png"),
          crop(V18, "PMC11863984_main_Table_1_p3.png"),
          "caption: '' (empty)", "caption: 'Table 1 Overall characteristics\nof the patient cohort...'",
          "'no caption' → 'correct'   ·   PMC11863984_main p3")

# 7 icon strip
icons = [("PMC10297671_dermatopathology-10-00024_Figure_2_p1.png", "14 × 10 pt"),
         ("PMC10297671_dermatopathology-10-00024_Figure_3_p1.png", "22 × 11 pt"),
         ("PMC10297671_dermatopathology-10-00024_Figure_4_p1.png", "53 × 18 pt"),
         ("PMC10297671_dermatopathology-10-00024_Figure_5_p1.png", "59 × 22 pt"),
         ("PMC10297671_dermatopathology-10-00024_Figure_1_p1.png", "178 × 36 pt")]
fig, axes = plt.subplots(1, 5, figsize=(11.0, 2.7), dpi=170)
fig.suptitle("Fault mode: decorative icons emitted as figures", fontsize=14,
             fontweight="bold", y=.99)
fig.text(.5, .875, "min_figure_pts = 50 (width AND height)   ·   all five rated 'icon' by the "
         "annotator, all dropped   ·   PMC10297671 p1", ha="center", fontsize=9.5, color=GREY)
for ax, (fn, size) in zip(axes, icons):
    p = crop(OTS, fn)
    if p:
        ax.imshow(mpimg.imread(p))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(ORANGE)
        s.set_linewidth(1.6)
    ax.set_title(size, fontsize=10, color=ORANGE, pad=4)
fig.tight_layout(rect=[0, 0, 1, .84])
fig.savefig(OUT / "07_icons.png", facecolor="white")
plt.close(fig)


# 8 / 9 caption fixes — same pixels, caption string differs
def caps(variant, pmcid):
    d = json.load(open(R / "out/sweeps" / variant / "json" / f"{pmcid}_media.json"))
    out = {}
    for k in ("figures", "tables"):
        for it in d.get(k) or []:
            out[Path(it["image_path"]).name] = (it.get("caption") or "").strip()
    return out

for key, pmcid, bf, vf, ttl in [
    ("08_figure_caption", "PMC10297671_dermatopathology-10-00024",
     "PMC10297671_dermatopathology-10-00024_Figure_7_p2.png",
     "PMC10297671_dermatopathology-10-00024_Figure_1_p2.png",
     "Fault mode: figure caption not attached"),
    ("09_side_caption", "PMC12129649_NEUP-45-177",
     "PMC12129649_NEUP-45-177_Figure_3_p4.png",
     "PMC12129649_NEUP-45-177_Figure_2_p4.png",
     "Fault mode: caption sits to the right of the figure")]:
    cb, cv = caps(OTS, pmcid), caps(V18, pmcid)
    fig = plt.figure(figsize=(9.6, 5.2), dpi=170)
    fig.suptitle(ttl, fontsize=14, fontweight="bold", y=.972)
    fig.text(.5, .905, f"custom caption matcher   ·   identical pixels (IoU 1.0), the caption "
             f"string changes   ·   {pmcid} ", ha="center", fontsize=9.5, color=GREY)
    ax = fig.add_axes([0.06, 0.30, 0.40, 0.54])
    panel(ax, crop(OTS, bf), "#B0B0B0", "Docling baseline")
    ax2 = fig.add_axes([0.54, 0.30, 0.40, 0.54])
    panel(ax2, crop(V18, vf), BLUE, "Proposed pipeline")
    bt = cb.get(bf, "") or "(empty)"
    vt = cv.get(vf, "") or "(empty)"
    fig.text(.06, .22, f"caption: {bt[:70]!r}", fontsize=10, color="#8C8C8C")
    fig.text(.06, .13, f"caption: {vt[:88]!r}", fontsize=10, color=BLUE, fontweight="bold")
    fig.savefig(OUT / f"{key}.png", facecolor="white")
    plt.close(fig)

for f in sorted(OUT.glob("*.png")):
    print(f"   {f.name:26s} {f.stat().st_size:>8,} B")
print(f"\n   manifest: {OUT/'manifest.json'}")
