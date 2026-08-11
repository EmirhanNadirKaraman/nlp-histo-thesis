#!/usr/bin/env python3
"""Before/after text panels for the two fault modes that produce no crop:
invisible / hidden text nodes, and paragraph fragmentation."""
import json
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nlp_histo.pipeline.stages.pdf_text_extraction.config import PipelineConfig
from nlp_histo.pipeline.stages.pdf_text_extraction.components.node_scorer import NodeScorer
from nlp_histo.pipeline.stages.pdf_text_extraction.components.evidence_gatherer import (
    PyMuPDFEvidenceGatherer)
from nlp_histo.pipeline.stages.pdf_text_extraction.models.dto import LayoutElement, BoundingBox

R = Path("/Users/emir/Documents/GitHub/nlp-histo")
OUT = R / "presentation/assets/fault_modes/text"
OUT.mkdir(parents=True, exist_ok=True)
V18 = "18_hybrid_best_family_fixes_footnote_expand_1_2"
BLUE, ORANGE, PURPLE, GREY = "#005293", "#E37222", "#9B4F96", "#595959"
MONO = {"family": "monospace", "fontsize": 8.4}


def score(pmcid):
    lay = [p for p in (R / "out/sweeps" / V18 / "docling_full").glob(f"{pmcid}_tbl*_layout.json")
           if "twopass" not in p.name][0]
    L = json.load(open(lay))
    dims = {int(k): v for k, v in (L.get("page_dims") or {}).items()}
    els = [LayoutElement(type=e["type"], page=e["page"],
                         bbox=BoundingBox(page=e["page"], x1=e["bbox"]["x1"], y1=e["bbox"]["y1"],
                                          x2=e["bbox"]["x2"], y2=e["bbox"]["y2"]),
                         text=e.get("text") or "", level=e.get("level"))
           for e in L["elements"] if e.get("bbox")]
    cfg = PipelineConfig()
    g = PyMuPDFEvidenceGatherer(R / "eval/pdfs" / f"{pmcid}.pdf", cfg.two_pass)
    ev = [g.gather(e, (dims.get(e.page) or {}).get("height", 842)) for e in els]
    g.close()
    return NodeScorer(cfg.two_pass, dims).score_all(els, ev)


def panel(fname, title, sub, left_title, left_lines, right_title, right_lines, note=""):
    fig = plt.figure(figsize=(11.0, 5.4), dpi=170, facecolor="white")
    fig.text(.012, .975, title, fontsize=14, fontweight="bold", va="top")
    fig.text(.012, .922, sub, fontsize=9.5, color=GREY, va="top")
    for x, ttl, lines, col in ((.012, left_title, left_lines, "#8C8C8C"),
                               (.515, right_title, right_lines, BLUE)):
        fig.text(x, .862, ttl, fontsize=11, fontweight="bold", color=col, va="top")
        y = .805
        for txt, style in lines:
            c = {"ghost": PURPLE, "keep": "#222222", "join": BLUE,
                 "err": ORANGE, "dim": "#AAAAAA"}[style]
            wrapped = textwrap.wrap(txt, 62) or [""]
            for i, ln in enumerate(wrapped):
                fig.text(x, y, ln, color=c, va="top",
                         fontweight="bold" if style in ("ghost", "join", "err") else "normal",
                         **MONO)
                y -= .036
            y -= .014
    if note:
        fig.text(.012, .045, note, fontsize=9, color=ORANGE, va="top")
    fig.savefig(OUT / fname, facecolor="white")
    plt.close(fig)
    print(f"  {fname}")


print("rendering text panels")

# ---------------------------------------------------------------- ghost text
pm = "PMC11755463_dermatopathology-12-00002"
scored = score(pm)
pg = 2
rej = [s for s in scored if not s.keep and s.element.page == pg]
kept = [s for s in scored if s.keep and s.element.page == pg]
left, right = [], []
for s in rej[:6]:
    left.append((f"[{s.element.type}] {(s.element.text or '').strip()[:52]!r}", "ghost"))
for s in kept[:4]:
    t = (s.element.text or "").strip()
    if t:
        left.append((f"[{s.element.type}] {t[:52]!r}", "keep"))
        right.append((f"[{s.element.type}] {t[:52]!r}", "keep"))
right.insert(0, (f"({len(rej)} invisible nodes redacted before pass 2)", "dim"))
panel("10_ghost_text_panel.png",
      "Fault mode 10 — invisible text nodes removed before knowledge extraction",
      f"{pm}, page {pg}  ·  two-pass NodeScorer, rule R1_blank_pixels  ·  "
      f"{len(rej)} of {len([s for s in scored if s.element.page == pg])} nodes on this page rejected",
      "Pass 1 — Docling on the original PDF",
      left,
      "Pass 2 — Docling on the redacted PDF",
      right,
      "The rejected strings are MDPI's pre-publication running header "
      "(\"... FOR PEER REVIEW\"), which sits in an invisible layer. Docling reads it as body text.")

# ------------------------------------------------------- hidden duplicate (R3)
pm2 = "PMC10296831_dermatopathology-10-00026"
s2 = score(pm2)
r3 = [s for s in s2 if not s.keep and "hidden text layer" in (s.rejection_reason or "")]
left2 = [(f"{(s.element.text or '').strip()[:150]}", "ghost") for s in r3[:2]]
left2 += [("...also extracted a second time as normal body text", "dim")]
right2 = [("(both hidden copies removed — the paragraph is extracted once)", "dim")]
panel("10b_hidden_duplicate_panel.png",
      "Fault mode 10b — a hidden duplicate of real body text",
      f"{pm2}  ·  two-pass NodeScorer, rule R3_dense_text  ·  "
      f"reason: {r3[0].rejection_reason if r3 else ''}",
      "Pass 1 — hidden duplicate present", left2,
      "Pass 2 — duplicate removed", right2,
      "Without this filter the paragraph reaches MAP twice, producing duplicate findings "
      "from a single source sentence.")

# ------------------------------------------------------------- text merging
CONNECT = re.compile(r"(-|,|and|or|of|the|with|in|to|et al\.|Fig\.|that|which|was|were)\s*$", re.I)


def merge_case(pmcid, tail_hint):
    raw = (R / "out/text_raw" / f"{pmcid}_raw.txt").read_text(encoding="utf-8", errors="replace")
    rl = [ln for ln in raw.splitlines() if ln.strip()]
    for i in range(len(rl) - 1):
        a = re.sub(r"^\[\w+\]\s*", "", rl[i]).strip()
        b = re.sub(r"^\[\w+\]\s*", "", rl[i + 1]).strip()
        if tail_hint in a:
            return a, b
    return None, None


a, b = merge_case("PMC11791726_HIS-86-485", "increases from LNM to ENE to TD")
panel("11_text_merge_panel.png",
      "Fault mode 11 — one paragraph split into two layout fragments",
      "PMC11791726_HIS-86-485  ·  ContextAwareStitcher  ·  a trailing connective "
      "(\"Furthermore,\") triggers the join",
      "out/text_raw — pre-assembly fragments",
      [("fragment 1:", "dim"), (a[-190:], "keep"), ("fragment 2:", "dim"), (b[:190], "keep")],
      "out/text — assembled",
      [("one continuous paragraph:", "dim"),
       ((a[-95:] + " " + b[:110]), "join")],
      "Also applied at assembly: bracketed citation markers stripped (14 -> 0 on "
      "PMC5803687), double spaces collapsed (200 -> 4).")

a2, b2 = merge_case("PMC12129649_NEUP-45-177", "prevent iron-dependent proliferation of")
panel("12_text_merge_error_panel.png",
      "Merge ERROR — body text glued to the journal copyright footer",
      "PMC12129649_NEUP-45-177  ·  ContextAwareStitcher  ·  the rule is purely syntactic, "
      "so a trailing \"of\" is enough",
      "out/text_raw — pre-assembly fragments",
      [("fragment 1 (body text):", "dim"), (a2[-190:], "keep"),
       ("fragment 2 (copyright footer):", "dim"), (b2[:190], "err")],
      "out/text — assembled (wrong)",
      [("body text and footer merged:", "dim"), ((a2[-85:] + " " + b2[:110]), "err")],
      "This is the thesis's own stated limitation: paragraph-merging errors occur "
      "\"without leaving a signal in the rubric-based evaluation\".")
