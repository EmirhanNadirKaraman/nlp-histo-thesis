"""
eval/precision_recall.py — Precision, recall and F1 for figure and table detection.

  TP  = correctly detected crop (label starts with "correct", or "missing footnotes" for tables)
  FP  = wrongly detected crop   (label is "incorrect", "icon", "crop is too big", "wrong caption", etc.)
  FN  = real figures/tables the pipeline missed (from ground_truth.csv)

For docling / docling_recon table variants, FN is derived from the full pipeline:
  total_actual_tables = TP_full + FN_full
  FN_variant          = max(0, total_actual_tables − TP_variant)

Usage:
    python eval/precision_recall.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

HERE     = Path(__file__).parent
ANN_DIR  = HERE / "annotations"
CSV_PATH = HERE / "ground_truth.csv"

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"


# ── Label classification ───────────────────────────────────────────────────────

def classify_figure(label: str) -> str:
    """'tp' | 'fp' | 'skip'"""
    lbl = label.lower().strip()
    if lbl.startswith("correct"):
        return "tp"
    if lbl in ("icon", "incorrect"):
        return "fp"
    return "skip"


def classify_table(label: str) -> str:
    """'tp' | 'fp' | 'skip'"""
    lbl = label.lower().strip()
    if lbl == "correct" or lbl.startswith("missing footnotes"):
        return "tp"
    if (lbl == "incorrect"
            or lbl.startswith("wrong caption")
            or lbl.startswith("crop is too big")
            or lbl.startswith("weird")):
        return "fp"
    return "skip"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_ann(mode: str) -> dict[str, str]:
    p = ANN_DIR / f"annotations_{mode}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_ground_truth() -> dict[str, dict[str, int]]:
    records: dict[str, dict[str, int]] = {}
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records[row["pmcid"]] = {
                "missed_figures": int(row.get("missed_figures") or 0),
                "missed_tables":  int(row.get("missed_tables")  or 0),
                "total_tables":   int(row.get("total_tables")   or 0),
            }
    return records


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else None
    r = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * p * r / (p + r)) if (p and r) else None
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f1)


def pct(v) -> str:
    return f"{v:.1%}" if v is not None else "  N/A "


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    gt = load_ground_truth()

    # ── Figures ───────────────────────────────────────────────────────────────
    fig_ann  = load_ann("json_figures")
    fig_tp   = sum(1 for v in fig_ann.values() if classify_figure(v) == "tp")
    fig_fp   = sum(1 for v in fig_ann.values() if classify_figure(v) == "fp")
    fig_fn   = sum(r["missed_figures"] for r in gt.values())
    fig_skip = sum(1 for v in fig_ann.values() if classify_figure(v) == "skip")
    fig_m    = metrics(fig_tp, fig_fp, fig_fn)

    # ── Tables ────────────────────────────────────────────────────────────────
    table_modes = {
        "full":          "json_tables_full",
        "docling":       "json_tables_docling",
        "docling_recon": "json_tables_docling_recon",
    }

    total_actual_tables = sum(r["total_tables"] for r in gt.values())
    table_results: dict[str, dict] = {}

    for variant, mode in table_modes.items():
        ann  = load_ann(mode)
        tp   = sum(1 for v in ann.values() if classify_table(v) == "tp")
        fp   = sum(1 for v in ann.values() if classify_table(v) == "fp")
        skip = sum(1 for v in ann.values() if classify_table(v) == "skip")
        fn   = max(0, total_actual_tables - tp)

        table_results[variant] = metrics(tp, fp, fn) | {"skip": skip}

    # ── Output ────────────────────────────────────────────────────────────────
    W = 68
    print(f"\n{BOLD}{'─' * W}{RESET}")
    print(f"  {'':22} {'Precision':>10} {'Recall':>10} {'F1':>8}   TP / FP / FN")
    print(f"{BOLD}{'─' * W}{RESET}")

    row = fig_m
    skip_note = f"  ({fig_skip} skipped)" if fig_skip else ""
    print(f"  {CYAN}{'Figures':<22}{RESET}"
          f" {pct(row['precision']):>10} {pct(row['recall']):>10} {pct(row['f1']):>8}"
          f"   {row['tp']} / {row['fp']} / {row['fn']}{skip_note}")

    print(f"{'─' * W}")

    labels = {"full": "Tables (hybrid/full)", "docling": "Tables (docling)", "docling_recon": "Tables (docling+recon)"}
    for variant, row in table_results.items():
        skip_note = f"  ({row['skip']} skipped)" if row["skip"] else ""
        print(f"  {CYAN}{labels[variant]:<22}{RESET}"
              f" {pct(row['precision']):>10} {pct(row['recall']):>10} {pct(row['f1']):>8}"
              f"   {row['tp']} / {row['fp']} / {row['fn']}{skip_note}")

    print(f"{BOLD}{'─' * W}{RESET}")
    print(f"  {DIM}total actual tables: {total_actual_tables} (from ground_truth.csv){RESET}")

    # ── Label breakdowns ──────────────────────────────────────────────────────
    sections = [
        ("Figures",               fig_ann,            classify_figure),
        ("Tables — hybrid/full",  load_ann("json_tables_full"),          classify_table),
        ("Tables — docling",      load_ann("json_tables_docling"),        classify_table),
        ("Tables — docling+recon",load_ann("json_tables_docling_recon"),  classify_table),
    ]
    for title, ann, clf in sections:
        counts: dict[str, int] = defaultdict(int)
        for v in ann.values():
            counts[v] += 1
        print(f"\n  {BOLD}{title}{RESET}")
        for label, n in sorted(counts.items(), key=lambda x: -x[1]):
            cls   = clf(label)
            color = GREEN if cls == "tp" else (RED if cls == "fp" else DIM)
            tag   = f"[{cls}]"
            print(f"    {color}{tag:6}{RESET}  {n:3}×  {label}")

    print()


if __name__ == "__main__":
    main()
