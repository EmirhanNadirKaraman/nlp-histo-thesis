#!/usr/bin/env python3
"""
score_pdf_variants.py — per-variant precision / recall / F1 for PDF
extraction sweeps.

Reads, for each sweep variant under ``out/sweeps/<variant>/``:

1. The variant's emitted crop set (from each ``*_media.json`` under
   ``out/sweeps/<variant>/json/``).
2. The variant's per-variant labels (from
   ``eval/annotations/<variant>/<mode>.json`` if it exists, else falls
   back to the legacy shared file ``eval/annotations/annotations_<mode>.json``).
3. Ground-truth recall counts (``eval/ground_truth.csv``).

Emits one Markdown row per variant:

    | variant | kind | P | R | F1 | TP | FP | FN | labelled | emitted |

Plus a machine-readable JSON next to it.

Label classification borrows from ``eval/precision_recall.py``:
* TP: label starts with "correct", or "missing footnotes" (tables only).
* FP: label is "incorrect", "icon", "wrong caption", "crop is too big", "weird…".
* Anything else: skipped (not counted toward TP or FP).

Usage::

    python scripts/eval/score_pdf_variants.py
    python scripts/eval/score_pdf_variants.py --md-out reports/variants_PR.md
    python scripts/eval/score_pdf_variants.py --json-out reports/variants_PR.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SWEEPS_ROOT = _REPO_ROOT / "out" / "sweeps"
_ANN_ROOT = _REPO_ROOT / "eval" / "annotations"
_GT_CSV = _REPO_ROOT / "eval" / "ground_truth.csv"


# ── Label classification (kept in sync with eval/precision_recall.py) ─────────


def classify_figure(label: str) -> str:
    lbl = label.lower().strip()
    if lbl.startswith("correct"):
        return "tp"
    if lbl in ("icon", "incorrect"):
        return "fp"
    return "skip"


def classify_table(label: str) -> str:
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


def load_emitted_crops(sweep_dir: Path) -> Tuple[set[str], set[str]]:
    """Return (figures, tables) crop filename sets for one sweep variant."""
    figures: set[str] = set()
    tables: set[str] = set()
    for media in (sweep_dir / "json").rglob("*_media.json"):
        try:
            data = json.loads(media.read_text())
        except Exception:
            continue
        for m in data.get("figures", []) or []:
            if m.get("image_path"):
                figures.add(Path(m["image_path"]).name)
        for m in data.get("tables", []) or []:
            if m.get("image_path"):
                tables.add(Path(m["image_path"]).name)
    return figures, tables


def load_labels(mode: str, variant: str, fallback_legacy: bool = True) -> Dict[str, str]:
    """Load label dict for a variant, falling back to the legacy shared file.

    Returns ``{}`` if neither path exists.
    """
    per_variant = _ANN_ROOT / variant / f"{mode}.json"
    if per_variant.exists():
        try:
            return json.loads(per_variant.read_text())
        except json.JSONDecodeError:
            pass
    if fallback_legacy:
        legacy = _ANN_ROOT / f"annotations_{mode}.json"
        if legacy.exists():
            try:
                return json.loads(legacy.read_text())
            except json.JSONDecodeError:
                pass
    return {}


def load_ground_truth() -> Dict[str, Dict[str, int]]:
    """Per-pmcid ground-truth counts.  Returns {} if the CSV is absent."""
    out: Dict[str, Dict[str, int]] = {}
    if not _GT_CSV.exists():
        return out
    with _GT_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["pmcid"]] = {
                "missed_figures": int(row.get("missed_figures") or 0),
                "missed_tables":  int(row.get("missed_tables")  or 0),
                "total_tables":   int(row.get("total_tables")   or 0),
            }
    return out


# ── Detector mode resolution ──────────────────────────────────────────────────


# Map sweep directory name → (table_mode, figure_mode) for label lookup.
# Sweeps using DOCLING-only get the docling table label file; everything
# else (HYBRID, TATR-only, threshold variants, no-two-pass) reads the
# "full" hybrid label set.  Figures share one label file across variants.
def detector_modes(variant_dir: Path) -> Tuple[str, str]:
    # Read the variant's manifest to discover its detector choice.
    manifests = sorted((variant_dir / "run_metadata").glob("run_*.json"))
    detector = "HYBRID"
    if manifests:
        try:
            data = json.loads(manifests[-1].read_text())
            detector = data.get("config", {}).get("table_detector", "HYBRID")
        except Exception:
            pass
    table_mode = "json_tables_docling" if detector.upper() == "DOCLING" else "json_tables_full"
    return table_mode, "json_figures"


# ── Metrics ───────────────────────────────────────────────────────────────────


def metrics(tp: int, fp: int, fn: int) -> Dict[str, Optional[float]]:
    p = tp / (tp + fp) if (tp + fp) else None
    r = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * p * r / (p + r)) if (p and r) else None
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f1}


def score_kind(
    emitted: set[str],
    labels: Dict[str, str],
    classify,
    *,
    total_actual: Optional[int] = None,
    fn_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute TP/FP/FN/P/R/F1 against the labels for crops the variant emitted.

    Recall-denominator semantics (choose ONE):

    * ``total_actual`` — total real items that exist (TP + FN ground truth);
      FN_variant = max(0, total_actual - TP_variant).  Use for tables, where
      ``ground_truth.csv.total_tables`` provides the corpus-wide total.
    * ``fn_count`` — explicit per-corpus FN (e.g. ``missed_figures`` summed
      across PDFs).  Use for figures.

    When neither is supplied recall + F1 are reported as ``None`` (no GT).
    """
    tp = fp = skip = unlabelled = 0
    for crop in emitted:
        if crop not in labels:
            unlabelled += 1
            continue
        v = classify(labels[crop])
        if v == "tp":
            tp += 1
        elif v == "fp":
            fp += 1
        else:
            skip += 1
    if fn_count is not None:
        fn = fn_count
        m = metrics(tp, fp, fn)
    elif total_actual is not None:
        fn = max(0, total_actual - tp)
        m = metrics(tp, fp, fn)
    else:
        fn = 0
        m = metrics(tp, fp, fn)
        m["recall"] = None
        m["f1"] = None
    m["fn"] = fn
    m["skipped"] = skip
    m["labelled"] = tp + fp + skip
    m["unlabelled"] = unlabelled
    m["emitted"] = len(emitted)
    return m


# ── Rendering ─────────────────────────────────────────────────────────────────


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.1%}" if abs(v) <= 1 else f"{v:.3f}"
    return str(v)


def render_markdown(rows: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    out.append("# Per-variant PDF-extraction P/R/F1")
    out.append("")
    out.append("Computed by `scripts/eval/score_pdf_variants.py`.")
    out.append("")
    out.append("Source labels are read from `eval/annotations/<variant>/<mode>.json` "
               "if a per-variant file exists, else from the legacy "
               "`annotations_<mode>.json`.")
    out.append("")
    out.append("| variant | kind | P | R | F1 | TP | FP | FN | labelled | unlabelled | emitted |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        out.append(
            f"| {r['variant']} | {r['kind']} | "
            f"{_fmt(r['precision'])} | {_fmt(r['recall'])} | {_fmt(r['f1'])} | "
            f"{r['tp']} | {r['fp']} | {r['fn']} | "
            f"{r['labelled']} | {r['unlabelled']} | {r['emitted']} |"
        )
    return "\n".join(out) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweeps-root", type=Path, default=_SWEEPS_ROOT)
    p.add_argument("--md-out", type=Path, default=None)
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--no-legacy-fallback", action="store_true",
                   help="Only use per-variant label files; do not fall back "
                        "to eval/annotations/annotations_<mode>.json.")
    args = p.parse_args(argv)

    gt = load_ground_truth()
    figures_fn_total = sum(g.get("missed_figures", 0) for g in gt.values())
    tables_fn_total = sum(g.get("total_tables", 0) for g in gt.values())

    rows: List[Dict[str, Any]] = []
    for variant_dir in sorted(p for p in args.sweeps_root.iterdir() if p.is_dir()):
        variant = variant_dir.name
        figures_emitted, tables_emitted = load_emitted_crops(variant_dir)
        table_mode, figure_mode = detector_modes(variant_dir)

        fig_labels = load_labels(figure_mode, variant,
                                  fallback_legacy=not args.no_legacy_fallback)
        tab_labels = load_labels(table_mode,  variant,
                                  fallback_legacy=not args.no_legacy_fallback)

        # Figures: FN is the sum of `missed_figures` from ground_truth.csv
        # (variant-independent corpus-wide miss count, not derived from TP).
        # Tables: FN = max(0, total_tables - TP_variant).
        fig_m = score_kind(figures_emitted, fig_labels, classify_figure,
                           fn_count=figures_fn_total)
        tab_m = score_kind(tables_emitted,  tab_labels, classify_table,
                           total_actual=tables_fn_total)

        fig_m["variant"] = variant
        fig_m["kind"] = "figures"
        tab_m["variant"] = variant
        tab_m["kind"] = "tables"
        rows.append(fig_m)
        rows.append(tab_m)

    md = render_markdown(rows)
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(md)
        print(f"wrote markdown → {args.md_out}", file=sys.stderr)
    else:
        sys.stdout.write(md)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2))
        print(f"wrote json → {args.json_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
