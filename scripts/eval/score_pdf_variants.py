#!/usr/bin/env python3
"""
score_pdf_variants.py — per-variant precision / recall / F1 for PDF
extraction sweeps, scored against a configurable label rubric.

Reads, for each sweep variant under ``out/sweeps/<variant>/``:

1. The variant's emitted crop set (from each ``*_media.json`` under
   ``out/sweeps/<variant>/json/``).
2. The variant's per-variant labels (from
   ``eval/annotations/<variant>/<mode>.json`` if it exists, else falls
   back to the legacy shared file ``eval/annotations/annotations_<mode>.json``).
3. Ground-truth recall counts (``eval/ground_truth.csv``).
4. Label rubric (``eval/label_rubric.yaml``) mapping each label string to
   per-dimension scores: ``(crop, caption, footnote)``.

Emits four metrics per (variant, kind):

  * **crop** F1   — bbox correctness, with GT-derived recall denominator
  * **caption** accuracy (P only) — caption-attachment correctness on
                                    detected items
  * **footnote** accuracy (P only) — footnote-inclusion correctness on
                                     detected items
  * **strict** F1 — all three dims must score 1.0 for TP

Compound labels (``"wrong caption + missing footnotes"``) are split on
``+`` and combined element-wise via MIN (most pessimistic per dim).
``n/a`` excludes the item from that dimension's denominator entirely
(use for "not a table" labels where caption/footnote are meaningless).
``skip`` excludes the item from all dimension denominators.

Usage::

    python scripts/eval/score_pdf_variants.py
    python scripts/eval/score_pdf_variants.py --rubric eval/label_rubric.yaml
    python scripts/eval/score_pdf_variants.py --md-out reports/variants_PR.md
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
_DEFAULT_RUBRIC_PATH = _REPO_ROOT / "eval" / "label_rubric.yaml"
_DIM_NAMES = ("crop", "caption", "footnote", "mask")
# Dims that have a corpus-level recall denominator (FN from ground truth).
# Caption / footnote are precision-only — recall is undefined corpus-wide.
_GT_RECALL_DIMS = frozenset({"crop", "mask"})


# ── Label classification (kept in sync with eval/precision_recall.py) ─────────


def classify_figure(label: str) -> str:
    """Legacy single-dim classifier (only used under --legacy mode).

    Exact match only (no prefix matching).  Labels not in the explicit
    tp/fp sets fall through to "skip".
    """
    lbl = label.lower().strip()
    if lbl == "correct":
        return "tp"
    if lbl in ("icon", "incorrect"):
        return "fp"
    return "skip"


def classify_table(label: str) -> str:
    """Legacy single-dim classifier (only used under --legacy mode).

    Exact match only (no prefix matching).  Labels not in the explicit
    tp/fp sets fall through to "skip".
    """
    lbl = label.lower().strip()
    if lbl in ("correct", "missing footnotes"):
        return "tp"
    if lbl in ("incorrect", "wrong caption"):
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
    elif "docling" in variant_dir.name.lower():
        # Manifest absent (e.g. scoring from eval/annotations after the
        # out/sweeps crops were deleted, B-068): infer the detector from the
        # variant name. Only docling-named variants use the docling table
        # label set; every other family ran HYBRID/TATR → "full". Validated
        # to reproduce reports/stage7_PR.md with 0 mismatches.
        detector = "DOCLING"
    table_mode = "json_tables_docling" if detector.upper() == "DOCLING" else "json_tables_full"
    return table_mode, "json_figures"


# ── Metrics ───────────────────────────────────────────────────────────────────


def metrics(tp: int, fp: int, fn: int) -> Dict[str, Optional[float]]:
    p = tp / (tp + fp) if (tp + fp) else None
    r = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * p * r / (p + r)) if (p and r) else None
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f1}


# ── Label rubric loading + parsing ────────────────────────────────────────────


_BUILTIN_RUBRIC = {
    "correct":           {"crop": 1.0, "caption": 1.0,   "footnote": 1.0,   "mask": 1.0},
    "incorrect":         {"crop": 0.0, "caption": "n/a", "footnote": "n/a", "mask": 0.0},
    "icon":              {"crop": 0.0, "caption": "n/a", "footnote": "n/a", "mask": 1.0},
    "wrong caption":     {"crop": 1.0, "caption": 0.0,   "footnote": 1.0,   "mask": 1.0},
    "missing footnotes": {"crop": 1.0, "caption": 1.0,   "footnote": 0.0,   "mask": 1.0},
    "crop is too big":   {"crop": 0.75,"caption": 1.0,   "footnote": 1.0,   "mask": 0.75},
    "crop is too small": {"crop": 0.25,"caption": 1.0,   "footnote": 1.0,   "mask": 0.25},
    "table_in_figure":   {"crop": 0.0, "caption": "n/a", "footnote": "n/a", "mask": 1.0},
    "weird":             {"crop": "skip","caption": "skip","footnote": "skip","mask": "skip"},
}


def load_rubric(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Load a YAML rubric file.  Returns a flat dict mapping label strings
    to per-dimension scores (merged across shared / figure / table blocks).

    Supports two YAML formats:
      * **New (2026-05-19):** three top-level blocks ``shared_labels:``,
        ``figure_labels:``, ``table_labels:``.  The flat dict merges all
        three so any label looked up by name still resolves.
      * **Legacy:** a single top-level ``labels:`` block.

    Falls back to the built-in rubric if the file is missing, YAML import
    fails, or no recognised block is present.
    """
    if path is None:
        path = _DEFAULT_RUBRIC_PATH
    if not path.exists():
        return dict(_BUILTIN_RUBRIC)
    try:
        import yaml  # type: ignore
    except ImportError:
        return dict(_BUILTIN_RUBRIC)
    try:
        data = yaml.safe_load(path.read_text())
    except Exception:
        return dict(_BUILTIN_RUBRIC)
    data = data or {}

    merged: Dict[str, Dict[str, Any]] = {}
    if "labels" in data:
        # Legacy format.
        merged.update(data.get("labels", {}) or {})
    for block in ("shared_labels", "figure_labels", "table_labels"):
        merged.update(data.get(block, {}) or {})

    if not merged:
        return dict(_BUILTIN_RUBRIC)
    # Lower-case keys for case-insensitive lookup
    return {str(k).lower(): v for k, v in merged.items()}


def load_rubric_per_kind(
    path: Optional[Path] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load the rubric split by kind.

    Returns ``{"figures": {...}, "tables": {...}}`` where each inner dict
    is ``shared_labels`` merged with the kind-specific block.  Used by
    the annotator to filter the ``[l]`` picker per current mode.

    Legacy single-block format gracefully degrades: both kinds receive
    the same merged dict.
    """
    if path is None:
        path = _DEFAULT_RUBRIC_PATH
    if not path.exists():
        builtin = dict(_BUILTIN_RUBRIC)
        return {"figures": builtin, "tables": builtin}
    try:
        import yaml  # type: ignore
    except ImportError:
        builtin = dict(_BUILTIN_RUBRIC)
        return {"figures": builtin, "tables": builtin}
    try:
        data = yaml.safe_load(path.read_text())
    except Exception:
        builtin = dict(_BUILTIN_RUBRIC)
        return {"figures": builtin, "tables": builtin}
    data = data or {}

    shared = dict((data.get("shared_labels", {}) or {}))
    fig    = dict((data.get("figure_labels", {}) or {}))
    tab    = dict((data.get("table_labels",  {}) or {}))

    if not (shared or fig or tab):
        legacy = dict((data.get("labels", {}) or {}))
        return {
            "figures": {str(k).lower(): v for k, v in legacy.items()},
            "tables":  {str(k).lower(): v for k, v in legacy.items()},
        }

    figures = {str(k).lower(): v for k, v in {**shared, **fig}.items()}
    tables  = {str(k).lower(): v for k, v in {**shared, **tab}.items()}
    return {"figures": figures, "tables": tables}




def parse_label_to_rubric(
    label: str,
    rubric: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Map a label string to per-dim scores via direct rubric lookup only.

    No prefix matching, no compound auto-derivation — the label must match
    a rubric entry case-insensitively or it returns ``None``.  Run the
    scorer with ``--review-unknown`` to add an entry interactively.
    """
    if not label or not label.strip():
        return None
    norm = label.lower().strip()
    if norm in rubric:
        return dict(rubric[norm])
    return None


_RUBRIC_VALID_TOKENS = {"n/a", "skip"}


def _prompt_rubric_score(label: str) -> Dict[str, Any]:
    """Interactively ask the user for per-dim scores for one new label.

    Each dim accepts a float in [0.0, 1.0], the literal ``n/a``, or
    ``skip``.  Re-prompts on invalid input.  KeyboardInterrupt re-raises.
    """
    print(f"\n  New label: {label!r}")
    print("  Enter score per dim (0.0–1.0, 'n/a', or 'skip'):")
    scored: Dict[str, Any] = {}
    for dim in _DIM_NAMES:
        while True:
            raw = input(f"    {dim}: ").strip().lower()
            if raw in _RUBRIC_VALID_TOKENS:
                scored[dim] = raw
                break
            try:
                f = float(raw)
            except ValueError:
                print("      invalid — enter 0.0–1.0, 'n/a', or 'skip'")
                continue
            if not (0.0 <= f <= 1.0):
                print("      out of range — must be in [0.0, 1.0]")
                continue
            scored[dim] = f
            break
    return scored


def _fmt_yaml_scalar(v: Any) -> str:
    """Format a single scalar for YAML output (string or numeric)."""
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        # Avoid trailing zeros: 1.0 → "1.0", 0.5 → "0.5"
        return f"{v:g}" if v != int(v) else f"{v:.1f}"
    return str(v)


def append_rubric_entry(path: Path, label: str, scored: Dict[str, Any]) -> None:
    """Append one ``labels:`` entry to the YAML file as raw text, preserving
    existing comments and ordering.  Key is single-quoted for safety.

    Idempotent only at the file level — caller is responsible for not
    appending duplicates.
    """
    if not path.exists():
        # Initialise a minimal file with the labels block.
        path.write_text("labels:\n")
    safe_label = label.replace("'", "''")
    lines = [f"\n  '{safe_label}':"]
    for dim in _DIM_NAMES:
        v = scored.get(dim, "n/a")
        lines.append(f"    {dim}: {_fmt_yaml_scalar(v)}")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def collect_unknown_labels(
    label_dicts: Iterable[Dict[str, str]],
    rubric: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Return sorted list of distinct labels (lower-cased, stripped) appearing
    in any of the supplied annotation dicts that have no rubric entry.
    """
    seen: set[str] = set()
    for ann in label_dicts:
        for raw in ann.values():
            if not raw:
                continue
            norm = raw.lower().strip()
            if norm and norm not in rubric and norm not in seen:
                seen.add(norm)
    return sorted(seen)


def score_kind_rubric(
    emitted: set[str],
    labels: Dict[str, str],
    rubric: Dict[str, Dict[str, Any]],
    *,
    total_actual: Optional[int] = None,
    fn_count: Optional[int] = None,
    dim_pass_threshold: float = 0.5,
    strict_threshold: float = 1.0,
    kind: Optional[str] = None,
) -> Dict[str, Any]:
    """Rubric-based scoring.

    For each emitted crop with a known label:
      * Each dimension's score is read from the rubric (0.0–1.0 / 'n/a' / 'skip').
      * Per-dimension: TP if score ≥ ``dim_pass_threshold`` (default 0.5);
        FP otherwise.  ``n/a`` excludes from that dim's denominator;
        ``skip`` excludes from all dims.
      * Strict overall: item is TP iff every dimension scores ≥
        ``strict_threshold`` (default 1.0) — i.e. perfect on every dim.
        ``n/a`` dims don't disqualify; ``skip`` does.

    Recall denominator (used by crop dim + strict):
      * ``total_actual`` → FN = max(0, total_actual - TP)
      * ``fn_count`` → FN = fn_count
      * neither → recall + F1 reported as None

    Kind override (Fix A, 2026-05-19):
      * ``kind="figures"`` forces ``footnote = "n/a"`` for every item
        regardless of what the rubric specifies.  Figures don't carry
        body footnotes, so the rubric's generic labels (``correct``,
        ``wrong caption``, ``crop too big minor``, …) which set
        ``footnote: 1.0`` would otherwise inflate the footnote TP count
        with figure items that should be excluded from that dim.
      * ``kind="tables"`` (or ``None``) leaves footnote scoring as the
        rubric dictates.
    """
    per_dim = {d: {"tp": 0, "fp": 0, "na": 0, "skipped": 0,
                    "fp_by_label": defaultdict(int)}
               for d in _DIM_NAMES}
    strict_tp = strict_fp = strict_skipped = 0
    unlabelled = unrecognised = 0
    labelled = 0
    icon_count = 0  # tally for "how much of crop precision loss is icons?"

    for crop in emitted:
        if crop not in labels:
            unlabelled += 1
            continue
        labelled += 1
        raw_label = labels[crop]
        scored = parse_label_to_rubric(raw_label, rubric)
        if scored is None:
            unrecognised += 1
            strict_skipped += 1
            for d in _DIM_NAMES:
                per_dim[d]["skipped"] += 1
            continue

        # Fix A (2026-05-19): figures don't have body footnotes.  Override
        # any rubric footnote score to "n/a" so generic labels like
        # `correct`, `wrong caption`, `crop too big minor` (which set
        # `footnote: 1.0`) don't inflate the footnote TP count when used
        # on figures.  Per-figure-label rubric entries that already set
        # `footnote: "n/a"` are unaffected.
        if kind == "figures":
            scored["footnote"] = "n/a"

        normalized_label = (raw_label or "").lower().strip()
        if normalized_label == "icon":
            icon_count += 1

        strict_pass = True
        any_dim_scored = False
        for d in _DIM_NAMES:
            v = scored.get(d, "n/a")
            if v == "skip":
                per_dim[d]["skipped"] += 1
                strict_pass = False
            elif v == "n/a":
                per_dim[d]["na"] += 1
                # n/a doesn't disqualify strict — the dim just doesn't apply
            else:
                try:
                    score = float(v)
                except (TypeError, ValueError):
                    per_dim[d]["skipped"] += 1
                    strict_pass = False
                    continue
                any_dim_scored = True
                if score >= dim_pass_threshold:
                    per_dim[d]["tp"] += 1
                else:
                    per_dim[d]["fp"] += 1
                    per_dim[d]["fp_by_label"][raw_label] += 1
                if score < strict_threshold:
                    strict_pass = False

        if not any_dim_scored:
            # Every dim was n/a or skip → item contributes nothing to strict
            strict_skipped += 1
        elif strict_pass:
            strict_tp += 1
        else:
            strict_fp += 1

    # ── Per-dim metrics ──────────────────────────────────────────────────────
    result_dims: Dict[str, Optional[Dict[str, Any]]] = {}
    for d in _DIM_NAMES:
        agg = per_dim[d]
        if agg["tp"] + agg["fp"] == 0:
            result_dims[d] = None
            continue
        # GT-recall dims (crop, mask): FN comes from ground_truth.csv —
        # missed_tables for tables, missed_figures for figures.  Caption /
        # footnote: precision-only, FN = 0, recall + F1 set to None.
        if d in _GT_RECALL_DIMS:
            if fn_count is not None:
                fn = fn_count
            elif total_actual is not None:
                fn = max(0, total_actual - agg["tp"])
            else:
                fn = 0
        else:
            fn = 0
        m = metrics(agg["tp"], agg["fp"], fn)
        if d not in _GT_RECALL_DIMS:
            # Caption / footnote: precision-only is meaningful.
            m["recall"] = None
            m["f1"] = None
        m["na"] = agg["na"]
        m["skipped"] = agg["skipped"]
        m["fp_by_label"] = dict(agg["fp_by_label"])
        result_dims[d] = m

    # ── Strict overall ───────────────────────────────────────────────────────
    if fn_count is not None:
        s_fn = fn_count
    elif total_actual is not None:
        s_fn = max(0, total_actual - strict_tp)
    else:
        s_fn = 0
    strict = metrics(strict_tp, strict_fp, s_fn)
    strict["skipped"] = strict_skipped

    return {
        "by_dim":      result_dims,
        "strict":      strict,
        "counts": {
            "emitted":     len(emitted),
            "labelled":    labelled,
            "unlabelled":  unlabelled,
            "unrecognised": unrecognised,
            "icon":        icon_count,
        },
    }


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


def render_aligned(rows: List[Dict[str, Any]], *, mode: str = "rubric") -> str:
    """Render rows as a tab-aligned ASCII table for terminal output.

    Each cell is space-padded to its column's max observed width and
    separated by a tab character.  Looks aligned in any monospace
    terminal regardless of tab-stop interpretation.

    ``mode`` selects the column set:
      * ``"rubric"`` — per-dim P/R/F1 + strict (default; matches the rubric
        scorer output).
      * ``"legacy"`` — simple variant × kind P/R/F1 (matches the legacy
        single-dim scorer output).
    """
    if not rows:
        return ""

    if mode == "legacy":
        columns: List[tuple[str, Any]] = [
            ("variant",    lambda r: r["variant"]),
            ("kind",       lambda r: r["kind"]),
            ("P",          lambda r: _fmt(r.get("precision"))),
            ("R",          lambda r: _fmt(r.get("recall"))),
            ("F1",         lambda r: _fmt(r.get("f1"))),
            ("TP",         lambda r: r.get("tp", 0)),
            ("FP",         lambda r: r.get("fp", 0)),
            ("FN",         lambda r: r.get("fn", 0)),
            ("labelled",   lambda r: r.get("labelled", 0)),
            ("unlabelled", lambda r: r.get("unlabelled", 0)),
            ("emitted",    lambda r: r.get("emitted", 0)),
        ]
    else:
        def _bd(r, dim, field):
            d = (r.get("by_dim") or {}).get(dim) or {}
            return d.get(field)

        def _tpfp(r, dim):
            d = (r.get("by_dim") or {}).get(dim) or {}
            return f"{d.get('tp', 0)}/{d.get('fp', 0)}"

        def _strict(r, field):
            s = r.get("strict") or {}
            return s.get(field)

        def _strict_tpfpfn(r):
            s = r.get("strict") or {}
            return f"{s.get('tp', 0)}/{s.get('fp', 0)}/{s.get('fn', 0)}"

        def _cnt(r, field):
            c = r.get("counts") or {}
            return c.get(field, 0)

        columns = [
            ("variant",          lambda r: r["variant"]),
            ("kind",             lambda r: r["kind"]),
            ("crop P",           lambda r: _fmt(_bd(r, "crop", "precision"))),
            ("crop R",           lambda r: _fmt(_bd(r, "crop", "recall"))),
            ("crop F1",          lambda r: _fmt(_bd(r, "crop", "f1"))),
            ("mask P",           lambda r: _fmt(_bd(r, "mask", "precision"))),
            ("mask R",           lambda r: _fmt(_bd(r, "mask", "recall"))),
            ("mask F1",          lambda r: _fmt(_bd(r, "mask", "f1"))),
            ("cap P",            lambda r: _fmt(_bd(r, "caption", "precision"))),
            ("cap tp/fp",        lambda r: _tpfp(r, "caption")),
            ("foot P",           lambda r: _fmt(_bd(r, "footnote", "precision"))),
            ("foot tp/fp",       lambda r: _tpfp(r, "footnote")),
            ("strict P",         lambda r: _fmt(_strict(r, "precision"))),
            ("strict R",         lambda r: _fmt(_strict(r, "recall"))),
            ("strict F1",        lambda r: _fmt(_strict(r, "f1"))),
            ("strict tp/fp/fn",  lambda r: _strict_tpfpfn(r)),
            ("emitted",          lambda r: _cnt(r, "emitted")),
            ("unlabelled",       lambda r: _cnt(r, "unlabelled")),
            ("unrecog",          lambda r: _cnt(r, "unrecognised")),
            ("icons",            lambda r: _cnt(r, "icon")),
        ]

    headers = [h for h, _ in columns]
    body = [[str(getter(r)) for _, getter in columns] for r in rows]

    widths = [max(len(h), *(len(row[i]) for row in body)) for i, h in enumerate(headers)]

    def _fmt_row(cells: List[str]) -> str:
        return "\t".join(c.ljust(w) for c, w in zip(cells, widths))

    lines = [
        _fmt_row(headers),
        _fmt_row(["-" * w for w in widths]),
    ]
    lines.extend(_fmt_row(row) for row in body)
    return "\n".join(lines) + "\n"


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


def render_markdown_rubric(rows: List[Dict[str, Any]], *,
                            rubric_path: Optional[Path] = None) -> str:
    """Render the rubric-based per-dim + strict report."""
    out: List[str] = []
    out.append("# Per-variant PDF-extraction — rubric-based P/R/F1")
    out.append("")
    out.append("Computed by `scripts/eval/score_pdf_variants.py` using the rubric at:")
    out.append(f"`{rubric_path}`" if rubric_path else "(built-in fallback rubric)")
    out.append("")
    out.append("**Crop F1** — bbox correctness vs ground-truth recall.")
    out.append("**Caption / Footnote P** — accuracy on detected items "
               "(no corpus-level recall denominator).")
    out.append("**Strict F1** — TP iff every dimension scores 1.0 on the rubric.")
    out.append("")
    out.append("| variant | kind | "
               "crop P | crop R | crop F1 | "
               "caption P (TP/FP) | footnote P (TP/FP) | "
               "strict P | strict R | strict F1 | strict TP/FP/FN | "
               "emitted | labelled | unrecog |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        crop = (r.get("by_dim") or {}).get("crop")
        cap  = (r.get("by_dim") or {}).get("caption")
        ft   = (r.get("by_dim") or {}).get("footnote")
        strict = r.get("strict") or {}
        out.append(
            f"| {r['variant']} | {r['kind']} | "
            f"{_fmt(crop['precision']) if crop else '—'} | "
            f"{_fmt(crop['recall']) if crop else '—'} | "
            f"{_fmt(crop['f1']) if crop else '—'} | "
            f"{_fmt(cap['precision']) if cap else '—'} "
            f"({cap['tp']}/{cap['fp']})" if cap else "—"
        )
        # Caption + footnote in one cell each; rebuild line cleanly:
    out = []  # rebuild simpler
    out.append("# Per-variant PDF-extraction — rubric-based P/R/F1")
    out.append("")
    if rubric_path:
        out.append(f"Rubric: `{rubric_path}`")
    else:
        out.append("Rubric: (built-in fallback)")
    out.append("")
    out.append("Per-dim semantics:")
    out.append("* **Crop F1** — figure / table output correctness with GT-derived recall.")
    out.append("* **Mask F1** — text-safety (was the region correctly masked from body text?).  "
               "Same FN source as crop (missed_figures / missed_tables).")
    out.append("* **Caption / Footnote precision** — accuracy on detected items "
               "(no recall denominator).  TP/FP counts in parens.")
    out.append("* **Strict F1** — TP iff every dim scores ≥ 1.0.")
    out.append("* **Icon count** — emitted crops labelled `icon` (mask-OK but crop-FP).  "
               "Helps quantify how much of crop-precision loss is icons specifically.")
    out.append("")
    out.append("| variant | kind | crop P | crop R | crop F1 | mask P | mask R | mask F1 | "
               "caption P | caption tp/fp | footnote P | footnote tp/fp | "
               "strict P | strict R | strict F1 | strict tp/fp/fn | "
               "emitted | unlabelled | unrecog | icons |")
    out.append("|" + "---|" * 20)
    for r in rows:
        bd = r.get("by_dim") or {}
        crop = bd.get("crop") or {}
        mask = bd.get("mask") or {}
        cap  = bd.get("caption") or {}
        ft   = bd.get("footnote") or {}
        strict = r.get("strict") or {}
        cnt = r.get("counts") or {}
        out.append("| "
            f"{r['variant']} | {r['kind']} | "
            f"{_fmt(crop.get('precision'))} | "
            f"{_fmt(crop.get('recall'))} | "
            f"{_fmt(crop.get('f1'))} | "
            f"{_fmt(mask.get('precision'))} | "
            f"{_fmt(mask.get('recall'))} | "
            f"{_fmt(mask.get('f1'))} | "
            f"{_fmt(cap.get('precision'))} | "
            f"{cap.get('tp', '—')}/{cap.get('fp', '—')} | "
            f"{_fmt(ft.get('precision'))} | "
            f"{ft.get('tp', '—')}/{ft.get('fp', '—')} | "
            f"{_fmt(strict.get('precision'))} | "
            f"{_fmt(strict.get('recall'))} | "
            f"{_fmt(strict.get('f1'))} | "
            f"{strict.get('tp', 0)}/{strict.get('fp', 0)}/{strict.get('fn', 0)} | "
            f"{cnt.get('emitted', 0)} | "
            f"{cnt.get('unlabelled', 0)} | "
            f"{cnt.get('unrecognised', 0)} | "
            f"{cnt.get('icon', 0)} |")
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
    p.add_argument("--rubric", type=Path, default=_DEFAULT_RUBRIC_PATH,
                   help="YAML rubric mapping label strings to per-dim scores.  "
                        "See eval/label_rubric.yaml for the default.")
    p.add_argument("--legacy", action="store_true",
                   help="Use the old binary classifier instead of the rubric "
                        "(emits the original P/R/F1 single column).")
    p.add_argument("--dim-pass-threshold", type=float, default=0.5,
                   help="Per-dim TP threshold: score >= this counts as TP.")
    p.add_argument("--strict-threshold", type=float, default=1.0,
                   help="Strict-overall threshold: every dim must score >= this.")
    p.add_argument("--review-unknown", action="store_true",
                   help="Before scoring, list every label that doesn't have a "
                        "rubric entry and interactively prompt for per-dim "
                        "scores.  New entries are appended to the rubric YAML.")
    args = p.parse_args(argv)

    gt = load_ground_truth()
    figures_fn_total = sum(g.get("missed_figures", 0) for g in gt.values())
    tables_fn_total = sum(g.get("total_tables", 0) for g in gt.values())

    rubric = load_rubric(args.rubric) if not args.legacy else {}
    per_kind_rubric = (
        load_rubric_per_kind(args.rubric)
        if not args.legacy
        else {"figures": {}, "tables": {}}
    )

    # ── Optional pre-scoring review of unknown labels ────────────────────────
    if args.review_unknown and not args.legacy:
        all_label_dicts: List[Dict[str, str]] = []
        for variant_dir in sorted(p for p in args.sweeps_root.iterdir() if p.is_dir()):
            variant = variant_dir.name
            table_mode, figure_mode = detector_modes(variant_dir)
            all_label_dicts.append(load_labels(
                figure_mode, variant,
                fallback_legacy=not args.no_legacy_fallback))
            all_label_dicts.append(load_labels(
                table_mode, variant,
                fallback_legacy=not args.no_legacy_fallback))
        unknowns = collect_unknown_labels(all_label_dicts, rubric)
        if unknowns:
            print(f"\n{len(unknowns)} label(s) not in rubric "
                  f"({args.rubric}):", file=sys.stderr)
            for u in unknowns:
                print(f"  - {u!r}", file=sys.stderr)
            print("\nPrompting for scores. Ctrl-C to abort.\n", file=sys.stderr)
            try:
                for label in unknowns:
                    scored = _prompt_rubric_score(label)
                    append_rubric_entry(args.rubric, label, scored)
                    rubric[label] = scored
                    print(f"  → appended to {args.rubric}", file=sys.stderr)
            except KeyboardInterrupt:
                print("\n  Aborted by user — partial entries already saved.",
                      file=sys.stderr)
                return 130
        else:
            print("No unknown labels — rubric covers everything.", file=sys.stderr)

    rows: List[Dict[str, Any]] = []
    for variant_dir in sorted(p for p in args.sweeps_root.iterdir() if p.is_dir()):
        variant = variant_dir.name
        if variant.startswith("_"):
            continue  # archive / internal dir (e.g. _archive_YYYY-MM-DD), not a variant
        figures_emitted, tables_emitted = load_emitted_crops(variant_dir)
        table_mode, figure_mode = detector_modes(variant_dir)

        fig_labels = load_labels(figure_mode, variant,
                                  fallback_legacy=not args.no_legacy_fallback)
        tab_labels = load_labels(table_mode,  variant,
                                  fallback_legacy=not args.no_legacy_fallback)

        # Fallback when the sweep crops were deleted (B-068): no *_media.json on
        # disk → derive the emitted set from the annotated crop ids. The scorer
        # skips any emitted crop without a label, so emitted == set(labels) is
        # exact (an emitted-but-unlabelled crop contributes to no metric).
        if not figures_emitted and not tables_emitted:
            figures_emitted = set(fig_labels)
            tables_emitted = set(tab_labels)

        if args.legacy:
            fig_m = score_kind(figures_emitted, fig_labels, classify_figure,
                               fn_count=figures_fn_total)
            tab_m = score_kind(tables_emitted,  tab_labels, classify_table,
                               total_actual=tables_fn_total)
        else:
            # Use the per-kind rubric so figures get their kind-specific
            # footnote scores (n/a) directly from the rubric, not from a
            # runtime override.  Falls back to the flat dict on legacy
            # rubrics without `figure_labels` / `table_labels` blocks.
            fig_m = score_kind_rubric(
                figures_emitted, fig_labels, per_kind_rubric["figures"],
                fn_count=figures_fn_total,
                dim_pass_threshold=args.dim_pass_threshold,
                strict_threshold=args.strict_threshold,
                kind="figures",
            )
            tab_m = score_kind_rubric(
                tables_emitted, tab_labels, per_kind_rubric["tables"],
                total_actual=tables_fn_total,
                dim_pass_threshold=args.dim_pass_threshold,
                strict_threshold=args.strict_threshold,
                kind="tables",
            )

        fig_m["variant"] = variant
        fig_m["kind"] = "figures"
        tab_m["variant"] = variant
        tab_m["kind"] = "tables"
        rows.append(fig_m)
        rows.append(tab_m)

    # Markdown goes to --md-out (file); stdout gets the tab-aligned format.
    if args.md_out:
        md = (render_markdown(rows) if args.legacy
              else render_markdown_rubric(rows, rubric_path=args.rubric))
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(md)
        print(f"wrote markdown → {args.md_out}", file=sys.stderr)

    aligned = render_aligned(
        rows,
        mode="legacy" if args.legacy else "rubric",
    )
    sys.stdout.write(aligned)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2))
        print(f"wrote json → {args.json_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
