#!/usr/bin/env python3
"""Recompute the PDF-extraction variant ranking from the surviving annotation
labels + ground_truth.csv, with and without a given pmcid, to check whether
dropping that paper changes the winning config.

Background (BUGS.md B-068): the sweep crop outputs (out/sweeps/<variant>/json/
*_media.json) and run manifests were deleted, so score_pdf_variants.py can no
longer run as-is. But the metric is a pure function of the per-crop labels
(eval/annotations/<variant>/*.json) + ground_truth.csv, because the scorer
skips any emitted crop without a label. We therefore reconstruct the emitted
set as set(labels) and the table-label file from the variant name (docling
variants score json_tables_docling, everything else json_tables_full).

Two assumptions are baked in:
  (1) emitted == set(labels)  — valid iff every emitted crop was labelled
                                (reports/*_PR.md show unlabelled=0).
  (2) detector mode from name — docling-named variants only.
Both are validated by the --gate check: the full (all-PDFs) reconstruction
must reproduce reports/stage7_PR.md. If it doesn't, do not trust the dropped
numbers.

Reuses score_pdf_variants' scoring functions verbatim so the metric math is
byte-identical to the original.

Usage:
    python3 scripts/eval/recheck_drop_pmcid.py PMC11863705
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_EVAL_DIR = _REPO_ROOT / "scripts" / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import score_pdf_variants as S  # noqa: E402

EXCLUDE = sys.argv[1] if len(sys.argv) > 1 else "PMC11863705"
GATE_REPORT = _REPO_ROOT / "reports" / "stage7_PR.md"


def table_mode(variant: str) -> str:
    """Replicates score_pdf_variants.detector_modes without the (deleted)
    run manifest: only docling-detector variants read the docling table file."""
    return "json_tables_docling" if "docling" in variant.lower() else "json_tables_full"


def filt(labels: dict, exclude: str | None) -> dict:
    if not exclude:
        return labels
    return {k: v for k, v in labels.items() if not k.startswith(exclude + "_")}


def gt_sums(gt: dict, exclude: str | None):
    items = {p: g for p, g in gt.items()
             if not (exclude and p.startswith(exclude + "_") or p == (exclude or "") + "_main")}
    # be explicit: drop any pmcid whose PMC-prefix matches `exclude`
    if exclude:
        items = {p: g for p, g in gt.items()
                 if re.match(r"(PMC\d+)", p).group(1) != exclude}
    fig_fn = sum(g["missed_figures"] for g in items.values())
    tab_fn = sum(g["total_tables"] for g in items.values())
    return fig_fn, tab_fn


def run(exclude: str | None) -> dict:
    rubric_pk = S.load_rubric_per_kind(S._DEFAULT_RUBRIC_PATH)
    gt = S.load_ground_truth()
    fig_fn, tab_fn = gt_sums(gt, exclude)
    rows: dict[str, dict] = {}
    for vdir in sorted(p for p in S._ANN_ROOT.iterdir() if p.is_dir()):
        variant = vdir.name
        if not (vdir / "json_figures.json").exists():
            continue  # not a scored variant dir (e.g. share_map.json lives at root)
        tmode = table_mode(variant)
        fig_labels = filt(S.load_labels("json_figures", variant), exclude)
        tab_labels = filt(S.load_labels(tmode, variant), exclude)
        figm = S.score_kind_rubric(
            set(fig_labels), fig_labels, rubric_pk["figures"],
            fn_count=fig_fn, kind="figures",
        )
        tabm = S.score_kind_rubric(
            set(tab_labels), tab_labels, rubric_pk["tables"],
            total_actual=tab_fn, kind="tables",
        )
        rows[variant] = {"fig": figm, "tab": tabm}
    return rows


def _pct(x):
    return f"{x*100:.1f}%" if x is not None else "—"


def parse_gate(path: Path) -> dict:
    """Pull reported table strict-F1 per variant from a *_PR.md scorer dump."""
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if "| tables |" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        variant = cells[1]
        # locate the strict-F1 cell: header is the source of truth, but the
        # tables row has 'strict F1' as the 15th column in the emitted dump.
        # Match the first standalone percentage after the 'strict R' group.
        pcts = [c for c in cells if c.endswith("%")]
        # columns: cropP cropR cropF1 maskP maskR maskF1 captionP footnoteP
        #          strictP strictR strictF1  → strict F1 is the 11th pct
        if len(pcts) >= 11:
            try:
                out[variant] = float(pcts[10].rstrip("%")) / 100.0
            except ValueError:
                pass
    return out


def main() -> int:
    full = run(None)
    dropped = run(EXCLUDE)
    gate = parse_gate(GATE_REPORT)

    # Rank by TABLE strict F1 — the headline metric the winner is chosen on
    # (figures are detector-invariant).
    def tbl_f1(rows, v):
        return rows[v]["tab"]["strict"]["f1"] or 0.0

    order_full = sorted(full, key=lambda v: tbl_f1(full, v), reverse=True)
    order_drop = sorted(dropped, key=lambda v: tbl_f1(dropped, v), reverse=True)

    print(f"\nRecompute from labels — exclude = {EXCLUDE}\n")
    print("GATE: full (all-PDF) reconstruction vs reports/stage7_PR.md "
          "(table strict F1)")
    mism = 0
    for v in order_full:
        rep = gate.get(v)
        mine = tbl_f1(full, v)
        flag = ""
        if rep is not None:
            if abs(rep - mine) > 0.0005:
                flag = f"  <-- MISMATCH (reported {_pct(rep)})"
                mism += 1
        else:
            flag = "  (not in stage7 report)"
        print(f"  {v:<52} {_pct(mine)}{flag}")
    print(f"\nGATE result: {mism} mismatch(es) vs stage7_PR.md "
          f"on overlapping variants.\n")

    print("IMPACT: table strict F1 — full(28) vs dropped(27), ranked by full")
    print(f"  {'variant':<52} {'full':>7} {'drop':>7}  {'Δpp':>6}")
    for v in order_full:
        f = tbl_f1(full, v)
        d = tbl_f1(dropped, v)
        print(f"  {v:<52} {_pct(f):>7} {_pct(d):>7}  {(d-f)*100:>+6.1f}")

    print("\nWINNER (top by table strict F1):")
    print(f"  full(28):  {order_full[0]}  ({_pct(tbl_f1(full, order_full[0]))})")
    print(f"  drop(27):  {order_drop[0]}  ({_pct(tbl_f1(dropped, order_drop[0]))})")
    print(f"  ranking changed: {order_full != order_drop}")
    print(f"  winner changed:  {order_full[0] != order_drop[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
