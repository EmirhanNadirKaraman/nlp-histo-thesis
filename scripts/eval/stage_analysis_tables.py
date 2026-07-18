#!/usr/bin/env python3
"""Regenerate per-stage analysis tables for the document-extraction sweep.

The sweep is staged (detector → table-in-figure → header-fix → footnote-screen
→ footnote-multiplier → merge-flags → reconstruction); each stage's decision is
made on a small set of variants. This script reads a `score_pdf_variants.py`
JSON report and emits one markdown analysis table per stage — just that stage's
variants, their table-rubric metrics, and the stage's leading variant flagged —
so the per-stage decision is legible without wading through all 31 rows.

The variant→stage mapping is imported from `run_all_sweeps.ALL_SWEEPS` (the
authoritative source), so new variants are picked up automatically.

Deciding metric per stage:
  * detector stages  → table crop F1 (which detector/threshold crops best);
  * every later stage → table strict F1 (the headline rubric metric).

By default it scores `out/sweeps/` fresh and analyses only the variants present
there — i.e. only the freshly re-run variants, never the old baseline. Stages
not yet re-run print "not scored yet". Pass a positional JSON to analyse a saved
report instead (e.g. the old full baseline).

Usage:
    # only the freshly re-run variants (scores out/sweeps):
    python3 scripts/eval/stage_analysis_tables.py
    python3 scripts/eval/stage_analysis_tables.py --per-stage-dir reports/stage_analysis

    # analyse a saved scored report instead (e.g. the old baseline):
    python3 scripts/eval/stage_analysis_tables.py reports/post_exclusion_PMC11863705/variants_PR.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for p in (str(_REPO), str(_REPO / "scripts" / "eval")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_all_sweeps as RS  # noqa: E402  — authoritative variant→stage map

# Human-readable heading + the metric that decides each stage. Detector
# substages (detector_docling/tatr/hybrid) are merged into one "Stage 1".
_STAGE_META = {
    "detector":            ("Stage 1 — detector / threshold selection",
                            "Which table detector (Docling / TATR / Hybrid) and "
                            "TATR threshold crops tables best.", "crop"),
    "table_in_figure":     ("Stage 2 — table-in-figure handling",
                            "Does dropping/pre-masking tables that fall inside "
                            "figure regions help?", "strict"),
    "header_fix":          ("Stage 3 — header-band drop",
                            "Does dropping detected tables in the top page band "
                            "(running-header false positives) help?", "strict"),
    "footnote_screen":     ("Stage 4 — footnote expansion (family pick)",
                            "Footnote expansion ON at multiplier 1.2, compared "
                            "across detector families.", "strict"),
    "footnote_multiplier": ("Stage 5 — footnote multiplier",
                            "Best footnote_threshold_multiplier per family.",
                            "strict"),
    "merge_flags":         ("Stage 6 — caption-merge flags",
                            "Do merge_tables_by_caption / merge_figures_by_caption "
                            "help?", "strict"),
    "reconstruction":      ("Stage 7 — table reconstruction",
                            "Does reconstruct_tables_from_lists help?", "strict"),
}


def _stage_group(stage: str) -> str:
    return "detector" if stage.startswith("detector") else stage


def _ordered_groups() -> list[str]:
    """Stage groups in sweep order, from ALL_SWEEPS."""
    seen: list[str] = []
    for spec in RS.ALL_SWEEPS:
        g = _stage_group(spec.stage)
        if g not in seen:
            seen.append(g)
    return seen


def _variant_groups() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for spec in RS.ALL_SWEEPS:
        groups.setdefault(_stage_group(spec.stage), []).append(spec.name)
    return groups


def _score_sweeps(sweeps_root: Path) -> Path:
    """Score a sweeps root fresh → temp JSON, covering only the variant dirs
    present there (i.e. only the freshly re-extracted variants). Reuses
    score_pdf_variants.py verbatim so the metrics are identical."""
    if not sweeps_root.exists() or not any(p.is_dir() for p in sweeps_root.iterdir()):
        sys.exit(f"error: {sweeps_root} has no variant dirs — run "
                 f"run_all_sweeps.py first (or pass a scored --json).")
    tmp = Path(tempfile.mkdtemp(prefix="stage_analysis_")) / "scored.json"
    cmd = [sys.executable,
           str(_REPO / "scripts" / "eval" / "score_pdf_variants.py"),
           "--sweeps-root", str(sweeps_root), "--json-out", str(tmp)]
    res = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True)
    if res.returncode != 0 or not tmp.exists():
        sys.exit(f"error: scoring {sweeps_root} failed:\n{res.stderr[-2000:]}")
    return tmp


def _rel(p: Path) -> str:
    """Repo-relative display path, robust to relative/outside inputs."""
    try:
        return str(p.resolve().relative_to(_REPO))
    except ValueError:
        return str(p)


def _pct(x):
    return f"{x*100:.1f}" if x is not None else "—"


def _figures_line(tables_json: list) -> str | None:
    figs = [r for r in tables_json if r["kind"] == "figures"]
    if not figs:
        return None
    f = figs[0]["by_dim"]
    s = figs[0]["strict"]
    return (f"_Figures are detector-invariant across all variants: crop F1 "
            f"{_pct(f['crop']['f1'])}, mask F1 {_pct(f['mask']['f1'])}, "
            f"strict F1 {_pct(s['f1'])}, {figs[0]['counts']['emitted']} emitted._")


def render_stage(group: str, variants: list[str], scored: dict[str, dict]) -> str:
    title, blurb, metric = _STAGE_META.get(
        group, (f"Stage — {group}", "", "strict"))
    present = [v for v in variants if v in scored]
    out = [f"## {title}", "", f"*{blurb}*", ""]
    if not present:
        out.append("_No scored variants for this stage yet "
                   "(run the stage and re-score)._\n")
        return "\n".join(out)

    def key(v):
        r = scored[v]
        return (r["by_dim"]["crop"]["f1"] if metric == "crop"
                else r["strict"]["f1"]) or 0.0
    winner = max(present, key=key)

    out.append("| variant | crop P | crop R | crop F1 | mask F1 | strict F1 "
               "| emitted | unlab |")
    out.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for v in sorted(present, key=key, reverse=True):
        r = scored[v]
        c, m, s = r["by_dim"]["crop"], r["by_dim"]["mask"], r["strict"]
        star = " ⭐" if v == winner else ""
        # bold the deciding metric
        cf = f"**{_pct(c['f1'])}**" if metric == "crop" else _pct(c['f1'])
        sf = f"**{_pct(s['f1'])}**" if metric == "strict" else _pct(s['f1'])
        out.append(f"| {v}{star} | {_pct(c['precision'])} | {_pct(c['recall'])} "
                   f"| {cf} | {_pct(m['f1'])} | {sf} | {r['counts']['emitted']} "
                   f"| {r['counts']['unlabelled']} |")
    mlabel = "table crop F1" if metric == "crop" else "table strict F1"
    win_val = (scored[winner]["by_dim"]["crop"]["f1"] if metric == "crop"
               else scored[winner]["strict"]["f1"])
    out.append("")
    out.append(f"**Winner: `{winner}`** — {mlabel} {_pct(win_val)}%.")
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", nargs="?", type=Path, default=None,
                    help="Analyse this pre-scored variants_*.json instead of "
                         "scoring out/sweeps (e.g. the old baseline).")
    ap.add_argument("--sweeps-root", type=Path, default=_REPO / "out" / "sweeps",
                    help="Score this sweeps root fresh and analyse only the "
                         "variants present there (default: out/sweeps — i.e. "
                         "only the freshly re-run variants).")
    ap.add_argument("--out", type=Path,
                    default=_REPO / "reports" / "stage_analysis.md")
    ap.add_argument("--per-stage-dir", type=Path, default=None,
                    help="Also write one file per stage into this dir.")
    args = ap.parse_args()

    # Default: score out/sweeps → only the freshly re-extracted variants.
    # Override with a positional JSON to analyse a saved report (e.g. baseline).
    if args.json:
        json_path = args.json.resolve()
        if not json_path.exists():
            print(f"error: {json_path} not found.", file=sys.stderr)
            return 2
    else:
        json_path = _score_sweeps(args.sweeps_root.resolve())

    all_rows = json.loads(json_path.read_text())
    scored = {r["variant"]: r for r in all_rows if r["kind"] == "tables"}
    vgroups = _variant_groups()

    header = [
        "# Document-extraction sweep — per-stage analysis tables", "",
        f"Source: `{_rel(json_path)}`  ·  "
        f"{len(scored)} variants scored.  ",
        "Tables show the **table** rubric (figure cropping is "
        "detector-invariant). ⭐ marks each stage's leading variant; the "
        "deciding metric is **bold**.", "",
    ]
    fl = _figures_line(all_rows)
    if fl:
        header += [fl, ""]

    sections = [render_stage(g, vgroups.get(g, []), scored)
                for g in _ordered_groups()]
    report = "\n".join(header + sections)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"wrote {_rel(args.out)}  ({len(scored)} variants)")

    if args.per_stage_dir:
        args.per_stage_dir.mkdir(parents=True, exist_ok=True)
        for g in _ordered_groups():
            p = args.per_stage_dir / f"{g}.md"
            p.write_text("\n".join(header[:1] + [""]) +
                         render_stage(g, vgroups.get(g, []), scored))
            print(f"  wrote {_rel(p)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
