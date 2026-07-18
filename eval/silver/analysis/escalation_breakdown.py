"""
Augment cascade-sweep CSVs with per-tier escalation rates + a price-weighted cost.

The MAP cascade sweeps (``run_new_summarization_sweeps.py`` /
``map_theta_sweep.py``) log per-tier invocation counts — ``n_l1_invoked``,
``n_l2_invoked``, ``n_l3_invoked`` — and voter counts (``n_l1_voters``,
``n_l2_voters``), but the only named cost column is ``escalate_rate`` (the
L2→L3 marginal rate). This helper derives the rest so thesis tables / variant
comparison can cite them directly.

Derived columns (per row); ``n_l1_invoked == n_chunks`` (every chunk runs L1):

  l1_escalate_rate       = n_l2_invoked / n_l1_invoked   # L1 disagreed → L2
  l2_escalate_rate       = n_l3_invoked / n_l1_invoked   # reached L3 (== escalate_rate)
  l2_escalate_rate_cond  = n_l3_invoked / n_l2_invoked   # of L2-reached, fraction → L3
  cost_frac              = (9.6·n_l2 + 18·n_l3) / (27.6·n_chunks)   # price-weighted [0,1]
  total_calls            = n_l1·n_l1_voters + n_l2·n_l2_voters + n_l3·1

``cost_frac`` mirrors ``run_new_summarization_sweeps._cost_frac`` (blended in+out
$/1M per tier, L1=1.75 / L2=9.60 / L3=18.00, L1 term dropped as constant). Unlike
``escalate_rate`` it charges the L1→L2 traffic too (L2 ≈ 5.5× L1/call).

Usage:
  python -m eval.silver.analysis.escalation_breakdown <csv>            # write <name>_escalation.csv
  python -m eval.silver.analysis.escalation_breakdown <csv> --print    # summary table, no write
  python -m eval.silver.analysis.escalation_breakdown <csv> --rank     # cost-efficiency Pareto frontier
  python -m eval.silver.analysis.escalation_breakdown <csv> --in-place # overwrite input

Default writes ``<name>_escalation.csv`` with all derived columns appended (this
is the safe "persisted cost_frac" — it never touches the engine's own CSV).
Idempotent: re-running overwrites the derived columns rather than duplicating.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Source columns the cascade writes.
_L1, _L2, _L3 = "n_l1_invoked", "n_l2_invoked", "n_l3_invoked"
_NCHUNKS, _L1V, _L2V = "n_chunks", "n_l1_voters", "n_l2_voters"
# Blended in+out $/1M per tier — see run_new_summarization_sweeps.TIER_COST_BLENDED.
_W2, _W3 = 9.60, 18.00
_DEFAULT_TIER_VOTERS = 3   # fallback when n_l{1,2}_voters columns are absent (legacy CSVs)
# Derived columns this helper adds (inserted after escalate_rate if present).
_DERIVED = ("l1_escalate_rate", "l2_escalate_rate", "l2_escalate_rate_cond",
            "cost_frac", "total_calls")
_METRIC = "strict_f1_optimal"   # quality axis for --rank
# Columns used to label rows (only those that exist are shown).
_LABEL_COLS = (
    "embedder", "scorer_kind", "alignment_strategy", "variant",
    "theta", "reject_theta", "voter_subset",
)


def _rate(num: str, den: str) -> str:
    """num/den as a 4-dp string; blank when the denominator is 0/missing."""
    try:
        d = float(den)
        if d == 0:
            return ""
        return f"{float(num) / d:.4f}"
    except (TypeError, ValueError):
        return ""


def _num(r: dict, key: str, default: float = 0.0) -> float:
    try:
        v = r.get(key)
        return float(v) if v not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _cost_frac(r: dict) -> str:
    """Price-weighted escalation cost in [0,1]: (w2·n_l2 + w3·n_l3)/((w2+w3)·n_chunks)."""
    n = _num(r, _NCHUNKS)
    if n <= 0:
        return ""
    return f"{(_W2 * _num(r, _L2) + _W3 * _num(r, _L3)) / ((_W2 + _W3) * n):.4f}"


def _total_calls(r: dict) -> str:
    """LLM calls = n_l1·v1 + n_l2·v2 + n_l3·1 (L3 is a single voter)."""
    v1 = _num(r, _L1V, _DEFAULT_TIER_VOTERS)
    v2 = _num(r, _L2V, _DEFAULT_TIER_VOTERS)
    return f"{int(_num(r, _L1) * v1 + _num(r, _L2) * v2 + _num(r, _L3))}"


def _augment_rows(rows: list[dict], src: Path) -> list[dict]:
    """Fill the derived columns on each row."""
    for r in rows:
        r["l1_escalate_rate"] = _rate(r.get(_L2, ""), r.get(_L1, ""))
        r["l2_escalate_rate"] = _rate(r.get(_L3, ""), r.get(_L1, ""))
        r["l2_escalate_rate_cond"] = _rate(r.get(_L3, ""), r.get(_L2, ""))
        r["cost_frac"] = _cost_frac(r)
        r["total_calls"] = _total_calls(r)
        # Cross-check l2_escalate_rate vs the harness's own escalate_rate.
        existing, derived = r.get("escalate_rate"), r.get("l2_escalate_rate")
        if existing and derived:
            try:
                if abs(float(existing) - float(derived)) > 0.0002:
                    print(f"  [warn] {src.name}: l2_escalate_rate {derived} != "
                          f"escalate_rate {existing} — column semantics may have drifted",
                          file=sys.stderr)
            except ValueError:
                pass
    return rows


def _fieldnames_with_derived(original: list[str]) -> list[str]:
    base = [c for c in original if c not in _DERIVED]
    if "escalate_rate" in base:
        i = base.index("escalate_rate") + 1
        return base[:i] + list(_DERIVED) + base[i:]
    return base + list(_DERIVED)


def process(path: Path, *, in_place: bool, do_print: bool, do_rank: bool, suffix: str) -> Path | None:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise SystemExit(f"{path}: empty CSV / no header")
        missing = [c for c in (_L1, _L2, _L3) if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f"{path}: not a cascade sweep CSV — missing {', '.join(missing)}")
        original_fields = list(reader.fieldnames)
        rows = list(reader)

    rows = _augment_rows(rows, path)

    if do_rank:
        _print_rank(path, rows)
        return None
    if do_print:
        _print_table(path, rows)
        return None

    out_fields = _fieldnames_with_derived(original_fields)
    out_path = path if in_place else path.with_name(f"{path.stem}{suffix}{path.suffix}")
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{path}  ({len(rows)} rows)  →  {out_path}")
    return out_path


def _print_table(path: Path, rows: list[dict]) -> None:
    labels = [c for c in _LABEL_COLS if rows and c in rows[0]]
    cols = ["l1_esc", "l2_esc", "l2|L2", "cost", "calls"]
    keys = ["l1_escalate_rate", "l2_escalate_rate", "l2_escalate_rate_cond", "cost_frac", "total_calls"]
    header = labels + cols
    widths = [max(len(h), 6) for h in header]
    print(f"\n{path}  ({len(rows)} rows)")
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for r in rows:
        cells = [str(r.get(c, "")) for c in labels] + [r.get(k, "") for k in keys]
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))


def _print_rank(path: Path, rows: list[dict]) -> None:
    """Print the cost-efficiency Pareto frontier: cells non-dominated on
    (``strict_f1_optimal`` ↑, ``cost_frac`` ↓), sorted by quality. These are the
    only cells worth considering when chasing a cheaper variant — everything
    else is beaten on both quality and cost."""
    if not rows or _METRIC not in rows[0]:
        raise SystemExit(f"{path}: no '{_METRIC}' column — --rank needs a quality metric")
    pts = [(r, _num(r, _METRIC), _num(r, "cost_frac")) for r in rows]
    front = []
    for r, q, c in pts:
        if not any(q2 >= q and c2 <= c and (q2 > q or c2 < c) for _, q2, c2 in pts):
            front.append((r, q, c))
    front.sort(key=lambda t: -t[1])   # quality desc

    labels = [c for c in _LABEL_COLS if c in rows[0]]
    header = labels + ["strictF1", "cost", "l1_esc", "l2_esc", "calls"]
    widths = [max(len(h), 6) for h in header]
    print(f"\n{path}  — cost-efficiency Pareto frontier "
          f"({len(front)} of {len(rows)} cells; strictF1 ↑ / cost ↓)")
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for r, q, c in front:
        cells = [str(r.get(x, "")) for x in labels] + [
            f"{q:.4f}", f"{c:.4f}", r.get("l1_escalate_rate", ""),
            r.get("l2_escalate_rate", ""), r.get("total_calls", "")]
        print("  ".join(x.ljust(w) for x, w in zip(cells, widths)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", type=Path, help="cascade sweep CSV(s)")
    ap.add_argument("--in-place", action="store_true", help="overwrite the input CSV instead of a copy")
    ap.add_argument("--print", dest="do_print", action="store_true", help="print a per-row table; write nothing")
    ap.add_argument("--rank", dest="do_rank", action="store_true",
                    help="print the cost-efficiency Pareto frontier (strictF1 ↑ / cost ↓); write nothing")
    ap.add_argument("--suffix", default="_escalation", help="output filename suffix (default: _escalation)")
    args = ap.parse_args(argv)

    if sum([args.in_place, args.do_print, args.do_rank]) > 1:
        ap.error("--in-place / --print / --rank are mutually exclusive")

    for p in args.csv:
        if not p.exists():
            raise SystemExit(f"{p}: no such file")
        process(p, in_place=args.in_place, do_print=args.do_print, do_rank=args.do_rank, suffix=args.suffix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
