#!/usr/bin/env python3
"""
Compare two routing policy runs on the same chunks.

Aligns records by (pmcid, chunk_id) and builds a 2×2 KEEP/DEFERRED confusion
matrix.  Both escalate and reject collapse to DEFERRED — both require the
expensive path, so the primary cost distinction is KEEP vs. DEFERRED.

Usage
-----
python scripts/compare_policies.py \\
    --policy-a out/routing_records_policy_a.jsonl  --id-a "emb_theta07" \\
    --policy-b out/routing_records_labeled.jsonl   --id-b "sem_theta074" \\
    --out-dir out/policy_comparison/

Outputs
-------
out_dir/compare_matrix.csv  — 2×2 KEEP/DEFERRED matrix with counts + FAR
out_dir/compare_report.json — matched pairs, per-policy deferral rates,
                              raw decision breakdown, disagreement FAR

Works without labels
--------------------
Decision comparison, deferral rates, and confusion counts are available for any
two routing record files.  Quality metrics (false_accept_rate cells) are null
when keep_ok labels are absent.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from pipeline.stages.knowledge_extraction.routing.routing_dataset import (  # noqa: E402
    RoutingDataset,
    RoutingRecord,
)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--policy-a", required=True, help="Path to policy A routing records JSONL")
    p.add_argument("--id-a", required=True, help="Policy ID label for policy A")
    p.add_argument("--policy-b", required=True, help="Path to policy B routing records JSONL")
    p.add_argument("--id-b", required=True, help="Policy ID label for policy B")
    p.add_argument("--out-dir", default="out/policy_comparison/", help="Output directory")
    return p.parse_args()


# ── Decision collapsing ────────────────────────────────────────────────────────

def _collapse(decision: str) -> str:
    """Collapse raw decision to KEEP or DEFERRED (escalate + reject → DEFERRED)."""
    return "KEEP" if decision == "keep" else "DEFERRED"


# ── Loading ────────────────────────────────────────────────────────────────────

def _load_indexed(path: str) -> dict[tuple[str, str], RoutingRecord]:
    records = list(RoutingDataset.iter_records(path))
    return {(r.pmcid, r.chunk_id): r for r in records}


# ── Matrix cell ────────────────────────────────────────────────────────────────

def _make_cell() -> dict:
    return {
        "n": 0,
        "n_keep_ok_true": 0,
        "n_keep_ok_false": 0,
        "false_accept_rate": None,  # filled after accumulation
    }


def _fill_far(cell: dict) -> None:
    """Fill false_accept_rate in place if any keep_ok labels are present."""
    total = cell["n_keep_ok_true"] + cell["n_keep_ok_false"]
    if total > 0:
        cell["false_accept_rate"] = cell["n_keep_ok_false"] / total


def _update_cell(cell: dict, rec_a: RoutingRecord, rec_b: RoutingRecord) -> None:
    """Accumulate one matched pair into a matrix cell."""
    cell["n"] += 1
    # Use keep_ok from whichever record has it; prefer record from the file that
    # has the label (type A/B from label_routing_records.py).  If both have it
    # and they disagree, use rec_a's value (arbitrary but consistent).
    keep_ok = None
    if rec_a.keep_ok is not None:
        keep_ok = rec_a.keep_ok
    elif rec_b.keep_ok is not None:
        keep_ok = rec_b.keep_ok

    if keep_ok is True:
        cell["n_keep_ok_true"] += 1
    elif keep_ok is False:
        cell["n_keep_ok_false"] += 1


# ── Core comparison ────────────────────────────────────────────────────────────

def compare(
    records_a: dict[tuple[str, str], RoutingRecord],
    records_b: dict[tuple[str, str], RoutingRecord],
    id_a: str,
    id_b: str,
) -> dict:
    """
    Build the 2×2 KEEP/DEFERRED matrix and summary report.

    Matrix cells (primary axis = policy A rows, policy B columns):
        concordant_keep     — both KEEP
        a_only_keep         — A KEEP, B DEFERRED
        b_only_keep         — A DEFERRED, B KEEP
        concordant_deferred — both DEFERRED
    """
    keys_a = set(records_a)
    keys_b = set(records_b)
    matched_keys = keys_a & keys_b
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a

    # 2×2 matrix
    concordant_keep = _make_cell()
    a_only_keep = _make_cell()
    b_only_keep = _make_cell()
    concordant_deferred = _make_cell()

    raw_decision_a: dict[str, int] = {}
    raw_decision_b: dict[str, int] = {}

    for key in sorted(matched_keys):
        ra = records_a[key]
        rb = records_b[key]
        da = _collapse(ra.decision)
        db = _collapse(rb.decision)

        raw_decision_a[ra.decision] = raw_decision_a.get(ra.decision, 0) + 1
        raw_decision_b[rb.decision] = raw_decision_b.get(rb.decision, 0) + 1

        if da == "KEEP" and db == "KEEP":
            _update_cell(concordant_keep, ra, rb)
        elif da == "KEEP" and db == "DEFERRED":
            _update_cell(a_only_keep, ra, rb)
        elif da == "DEFERRED" and db == "KEEP":
            _update_cell(b_only_keep, ra, rb)
        else:
            _update_cell(concordant_deferred, ra, rb)

    for cell in [concordant_keep, a_only_keep, b_only_keep, concordant_deferred]:
        _fill_far(cell)

    n_matched = len(matched_keys)

    # Per-policy deferral rate (on matched set)
    def _deferral_rate(records: dict, keys) -> float | None:
        if not keys:
            return None
        return sum(1 for k in keys if _collapse(records[k].decision) == "DEFERRED") / len(keys)

    deferral_rate_a = _deferral_rate(records_a, matched_keys)
    deferral_rate_b = _deferral_rate(records_b, matched_keys)

    # disagreement_far_a_over_b:
    # Of chunks A accepted (KEEP) that B correctly deferred, what fraction were
    # bad (keep_ok=False)?  Requires keep_ok labels.
    labeled_a_only = a_only_keep["n_keep_ok_true"] + a_only_keep["n_keep_ok_false"]
    disagreement_far_a_over_b = (
        a_only_keep["n_keep_ok_false"] / labeled_a_only
        if labeled_a_only > 0
        else None
    )
    labeled_b_only = b_only_keep["n_keep_ok_true"] + b_only_keep["n_keep_ok_false"]
    disagreement_far_b_over_a = (
        b_only_keep["n_keep_ok_false"] / labeled_b_only
        if labeled_b_only > 0
        else None
    )

    return {
        "id_a": id_a,
        "id_b": id_b,
        "n_matched": n_matched,
        "n_only_a": len(only_a),
        "n_only_b": len(only_b),
        "matrix": {
            "concordant_keep": concordant_keep,
            "a_only_keep": a_only_keep,
            "b_only_keep": b_only_keep,
            "concordant_deferred": concordant_deferred,
        },
        "deferral_rate_a": deferral_rate_a,
        "deferral_rate_b": deferral_rate_b,
        "raw_decision_breakdown_a": raw_decision_a,
        "raw_decision_breakdown_b": raw_decision_b,
        "disagreement_far_a_over_b": disagreement_far_a_over_b,
        "disagreement_far_b_over_a": disagreement_far_b_over_a,
    }


# ── I/O ────────────────────────────────────────────────────────────────────────

def write_matrix_csv(report: dict, out_dir: Path) -> None:
    """Write 2×2 confusion matrix as CSV."""
    path = out_dir / "compare_matrix.csv"
    matrix = report["matrix"]
    id_a = report["id_a"]
    id_b = report["id_b"]

    def _far(cell: dict) -> str:
        v = cell["false_accept_rate"]
        return f"{v:.4f}" if v is not None else ""

    fieldnames = ["", f"{id_b} KEEP", f"{id_b} DEFERRED"]
    rows = [
        {
            "": f"{id_a} KEEP",
            f"{id_b} KEEP": (
                f"n={matrix['concordant_keep']['n']} "
                f"far={_far(matrix['concordant_keep'])}"
            ),
            f"{id_b} DEFERRED": (
                f"n={matrix['a_only_keep']['n']} "
                f"far={_far(matrix['a_only_keep'])}"
            ),
        },
        {
            "": f"{id_a} DEFERRED",
            f"{id_b} KEEP": (
                f"n={matrix['b_only_keep']['n']} "
                f"far={_far(matrix['b_only_keep'])}"
            ),
            f"{id_b} DEFERRED": (
                f"n={matrix['concordant_deferred']['n']} "
                f"far={_far(matrix['concordant_deferred'])}"
            ),
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[matrix]  {path}")


def write_compare_report(report: dict, out_dir: Path) -> None:
    path = out_dir / "compare_report.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"[report]  {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records_a = _load_indexed(args.policy_a)
    records_b = _load_indexed(args.policy_b)

    print("\n=== Input ===")
    print(f"  {args.id_a}  : {len(records_a)} records  ({args.policy_a})")
    print(f"  {args.id_b}  : {len(records_b)} records  ({args.policy_b})")

    report = compare(records_a, records_b, args.id_a, args.id_b)

    print(f"\n=== Matched pairs: {report['n_matched']} ===")
    print(f"  only in {args.id_a}: {report['n_only_a']}")
    print(f"  only in {args.id_b}: {report['n_only_b']}")

    matrix = report["matrix"]
    id_a = args.id_a
    id_b = args.id_b
    print(f"\n  2×2 KEEP/DEFERRED matrix (rows={id_a}, cols={id_b})")
    print(f"                    {id_b} KEEP          {id_b} DEFERRED")

    def _cell_str(cell: dict) -> str:
        far = cell["false_accept_rate"]
        far_str = f"  far={far:.3f}" if far is not None else ""
        return f"n={cell['n']:>5}{far_str}"

    print(f"  {id_a} KEEP      {_cell_str(matrix['concordant_keep']):<30} {_cell_str(matrix['a_only_keep'])}")
    print(f"  {id_a} DEFERRED  {_cell_str(matrix['b_only_keep']):<30} {_cell_str(matrix['concordant_deferred'])}")

    dr_a = report["deferral_rate_a"]
    dr_b = report["deferral_rate_b"]
    print(f"\n  deferral_rate  {id_a}: {dr_a:.3f}" if dr_a is not None else f"\n  deferral_rate  {id_a}: —")
    print(f"  deferral_rate  {id_b}: {dr_b:.3f}" if dr_b is not None else f"  deferral_rate  {id_b}: —")

    dfar_ab = report["disagreement_far_a_over_b"]
    dfar_ba = report["disagreement_far_b_over_a"]
    if dfar_ab is not None:
        print(f"\n  disagreement FAR ({id_a} over {id_b}): {dfar_ab:.3f}")
    else:
        print(f"\n  disagreement FAR ({id_a} over {id_b}): — (no labels)")
    if dfar_ba is not None:
        print(f"  disagreement FAR ({id_b} over {id_a}): {dfar_ba:.3f}")
    else:
        print(f"  disagreement FAR ({id_b} over {id_a}): — (no labels)")

    write_matrix_csv(report, out_dir)
    write_compare_report(report, out_dir)


if __name__ == "__main__":
    main()
