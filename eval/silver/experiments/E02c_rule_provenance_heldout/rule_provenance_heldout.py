#!/usr/bin/env python3
"""E02c — final-rule provenance carry-rate on the HELD-OUT set (RQ5 / RQ2).

Generalization counterpart to E02b (which measured the ``related15`` calibration
cluster over the DB ``sum_*`` tables). The held-out knowledge_extraction runs were never
persisted to the DB (offline replay), so this experiment reads the on-disk
pipeline summaries and resolves each rule's paragraph pointer against the DB.

Data source per paper: ``out/summaries/<set>/summaries/<pmcid>.json`` — the frozen
pipeline output (same ``pipeline_config_hash`` as E14, verified at run time),
carrying ``final_rules`` and ``canonical_rules``. Each ``CanonicalRule`` serializes
``representative_text_element_id`` — the paragraph pointer of its best-grounded
member (the exact "identifier of the paragraph it came from" that
``09_provenance_model.tex`` describes). The chain measured is:

  A. FinalRule -> CanonicalRule   ``final_rule.canonical_id`` resolves to a
                                  ``canonical_rules`` row in the same paper.
  B. CanonicalRule -> paragraph   ``representative_text_element_id`` is non-null.
  C. paragraph -> TextElement     that id resolves to a real ``text_elements`` row.
  D. TextElement -> Document      its Document PMCID == the rule's paper (same-paper).

Headline = fraction of final rules reaching a real source paragraph (A-C).
``carried & same-paper`` (A-D) is reported alongside.

This resolves via the canonical rule's representative paragraph rather than the
full member-span set E02b walked (per-finding spans were not persisted for
held-out); both establish "final rule -> >=1 real source paragraph". ``related15``
is re-measured here by the same JSON path as a cross-check against E02b's
DB-path 100%.

Read-only (JSON + one read-only DB query). No API, no pipeline re-run.

Usage:
  python -m eval.silver.experiments.E02c_rule_provenance_heldout.rule_provenance_heldout
"""
from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from eval.paths import REPO_ROOT as _REPO_ROOT  # noqa: E402

from dotenv import load_dotenv

load_dotenv(str(_REPO_ROOT / ".env"))

from nlp_histo.database import Document, TextElement, get_db_connection

REPORTS_DIR = _REPO_ROOT / "eval" / "reports"

# set label -> summaries glob (headline first).
SETS = {
    "heldout15": _REPO_ROOT / "out" / "summaries" / "heldout15" / "summaries" / "*.json",
    "related15": _REPO_ROOT / "out" / "summaries" / "summaries" / "*.json",
}

LEVELS = ["A_canonical", "B_rep_te", "C_text_element"]


def _bare_pmcid(pmcid: str) -> str:
    return pmcid.split("_", 1)[0]


def main() -> None:
    db = get_db_connection()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    with db.session_scope() as s:
        te_to_pmcid: dict[int, str] = {
            te_id: pmcid
            for te_id, pmcid in s.query(TextElement.id, Document.pmcid).join(
                Document, TextElement.document_id == Document.id
            )
        }
    valid_te = set(te_to_pmcid)

    per_paper: list[dict] = []
    set_totals: dict[str, dict] = {}
    config_hashes: dict[str, set] = defaultdict(set)

    for set_name, pattern in SETS.items():
        files = sorted(glob.glob(str(pattern)))
        fail_funnel: dict[str, int] = defaultdict(int)
        n_final = n_carried = cross_paper = 0
        n_with_mnids = 0  # final rules that also carry member_normal_ids (alt path)

        for fp in files:
            d = json.load(open(fp))
            config_hashes[set_name].add(d.get("pipeline_config_hash"))
            paper = _bare_pmcid(d.get("pmcid", Path(fp).stem))
            canon = {c["canonical_id"]: c for c in d.get("canonical_rules") or []}
            finals = d.get("final_rules") or []
            p_final = len(finals)
            p_carried = 0

            for fr in finals:
                if fr.get("member_normal_ids"):
                    n_with_mnids += 1
                # A: FinalRule -> CanonicalRule
                cr = canon.get(fr.get("canonical_id"))
                if cr is None:
                    fail_funnel["A_canonical"] += 1
                    continue
                # B: CanonicalRule -> representative paragraph pointer
                te_id = cr.get("representative_text_element_id")
                if te_id is None:
                    fail_funnel["B_rep_te"] += 1
                    continue
                # C: paragraph id -> real TextElement
                if te_id not in valid_te:
                    fail_funnel["C_text_element"] += 1
                    continue
                # D: same-paper (secondary integrity)
                p_carried += 1
                if _bare_pmcid(te_to_pmcid[te_id]) != paper:
                    cross_paper += 1

            n_final += p_final
            n_carried += p_carried
            per_paper.append({
                "set": set_name, "pmcid": paper,
                "final_rules": p_final, "carried": p_carried,
                "carry_rate": round(p_carried / p_final, 4) if p_final else None,
            })

        set_totals[set_name] = {
            "papers": len(files), "final_rules": n_final, "carried": n_carried,
            "carry_rate": round(n_carried / n_final, 4) if n_final else None,
            "fail_funnel": dict(fail_funnel), "cross_paper": cross_paper,
            "with_mnids": n_with_mnids,
        }

    # report
    def pct(n: int, d: int) -> str:
        return f"{n / d:7.2%}" if d else "    n/a"

    print("\n" + "=" * 72)
    print("E02c — final-rule provenance carry-rate (held-out; related15 cross-check)")
    print("=" * 72)
    for set_name in SETS:
        t = set_totals[set_name]
        hashes = config_hashes[set_name]
        print(f"\n[{set_name}]  {t['papers']} papers  config_hash={','.join(map(str, hashes))}")
        for r in [r for r in per_paper if r["set"] == set_name]:
            print(f"  {r['pmcid']:15} {r['final_rules']:>6} rules  "
                  f"carried {r['carried']:>6}  {pct(r['carried'], r['final_rules'])}")
        print(f"  {'TOTAL':15} {t['final_rules']:>6} rules  "
              f"carried {t['carried']:>6}  {pct(t['carried'], t['final_rules'])}")
        if sum(t["fail_funnel"].values()) == 0:
            print("  failure funnel: none — every final rule reaches a source paragraph.")
        else:
            for lvl in LEVELS:
                if t["fail_funnel"].get(lvl):
                    print(f"  fail {lvl:16} {t['fail_funnel'][lvl]:>6}")
        print(f"  cross-paper (carried, paragraph != rule PMCID): {t['cross_paper']}")
        print(f"  final rules also carrying member_normal_ids: {t['with_mnids']}/{t['final_rules']}")

    hl = set_totals["heldout15"]
    print("\nHeadline (held-out generalization):")
    print(f"  carry-rate {hl['carried']}/{hl['final_rules']}  {pct(hl['carried'], hl['final_rules'])}")

    # CSV
    out_dir = REPORTS_DIR / "E02c_rule_provenance_heldout"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"rule_provenance_heldout_{ts}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["set", "pmcid", "final_rules", "carried", "carry_rate"])
        w.writeheader()
        w.writerows(per_paper)
        for set_name in SETS:
            t = set_totals[set_name]
            w.writerow({"set": set_name, "pmcid": "TOTAL",
                        "final_rules": t["final_rules"], "carried": t["carried"],
                        "carry_rate": t["carry_rate"]})
    print(f"\nCSV -> {csv_path}\n")


if __name__ == "__main__":
    main()
