#!/usr/bin/env python3
"""E02b — final-rule provenance carry-rate (RQ2).

Quantifies the knowledge-extraction provenance claim of
``03_methodology/04_knowledge_extraction/09_provenance_model.tex``: every
emitted ``FinalRule`` can be traced back to at least one real source paragraph.

E02 measures *document/media* provenance (paragraphs' ``unique_path``, figure/
table caption+crop+bbox). This experiment measures the rule side, walking the
four-link chain the thesis documents, entirely over persisted ``sum_*`` rows:

  A. FinalRule  -> CanonicalRule   ``sum_final_rules.canonical_id``
                                    resolves to a ``sum_canonical_rules`` row
                                    in the same ``pipeline_run_id``.
  B. CanonicalRule -> NormalFinding ``member_normal_ids`` resolve to >=1
                                    ``sum_normal_findings`` row in the run.
  C. NormalFinding -> span          >=1 resolved NF has >=1
                                    ``sum_normal_finding_spans`` row with a
                                    non-null ``text_element_id``.
  D. span -> TextElement            that ``text_element_id`` resolves to a real
                                    ``text_elements`` row.
  E. TextElement -> Document        ``text_elements.document_id`` -> PMCID
                                    (FK NOT NULL — structurally guaranteed).

IDs (``canonical_id`` / ``normal_id``) are unique per ``pipeline_run_id``, so
every resolution is run-scoped. Provenance is a per-run structural property, so
the carry-rate is computed per run and the production run set (the
``related15`` papers) is headlined; the per-run table shows it is not
cherry-picked.

Headline metric   = fraction of final rules that reach >=1 real source paragraph.
Completeness (2nd) = fraction of ``member_normal_ids`` that resolve, and fraction
                     of spans carrying a non-null ``text_element_id`` — so a rule
                     that traces via 1 of N members is still visible.

Read-only DB queries; no API calls.

Usage:
  python -m eval.silver.experiments.E02b_rule_provenance.rule_provenance_metric
"""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone

from eval.paths import REPO_ROOT as _REPO_ROOT  # noqa: E402

from dotenv import load_dotenv

load_dotenv(str(_REPO_ROOT / ".env"))

from nlp_histo.database import Document, TextElement, get_db_connection
from nlp_histo.database.models import (
    PipelineRun,
    SumCanonicalRule,
    SumFinalRule,
    SumNormalFinding,
    SumNormalFindingSpan,
)

REPORTS_DIR = _REPO_ROOT / "eval" / "reports"

# The 15 ILP calibration papers (configs/paper_selection/related15.yaml).
RELATED15 = {
    "PMC10100421", "PMC10108312", "PMC11649512", "PMC11649514", "PMC12605734",
    "PMC2253716", "PMC3564399", "PMC4282325", "PMC4282326", "PMC4329418",
    "PMC6635746", "PMC6850320", "PMC7539961", "PMC7540531", "PMC9826086",
}

# The five ordered chain links; a rule fails at the first one that does not hold.
LEVELS = ["A_canonical", "B_normal", "C_span_te", "D_text_element", "E_document"]


def _bare_pmcid(pmcid: str) -> str:
    """`PMC10100421_HIS-82-393` -> `PMC10100421` (run pmcids carry a suffix)."""
    return pmcid.split("_", 1)[0]


def main() -> None:
    db = get_db_connection()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    with db.session_scope() as s:
        # Global paragraph identity: te_id -> document PMCID (bare).
        te_to_pmcid: dict[int, str] = {
            te_id: pmcid
            for te_id, pmcid in s.query(TextElement.id, Document.pmcid).join(
                Document, TextElement.document_id == Document.id
            )
        }
        valid_te = set(te_to_pmcid)

        # Only runs that actually emitted final rules carry rule-provenance.
        run_ids = [
            rid for (rid,) in s.query(SumFinalRule.pipeline_run_id).distinct()
        ]
        runs = {
            r.id: r
            for r in s.query(PipelineRun).filter(PipelineRun.id.in_(run_ids))
        }

        per_run: list[dict] = []
        fail_funnel: dict[str, int] = defaultdict(int)  # level -> #rules failing there
        n_mnid_total = n_mnid_resolved = 0
        n_span_total = n_span_with_te = 0
        cross_paper = 0  # carried rule whose paragraph PMCID != rule PMCID

        for rid, run in sorted(runs.items()):
            canon = {c.canonical_id: c for c in
                     s.query(SumCanonicalRule).filter_by(pipeline_run_id=rid)}
            nfs = {nf.normal_id: nf for nf in
                   s.query(SumNormalFinding).filter_by(pipeline_run_id=rid)}
            nf_ids = [nf.id for nf in nfs.values()]
            spans_by_nf: dict[int, list] = defaultdict(list)
            if nf_ids:
                for sp in s.query(SumNormalFindingSpan).filter(
                    SumNormalFindingSpan.normal_finding_id.in_(nf_ids)
                ):
                    spans_by_nf[sp.normal_finding_id].append(sp)

            finals = s.query(SumFinalRule).filter_by(pipeline_run_id=rid).all()
            carried = 0
            run_fail: dict[str, int] = defaultdict(int)

            for fr in finals:
                # A: FinalRule -> CanonicalRule
                cr = canon.get(fr.canonical_id)
                if cr is None:
                    run_fail["A_canonical"] += 1
                    continue
                # B: CanonicalRule -> NormalFinding(s)
                mnids = cr.member_normal_ids or []
                n_mnid_total += len(mnids)
                resolved = [nfs[m] for m in mnids if m in nfs]
                n_mnid_resolved += len(resolved)
                if not resolved:
                    run_fail["B_normal"] += 1
                    continue
                # C: NormalFinding -> span with a non-null text_element_id
                te_hits: list[int] = []
                for nf in resolved:
                    for sp in spans_by_nf.get(nf.id, []):
                        n_span_total += 1
                        if sp.text_element_id is not None:
                            n_span_with_te += 1
                            te_hits.append(sp.text_element_id)
                if not te_hits:
                    run_fail["C_span_te"] += 1
                    continue
                # D: span te_id -> real TextElement
                te_valid = [t for t in te_hits if t in valid_te]
                if not te_valid:
                    run_fail["D_text_element"] += 1
                    continue
                # E: TextElement -> Document (FK NOT NULL) — always resolves.
                carried += 1
                # Secondary integrity: does the paragraph belong to the rule's paper?
                rule_pmc = _bare_pmcid(fr.pmcid)
                if not any(_bare_pmcid(te_to_pmcid[t]) == rule_pmc for t in te_valid):
                    cross_paper += 1

            n_final = len(finals)
            for lvl, c in run_fail.items():
                fail_funnel[lvl] += c
            per_run.append({
                "pipeline_run_id": rid,
                "pmcid": _bare_pmcid(run.pmcid),
                "in_related15": _bare_pmcid(run.pmcid) in RELATED15,
                "final_rules": n_final,
                "carried": carried,
                "carry_rate": round(carried / n_final, 4) if n_final else None,
            })

    # aggregate
    total = sum(r["final_rules"] for r in per_run)
    carried_total = sum(r["carried"] for r in per_run)
    rel = [r for r in per_run if r["in_related15"]]
    rel_total = sum(r["final_rules"] for r in rel)
    rel_carried = sum(r["carried"] for r in rel)

    def pct(n: int, d: int) -> str:
        return f"{n / d:7.2%}" if d else "    n/a"

    print("\n" + "=" * 72)
    print("E02b — final-rule provenance carry-rate")
    print("=" * 72)
    print(f"\nPer-run carry-rate ({len(per_run)} runs with >=1 final rule):\n")
    print(f"  {'run':>4} {'pmcid':15} {'rel15':5} {'final':>6} {'carried':>8} {'rate':>8}")
    for r in per_run:
        print(f"  {r['pipeline_run_id']:>4} {r['pmcid']:15} "
              f"{'yes' if r['in_related15'] else '   ':5} "
              f"{r['final_rules']:>6} {r['carried']:>8} "
              f"{pct(r['carried'], r['final_rules'])}")

    print("\nFailure funnel (rule fails at first broken link):")
    if sum(fail_funnel.values()) == 0:
        print("  none — every final rule traces to >=1 source paragraph.")
    else:
        for lvl in LEVELS:
            if fail_funnel.get(lvl):
                print(f"  {lvl:16} {fail_funnel[lvl]:>6}")

    print("\nCompleteness (secondary):")
    print(f"  member_normal_ids resolved   {n_mnid_resolved:>6} / {n_mnid_total:<6} "
          f"{pct(n_mnid_resolved, n_mnid_total)}")
    print(f"  spans with text_element_id   {n_span_with_te:>6} / {n_span_total:<6} "
          f"{pct(n_span_with_te, n_span_total)}")
    print(f"  carried rules cross-paper    {cross_paper:>6} "
          f"(paragraph PMCID != rule PMCID)")

    print("\nHeadline:")
    print(f"  related15 production runs    {rel_carried:>6} / {rel_total:<6} "
          f"{pct(rel_carried, rel_total)}")
    print(f"  all runs with final rules    {carried_total:>6} / {total:<6} "
          f"{pct(carried_total, total)}")

    # CSV
    out_dir = REPORTS_DIR / "E02b_rule_provenance"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"rule_provenance_{ts}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "pipeline_run_id", "pmcid", "in_related15",
            "final_rules", "carried", "carry_rate",
        ])
        w.writeheader()
        w.writerows(per_run)
        w.writerow({
            "pipeline_run_id": "TOTAL_related15", "pmcid": "", "in_related15": True,
            "final_rules": rel_total, "carried": rel_carried,
            "carry_rate": round(rel_carried / rel_total, 4) if rel_total else None,
        })
        w.writerow({
            "pipeline_run_id": "TOTAL_all", "pmcid": "", "in_related15": "",
            "final_rules": total, "carried": carried_total,
            "carry_rate": round(carried_total / total, 4) if total else None,
        })
    print(f"\nCSV -> {csv_path}\n")


if __name__ == "__main__":
    main()
