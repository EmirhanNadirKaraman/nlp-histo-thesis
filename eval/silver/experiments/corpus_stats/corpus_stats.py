#!/usr/bin/env python3
"""
corpus_stats.py — descriptive (zero-API) statistics over pipeline output.

Reads from either source (no LLM spend, no model loading):
  * ``--source files`` (default) — on-disk per-paper summary JSONs
    (``out/summaries/summaries/*.json``) + ``corpus_relations.json``. Works with
    no database; the default dir is the related15 calibration cluster.
  * ``--source db`` — the summarisation tables (sum_* + pipeline_runs).
Both paths produce identical CSV/markdown/PNG output.

Two report families:

  1. FUNNEL / YIELD — per-paper attrition through the knowledge_extraction cascade
     (MAP pre-grounding → MAP post-grounding → NORMALIZE → GROUP → CANONICALIZE
     → RESOLVE), plus grounding / non-groupable rejection breakdowns, the
     dedup factor (raw findings collapsed per CanonicalRule) and the overall
     compression ratio (FinalRules per pre-grounding MAP finding).

  2. KNOWLEDGE GRAPH — the cross-/intra-paper relation graph from the latest
     CorpusRelateStage run: SUPPORT vs CONTRADICT edge counts, node/edge totals,
     connected components, hub entities, the most-corroborated rules, and the
     contradiction pairs.

Outputs land under ``eval/reports/corpus_stats/`` (override with ``--out``): one CSV
per table, an optional PNG per chart, and a ``corpus_stats.md`` headline summary.

Usage
-----
    python -m eval.silver.experiments.corpus_stats.corpus_stats                       # latest successful run per paper
    python -m eval.silver.experiments.corpus_stats.corpus_stats                       # files mode (default)
    python -m eval.silver.experiments.corpus_stats.corpus_stats --no-plots            # CSV + markdown only
    python -m eval.silver.experiments.corpus_stats.corpus_stats --summaries-dir out/summaries/summaries
    python -m eval.silver.experiments.corpus_stats.corpus_stats --source db           # read the database instead
    python -m eval.silver.experiments.corpus_stats.corpus_stats --source db --corpus-run-id 20260617T101500
    python -m eval.silver.experiments.corpus_stats.corpus_stats --out out/my_stats --top 30

Run from the repo root. ``--source db`` reads DB config from .env
(see database/db_connection.py); files mode needs no database.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import func  # noqa: E402

from nlp_histo.database import (  # noqa: E402
    get_db_connection,
    PipelineRun,
    SumMapFinding,
    SumNormalFinding,
    SumFindingGroup,
    SumGroupMember,
    SumCanonicalRule,
    SumFinalRule,
    SumRejectionSummary,
    SumCorpusRelation,
)

# Ordered funnel stages: (column key, human label). The attrition plot/CSV walk
# these in order; map_pre is the denominator for fraction-of-MAP normalisation.
FUNNEL_STAGES = [
    ("map_pre", "MAP (pre-grounding)"),
    ("map_post", "MAP (post-grounding)"),
    ("normal", "NORMALIZE"),
    ("group_members", "GROUP (members)"),
    ("canonical", "CANONICALIZE"),
    ("final", "RESOLVE (final rules)"),
]


# run selection


def select_runs(session, status: str, all_runs: bool):
    """Return the PipelineRun rows to analyse.

    Default: the latest run per pmcid whose status == ``status`` (one row per
    paper, so per-paper means are not skewed by re-runs). ``--all-runs`` keeps
    every run regardless of status.
    """
    q = session.query(PipelineRun)
    if not all_runs:
        q = q.filter(PipelineRun.status == status)
    runs = q.order_by(PipelineRun.pmcid, PipelineRun.started_at.desc()).all()
    if all_runs:
        return runs
    latest: dict[str, PipelineRun] = {}
    for r in runs:  # ordered started_at DESC → first seen per pmcid is latest
        latest.setdefault(r.pmcid, r)
    return list(latest.values())


def _count_by_run(session, model):
    """{pipeline_run_id: row_count} for a model with a pipeline_run_id column."""
    rows = (
        session.query(model.pipeline_run_id, func.count())
        .group_by(model.pipeline_run_id)
        .all()
    )
    return {rid: n for rid, n in rows}


# funnel


def build_funnel(session, runs):
    """Per-run funnel dicts + a corpus-level aggregate dict."""
    run_ids = [r.id for r in runs]
    if not run_ids:
        return [], {}

    map_post = _count_by_run(session, SumMapFinding)
    normal = _count_by_run(session, SumNormalFinding)
    groups = _count_by_run(session, SumFindingGroup)
    canonical = _count_by_run(session, SumCanonicalRule)
    final = _count_by_run(session, SumFinalRule)

    # GROUP members: join member → group to attribute the run.
    member_rows = (
        session.query(SumFindingGroup.pipeline_run_id, func.count(SumGroupMember.id))
        .join(SumGroupMember, SumGroupMember.finding_group_id == SumFindingGroup.id)
        .group_by(SumFindingGroup.pipeline_run_id)
        .all()
    )
    group_members = {rid: n for rid, n in member_rows}

    # Avg findings collapsed per CanonicalRule = dedup factor.
    dedup_rows = (
        session.query(SumCanonicalRule.pipeline_run_id, func.avg(SumCanonicalRule.finding_count))
        .group_by(SumCanonicalRule.pipeline_run_id)
        .all()
    )
    dedup = {rid: (float(a) if a is not None else None) for rid, a in dedup_rows}

    # Rejection summaries (one row per run; older runs may lack one).
    rej = {
        rs.pipeline_run_id: rs
        for rs in session.query(SumRejectionSummary)
        .filter(SumRejectionSummary.pipeline_run_id.in_(run_ids))
        .all()
    }

    per_run = []
    for r in runs:
        rid = r.id
        rs = rej.get(rid)
        mpost = map_post.get(rid, 0)
        mpre = rs.map_findings_total if rs else mpost  # fall back when no summary
        ground_rej = rs.map_grounding_rejected if rs else 0
        row = {
            "pmcid": r.pmcid,
            "run_id": r.run_id,
            "status": r.status,
            "map_pre": mpre,
            "map_post": mpost,
            "grounding_rejected": ground_rej,
            "grounding_rejection_rate": (rs.grounding_rejection_rate if rs else None),
            "normal": normal.get(rid, 0),
            "group_members": group_members.get(rid, 0),
            "finding_groups": groups.get(rid, 0),
            "non_groupable": (rs.non_groupable_total if rs else None),
            "non_groupable_rate": (rs.non_groupable_rate if rs else None),
            "canonical": canonical.get(rid, 0),
            "final": final.get(rid, 0),
            "dedup_factor": dedup.get(rid),
            "compression_ratio": (final.get(rid, 0) / mpre if mpre else None),
        }
        per_run.append(row)

    agg = _aggregate_funnel(per_run)
    return per_run, agg


def _aggregate_funnel(per_run):
    n = len(per_run)
    if not n:
        return {}
    totals = {key: sum(r[key] or 0 for r in per_run) for key, _ in FUNNEL_STAGES}
    map_pre_tot = totals.get("map_pre") or 0
    map_post_tot = totals.get("map_post") or 0
    normal_tot = totals.get("normal") or 0
    final_tot = totals.get("final") or 0
    nongroup_tot = sum(r["non_groupable"] or 0 for r in per_run)

    # Mean per-paper fraction of pre-grounding MAP at each stage (papers with
    # map_pre==0 are excluded from the fraction average for that stage).
    frac_mean = {}
    for key, _ in FUNNEL_STAGES:
        fracs = [r[key] / r["map_pre"] for r in per_run if r["map_pre"]]
        frac_mean[key] = statistics.mean(fracs) if fracs else None

    # Pooled fraction of pre-grounding MAP = ratio of corpus sums (NOT mean of
    # per-paper ratios). This is what the thesis funnel reports (E04 / §02 / §05 /
    # §06) — e.g. grounding retention 1923/2294 = 83.8 %. The report shows pooled
    # (so it agrees with the thesis headline); the per-paper mean stays in
    # funnel_corpus.csv for the per-paper view.
    frac_pooled = {
        key: (totals[key] / map_pre_tot if map_pre_tot else None)
        for key, _ in FUNNEL_STAGES
    }

    comp = [r["compression_ratio"] for r in per_run if r["compression_ratio"] is not None]
    dedup = [r["dedup_factor"] for r in per_run if r["dedup_factor"] is not None]
    grej = [r["grounding_rejection_rate"] for r in per_run if r["grounding_rejection_rate"] is not None]
    ngrej = [r["non_groupable_rate"] for r in per_run if r["non_groupable_rate"] is not None]

    return {
        "n_papers": n,
        "totals": totals,
        "frac_mean": frac_mean,
        "frac_pooled": frac_pooled,
        "compression_ratio_median": statistics.median(comp) if comp else None,
        "compression_ratio_mean": statistics.mean(comp) if comp else None,
        "compression_ratio_pooled": (final_tot / map_pre_tot) if map_pre_tot else None,
        "dedup_factor_mean": statistics.mean(dedup) if dedup else None,
        "grounding_rejection_rate_mean": statistics.mean(grej) if grej else None,
        "grounding_rejection_rate_pooled": (
            (map_pre_tot - map_post_tot) / map_pre_tot if map_pre_tot else None
        ),
        "non_groupable_rate_mean": statistics.mean(ngrej) if ngrej else None,
        "non_groupable_rate_pooled": (nongroup_tot / normal_tot) if normal_tot else None,
    }


def aggregate_rejections(session, runs):
    """Corpus-wide grounding-by-category + non-groupable-by-reason tallies."""
    run_ids = [r.id for r in runs]
    by_category: dict[str, int] = defaultdict(int)
    reasons = {"no_subject": 0, "no_outcome": 0, "unclear_relation": 0}
    if not run_ids:
        return by_category, reasons
    for rs in (
        session.query(SumRejectionSummary)
        .filter(SumRejectionSummary.pipeline_run_id.in_(run_ids))
        .all()
    ):
        for cat, n in (rs.grounding_rejected_by_category or {}).items():
            by_category[cat] += int(n)
        reasons["no_subject"] += rs.non_groupable_no_subject or 0
        reasons["no_outcome"] += rs.non_groupable_no_outcome or 0
        reasons["unclear_relation"] += rs.non_groupable_unclear_relation or 0
    return by_category, reasons


def build_funnel_from_files(summaries_dir: Path, status: str, all_runs: bool):
    """Funnel + rejection tallies from on-disk per-paper summary JSONs.

    One file == one paper (the dir is already latest-per-pmcid), so no run
    selection is needed. Stage counts are reconstructed from the persisted
    ``rejection_summary`` + ``canonical_rules`` + ``final_rules`` blocks, giving
    the same per-run schema as the DB path so the writers/plots are shared.
    """
    per_run = []
    by_category: dict[str, int] = defaultdict(int)
    reasons = {"no_subject": 0, "no_outcome": 0, "unclear_relation": 0}

    for path in sorted(summaries_dir.glob("*.json")):
        d = json.loads(path.read_text())
        if not all_runs and d.get("status") != status:
            continue
        rs = d.get("rejection_summary") or {}
        canon = d.get("canonical_rules") or []
        final = d.get("final_rules") or []

        mpre = rs.get("map_findings_total")
        grej = rs.get("map_grounding_rejected") or 0
        normal = rs.get("normal_findings_total")
        nongroup = rs.get("non_groupable_total")
        members = (normal - nongroup) if (normal is not None and nongroup is not None) else None
        fcs = [c.get("finding_count") for c in canon if c.get("finding_count") is not None]

        per_run.append({
            "pmcid": d.get("pmcid"),
            "run_id": d.get("run_id"),
            "status": d.get("status"),
            "map_pre": mpre,
            "map_post": (mpre - grej) if mpre is not None else None,
            "grounding_rejected": grej,
            "grounding_rejection_rate": rs.get("grounding_rejection_rate"),
            "normal": normal,
            "group_members": members,
            "finding_groups": len({c.get("group_id") for c in canon}),
            "non_groupable": nongroup,
            "non_groupable_rate": rs.get("non_groupable_rate"),
            "canonical": len(canon),
            "final": len(final),
            "dedup_factor": (statistics.mean(fcs) if fcs else None),
            "compression_ratio": (len(final) / mpre if mpre else None),
        })

        for cat, n in (rs.get("grounding_rejected_by_category") or {}).items():
            by_category[cat] += int(n)
        reasons["no_subject"] += rs.get("non_groupable_no_subject") or 0
        reasons["no_outcome"] += rs.get("non_groupable_no_outcome") or 0
        reasons["unclear_relation"] += rs.get("non_groupable_unclear_relation") or 0

    return per_run, _aggregate_funnel(per_run), dict(by_category), reasons


# knowledge graph


class _UnionFind:
    """Minimal union-find for connected components (avoids a networkx dep)."""

    def __init__(self):
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def corpus_records_from_db(session, corpus_run_id: str | None):
    """Latest (or pinned) CorpusRelateStage edges as plain dicts + the run label."""
    if corpus_run_id is None:
        corpus_run_id = session.query(func.max(SumCorpusRelation.corpus_run_id)).scalar()
    if corpus_run_id is None:
        return [], None
    rows = (
        session.query(SumCorpusRelation)
        .filter(SumCorpusRelation.corpus_run_id == corpus_run_id)
        .all()
    )
    keys = ("rule_id_a", "rule_id_b", "relation_type", "comparison_scope", "same_paper",
            "subject_entity", "outcome_entity", "predicate_a", "predicate_b",
            "supporting_pmcids_a", "supporting_pmcids_b", "finding_count_a", "finding_count_b",
            "nli_score_a_to_b", "nli_score_b_to_a", "pmcid_a", "pmcid_b")
    return [{k: getattr(r, k) for k in keys} for r in rows], corpus_run_id


def corpus_records_from_file(path: str):
    """CorpusRelateStage edges from an on-disk corpus_relations.json + a label.

    The JSON ``relations`` entries already use the same field names as the
    SumCorpusRelation columns, so they feed build_graph unchanged.
    """
    p = Path(path)
    if not p.exists():
        return [], None
    d = json.loads(p.read_text())
    return d.get("relations", []), (d.get("generated_at") or p.name)


def build_graph(records, top: int, run_label):
    """Build graph stats from a list of relation dicts (DB rows or JSON edges)."""
    if not records:
        return {"corpus_run_id": run_label, "n_edges": 0}

    # Edge counts by (relation_type, comparison_scope).
    edge_counts: dict[tuple, int] = defaultdict(int)
    degree: dict[str, int] = defaultdict(int)
    uf = _UnionFind()
    rule_meta: dict[str, dict] = {}
    contradictions = []

    for e in records:
        edge_counts[(e["relation_type"], e["comparison_scope"])] += 1
        a, b = e["rule_id_a"], e["rule_id_b"]
        degree[a] += 1
        degree[b] += 1
        uf.union(a, b)

        for rid, pred, supp, fc in (
            (a, e["predicate_a"], e.get("supporting_pmcids_a"), e.get("finding_count_a")),
            (b, e["predicate_b"], e.get("supporting_pmcids_b"), e.get("finding_count_b")),
        ):
            m = rule_meta.setdefault(
                rid,
                {"subject": e["subject_entity"], "outcome": e["outcome_entity"],
                 "predicate": pred, "supporting_pmcids": set(), "finding_count": fc or 0},
            )
            if supp:
                m["supporting_pmcids"].update(supp)
            m["finding_count"] = max(m["finding_count"], fc or 0)

        if e["relation_type"] == "CONTRADICT":
            contradictions.append({
                "pmcid_a": e["pmcid_a"],
                "pmcid_b": e["pmcid_b"],
                "same_paper": e["same_paper"],
                "subject": e["subject_entity"],
                "outcome": e["outcome_entity"],
                "predicate_a": e["predicate_a"],
                "predicate_b": e["predicate_b"],
                "nli_a_to_b": e["nli_score_a_to_b"],
                "nli_b_to_a": e["nli_score_b_to_a"],
            })

    nodes = set(degree)
    components = defaultdict(int)
    for nd in nodes:
        components[uf.find(nd)] += 1
    comp_sizes = sorted(components.values(), reverse=True)

    top_hubs = sorted(
        ({"rule_id": rid, "degree": d,
          "subject": rule_meta.get(rid, {}).get("subject"),
          "outcome": rule_meta.get(rid, {}).get("outcome")}
         for rid, d in degree.items()),
        key=lambda x: x["degree"], reverse=True,
    )[:top]

    top_corroborated = sorted(
        ({"rule_id": rid, "subject": m["subject"], "outcome": m["outcome"],
          "predicate": m["predicate"], "n_supporting_pmcids": len(m["supporting_pmcids"]),
          "finding_count": m["finding_count"]}
         for rid, m in rule_meta.items()),
        key=lambda x: (x["n_supporting_pmcids"], x["finding_count"]), reverse=True,
    )[:top]

    contradictions.sort(key=lambda c: max(c["nli_a_to_b"], c["nli_b_to_a"]), reverse=True)

    return {
        "corpus_run_id": run_label,
        "n_edges": len(records),
        "n_nodes": len(nodes),
        "edge_counts": dict(edge_counts),
        "n_components": len(comp_sizes),
        "largest_component": comp_sizes[0] if comp_sizes else 0,
        "degree": dict(degree),
        "top_hubs": top_hubs,
        "top_corroborated": top_corroborated,
        "contradictions": contradictions[:top],
        "n_contradictions": sum(1 for e in records if e["relation_type"] == "CONTRADICT"),
    }


# output


def _write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def write_funnel_csvs(out: Path, per_run, agg, by_category, reasons):
    if per_run:
        _write_csv(out / "funnel_per_paper.csv", list(per_run[0].keys()), per_run)

    # Corpus attrition: stage, total, mean fraction-of-MAP.
    attrition = [
        {"stage": label, "key": key,
         "total": agg.get("totals", {}).get(key),
         "pooled_fraction_of_map": agg.get("frac_pooled", {}).get(key),
         "mean_fraction_of_map": agg.get("frac_mean", {}).get(key)}
        for key, label in FUNNEL_STAGES
    ]
    _write_csv(out / "funnel_corpus.csv",
               ["stage", "key", "total", "pooled_fraction_of_map", "mean_fraction_of_map"],
               attrition)

    _write_csv(out / "rejections_by_category.csv", ["category", "grounding_rejected"],
               [{"category": c, "grounding_rejected": n}
                for c, n in sorted(by_category.items(), key=lambda x: -x[1])])
    _write_csv(out / "rejections_by_reason.csv", ["reason", "count"],
               [{"reason": k, "count": v} for k, v in reasons.items()])


def write_graph_csvs(out: Path, g):
    _write_csv(out / "graph_edge_counts.csv", ["relation_type", "comparison_scope", "count"],
               [{"relation_type": rt, "comparison_scope": sc, "count": n}
                for (rt, sc), n in sorted(g["edge_counts"].items())])
    _write_csv(out / "graph_top_hubs.csv", ["rule_id", "degree", "subject", "outcome"], g["top_hubs"])
    _write_csv(out / "graph_top_corroborated.csv",
               ["rule_id", "subject", "outcome", "predicate", "n_supporting_pmcids", "finding_count"],
               g["top_corroborated"])
    _write_csv(out / "graph_contradictions.csv",
               ["pmcid_a", "pmcid_b", "same_paper", "subject", "outcome",
                "predicate_a", "predicate_b", "nli_a_to_b", "nli_b_to_a"],
               g["contradictions"])


def write_markdown(out: Path, agg, g):
    lines = ["# Corpus statistics — funnel + knowledge graph", ""]
    if agg:
        t = agg["totals"]
        lines += [
            f"**Papers analysed:** {agg['n_papers']}", "",
            "## Funnel (corpus totals)", "",
            "| Stage | Total | % of MAP (pooled) |",
            "|---|---|---|",
        ]
        for key, label in FUNNEL_STAGES:
            frac = agg["frac_pooled"].get(key)
            fp = f"{frac * 100:.1f}%" if frac is not None else "—"
            lines.append(f"| {label} | {t.get(key, 0)} | {fp} |")
        lines += [
            "",
            f"- Compression ratio (final / MAP-pre, pooled): "
            f"{_fmt(agg['compression_ratio_pooled'])}",
            f"- Mean dedup factor (raw findings / CanonicalRule): "
            f"{_fmt(agg['dedup_factor_mean'])}",
            f"- Grounding rejection rate (pooled): {_fmt(agg['grounding_rejection_rate_pooled'])}",
            f"- Non-groupable rate (pooled): {_fmt(agg['non_groupable_rate_pooled'])}",
            "",
        ]
    if g and g.get("n_edges"):
        lines += [
            "## Knowledge graph", "",
            f"- corpus_run_id: `{g['corpus_run_id']}`",
            f"- nodes (rules): {g['n_nodes']} · edges: {g['n_edges']}",
            f"- connected components: {g['n_components']} "
            f"(largest: {g['largest_component']} nodes)",
            f"- contradiction edges: {g['n_contradictions']}",
            "",
            "| relation_type | scope | count |",
            "|---|---|---|",
        ]
        for (rt, sc), n in sorted(g["edge_counts"].items()):
            lines.append(f"| {rt} | {sc} | {n} |")
        lines.append("")
    (out / "corpus_stats.md").write_text("\n".join(lines))


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


# plots


def make_plots(out: Path, agg, g):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[plots] matplotlib unavailable ({e}); skipping PNGs", file=sys.stderr)
        return

    if agg and agg.get("totals"):
        labels = [lab for _, lab in FUNNEL_STAGES]
        totals = [agg["totals"].get(k, 0) for k, _ in FUNNEL_STAGES]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(range(len(labels)), totals, color="#4C72B0")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("count (corpus total)")
        ax.set_title("Summarization funnel — signal attrition")
        for i, v in enumerate(totals):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "funnel.png", dpi=150)
        plt.close(fig)

    if g and g.get("degree"):
        degs = list(g["degree"].values())
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(degs, bins=range(1, max(degs) + 2), color="#55A868", align="left")
        ax.set_xlabel("node degree (relations per rule)")
        ax.set_ylabel("number of rules")
        ax.set_title(f"Knowledge-graph degree distribution "
                     f"({g['n_nodes']} nodes, {g['n_edges']} edges)")
        fig.tight_layout()
        fig.savefig(out / "graph_degree_hist.png", dpi=150)
        plt.close(fig)

    if g and g.get("edge_counts"):
        # SUPPORT vs CONTRADICT, split by scope.
        scopes = sorted({sc for _, sc in g["edge_counts"]})
        rtypes = sorted({rt for rt, _ in g["edge_counts"]})
        fig, ax = plt.subplots(figsize=(7, 5))
        width = 0.8 / max(len(rtypes), 1)
        for i, rt in enumerate(rtypes):
            vals = [g["edge_counts"].get((rt, sc), 0) for sc in scopes]
            ax.bar([x + i * width for x in range(len(scopes))], vals, width, label=rt)
        ax.set_xticks([x + width * (len(rtypes) - 1) / 2 for x in range(len(scopes))])
        ax.set_xticklabels(scopes)
        ax.set_ylabel("edge count")
        ax.set_title("Relation edges by type and scope")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "graph_edge_breakdown.png", dpi=150)
        plt.close(fig)


# main


def main():
    ap = argparse.ArgumentParser(description="Descriptive funnel + graph stats over pipeline output.")
    ap.add_argument("--source", choices=["files", "db"], default="files",
                    help="read on-disk summary JSONs (default) or the database")
    ap.add_argument("--out", default="eval/reports/corpus_stats", help="output directory")
    ap.add_argument("--status", default="success", help="run status to select (default: success)")
    ap.add_argument("--all-runs", action="store_true",
                    help="include every run/paper (default: status-filtered, latest per paper)")
    # files source
    ap.add_argument("--summaries-dir", default="out/summaries/summaries",
                    help="[files] per-paper summary JSON directory")
    ap.add_argument("--corpus-relations", default="out/summaries/corpus_relations.json",
                    help="[files] corpus_relations.json path")
    # db source
    ap.add_argument("--corpus-run-id", default=None,
                    help="[db] graph corpus_run_id (default: latest persisted)")
    ap.add_argument("--top", type=int, default=20, help="rows for hub/corroborated/contradiction tables")
    ap.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True,
                    help="write PNG charts (default: on)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.source == "files":
        summaries_dir = Path(args.summaries_dir)
        per_run, agg, by_category, reasons = build_funnel_from_files(
            summaries_dir, args.status, args.all_runs)
        print(f"[files] {len(per_run)} paper(s) from {summaries_dir}/ "
              f"({'all statuses' if args.all_runs else 'status=' + args.status}).")
        records, run_label = corpus_records_from_file(args.corpus_relations)
    else:
        db = get_db_connection()
        with db.session_scope() as session:
            runs = select_runs(session, args.status, args.all_runs)
            print(f"[db] selected {len(runs)} run(s) "
                  f"({'all' if args.all_runs else 'latest per paper, status=' + args.status}).")
            per_run, agg = build_funnel(session, runs)
            by_category, reasons = aggregate_rejections(session, runs)
            records, run_label = corpus_records_from_db(session, args.corpus_run_id)

    g = build_graph(records, args.top, run_label) if run_label is not None else None

    write_funnel_csvs(out, per_run, agg, by_category, reasons)
    if g and g.get("n_edges"):
        write_graph_csvs(out, g)
    write_markdown(out, agg, g)
    if args.plots:
        make_plots(out, agg, g)

    # Stdout summary.
    if agg:
        print(f"\nFunnel ({agg['n_papers']} papers):")
        for key, label in FUNNEL_STAGES:
            frac = agg["frac_pooled"].get(key)
            fp = f"{frac * 100:5.1f}% of MAP" if frac is not None else "        —"
            print(f"  {label:<24} {agg['totals'].get(key, 0):>7}  ({fp})")
        print(f"  compression ratio (pooled): {_fmt(agg['compression_ratio_pooled'])}")
        print(f"  dedup factor (mean):        {_fmt(agg['dedup_factor_mean'])}")
    if g is None:
        print("\nGraph: no corpus relations found "
              "(run CorpusRelateStage / check --corpus-relations path).")
    elif not g.get("n_edges"):
        print(f"\nGraph: corpus_run_id={g['corpus_run_id']} has 0 edges.")
    else:
        print(f"\nGraph (corpus_run_id={g['corpus_run_id']}): "
              f"{g['n_nodes']} nodes, {g['n_edges']} edges, "
              f"{g['n_components']} components (largest {g['largest_component']}), "
              f"{g['n_contradictions']} contradictions.")
    print(f"\nWrote CSVs + markdown{' + plots' if args.plots else ''} to {out}/")


if __name__ == "__main__":
    main()
