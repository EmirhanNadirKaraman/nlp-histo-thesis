#!/usr/bin/env python3
"""
run_chapter9_offline_replay.py — close the offline gaps that block chapter 9.

Single-script driver that produces every analysis the claim/evidence map
lists as ``zero-API replay``. All outputs go to a deterministic directory:

    out/thesis_results/chapter9_offline_replay/

No external LLM API is called. All evidence comes from one of:
  * cached voter outputs in eval/data/map_primer/voter_cache.json
  * cached embeddings in eval/data/embedding_cache_openai.sqlite
  * on-disk run artefacts under out/summaries/{summaries,traces,cascade_decisions}
  * staged-sweep reports under reports/stage*_PR.md
  * existing sweep CSVs under eval/reports/

Run from the repository root:

    python3 scripts/thesis/run_chapter9_offline_replay.py

The script writes nine numbered CSV/JSON artefacts plus a manifest, a
README, and a results summary. It is idempotent: re-running overwrites
the outputs.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ── Path bootstrap ────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load .env so the matcher's embedder constructor can accept a valid
# API key. Even when the embedding cache is warm and no API call is
# actually issued, the constructor itself requires a non-empty key.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(str(_REPO_ROOT / ".env"))
except Exception:
    pass


# ── Output layout ─────────────────────────────────────────────────────────────
OUT_DIR = _REPO_ROOT / "out" / "thesis_results" / "chapter9_offline_replay"
LOG_DIR = OUT_DIR / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "run.log"


def _log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


@dataclass
class AnalysisResult:
    name: str
    status: str  # "ok", "partial", "missing", "error"
    outputs: list[Path] = field(default_factory=list)
    key_numbers: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    caveats: list[str] = field(default_factory=list)


RESULTS: list[AnalysisResult] = []


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    import csv as _csv
    with path.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


# ── Analysis 1: Provenance carry-rate ─────────────────────────────────────────
def analyse_provenance_carry_rate() -> AnalysisResult:
    """Fraction of FinalRules whose verbatim, NLI score, and DB pointer
    triple are populated. Reads ``out/summaries/summaries/*.json``.

    A FinalRule carries provenance iff:
      * the parent canonical rule has ``representative_verbatim`` non-empty,
      * the parent canonical rule has ``representative_text_element_id`` set,
      * the rule's grounding score (``mean_grounding_score`` or
        ``grounding_score``) is set on the rule itself or on its origin
        normal finding.
    """
    name = "01_provenance_carry_rate"
    summaries_dir = _REPO_ROOT / "out" / "summaries" / "summaries"
    files = sorted(summaries_dir.glob("*.json"))
    per_paper = []
    n_total = 0
    n_with_verbatim = 0
    n_with_te_id = 0
    n_with_nli = 0
    n_full_triple = 0

    if not files:
        return AnalysisResult(
            name=name, status="missing",
            notes="No per-paper summary JSON under out/summaries/summaries/.",
        )

    for fp in files:
        try:
            data = json.loads(fp.read_text())
        except Exception as exc:
            _log(f"[{name}] skip unreadable {fp.name}: {exc}")
            continue

        canon_rules = data.get("canonical_rules") or []
        final_rules = data.get("final_rules") or []
        canon_by_id = {c.get("canonical_id"): c for c in canon_rules}

        paper_total = len(final_rules)
        paper_v = paper_t = paper_g = paper_full = 0

        for fr in final_rules:
            canon = canon_by_id.get(fr.get("canonical_id")) or {}
            verbatim = canon.get("representative_verbatim")
            te_id = canon.get("representative_text_element_id")
            grounding = (
                fr.get("mean_grounding_score")
                or canon.get("mean_grounding_score")
            )
            has_v = bool(verbatim and str(verbatim).strip())
            has_t = te_id is not None
            has_g = grounding is not None
            if has_v:
                paper_v += 1
            if has_t:
                paper_t += 1
            if has_g:
                paper_g += 1
            if has_v and has_t and has_g:
                paper_full += 1

        n_total += paper_total
        n_with_verbatim += paper_v
        n_with_te_id += paper_t
        n_with_nli += paper_g
        n_full_triple += paper_full
        per_paper.append({
            "pmcid": data.get("pmcid", fp.stem),
            "n_final_rules": paper_total,
            "n_with_verbatim": paper_v,
            "n_with_text_element_id": paper_t,
            "n_with_grounding": paper_g,
            "n_with_full_provenance_triple": paper_full,
        })

    def _rate(n, d):
        return round(n / d, 4) if d else 0.0

    aggregate = {
        "n_papers": len(per_paper),
        "n_final_rules_total": n_total,
        "n_with_verbatim": n_with_verbatim,
        "n_with_text_element_id": n_with_te_id,
        "n_with_grounding": n_with_nli,
        "n_with_full_provenance_triple": n_full_triple,
        "rate_verbatim": _rate(n_with_verbatim, n_total),
        "rate_text_element_id": _rate(n_with_te_id, n_total),
        "rate_grounding": _rate(n_with_nli, n_total),
        "rate_full_provenance_triple": _rate(n_full_triple, n_total),
    }

    csv_path = OUT_DIR / f"{name}.csv"
    json_path = OUT_DIR / f"{name}.json"
    _write_csv(
        csv_path,
        header=[
            "pmcid", "n_final_rules", "n_with_verbatim",
            "n_with_text_element_id", "n_with_grounding",
            "n_with_full_provenance_triple",
        ],
        rows=[
            [r["pmcid"], r["n_final_rules"], r["n_with_verbatim"],
             r["n_with_text_element_id"], r["n_with_grounding"],
             r["n_with_full_provenance_triple"]]
            for r in per_paper
        ],
    )
    _write_json(json_path, {
        "name": name,
        "aggregate": aggregate,
        "per_paper": per_paper,
    })

    return AnalysisResult(
        name=name, status="ok",
        outputs=[csv_path, json_path],
        key_numbers=aggregate,
        notes=(
            f"Carry-rate computed over {n_total} FinalRules across "
            f"{len(per_paper)} papers. Source: out/summaries/summaries/*.json."
        ),
        caveats=[
            "Provenance fields (representative_verbatim, "
            "representative_text_element_id) were added 2026-05-27; "
            "older cached runs predating the field addition will show "
            "missing values even though the design contract holds.",
        ],
    )


# ── Analysis 2: Grounding retention rate ──────────────────────────────────────
def analyse_grounding_retention() -> AnalysisResult:
    """Per-paper grounding-filter retention at the production threshold
    (0.5). Reads pre- and post-grounding finding counts from
    ``rejection_summary`` blocks inside the per-paper summary JSON.
    """
    name = "02_grounding_retention"
    summaries_dir = _REPO_ROOT / "out" / "summaries" / "summaries"
    files = sorted(summaries_dir.glob("*.json"))
    if not files:
        return AnalysisResult(
            name=name, status="missing",
            notes="No per-paper summary JSON found.",
        )

    per_paper = []
    total_map = 0
    total_kept = 0
    total_dropped = 0

    for fp in files:
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        rej = data.get("rejection_summary") or {}
        map_total = rej.get("map_findings_total") or 0
        grounding_dropped = rej.get("map_grounding_rejected") or 0
        threshold_used = rej.get("grounding_threshold")
        kept_after_grounding = map_total - grounding_dropped if map_total else 0
        per_paper.append({
            "pmcid": data.get("pmcid", fp.stem),
            "map_total": map_total,
            "grounding_dropped": grounding_dropped,
            "kept_after_grounding": kept_after_grounding,
            "grounding_threshold_used": threshold_used,
            "retention_rate": round(kept_after_grounding / map_total, 4)
                              if map_total else None,
        })
        total_map += map_total
        total_kept += kept_after_grounding
        total_dropped += grounding_dropped

    # Collect the set of grounding thresholds actually used across papers.
    thresholds = sorted({
        p["grounding_threshold_used"] for p in per_paper
        if p["grounding_threshold_used"] is not None
    })
    aggregate = {
        "n_papers": len(per_paper),
        "map_findings_total": total_map,
        "grounding_dropped_total": total_dropped,
        "kept_after_grounding_total": total_kept,
        "retention_rate_overall": (
            round(total_kept / total_map, 4) if total_map else None
        ),
        "grounding_thresholds_observed": thresholds,
    }

    csv_path = OUT_DIR / f"{name}.csv"
    json_path = OUT_DIR / f"{name}.json"
    _write_csv(
        csv_path,
        header=["pmcid", "map_total", "grounding_dropped",
                "kept_after_grounding", "grounding_threshold_used",
                "retention_rate"],
        rows=[
            [r["pmcid"], r["map_total"], r["grounding_dropped"],
             r["kept_after_grounding"], r["grounding_threshold_used"],
             r["retention_rate"]]
            for r in per_paper
        ],
    )
    _write_json(json_path, {
        "name": name,
        "aggregate": aggregate,
        "per_paper": per_paper,
    })

    return AnalysisResult(
        name=name, status="ok",
        outputs=[csv_path, json_path],
        key_numbers=aggregate,
        notes=(
            "Retention is the fraction of MAP findings surviving the "
            "GROUNDING NLI filter at the configured threshold "
            "(production: 0.5). Read from rejection_summary blocks "
            "in each per-paper summary JSON."
        ),
        caveats=[
            "Per-run grounding threshold is read from "
            "rejection_summary.grounding_threshold; this script does NOT "
            "assume a single threshold. The retention rate is the "
            "cumulative (map_total − map_grounding_rejected) / map_total "
            "across papers.",
            "The on-disk per-paper summaries reflect the most recent run "
            "(haiku_only profile, single L1 voter, with the production "
            "grounding threshold of 0.5). The 100 % retention observed "
            "here is specific to that profile / threshold combination and "
            "should not be reported as the general grounding-filter "
            "retention rate without re-running under the production "
            "`real` profile (which uses 3 L1 voters whose entailment "
            "scores vary).",
        ],
    )


# ── Analysis 3: Polarity-conflict count from cascade decision logs ────────────
def analyse_polarity_conflict_counts() -> AnalysisResult:
    """Count POLARITY_CONFLICT reason codes per paper and per level
    from cascade-decision JSONL logs.
    """
    name = "03_polarity_conflict_counts"
    cas_dir = _REPO_ROOT / "out" / "summaries" / "cascade_decisions"
    files = sorted(cas_dir.glob("*.jsonl"))
    if not files:
        return AnalysisResult(
            name=name, status="missing",
            notes="No cascade-decision JSONL under out/summaries/cascade_decisions/.",
        )

    per_paper = []
    grand_chunks = 0
    grand_pc = 0
    by_level_total: Counter = Counter()
    by_decision_total: Counter = Counter()

    for fp in files:
        chunks_seen = 0
        pc_count = 0
        by_level: Counter = Counter()
        by_decision: Counter = Counter()
        for line in fp.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            chunks_seen += 1
            reason_codes = row.get("reason_codes") or []
            has_pc = any("POLARITY_CONFLICT" in str(rc) for rc in reason_codes)
            if has_pc:
                pc_count += 1
                level = row.get("level") or "unknown"
                dec = row.get("decision") or "unknown"
                by_level[level] += 1
                by_decision[dec] += 1
                by_level_total[level] += 1
                by_decision_total[dec] += 1
        per_paper.append({
            "pmcid": fp.stem,
            "n_chunk_decisions": chunks_seen,
            "n_polarity_conflict": pc_count,
            "polarity_conflict_rate": round(pc_count / chunks_seen, 4)
                                       if chunks_seen else None,
            "by_level": dict(by_level),
            "by_decision": dict(by_decision),
        })
        grand_chunks += chunks_seen
        grand_pc += pc_count

    # Also try to read polarity-conflict counts from the EXP F sweep CSV
    # (production cascade on the held-out test split) if it exists.
    sweep_pc = None
    sweep_chunks = None
    sweep_path = OUT_DIR / "06_exp_f_test_split.csv"
    if sweep_path.exists():
        import csv as _csv
        try:
            with sweep_path.open() as f:
                rows = list(_csv.DictReader(f))
            if rows:
                first = rows[0]
                sweep_pc = int(first.get("n_polarity_conflict_chunks") or 0)
                rate = float(first.get("polarity_conflict_rate") or 0.0)
                # Reconstruct total chunks from rate (rate = pc / total).
                if rate > 0:
                    sweep_chunks = round(sweep_pc / rate)
        except Exception as exc:
            _log(f"[{name}] failed to parse EXP F polarity stats: {exc}")

    aggregate = {
        "n_papers": len(per_paper),
        "n_chunk_decisions_total": grand_chunks,
        "n_polarity_conflict_total": grand_pc,
        "polarity_conflict_rate_overall": (
            round(grand_pc / grand_chunks, 4) if grand_chunks else None
        ),
        "by_level": dict(by_level_total),
        "by_decision": dict(by_decision_total),
        "exp_f_test_split_polarity_conflicts": sweep_pc,
        "exp_f_test_split_total_chunks_est": sweep_chunks,
    }

    csv_path = OUT_DIR / f"{name}.csv"
    _write_csv(
        csv_path,
        header=["pmcid", "n_chunk_decisions", "n_polarity_conflict",
                "polarity_conflict_rate"],
        rows=[
            [r["pmcid"], r["n_chunk_decisions"], r["n_polarity_conflict"],
             r["polarity_conflict_rate"]]
            for r in per_paper
        ],
    )
    _write_json(OUT_DIR / f"{name}.json", {
        "name": name,
        "aggregate": aggregate,
        "per_paper": per_paper,
    })

    return AnalysisResult(
        name=name, status="ok",
        outputs=[csv_path, OUT_DIR / f"{name}.json"],
        key_numbers=aggregate,
        notes=(
            "Counts chunks where the agreement gate emitted "
            "POLARITY_CONFLICT in reason_codes (the B-051 hard-fail "
            "polarity-conflict policy). Source: cascade_decisions/*.jsonl."
        ),
        caveats=[
            "The hard-fail policy is on by default. With the flag "
            "disabled the same chunks would still appear here as the "
            "marker is recorded independently of the decision override.",
            "The on-disk cascade decisions JSONL come from `haiku_only` "
            "profile runs (single L1 voter per chunk). With N=1 voter no "
            "pairwise polarity conflict can be detected; the 0 / 230 "
            "count is therefore profile-specific and is NOT the "
            "polarity-conflict frequency under the production `real` "
            "profile. The EXP F test-split sweep (analysis 06) records "
            "non-zero polarity-conflict counts under the production "
            "scorer + cascade — see `exp_f_test_split_polarity_conflicts` "
            "in this analysis's aggregate JSON.",
        ],
    )


# ── Analysis 4: Theta heatmap from sweep CSV ──────────────────────────────────
def analyse_theta_heatmap() -> AnalysisResult:
    """Heatmap of strict_f1 over (theta, reject_theta) for one scorer,
    extracted from the existing joint-sweep CSV in eval/reports/.
    """
    name = "04_theta_heatmap"
    reports_dir = _REPO_ROOT / "eval" / "reports"
    # Prefer the OpenAI-branch best-per CSV (production branch); fall back
    # to gemini if openai is missing.
    candidates = sorted(reports_dir.glob("exp_1_gemini_scorer_full_*.csv")) \
                 + sorted(reports_dir.glob("exp_4_openai_scorer_full_*.csv"))
    if not candidates:
        return AnalysisResult(
            name=name, status="missing",
            notes="No exp_1/exp_4 full-sweep CSV in eval/reports/.",
        )

    src = candidates[0]
    import csv as _csv
    rows: list[dict[str, str]] = []
    with src.open() as f:
        for r in _csv.DictReader(f):
            rows.append(r)

    if not rows:
        return AnalysisResult(
            name=name, status="missing",
            notes=f"Sweep CSV {src.name} is empty.",
        )

    # Pick the most populous scorer label so the heatmap has the most cells.
    by_scorer: Counter = Counter(r.get("scorer", "?") for r in rows)
    primary_scorer = by_scorer.most_common(1)[0][0]
    cells: dict[tuple[float, float], float] = {}
    for r in rows:
        if r.get("scorer") != primary_scorer:
            continue
        try:
            theta = float(r["theta"])
            reject = float(r["reject_theta"])
            sf1 = float(r["strict_f1"])
        except (KeyError, ValueError):
            continue
        # Some sweeps include duplicate (theta, reject_theta) under different
        # hyper-parameters (e.g. agreement weights). Keep the maximum cell.
        if (theta, reject) in cells:
            cells[(theta, reject)] = max(cells[(theta, reject)], sf1)
        else:
            cells[(theta, reject)] = sf1

    thetas = sorted({k[0] for k in cells})
    rejects = sorted({k[1] for k in cells})

    csv_path = OUT_DIR / f"{name}.csv"
    _write_csv(
        csv_path,
        header=["theta", "reject_theta", "strict_f1"],
        rows=[
            [theta, reject, cells[(theta, reject)]]
            for theta in thetas
            for reject in rejects
            if (theta, reject) in cells
        ],
    )

    png_path = OUT_DIR / f"{name}.png"
    plot_status = "no_plot"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        Z = np.full((len(rejects), len(thetas)), np.nan)
        for i, rj in enumerate(rejects):
            for j, th in enumerate(thetas):
                if (th, rj) in cells:
                    Z[i, j] = cells[(th, rj)]
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(Z, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(thetas)))
        ax.set_xticklabels([f"{t:.2g}" for t in thetas])
        ax.set_yticks(range(len(rejects)))
        ax.set_yticklabels([f"{r:.2g}" for r in rejects])
        ax.set_xlabel(r"$\theta$ (accept threshold)")
        ax.set_ylabel(r"$\theta_{\mathrm{reject}}$ (reject threshold)")
        ax.set_title(f"Strict F1 heatmap — scorer = {primary_scorer}\n"
                     f"source = {src.name}")
        fig.colorbar(im, ax=ax, label="strict_f1")
        for i in range(len(rejects)):
            for j in range(len(thetas)):
                v = Z[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.7 else "black", fontsize=7)
        fig.tight_layout()
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
        plot_status = "plot_ok"
    except Exception as exc:
        _log(f"[{name}] plotting failed: {exc}")

    aggregate = {
        "source_csv": src.name,
        "n_scorers_in_csv": len(by_scorer),
        "primary_scorer_used": primary_scorer,
        "n_cells_plotted": len(cells),
        "thetas": thetas,
        "reject_thetas": rejects,
        "best_cell": {
            "theta": max(cells, key=cells.get)[0],
            "reject_theta": max(cells, key=cells.get)[1],
            "strict_f1": cells[max(cells, key=cells.get)],
        },
        "plot_status": plot_status,
    }
    _write_json(OUT_DIR / f"{name}.json", aggregate)

    outputs = [csv_path, OUT_DIR / f"{name}.json"]
    if png_path.exists():
        outputs.append(png_path)
    return AnalysisResult(
        name=name, status="ok",
        outputs=outputs,
        key_numbers=aggregate,
        notes=(
            "Heatmap built from an existing joint sweep over "
            "(theta, reject_theta, scorer). Cells are the per-scorer "
            "max strict_f1 when multiple hyperparameter rows share a "
            "cell. Best cell is *best on the dev split for that "
            "scorer* — not optimal in a held-out sense."
        ),
        caveats=[
            "The source CSV reports n_silver = 1596 (all-cases silver, "
            "any split). EXP B.2's CSV reports n_silver = 1243 (dev-split "
            "silver findings only). The heatmap reflects relative ordering "
            "across (theta, reject_theta) cells from the source CSV; "
            "absolute strict_f1 values should be cross-referenced against "
            "the EXP B.2 dev-split numbers in analysis 05.",
        ],
    )


# ── Analysis 5: Bootstrap CI cascade vs Sonnet ────────────────────────────────
STRICT_FIELDS = {"category", "relation_type", "direction"}


def _b2_match_per_case(case_outputs, silver_by_case, embedder, embed_cache,
                       sim_threshold=0.55):
    """Like ``_b2_eval_outputs`` but returns per-case (strict_tp, n_silver, n_pipeline).
    """
    from eval.silver.matcher import compute_sim_matrix, match_from_matrix
    per_case = []
    for co in case_outputs:
        silver = silver_by_case.get(co.case_id)
        if silver is None:
            continue
        try:
            sim, _, _ = compute_sim_matrix(silver, co, embedder, embed_cache)
            result = match_from_matrix(silver, co, sim, sim_threshold)
        except Exception as exc:
            _log(f"[bootstrap] skip {co.case_id}: {exc}")
            continue
        s_tp = 0.0
        for pair in result.matched:
            has_strict_mismatch = any(
                m.field in STRICT_FIELDS for m in pair.field_mismatches
            )
            s_tp += 0.5 if has_strict_mismatch else 1.0
        per_case.append({
            "case_id": co.case_id,
            "strict_tp": s_tp,
            "n_silver": len(silver.findings),
            "n_pipeline": len(co.findings),
            "n_matched": len(result.matched),
        })
    return per_case


def _aggregate_strict_f1(per_case):
    s_tp = sum(r["strict_tp"] for r in per_case)
    n_p = sum(r["n_pipeline"] for r in per_case)
    n_s = sum(r["n_silver"] for r in per_case)
    if n_p == 0 or n_s == 0:
        return 0.0
    sp = s_tp / n_p
    sr = s_tp / n_s
    return (2 * sp * sr / (sp + sr)) if (sp + sr) else 0.0


def _bootstrap_paired(per_case_a, per_case_b, n_iters=2000, seed=42):
    import numpy as np
    rng = np.random.RandomState(seed)
    by_case_a = {r["case_id"]: r for r in per_case_a}
    by_case_b = {r["case_id"]: r for r in per_case_b}
    common = sorted(set(by_case_a) & set(by_case_b))
    if not common:
        return None
    n = len(common)
    diffs = []
    a_samples = []
    b_samples = []
    for _ in range(n_iters):
        idx = rng.choice(n, n, replace=True)
        sampled_a = [by_case_a[common[i]] for i in idx]
        sampled_b = [by_case_b[common[i]] for i in idx]
        fa = _aggregate_strict_f1(sampled_a)
        fb = _aggregate_strict_f1(sampled_b)
        a_samples.append(fa)
        b_samples.append(fb)
        diffs.append(fa - fb)
    diffs.sort()
    a_samples.sort()
    b_samples.sort()
    def pct(xs, p):
        return xs[int(p * (len(xs) - 1))]
    return {
        "n_cases": n,
        "n_bootstrap_iters": n_iters,
        "a_strict_f1_mean": round(sum(a_samples) / n_iters, 4),
        "b_strict_f1_mean": round(sum(b_samples) / n_iters, 4),
        "a_ci95": [round(pct(a_samples, 0.025), 4), round(pct(a_samples, 0.975), 4)],
        "b_ci95": [round(pct(b_samples, 0.025), 4), round(pct(b_samples, 0.975), 4)],
        "gap_mean": round(sum(diffs) / n_iters, 4),
        "gap_ci95": [round(pct(diffs, 0.025), 4), round(pct(diffs, 0.975), 4)],
        "gap_overlaps_zero": pct(diffs, 0.025) <= 0 <= pct(diffs, 0.975),
    }


def analyse_bootstrap_ci() -> AnalysisResult:
    """Bootstrap a paired 95 % CI on the cascade-vs-Sonnet strict-F1 gap
    over dev-split cases. Uses cached embeddings; no API calls.
    """
    name = "05_bootstrap_ci_cascade_vs_sonnet"
    try:
        from eval.silver.embedders import OpenAIEmbedder
        from eval.silver.data.jsonl_utils import read_jsonl
        from eval.silver.data.schemas import SilverCaseResult
        from eval.silver.data.split import assign_split
    except Exception as exc:
        return AnalysisResult(
            name=name, status="error",
            notes=f"Import failure: {exc}",
        )

    silver_path = _REPO_ROOT / "eval" / "data" / "silver_findings_related15.jsonl"
    cache_path = _REPO_ROOT / "eval" / "data" / "map_primer" / "voter_cache.json"
    if not silver_path.exists() or not cache_path.exists():
        return AnalysisResult(
            name=name, status="missing",
            notes=(
                f"Required artefact missing — silver={silver_path.exists()}, "
                f"voter_cache={cache_path.exists()}."
            ),
        )

    voter_cache = json.loads(cache_path.read_text())
    silver_by_case = {
        rec.case_id: rec
        for rec in read_jsonl(silver_path, SilverCaseResult)
    }
    def dev_filter(cid):
        return assign_split(cid) == "dev"

    # Inject the b2 builder from the orchestrator using importlib so we
    # don't have to vendor it.
    import importlib.util
    orch_path = (_REPO_ROOT / "scripts" / "eval"
                 / "run_summarization_experiments.py")
    spec = importlib.util.spec_from_file_location("rse_orch", orch_path)
    if spec is None or spec.loader is None:
        return AnalysisResult(
            name=name, status="error",
            notes="Cannot import run_summarization_experiments.py.",
        )
    orch = importlib.util.module_from_spec(spec)
    sys.modules["rse_orch"] = orch
    spec.loader.exec_module(orch)

    # Build per-case outputs for two baselines:
    #   * l3_only (Sonnet always)
    #   * cascade_provisional_final (re-replayed via map_theta_sweep._replay)
    sonnet_outputs = orch._b2_build_voter_outputs(
        voter_cache, level="l3", voter_index=0, case_filter=dev_filter,
    )

    # For cascade — we use the EXP B.2 helper that replays the cascade
    # under a config. We re-use the cascade_provisional_final settings
    # documented in the EXP B.2 results doc:
    #   embedder=gemini, scorer=hybrid_entity_heavy, theta=0.9, reject=0.2
    # (these are the production-pinned values in the working tree).
    # Use the embedding-only scorer kind because the hybrid blend
    # weights are sweep-specific; this gives a cascade *baseline* whose
    # numbers should be cross-checked against EXP B.2's row.
    cascade_cfg = {
        "embedder": "openai",                # use cached openai-matcher
        "scorer": "embedding_default",
        "theta": 0.9,
        "reject_theta": 0.2,
        "polarity_flag": True,
        "single_voter_policy": "keep",
        "agreement_weights": None,
        "hybrid_weights": None,
    }

    try:
        from eval.silver.map_context import _load_map_context
        _load_map_context(cascade_cfg["embedder"], embed_cache_path=None)
        # Replay only dev-split cases.
        # _replay returns a dict; we adapt below.
        # Note: this path may not be straightforwardly per-case-emitting;
        # we degrade gracefully if it fails.
        cascade_outputs = []
        for case_id, entry in voter_cache.items():
            if not dev_filter(entry["case_id"]):
                continue
            # The cascade builds a per-chunk decision; for offline
            # bootstrap we approximate the cascade output by taking the
            # L3 voter's findings (since at theta=0.9 escalation is near-
            # universal on this corpus per EXP B.2). This is an
            # acknowledged approximation — see caveats.
            ce = entry
            from eval.silver.data.schemas import PipelineCaseOutput
            findings = []
            chunk_map = ce.get("chunk_map") or {}
            for chunk_id in chunk_map:
                l3 = (ce.get("l3") or {}).get(chunk_id)
                if not l3:
                    continue
                for f in l3.get("findings", []) or []:
                    findings.append(
                        orch._b2_finding_to_pipeline(
                            f, ce["pmcid"], chunk_id,
                            run_id="exp_b2_cascade_l3approx",
                        )
                    )
            cascade_outputs.append(PipelineCaseOutput(
                case_id=ce["case_id"], pmcid=ce["pmcid"], te_id=ce["te_id"],
                run_id="exp_b2_cascade_l3approx", pipeline_run_id=0,
                findings=findings,
            ))
    except Exception as exc:
        return AnalysisResult(
            name=name, status="error",
            notes=f"Cascade replay setup failed: {exc}",
        )

    # Run matcher per-case for both baselines under the OpenAI matcher
    # embedder (the production matcher).
    import os
    try:
        from eval.silver.matcher import (
            make_embedding_cache as _make_cache, DEFAULT_CACHE_PATH as _DCP
        )
        from eval.silver.embedders import OpenAIEmbedder
        cache_obj = _make_cache(_DCP)
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return AnalysisResult(
                name=name, status="error",
                notes=("OPENAI_API_KEY not set in the environment; matcher "
                       "embedder constructor requires a non-empty key even "
                       "though the cache is warm. Source `.env` and re-run."),
            )
        embedder = OpenAIEmbedder(api_key=api_key)
    except Exception as exc:
        return AnalysisResult(
            name=name, status="error",
            notes=f"Matcher setup failed: {exc}",
        )

    sonnet_per_case = _b2_match_per_case(
        sonnet_outputs, silver_by_case, embedder, cache_obj,
    )
    cascade_per_case = _b2_match_per_case(
        cascade_outputs, silver_by_case, embedder, cache_obj,
    )

    sonnet_agg = _aggregate_strict_f1(sonnet_per_case)
    cascade_agg = _aggregate_strict_f1(cascade_per_case)

    boot = _bootstrap_paired(cascade_per_case, sonnet_per_case,
                              n_iters=2000, seed=42)

    csv_path = OUT_DIR / f"{name}_per_case.csv"
    rows = []
    by_a = {r["case_id"]: r for r in cascade_per_case}
    by_b = {r["case_id"]: r for r in sonnet_per_case}
    for cid in sorted(set(by_a) | set(by_b)):
        ra = by_a.get(cid)
        rb = by_b.get(cid)
        rows.append([
            cid,
            ra["strict_tp"] if ra else "",
            ra["n_pipeline"] if ra else "",
            ra["n_silver"] if ra else "",
            rb["strict_tp"] if rb else "",
            rb["n_pipeline"] if rb else "",
            rb["n_silver"] if rb else "",
        ])
    _write_csv(
        csv_path,
        header=["case_id", "cascade_strict_tp", "cascade_n_pipeline",
                "cascade_n_silver", "sonnet_strict_tp",
                "sonnet_n_pipeline", "sonnet_n_silver"],
        rows=rows,
    )

    summary = {
        "cascade_baseline": "l3-only approximation",
        "sonnet_baseline": "l3_only (Sonnet)",
        "cascade_strict_f1_pointwise": round(cascade_agg, 4),
        "sonnet_strict_f1_pointwise": round(sonnet_agg, 4),
        "bootstrap": boot,
        "n_cases_cascade": len(cascade_per_case),
        "n_cases_sonnet": len(sonnet_per_case),
    }
    _write_json(OUT_DIR / f"{name}.json", summary)

    return AnalysisResult(
        name=name, status="partial",
        outputs=[csv_path, OUT_DIR / f"{name}.json"],
        key_numbers=summary,
        notes=(
            "Paired bootstrap (n=2000) of the strict-F1 gap between the "
            "cascade replay and the Sonnet-only (l3_only) baseline on the "
            "dev split. Embedding matcher uses cached OpenAI embeddings."
        ),
        caveats=[
            "The cascade baseline here is approximated by always using "
            "the L3 voter outputs (this is the same operation EXP B.2 "
            "performs for the l3_only row). The full cascade routing "
            "logic at (theta=0.9, reject_theta=0.2) under the "
            "production scorer is NOT reproduced inside this script; "
            "for the canonical cascade vs Sonnet comparison, cross-"
            "reference the EXP B.2 CSV in eval/reports/.",
            "Because the cascade approximation collapses to Sonnet on "
            "this corpus, the bootstrap CI here is uninformative as a "
            "stand-alone bound on the production cascade gap; it is "
            "retained for methodology demonstration only. The proper "
            "cascade bootstrap requires retaining per-case match outputs "
            "inside map_theta_sweep.run_sweep (see EXP A stub in the "
            "orchestrator).",
        ],
    )


# ── Analysis 6: EXP F on the held-out test split ──────────────────────────────
def analyse_exp_f_test_split() -> AnalysisResult:
    """Run the held-out test confirmation by calling map_theta_sweep.run_sweep
    with split='test' at the production cascade config. Replay-only; no
    API spend. Mirrors the orchestrator's _run_exp_f without requiring
    a PROVISIONAL_FINAL_MAP_CONFIG state file (we use the working-tree
    production-pinned config from EXP B.2 / configs/run.yaml).
    """
    name = "06_exp_f_test_split"
    try:
        from eval.silver.map_theta_sweep import run_sweep, ScorerSpec
        from eval.silver.map_context import _load_map_context
        from pipeline.stages.knowledge_extraction.config import (
            AgreementConfig, HybridConfig,
        )
    except Exception as exc:
        return AnalysisResult(
            name=name, status="error",
            notes=f"Import failure: {exc}",
        )

    # EXP B.2 / configs/run.yaml production pin (working tree, 2026-05-27):
    #   embedder=gemini, scorer=hybrid_entity_heavy, theta=0.9, reject=0.2,
    #   force_escalate_on_polarity_conflict=True, legacy_single_voter_policy=keep
    hybrid_cfg = HybridConfig(
        w_category=0.15, w_embedding=0.30, w_entity=0.45, w_evidence=0.10,
    )
    agreement_cfg = AgreementConfig(
        scorer_kind="hybrid", hybrid=hybrid_cfg,
        force_escalate_on_polarity_conflict=True,
    )
    spec = ScorerSpec("final_gemini_hybrid_entity_heavy", "hybrid", agreement_cfg)

    try:
        map_ctx = _load_map_context("gemini", embed_cache_path=None)
    except Exception as exc:
        return AnalysisResult(
            name=name, status="error",
            notes=f"Loading map context failed (gemini embedder): {exc}",
        )

    try:
        rows = run_sweep(
            voter_cache=map_ctx.voter_cache,
            silver_by_case=map_ctx.silver_by_case,
            embedder=map_ctx.embedder,
            embed_cache=map_ctx.embed_cache,
            sim_threshold=0.55,
            scorer_specs=[spec],
            thetas=[0.9],
            reject_thetas=[0.2],
            split="test",
            seed=42,
            dev_fraction=0.8,
            agreement_embed_fn=map_ctx.agreement_embed_fn,
            single_voter_policies=("keep",),
            force_escalate_on_polarity_conflict_grid=(True,),
        )
    except Exception as exc:
        return AnalysisResult(
            name=name, status="error",
            notes=f"run_sweep(split=test) failed: {exc}",
        )

    if not rows:
        return AnalysisResult(
            name=name, status="missing",
            notes="run_sweep returned no rows for the test split.",
        )

    # The first row is the production config on the test split.
    row = rows[0]
    csv_path = OUT_DIR / f"{name}.csv"
    keys = list(row.keys())
    _write_csv(csv_path, keys, [[row[k] for k in keys]])

    # Reference dev-split numbers from EXP B.2 row for delta computation
    dev_strict_f1 = 0.7434  # from eval/reports/exp_b2_cost_quality_20260527T005112.csv
    test_strict_f1 = float(row.get("strict_f1", 0.0) or 0.0)
    summary = {
        "config": "embedder=gemini, scorer=hybrid_entity_heavy, theta=0.9, "
                   "reject_theta=0.2, force_escalate=True, single_voter=keep",
        "split": row.get("split"),
        "test_strict_f1": round(test_strict_f1, 4),
        "test_f1": round(float(row.get("f1", 0.0) or 0.0), 4),
        "test_precision": round(float(row.get("precision", 0.0) or 0.0), 4),
        "test_recall": round(float(row.get("recall", 0.0) or 0.0), 4),
        "test_n_silver": row.get("n_silver"),
        "test_n_pipeline": row.get("n_pipeline"),
        "test_n_matched": row.get("n_matched"),
        "dev_strict_f1_for_reference": dev_strict_f1,
        "test_minus_dev_strict_f1_delta": round(
            test_strict_f1 - dev_strict_f1, 4
        ),
    }
    _write_json(OUT_DIR / f"{name}.json", summary)

    return AnalysisResult(
        name=name, status="ok",
        outputs=[csv_path, OUT_DIR / f"{name}.json"],
        key_numbers=summary,
        notes=(
            "Held-out test split confirmation at the working-tree "
            "production pin (gemini embedder, hybrid_entity_heavy scorer, "
            "theta=0.9, reject_theta=0.2). Replay-only over the primer "
            "cache; no API calls."
        ),
        caveats=[
            "The production pin is taken from EXP B.2's "
            "cascade_provisional_final row (eval/reports/"
            "exp_b2_cost_quality_20260527T005112.csv) rather than from a "
            "state file produced by the orchestrator. If the orchestrator "
            "is run end-to-end (EXP 1–7) the state file will pin a (possibly "
            "different) winning config; re-run this analysis under that "
            "config before drafting chapter 9.6.",
        ],
    )


# ── Analysis 7: NLI four-mode A/B ─────────────────────────────────────────────
def analyse_nli_ab() -> AnalysisResult:
    """A/B compare available corpus_relations.json outputs by label
    distribution and pairwise label-change matrix.
    """
    name = "07_nli_input_ab"
    src_dir = _REPO_ROOT / "out" / "summaries"
    candidates = {
        "predicate":           src_dir / "corpus_relations_predicate.json",
        "verbatim":            src_dir / "corpus_relations_verbatim.json",
        "scope_predicate":     src_dir / "corpus_relations_scope_predicate.json",
        "scope_verbatim":      src_dir / "corpus_relations_scope_verbatim.json",
        "current_production":  src_dir / "corpus_relations.json",
    }
    available = {k: p for k, p in candidates.items() if p.exists()}

    if not available:
        return AnalysisResult(
            name=name, status="missing",
            notes="No corpus_relations*.json files in out/summaries/.",
        )

    def _load(p: Path) -> list[dict]:
        try:
            data = json.loads(p.read_text())
        except Exception:
            return []
        if isinstance(data, list):
            return data
        return data.get("relations", []) or []

    loaded = {k: _load(p) for k, p in available.items()}

    def _pair_key(rel):
        a = rel.get("rule_id_a") or ""
        b = rel.get("rule_id_b") or ""
        return tuple(sorted([a, b]))

    # Per-mode label distribution and per-pair label index.
    distributions = {}
    pair_labels: dict[str, dict[tuple, str]] = {}
    for mode, rels in loaded.items():
        labels = Counter()
        per_pair = {}
        for r in rels:
            lab = (r.get("relation_type") or r.get("classified_label")
                   or "UNRELATED")
            labels[lab] += 1
            per_pair[_pair_key(r)] = lab
        distributions[mode] = dict(labels)
        pair_labels[mode] = per_pair

    # Flip matrix between every pair of modes that share at least 80% of
    # pair keys.
    pairs = sorted(available)
    flip_tables = {}
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            a, b = pairs[i], pairs[j]
            common = set(pair_labels[a]) & set(pair_labels[b])
            if not common:
                continue
            flip: Counter = Counter()
            for k in common:
                la = pair_labels[a][k]
                lb = pair_labels[b][k]
                flip[(la, lb)] += 1
            flip_tables[f"{a}__vs__{b}"] = {
                "n_common_pairs": len(common),
                "flip_counts": {f"{k[0]}→{k[1]}": v for k, v in flip.items()},
            }

    summary = {
        "available_modes": list(available),
        "missing_modes": [k for k in candidates if k not in available],
        "n_relations_per_mode": {k: len(v) for k, v in loaded.items()},
        "label_distribution_per_mode": distributions,
        "flip_tables": flip_tables,
    }
    _write_json(OUT_DIR / f"{name}.json", summary)

    csv_path = OUT_DIR / f"{name}.csv"
    csv_rows = []
    for mode, dist in distributions.items():
        for label, count in sorted(dist.items()):
            csv_rows.append([mode, label, count])
    _write_csv(csv_path, ["mode", "label", "count"], csv_rows)

    return AnalysisResult(
        name=name, status=("partial" if len(available) < 4 else "ok"),
        outputs=[csv_path, OUT_DIR / f"{name}.json"],
        key_numbers={
            "available_modes": list(available),
            "missing_modes": summary["missing_modes"],
            "n_relations_per_mode": summary["n_relations_per_mode"],
        },
        notes=(
            "Available NLI-input modes compared by label distribution and "
            "per-pair label flips. The complete A/B requires four files "
            "(predicate, scope_predicate, verbatim, scope_verbatim); when "
            "only a subset exists this analysis reports what is on disk."
        ),
        caveats=[
            "Missing modes can be produced via scripts/rebuild_from_cached_map.py "
            "with the RelateConfig flags toggled appropriately, then this "
            "analysis re-run.",
        ],
    )


# ── Analysis 8: Per-rubric variant-18 F1 breakdown ────────────────────────────
def analyse_variant18_rubric() -> AnalysisResult:
    """Extract variant 18's per-rubric-dimension F1 row from stage6_PR.md
    (the final completed staged-sweep report).
    """
    name = "08_per_rubric_variant18"
    stage6 = _REPO_ROOT / "reports" / "stage6_PR.md"
    if not stage6.exists():
        return AnalysisResult(
            name=name, status="missing",
            notes=f"{stage6} not found.",
        )
    txt = stage6.read_text()
    target = "18_hybrid_best_family_fixes_footnote_expand_1_2"
    rows = []
    for line in txt.splitlines():
        if line.startswith(f"| {target} ") and "|" in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            # Schema from stage6_PR.md header:
            #   variant | kind | crop P | crop R | crop F1 | mask P | mask R |
            #   mask F1 | caption P | caption tp/fp | footnote P |
            #   footnote tp/fp | strict P | strict R | strict F1 |
            #   strict tp/fp/fn | emitted | unlabelled | unrecog | icons
            if len(cells) >= 20:
                rows.append({
                    "variant":      cells[0],
                    "kind":         cells[1],
                    "crop_P":       cells[2],
                    "crop_R":       cells[3],
                    "crop_F1":      cells[4],
                    "mask_P":       cells[5],
                    "mask_R":       cells[6],
                    "mask_F1":      cells[7],
                    "caption_P":    cells[8],
                    "caption_tp_fp": cells[9],
                    "footnote_P":   cells[10],
                    "footnote_tp_fp": cells[11],
                    "strict_P":     cells[12],
                    "strict_R":     cells[13],
                    "strict_F1":    cells[14],
                    "strict_tp_fp_fn": cells[15],
                    "emitted":      cells[16],
                    "unlabelled":   cells[17],
                    "unrecog":      cells[18],
                    "icons":        cells[19],
                })

    if not rows:
        return AnalysisResult(
            name=name, status="missing",
            notes=f"variant 18 row not found in {stage6.name}.",
        )

    csv_path = OUT_DIR / f"{name}.csv"
    keys = list(rows[0].keys())
    _write_csv(csv_path, keys, [[r[k] for k in keys] for r in rows])
    _write_json(OUT_DIR / f"{name}.json", {
        "source": stage6.name,
        "rows": rows,
    })

    return AnalysisResult(
        name=name, status="ok",
        outputs=[csv_path, OUT_DIR / f"{name}.json"],
        key_numbers={
            "figures_strict_F1": next(
                (r["strict_F1"] for r in rows if r["kind"] == "figures"), None),
            "tables_strict_F1": next(
                (r["strict_F1"] for r in rows if r["kind"] == "tables"), None),
        },
        notes=(
            "Per-rubric-dimension F1 (crop, mask, caption, footnote, "
            "strict) for the final production configuration (variant 18: "
            "HYBRID at TATR threshold 0.99, header-zone filtering, "
            "in-figure filtering, footnote expansion, two-pass on). "
            "Source: reports/stage6_PR.md."
        ),
        caveats=[
            "The 27-PDF evaluation set is a uniform random sample of the "
            "ingested corpus (seeded RNG in eval/run.py), not stratified. "
            "(Originally 28 PDFs; PMC11863705 removed 2026-06-03 as a "
            "multi-PDF package never ingested into the corpus; see BUGS.md "
            "B-068.)",
        ],
    )


# ── Analysis 9: Provenance example ────────────────────────────────────────────
def analyse_provenance_example() -> AnalysisResult:
    """Pick one FinalRule with a complete provenance triple and dump the
    full trail (source paragraph → MAP finding → NLI score → canonical
    rule → final rule).
    """
    name = "09_provenance_example"
    summaries_dir = _REPO_ROOT / "out" / "summaries" / "summaries"
    files = sorted(summaries_dir.glob("*.json"))
    if not files:
        return AnalysisResult(
            name=name, status="missing",
            notes="No per-paper summary JSON found.",
        )
    for fp in files:
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        canon_rules = data.get("canonical_rules") or []
        final_rules = data.get("final_rules") or []
        if not canon_rules or not final_rules:
            continue
        canon_by_id = {c.get("canonical_id"): c for c in canon_rules}
        for fr in final_rules:
            canon = canon_by_id.get(fr.get("canonical_id"))
            if not canon:
                continue
            verbatim = canon.get("representative_verbatim")
            te_id = canon.get("representative_text_element_id")
            grounding = (
                fr.get("mean_grounding_score")
                or canon.get("mean_grounding_score")
            )
            if not (verbatim and te_id is not None and grounding is not None):
                continue
            example = {
                "pmcid": data.get("pmcid", fp.stem),
                "source_paragraph_te_id": te_id,
                "verbatim_source_sentence": verbatim,
                "canonical_rule": {
                    "canonical_id": canon.get("canonical_id"),
                    "subject_entity": canon.get("subject_entity"),
                    "outcome_entity": canon.get("outcome_entity"),
                    "relation_type": canon.get("relation_type"),
                    "direction": canon.get("direction"),
                    "category": canon.get("category"),
                    "predicate_text": canon.get("predicate_text"),
                    "scope": canon.get("scope"),
                    "mean_grounding_score": canon.get("mean_grounding_score"),
                    "finding_count": canon.get("finding_count"),
                },
                "final_rule": {
                    "canonical_id": fr.get("canonical_id"),
                    "final_score": fr.get("final_score"),
                    "support_count": fr.get("support_count"),
                    "contradict_count": fr.get("contradict_count"),
                    "study_coverage": fr.get("study_coverage"),
                    "is_conflicted": fr.get("is_conflicted"),
                },
                "trail_explanation": (
                    "The verbatim source sentence above is taken from the "
                    "text_elements row referenced by source_paragraph_te_id "
                    "in the project's PostgreSQL database. The MAP voter "
                    "produced a finding whose claim was filtered through "
                    "the NLI grounding model at threshold 0.5; the per-"
                    "rule mean_grounding_score above is the (averaged) "
                    "entailment score for the finding(s) backing this "
                    "rule. The canonical rule was then scored by RESOLVE "
                    "to produce the final rule."
                ),
            }
            json_path = OUT_DIR / f"{name}.json"
            _write_json(json_path, example)
            return AnalysisResult(
                name=name, status="ok",
                outputs=[json_path],
                key_numbers={
                    "pmcid": example["pmcid"],
                    "canonical_id": example["canonical_rule"]["canonical_id"],
                    "final_score": example["final_rule"]["final_score"],
                    "grounding_score": example["canonical_rule"]["mean_grounding_score"],
                },
                notes=(
                    "One end-to-end provenance trail: source sentence → "
                    "canonical rule → final rule, with grounding score."
                ),
                caveats=[
                    "The full source paragraph (text_elements.text_content) "
                    "requires a DB query via provenance/paragraph_lookup.py; "
                    "this script reports only the verbatim source sentence "
                    "carried on the canonical rule.",
                ],
            )

    return AnalysisResult(
        name=name, status="missing",
        notes=(
            "No FinalRule in the on-disk summaries carries a complete "
            "provenance triple (verbatim + te_id + grounding). The "
            "summaries predate the 2026-05-27 CanonicalRule field "
            "addition; re-run the pipeline to populate the new fields."
        ),
    )


# ── Analysis 10: Paired bootstrap CI on the cascade-vs-Sonnet gap ─────────────
def _build_cascade_outputs(voter_cache, agreement_embed_fn, dev_filter):
    """Replay the production cascade per chunk and return per-case
    PipelineCaseOutputs filtered to ``dev_filter`` cases.

    Production pin (EXP B.2 cascade_provisional_final / configs/run.yaml
    working tree, 2026-05-27):
      embedder = gemini, scorer = hybrid_entity_heavy
      (w_category 0.15 / w_embedding 0.30 / w_entity 0.45 / w_evidence 0.10),
      theta = 0.9, reject_theta = 0.2,
      force_escalate_on_polarity_conflict = True,
      legacy_single_voter_policy = "keep".
    """
    from eval.silver.map_theta_sweep import _replay, ScorerSpec, _build_scorer
    from pipeline.stages.knowledge_extraction.config import (
        AgreementConfig, HybridConfig,
    )
    hybrid_cfg = HybridConfig(
        w_category=0.15, w_embedding=0.30, w_entity=0.45, w_evidence=0.10,
    )
    agreement_cfg = AgreementConfig(
        scorer_kind="hybrid", hybrid=hybrid_cfg,
        force_escalate_on_polarity_conflict=True,
    )
    spec = ScorerSpec(
        name="final_gemini_hybrid_entity_heavy",
        kind="hybrid",
        weights=agreement_cfg,
    )
    scorer = _build_scorer(spec, agreement_embed_fn)
    case_outputs, accept_counts, early_chunks, n_pol = _replay(
        voter_cache, scorer, theta=0.9, reject_theta=0.2,
        single_voter_policy="keep",
        force_escalate_on_polarity_conflict=True,
    )
    # Filter to dev cases only.
    filtered = [co for co in case_outputs if dev_filter(co.case_id)]
    return filtered, accept_counts, n_pol


def _build_sonnet_only_outputs(voter_cache, dev_filter):
    """Always-L3 baseline outputs from voter_cache."""
    import importlib.util
    orch_path = (_REPO_ROOT / "scripts" / "eval"
                 / "run_summarization_experiments.py")
    spec = importlib.util.spec_from_file_location("rse_orch", orch_path)
    orch = importlib.util.module_from_spec(spec)
    sys.modules["rse_orch"] = orch
    spec.loader.exec_module(orch)
    outputs = orch._b2_build_voter_outputs(
        voter_cache, level="l3", voter_index=0, case_filter=dev_filter,
    )
    return outputs


def analyse_paired_bootstrap_ci() -> AnalysisResult:
    """Paired bootstrap (n=2000) on the cascade-vs-Sonnet strict-F1 gap
    over the 221 dev-split cases. The cascade is reproduced by calling
    ``map_theta_sweep._replay`` at the production pin; per-case match
    outputs are computed here in-script (the ``run_sweep`` path
    discards them by design).
    """
    name = "10_cascade_vs_sonnet_gap_ci"
    import os
    try:
        from eval.silver.data.jsonl_utils import read_jsonl
        from eval.silver.data.schemas import SilverCaseResult
        from eval.silver.data.split import assign_split
        from eval.silver.map_context import _load_map_context
    except Exception as exc:
        return AnalysisResult(
            name=name, status="error",
            notes=f"Import failure: {exc}",
        )

    voter_cache_path = _REPO_ROOT / "eval" / "data" / "map_primer" / "voter_cache.json"
    silver_path      = _REPO_ROOT / "eval" / "data" / "silver_findings_related15.jsonl"
    if not voter_cache_path.exists() or not silver_path.exists():
        return AnalysisResult(
            name=name, status="missing",
            notes="voter_cache.json or silver_findings_related15.jsonl missing.",
        )

    voter_cache = json.loads(voter_cache_path.read_text())
    silver_by_case = {
        rec.case_id: rec
        for rec in read_jsonl(silver_path, SilverCaseResult)
    }
    def dev_filter(cid):
        return assign_split(cid) == "dev"

    # Load matcher (OpenAI embedder + cache).
    try:
        from eval.silver.matcher import (
            make_embedding_cache, DEFAULT_CACHE_PATH,
        )
        from eval.silver.embedders import OpenAIEmbedder
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return AnalysisResult(
                name=name, status="error",
                notes="OPENAI_API_KEY missing; matcher embedder needs it.",
            )
        matcher_cache = make_embedding_cache(DEFAULT_CACHE_PATH)
        matcher_embedder = OpenAIEmbedder(api_key=api_key)
    except Exception as exc:
        return AnalysisResult(
            name=name, status="error",
            notes=f"Matcher setup failed: {exc}",
        )

    # Load agreement embedder (Gemini, cached).
    try:
        map_ctx = _load_map_context("gemini", embed_cache_path=None)
        agreement_embed_fn = map_ctx.agreement_embed_fn
    except Exception as exc:
        return AnalysisResult(
            name=name, status="error",
            notes=f"Agreement embedder setup failed: {exc}",
        )

    # Build cascade and Sonnet-only outputs for dev cases.
    _log(f"[{name}] replaying production cascade on dev split …")
    cascade_outputs, accept_counts, n_pol_cascade = _build_cascade_outputs(
        voter_cache, agreement_embed_fn, dev_filter,
    )
    cascade_outputs = [
        co for co in cascade_outputs if co.case_id in silver_by_case
    ]
    sonnet_outputs = _build_sonnet_only_outputs(voter_cache, dev_filter)
    sonnet_outputs = [
        co for co in sonnet_outputs if co.case_id in silver_by_case
    ]

    cascade_per_case = _b2_match_per_case(
        cascade_outputs, silver_by_case, matcher_embedder, matcher_cache,
    )
    sonnet_per_case = _b2_match_per_case(
        sonnet_outputs, silver_by_case, matcher_embedder, matcher_cache,
    )

    cascade_point = _aggregate_strict_f1(cascade_per_case)
    sonnet_point  = _aggregate_strict_f1(sonnet_per_case)
    boot = _bootstrap_paired(cascade_per_case, sonnet_per_case,
                              n_iters=2000, seed=42)

    csv_path = OUT_DIR / f"{name}_per_case.csv"
    by_a = {r["case_id"]: r for r in cascade_per_case}
    by_b = {r["case_id"]: r for r in sonnet_per_case}
    rows = []
    for cid in sorted(set(by_a) | set(by_b)):
        ra = by_a.get(cid)
        rb = by_b.get(cid)
        rows.append([
            cid,
            ra["strict_tp"] if ra else "",
            ra["n_pipeline"] if ra else "",
            ra["n_silver"] if ra else "",
            ra["n_matched"] if ra else "",
            rb["strict_tp"] if rb else "",
            rb["n_pipeline"] if rb else "",
            rb["n_silver"] if rb else "",
            rb["n_matched"] if rb else "",
        ])
    _write_csv(
        csv_path,
        header=["case_id",
                "cascade_strict_tp", "cascade_n_pipeline",
                "cascade_n_silver", "cascade_n_matched",
                "sonnet_strict_tp", "sonnet_n_pipeline",
                "sonnet_n_silver", "sonnet_n_matched"],
        rows=rows,
    )

    summary = {
        "config_cascade": (
            "embedder=gemini, scorer=hybrid_entity_heavy "
            "(w_category=0.15, w_embedding=0.30, w_entity=0.45, "
            "w_evidence=0.10), theta=0.9, reject_theta=0.2, "
            "force_escalate=True, single_voter=keep"
        ),
        "config_sonnet": "always-L3 (claude-sonnet-4-6 @ T=0)",
        "split": "dev (hash-based, seed=42, dev_fraction=0.8 → 221 cases)",
        "cascade_strict_f1_pointwise": round(cascade_point, 4),
        "sonnet_strict_f1_pointwise":  round(sonnet_point, 4),
        "gap_pointwise_cascade_minus_sonnet": round(
            cascade_point - sonnet_point, 4
        ),
        "cascade_tier_distribution": accept_counts,
        "cascade_n_polarity_conflicts": int(n_pol_cascade),
        "n_cases_cascade":  len(cascade_per_case),
        "n_cases_sonnet":   len(sonnet_per_case),
        "n_cases_paired_common": (
            boot["n_cases"] if boot is not None else 0
        ),
        "bootstrap": boot,
        "interpretation": _bootstrap_interpretation(boot),
    }
    _write_json(OUT_DIR / f"{name}.json", summary)

    status = "ok" if boot is not None else "partial"
    return AnalysisResult(
        name=name, status=status,
        outputs=[csv_path, OUT_DIR / f"{name}.json"],
        key_numbers=summary,
        notes=(
            "Paired bootstrap of the cascade-vs-Sonnet strict-F1 gap on "
            "the dev split. Cascade replayed at the production pin via "
            "map_theta_sweep._replay; per-case match outputs computed "
            "in this script (zero-API matcher). Paired bootstrap is "
            "preferred over independent bootstrap because both baselines "
            "share the same per-case silver."
        ),
        caveats=[
            "The cascade replay uses _replay's tier-selection logic; "
            "this is the same path EXP F uses on the test split. The "
            "Sonnet-only baseline is the L3-voter findings per chunk, "
            "matching the EXP B.2 l3_only row.",
            "The matcher denominator is `n_pipeline` per case (a "
            "per-case aggregate); paired-bootstrap resamples cases not "
            "findings, so each bootstrap iterate is a case-level "
            "aggregate strict-F1.",
            "If the 95 % CI on the gap straddles zero the comparison "
            "is descriptive, not significant; do not write 'cascade "
            "outperforms Sonnet' in that case.",
        ],
    )


def _bootstrap_interpretation(boot) -> str:
    if boot is None:
        return "n/a"
    lo, hi = boot["gap_ci95"]
    if lo > 0:
        return ("Gap CI is entirely positive — cascade > Sonnet at the "
                "95 % bootstrap level on dev. Best-on-dev-split only; "
                "see chapter 9.6 for held-out confirmation.")
    if hi < 0:
        return ("Gap CI is entirely negative — Sonnet > cascade at the "
                "95 % bootstrap level on dev. Reconsider the cascade "
                "framing.")
    return ("Gap CI straddles zero — cascade and Sonnet are "
            "indistinguishable at the 95 % bootstrap level on dev. "
            "Report the gap as descriptive, not significant.")


# ── Analysis 11: NLI four-mode A/B (regenerate scope_* modes) ─────────────────
def _regenerate_corpus_relations(mode_name: str, *,
                                 scope_aware_nli: bool,
                                 use_verbatim_for_nli: bool,
                                 output_path: Path,
                                 summaries_dir: Path,
                                 ) -> None:
    """Run CorpusRelateStage with the given flag combo and write its
    corpus_relations JSON to ``output_path``. Reuses the existing
    canonical_rules cached in ``summaries_dir/*.json`` — no MAP re-run.
    """
    from pipeline.stages.knowledge_extraction.stages.corpus_relate import (
        CorpusRelateStage,
    )
    stage = CorpusRelateStage(
        entailment_threshold=0.50,
        contradiction_threshold=0.50,
        db=None,
        scope_aware_nli=scope_aware_nli,
        use_verbatim_for_nli=use_verbatim_for_nli,
    )
    stage.relate_from_dir(
        source_dir=summaries_dir,
        output_path=output_path,
        run_selection="latest_per_pmcid",
    )


def analyse_nli_four_mode_ab() -> AnalysisResult:
    """Regenerate the four NLI-input modes (predicate / scope_predicate /
    verbatim / scope_verbatim) by re-running CorpusRelateStage on the
    cached canonical_rules in out/summaries/summaries/*.json. Compare
    label distributions and per-pair label flips across modes.
    """
    name = "11_nli_input_four_mode_ab"
    summaries_dir = _REPO_ROOT / "out" / "summaries" / "summaries"
    if not summaries_dir.exists() or not list(summaries_dir.glob("*.json")):
        return AnalysisResult(
            name=name, status="missing",
            notes=f"No canonical rules under {summaries_dir}.",
        )

    mode_configs = [
        ("predicate",       False, False),
        ("scope_predicate", True,  False),
        ("verbatim",        False, True),
        ("scope_verbatim",  True,  True),
    ]

    mode_outputs: dict[str, Path] = {}
    for mode_name, scope, verbatim in mode_configs:
        out_json = OUT_DIR / f"corpus_relations_{mode_name}.json"
        try:
            _log(f"[{name}] regenerating mode={mode_name} "
                 f"(scope={scope}, verbatim={verbatim}) …")
            _regenerate_corpus_relations(
                mode_name,
                scope_aware_nli=scope,
                use_verbatim_for_nli=verbatim,
                output_path=out_json,
                summaries_dir=summaries_dir,
            )
            mode_outputs[mode_name] = out_json
        except Exception as exc:
            _log(f"[{name}] mode {mode_name} failed: {exc}\n"
                 f"{traceback.format_exc()}")

    if not mode_outputs:
        return AnalysisResult(
            name=name, status="error",
            notes="All four mode regenerations failed.",
        )

    # Load and compare.
    def _load(p: Path) -> list[dict]:
        try:
            data = json.loads(p.read_text())
        except Exception:
            return []
        if isinstance(data, list):
            return data
        return data.get("relations", []) or []

    loaded = {m: _load(p) for m, p in mode_outputs.items()}

    def _pair_key(rel):
        a = rel.get("rule_id_a") or ""
        b = rel.get("rule_id_b") or ""
        return tuple(sorted([a, b]))

    def _label(rel):
        return (rel.get("relation_type") or rel.get("classified_label")
                or "UNRELATED")

    # Per-mode distribution and per-pair label index.
    distributions = {}
    pair_labels: dict[str, dict[tuple, str]] = {}
    for mode, rels in loaded.items():
        labs = Counter()
        per_pair = {}
        for r in rels:
            lab = _label(r)
            labs[lab] += 1
            per_pair[_pair_key(r)] = lab
        distributions[mode] = dict(labs)
        pair_labels[mode] = per_pair

    # Pairwise flip matrix (only present pairs from each mode contribute).
    pairs = sorted(mode_outputs)
    flip_tables = {}
    stability_metrics = {}
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            a, b = pairs[i], pairs[j]
            common = set(pair_labels[a]) & set(pair_labels[b])
            if not common:
                continue
            flip: Counter = Counter()
            unchanged = 0
            for k in common:
                la = pair_labels[a][k]
                lb = pair_labels[b][k]
                if la == lb:
                    unchanged += 1
                flip[(la, lb)] += 1
            flip_tables[f"{a}__vs__{b}"] = {
                "n_common_pairs": len(common),
                "n_unchanged": unchanged,
                "stability": (round(unchanged / len(common), 4)
                              if common else None),
                "flip_counts": {f"{k[0]}->{k[1]}": v
                                 for k, v in flip.items()},
            }
            stability_metrics[f"{a}__vs__{b}"] = (
                round(unchanged / len(common), 4) if common else None
            )

    summary = {
        "modes_attempted": [m for m, *_ in mode_configs],
        "modes_succeeded": list(mode_outputs.keys()),
        "modes_failed":    [m for m, *_ in mode_configs
                            if m not in mode_outputs],
        "mode_output_files": {m: str(p.name)
                              for m, p in mode_outputs.items()},
        "n_relations_per_mode": {m: len(v) for m, v in loaded.items()},
        "label_distribution_per_mode": distributions,
        "stability_pairwise": stability_metrics,
        "flip_tables": flip_tables,
    }
    _write_json(OUT_DIR / f"{name}.json", summary)

    csv_path = OUT_DIR / f"{name}.csv"
    rows = []
    for mode, dist in distributions.items():
        for lab, count in sorted(dist.items()):
            rows.append([mode, lab, count])
    _write_csv(csv_path, ["mode", "label", "count"], rows)

    return AnalysisResult(
        name=name,
        status=("ok" if len(mode_outputs) == 4 else "partial"),
        outputs=[csv_path, OUT_DIR / f"{name}.json"]
                + list(mode_outputs.values()),
        key_numbers={
            "modes_succeeded": list(mode_outputs.keys()),
            "modes_failed":    [m for m, *_ in mode_configs
                                if m not in mode_outputs],
            "n_relations_per_mode": {m: len(v) for m, v in loaded.items()},
            "stability_pairwise":   stability_metrics,
        },
        notes=(
            "Four-mode NLI-input ablation regenerated from cached "
            "canonical rules. Each mode toggles the two RelateConfig "
            "flags (scope_aware_nli, use_verbatim_for_nli) and re-runs "
            "CorpusRelateStage over the 15-paper cache. The flip tables "
            "report per-pair label changes between every adjacent mode "
            "pair; the stability metric is the fraction of common "
            "(rule_a, rule_b) pairs that agree on the relation label "
            "between two modes."
        ),
        caveats=[
            "Relation counts are small (typically 10–20 per mode) on "
            "the 15-paper corpus; label-flip rates are reported as raw "
            "counts. Bootstrap CIs on per-mode F1 against a silver "
            "subset of cross-paper pairs are not computed by this "
            "script (no silver labels exist for cross-paper relations).",
            "The verbatim modes require populated "
            "CanonicalRule.representative_verbatim fields; legacy "
            "summaries without the field fall back to predicate_text "
            "(documented silent fallback in relate_stage._build_nli_text).",
        ],
    )


# ── Analysis 12: Real-profile grounding and polarity evidence ─────────────────
def _build_real_profile_findings_per_chunk(voter_cache, dev_filter,
                                             agreement_embed_fn):
    """Replay the production cascade on the dev split via _replay and
    return per-chunk metadata: (case_id, pmcid, chunk_id, source_text,
    accepted_findings, accept_level).
    """
    from eval.silver.map_theta_sweep import ScorerSpec, _build_scorer
    from pipeline.stages.knowledge_extraction.config import (
        AgreementConfig, HybridConfig,
    )
    hybrid_cfg = HybridConfig(
        w_category=0.15, w_embedding=0.30, w_entity=0.45, w_evidence=0.10,
    )
    agreement_cfg = AgreementConfig(
        scorer_kind="hybrid", hybrid=hybrid_cfg,
        force_escalate_on_polarity_conflict=True,
    )
    spec = ScorerSpec(
        name="final_gemini_hybrid_entity_heavy",
        kind="hybrid",
        weights=agreement_cfg,
    )
    scorer = _build_scorer(spec, agreement_embed_fn)
    # Replay returns case-level outputs aggregated across chunks; for the
    # GROUNDING re-run we need per-chunk findings. Re-implement the
    # per-chunk loop locally so we can capture the accept_level + source
    # text for each chunk.
    from pipeline.stages.knowledge_extraction.agreement import AgreementChecker
    from pipeline.stages.knowledge_extraction.interfaces.scoring import ChunkDecision
    from pipeline.stages.knowledge_extraction.models import AuditableSummary
    checker = AgreementChecker(
        scorer, theta=0.9, reject_theta=0.2,
        single_voter_policy="keep",
        force_escalate_on_polarity_conflict=True,
    )
    per_chunk: list[dict] = []
    accept_counts: Counter = Counter()
    n_polarity_conflicts = 0
    for entry in voter_cache.values():
        if not dev_filter(entry["case_id"]):
            continue
        for chunk_id in entry.get("chunk_map", {}):
            source_text = entry.get("source_texts", {}).get(chunk_id, "")
            l1_raw = entry.get("l1", {}).get(chunk_id) or []
            l1_out = [AuditableSummary.model_validate(d) for d in l1_raw if d]
            selected = None
            accept_level = "no_data"
            had_pc = False
            if l1_out:
                b1 = checker.compute(l1_out, source_text=source_text)
                if (b1.score_details or {}).get("hard_fail_reason") == "polarity_conflict":
                    had_pc = True
                if b1.decision == ChunkDecision.KEEP:
                    selected = checker.best(l1_out, b1).model_dump()
                    accept_level = "l1"
                elif b1.decision == ChunkDecision.REJECT:
                    accept_level = "reject"
            if selected is None and accept_level == "no_data":
                l2_raw = entry.get("l2", {}).get(chunk_id) or []
                l2_out = [AuditableSummary.model_validate(d) for d in l2_raw if d]
                if l2_out:
                    b2 = checker.compute(l2_out, source_text=source_text)
                    if (not had_pc and
                        (b2.score_details or {}).get("hard_fail_reason")
                          == "polarity_conflict"):
                        had_pc = True
                    if b2.decision == ChunkDecision.KEEP:
                        selected = checker.best(l2_out, b2).model_dump()
                        accept_level = "l2"
                    elif b2.decision == ChunkDecision.REJECT:
                        accept_level = "reject"
                    else:
                        l3_raw = entry.get("l3", {}).get(chunk_id)
                        if l3_raw:
                            selected = l3_raw
                        accept_level = "l3"
                else:
                    l3_raw = entry.get("l3", {}).get(chunk_id)
                    if l3_raw:
                        selected = l3_raw
                    accept_level = "l3"
            accept_counts[accept_level] += 1
            if had_pc:
                n_polarity_conflicts += 1
            per_chunk.append({
                "case_id": entry["case_id"],
                "pmcid":   entry["pmcid"],
                "chunk_id": chunk_id,
                "source_text": source_text,
                "selected": selected,
                "accept_level": accept_level,
                "had_polarity_conflict": had_pc,
            })
    return per_chunk, dict(accept_counts), n_polarity_conflicts


def analyse_real_profile_grounding_polarity() -> AnalysisResult:
    """Replay the production cascade on the dev split and apply the
    GROUNDING NLI filter at threshold 0.5 to the accepted findings.
    Reports per-paper retention rates, cascade tier distribution, and
    polarity-conflict counts.
    """
    name = "12_real_profile_grounding_polarity"
    try:
        from eval.silver.data.split import assign_split
        from eval.silver.map_context import _load_map_context
        from pipeline.stages.knowledge_extraction.models import (
            AuditableSummary,
        )
        from pipeline.stages.knowledge_extraction.grounding.grounding_filter import (
            GroundingFilter,
        )
    except Exception as exc:
        return AnalysisResult(
            name=name, status="error",
            notes=f"Import failure: {exc}",
        )

    voter_cache_path = _REPO_ROOT / "eval" / "data" / "map_primer" / "voter_cache.json"
    if not voter_cache_path.exists():
        return AnalysisResult(
            name=name, status="missing",
            notes=f"voter_cache.json missing at {voter_cache_path}.",
        )

    voter_cache = json.loads(voter_cache_path.read_text())
    def dev_filter(cid):
        return assign_split(cid) == "dev"

    try:
        map_ctx = _load_map_context("gemini", embed_cache_path=None)
        agreement_embed_fn = map_ctx.agreement_embed_fn
    except Exception as exc:
        return AnalysisResult(
            name=name, status="error",
            notes=f"Agreement embedder setup failed: {exc}",
        )

    _log(f"[{name}] replaying production cascade for grounding pass …")
    per_chunk, accept_counts, n_pc = _build_real_profile_findings_per_chunk(
        voter_cache, dev_filter, agreement_embed_fn,
    )

    # Collect all accepted findings into AuditableSummary objects (one
    # per chunk) so the GroundingFilter can process them in batch.
    # Findings with missing/empty verbatim are kept for visibility and
    # counted separately.
    grounder = GroundingFilter(threshold=0.5)

    per_paper = defaultdict(lambda: {
        "pmcid": None,
        "n_chunks_total":         0,
        "n_chunks_accepted":      0,
        "n_chunks_rejected":      0,
        "n_chunks_no_data":       0,
        "n_chunks_l1":            0,
        "n_chunks_l2":            0,
        "n_chunks_l3":            0,
        "n_chunks_polarity_conflict": 0,
        "n_findings_pre_grounding":   0,
        "n_findings_post_grounding":  0,
    })

    n_findings_pre  = 0
    n_findings_post = 0
    for c in per_chunk:
        rec = per_paper[c["pmcid"]]
        rec["pmcid"] = c["pmcid"]
        rec["n_chunks_total"] += 1
        rec[f"n_chunks_{c['accept_level']}"] = (
            rec.get(f"n_chunks_{c['accept_level']}", 0) + 1
        )
        if c["had_polarity_conflict"]:
            rec["n_chunks_polarity_conflict"] += 1
        if c["accept_level"] == "reject" or c["accept_level"] == "no_data":
            if c["accept_level"] == "reject":
                rec["n_chunks_rejected"] += 1
            else:
                rec["n_chunks_no_data"] += 1
            continue
        rec["n_chunks_accepted"] += 1
        selected = c["selected"]
        if not selected:
            continue
        try:
            summary = AuditableSummary.model_validate(selected)
        except Exception:
            continue
        findings = summary.findings or []
        n_findings_pre += len(findings)
        rec["n_findings_pre_grounding"] += len(findings)
        if not findings:
            continue
        try:
            kept_summary, dropped = grounder.filter_findings_with_scores(summary)
            kept = kept_summary.findings or []
        except Exception as exc:
            _log(f"[{name}] grounding failed for chunk {c['chunk_id']}: {exc}")
            continue
        n_findings_post += len(kept)
        rec["n_findings_post_grounding"] += len(kept)

    paper_rows = sorted(per_paper.values(), key=lambda r: r["pmcid"] or "")
    csv_path = OUT_DIR / f"{name}.csv"
    header = [
        "pmcid", "n_chunks_total", "n_chunks_accepted",
        "n_chunks_rejected", "n_chunks_no_data",
        "n_chunks_l1", "n_chunks_l2", "n_chunks_l3",
        "n_chunks_polarity_conflict",
        "n_findings_pre_grounding", "n_findings_post_grounding",
        "grounding_retention_rate",
    ]
    rows = []
    for r in paper_rows:
        pre = r["n_findings_pre_grounding"]
        post = r["n_findings_post_grounding"]
        retention = round(post / pre, 4) if pre else None
        rows.append([
            r["pmcid"], r["n_chunks_total"], r["n_chunks_accepted"],
            r["n_chunks_rejected"], r["n_chunks_no_data"],
            r.get("n_chunks_l1", 0), r.get("n_chunks_l2", 0),
            r.get("n_chunks_l3", 0), r["n_chunks_polarity_conflict"],
            pre, post, retention,
        ])
    _write_csv(csv_path, header, rows)

    aggregate = {
        "config": (
            "embedder=gemini, scorer=hybrid_entity_heavy, theta=0.9, "
            "reject_theta=0.2, force_escalate=True, single_voter=keep; "
            "grounding threshold=0.5"
        ),
        "split": "dev (hash-based, seed=42, dev_fraction=0.8 → 221 cases)",
        "n_papers": len(paper_rows),
        "cascade_tier_distribution": accept_counts,
        "n_chunks_total": sum(accept_counts.values()),
        "n_polarity_conflicts_total": int(n_pc),
        "polarity_conflict_rate": (
            round(n_pc / sum(accept_counts.values()), 4)
            if sum(accept_counts.values()) else None
        ),
        "n_findings_pre_grounding_total":  n_findings_pre,
        "n_findings_post_grounding_total": n_findings_post,
        "grounding_retention_rate_overall": (
            round(n_findings_post / n_findings_pre, 4)
            if n_findings_pre else None
        ),
    }
    _write_json(OUT_DIR / f"{name}.json", {
        "aggregate": aggregate,
        "per_paper": paper_rows,
    })

    return AnalysisResult(
        name=name, status="ok",
        outputs=[csv_path, OUT_DIR / f"{name}.json"],
        key_numbers=aggregate,
        notes=(
            "Production-cascade replay on the dev split (221 cases) "
            "followed by an inline GROUNDING NLI pass at threshold "
            "0.5. Replays the same configuration as EXP F (analysis 06) "
            "but on the dev split, and crucially applies the GROUNDING "
            "filter step that map_theta_sweep does not run."
        ),
        caveats=[
            "GROUNDING NLI runs on each chunk's accepted findings (the "
            "level the cascade chose). When the cascade rejects or has "
            "no L1/L2/L3 data the chunk contributes zero pre-grounding "
            "findings; this is the production behaviour.",
            "Polarity-conflict counts here come from the same _replay "
            "logic the EXP F sweep uses; the dev-split count "
            "supersedes analysis 03's haiku-only 0/230 figure as the "
            "production-cited number.",
            "The NLI model is the same pritamdeka/PubMedBERT-MNLI-MedNLI "
            "used by GROUNDING in production; no API calls.",
        ],
    )


# ── Driver ────────────────────────────────────────────────────────────────────
ANALYSES: list[tuple[str, Callable[[], AnalysisResult]]] = [
    ("01_provenance_carry_rate",          analyse_provenance_carry_rate),
    ("02_grounding_retention",            analyse_grounding_retention),
    ("03_polarity_conflict_counts",       analyse_polarity_conflict_counts),
    ("04_theta_heatmap",                  analyse_theta_heatmap),
    ("05_bootstrap_ci_cascade_vs_sonnet", analyse_bootstrap_ci),
    ("06_exp_f_test_split",               analyse_exp_f_test_split),
    ("07_nli_input_ab",                   analyse_nli_ab),
    ("08_per_rubric_variant18",           analyse_variant18_rubric),
    ("09_provenance_example",             analyse_provenance_example),
    ("10_cascade_vs_sonnet_gap_ci",       analyse_paired_bootstrap_ci),
    ("11_nli_input_four_mode_ab",         analyse_nli_four_mode_ab),
    ("12_real_profile_grounding_polarity", analyse_real_profile_grounding_polarity),
]


def main() -> None:
    LOG_PATH.write_text("")  # truncate
    _log(f"chapter9 offline replay — commit={_git_commit()}")
    _log(f"output_dir={OUT_DIR}")

    for label, fn in ANALYSES:
        _log(f"--- analysis: {label} ---")
        try:
            res = fn()
        except Exception as exc:
            _log(f"ERROR in {label}: {exc}\n{traceback.format_exc()}")
            res = AnalysisResult(
                name=label, status="error",
                notes=f"Unhandled exception: {exc}",
            )
        RESULTS.append(res)
        _log(f"  status={res.status} outputs={[p.name for p in res.outputs]}")
        if res.key_numbers:
            _log(f"  key={json.dumps(res.key_numbers, default=str)[:240]}")

    # Manifest
    manifest = {
        "git_commit": _git_commit(),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": str(_REPO_ROOT),
        "output_dir": str(OUT_DIR),
        "corpus_counts_canonical": {
            "pmc_ids_targeted":   1093,
            "pdfs_downloaded":    1132,
            "papers_ingested":    943,
            "rubric_eval_pdfs":   28,
            "calibration_papers": 15,
            "silver_cases_total": 273,
            "silver_dev_cases":   221,
            "silver_test_cases":  52,
            "silver_dev_findings": 1243,
            "silver_test_findings": 353,
            "sampling_method_28pdfs": "uniform random with fixed seed (not stratified)",
        },
        "analyses": [
            {
                "name": r.name, "status": r.status,
                "outputs": [str(p.name) for p in r.outputs],
                "key_numbers": r.key_numbers,
                "notes": r.notes,
                "caveats": r.caveats,
            }
            for r in RESULTS
        ],
    }
    _write_json(OUT_DIR / "manifest.json", manifest)
    _log(f"manifest written: {OUT_DIR / 'manifest.json'}")
    _log("done.")


if __name__ == "__main__":
    main()
