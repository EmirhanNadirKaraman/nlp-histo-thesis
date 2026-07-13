#!/usr/bin/env python3
"""E10 — single-model baselines vs the cascade (RQ3), offline.

The "leave-all-but-one-IN" complement to E12's leave-one-out: instead of dropping a
voter, run EACH model ALONE as the whole summariser (no voting, no escalation) and
score it against silver, then compare to the multi-voter cascade. Answers the crux
question — does the cascade actually beat just running a single strong model?

Every model's per-chunk output is already in the voter_cache (L3/Sonnet present for
all chunks, L1/L2 for ~all), so this is fully offline — no API. The cascade is
recomputed in-run at the frozen config (θ0.9/r0.1, hybrid/greedy) so all rows are
scored identically.

Cost/chunk: single model = 1 call × price(model); cascade = price-weighted per-tier
invocations (same model_prices.json basis as E09).

Usage:
  python -m eval.silver.experiments.E10_baselines.single_model_baselines
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(str(_REPO_ROOT / ".env"))

from eval.silver.map_context import _load_map_context
from eval.silver.map_theta_sweep import (
    REPORTS_DIR,
    ScorerSpec,
    _build_scorer,
    _finding_to_pipeline,
    _make_voters,
    _replay,
)
from eval.silver.matcher import SIMILARITY_THRESHOLD
from eval.silver.pipeline_sweep import _evaluate_outputs
from eval.silver.run_new_summarization_sweeps import (
    BEST_REJECT_THETA, BEST_THETA, BEST_VOTER_SUBSET, _filtered_voter_cache, _subset_meta,
)
from eval.silver.data.schemas import PipelineCaseOutput
from pipeline.stages.knowledge_extraction.config import AgreementConfig, HybridConfig
from pipeline.stages.knowledge_extraction.models import AuditableSummary

THETA, REJECT = BEST_THETA, BEST_REJECT_THETA   # calibrated 5-voter operating point (E07: θ0.9 / reject0.2)
_ANCHOR_CASCADE_STRICT_F1 = 0.7160  # 5-voter (drop_l2_2) related15, θ0.9/r0.2, single_voter_policy=escalate (E08)

_OUT_FIELDS = [
    "system", "model", "strict_f1_optimal", "delta_vs_cascade", "f1_optimal",
    "precision_optimal", "recall_optimal", "n_pipeline", "cost_per_chunk", "strict_f1_per_cost",
]


def _frozen_spec() -> ScorerSpec:
    cfg = AgreementConfig(
        scorer_kind="hybrid", alignment_strategy="greedy",
        tau=0.15, count_alpha=0.25, reuse_weight=0.15, contradiction_weight=0.20,
        hybrid=HybridConfig(w_category=0.15, w_embedding=0.30, w_entity=0.50, w_evidence=0.05),
    )
    return ScorerSpec("hybrid", "hybrid", cfg)


def _price_lookup():
    raw = json.loads((_REPO_ROOT / "configs" / "model_prices.json").read_text())["models"]
    def price(model: str) -> float:
        if model in raw:
            m = raw[model]
        else:
            cand = sorted((k for k in raw if model.startswith(k) or k.startswith(model)),
                          key=len, reverse=True)
            if not cand:
                raise SystemExit(f"no price for model {model!r}")
            m = raw[cand[0]]
        return float(m["input_per_1m"]) + float(m["output_per_1m"])
    return price


def _single_model_outputs(voter_cache, silver_by_case, level, idx) -> list[PipelineCaseOutput]:
    """Treat one voter's per-chunk output as the whole summary — no agreement, no escalation."""
    outs = []
    for e in voter_cache.values():
        if e["case_id"] not in silver_by_case:
            continue
        findings = []
        for cid in e.get("chunk_map", {}):
            if level == "l3":
                raw = e.get("l3", {}).get(cid)
            else:
                lst = e.get(level, {}).get(cid) or []
                raw = lst[idx] if idx < len(lst) else None
            if not raw:
                continue
            sel = AuditableSummary.model_validate(raw).model_dump()
            for f in sel.get("findings", []):
                findings.append(_finding_to_pipeline(f, e["pmcid"], cid, 0.0))
        outs.append(PipelineCaseOutput(case_id=e["case_id"], pmcid=e["pmcid"],
                                       te_id=e["te_id"], run_id=f"single_{level}{idx}",
                                       pipeline_run_id=0, findings=findings))
    return outs


def main() -> None:
    print(f"E10 single-model baselines vs cascade — related15, frozen cascade θ{THETA}/r{REJECT} (offline)")
    ctx = _load_map_context("gemini", embed_cache_path=None)
    price = _price_lookup()

    def score(case_outputs):
        co = [c for c in case_outputs if c.case_id in ctx.silver_by_case]
        return _evaluate_outputs(co, ctx.silver_by_case, ctx.embedder, ctx.embed_cache, SIMILARITY_THRESHOLD)

    # ── cascade (in-run, calibrated SHIPPED config) ──
    # Cascade arm = the calibrated voter selection (BEST_VOTER_SUBSET=drop_l2_2 → 5-voter,
    # Claude-Haiku dropped), filtered the SAME way the sweep does. The single-model baselines
    # below stay on the FULL 6-voter cache (Haiku is still a legitimate single-model baseline).
    scorer = _build_scorer(_frozen_spec(), ctx.agreement_embed_fn)
    cascade_cache = _filtered_voter_cache(ctx.voter_cache, BEST_VOTER_SUBSET)
    print(f"  cascade voter subset = {BEST_VOTER_SUBSET}")
    casc_outputs, _accept, _e, _n, invoked = _replay(
        cascade_cache, scorer, THETA, REJECT, single_voter_policy="escalate",  # E08: escalate best for 5-voter
        force_escalate_on_polarity_conflict=True)
    casc = score(casc_outputs)
    L1, L2, L3 = _make_voters()
    L3_list = L3 if isinstance(L3, (list, tuple)) else [L3]
    # Per-tier $ EXCLUDING the dropped voter, so the cascade cost reflects the SHIPPED voter set
    # (a dropped L2 voter lowers the L2 price paid on every escalated chunk).
    _dl, _di, _ = _subset_meta(BEST_VOTER_SUBSET)
    def _tier_price(voters, lvl):
        return sum(price(v.model) for i, v in enumerate(voters) if not (lvl == _dl and i == _di))
    L1f, L2f, L3f = _tier_price(L1, "l1"), _tier_price(L2, "l2"), _tier_price(L3_list, "l3")
    n_chunks = sum(len(e.get("chunk_map", {})) for e in ctx.voter_cache.values()) or 1
    casc_cost = (invoked["l1"] * L1f + invoked["l2"] * L2f + invoked["l3"] * L3f) / n_chunks

    n_voters = len(L1) + len(L2) - (1 if _dl in ("l1", "l2") else 0)
    rows = [{"system": f"CASCADE ({n_voters}-voter)", "model": "gemini/hybrid/greedy θ0.9", **casc,
             "cost_per_chunk": round(casc_cost, 2)}]

    # ── verification anchor ──
    flag = "OK" if abs(casc["strict_f1_optimal"] - _ANCHOR_CASCADE_STRICT_F1) < 5e-4 else "⚠️ MISMATCH"
    print(f"verification: cascade strict_f1={casc['strict_f1_optimal']:.4f} "
          f"vs E07 {_ANCHOR_CASCADE_STRICT_F1}  [{flag}]")

    # ── each model alone ──
    voters = [("l1", i, v) for i, v in enumerate(L1)] + [("l2", i, v) for i, v in enumerate(L2)] \
        + [("l3", i, v) for i, v in enumerate(L3_list)]
    for level, idx, v in voters:
        m = score(_single_model_outputs(ctx.voter_cache, ctx.silver_by_case, level, idx))
        rows.append({"system": f"single {level.upper()}", "model": f"{v.provider}/{v.model}",
                     **m, "cost_per_chunk": round(price(v.model), 2)})

    for r in rows:
        r["delta_vs_cascade"] = round(r["strict_f1_optimal"] - casc["strict_f1_optimal"], 4)
        r["strict_f1_per_cost"] = round(r["strict_f1_optimal"] / r["cost_per_chunk"], 4) if r["cost_per_chunk"] else 0.0

    print(f"\n{'system':18}{'model':36}{'strF1':>8}{'Δcasc':>8}{'f1':>7}{'P':>6}{'R':>6}"
          f"{'nPipe':>7}{'$/ch':>7}{'F1/$':>7}")
    for r in sorted(rows, key=lambda r: -r["strict_f1_optimal"]):
        print(f"{r['system']:18}{r['model'][:35]:36}{r['strict_f1_optimal']:>8.4f}"
              f"{r['delta_vs_cascade']:>+8.4f}{r['f1_optimal']:>7.4f}{r['precision_optimal']:>6.3f}"
              f"{r['recall_optimal']:>6.3f}{int(r['n_pipeline']):>7}{r['cost_per_chunk']:>7.2f}"
              f"{r['strict_f1_per_cost']:>7.4f}")

    sonnet = next((r for r in rows if "sonnet" in r["model"].lower()), None)
    if sonnet:
        d = sonnet["strict_f1_optimal"] - casc["strict_f1_optimal"]
        print(f"\nCASCADE vs single-Sonnet: ΔstrF1={d:+.4f}  cost {casc_cost:.2f} → {sonnet['cost_per_chunk']:.2f} "
              f"({sonnet['cost_per_chunk']/casc_cost:.0%} of cascade cost)")

    out_dir = REPORTS_DIR / "E10_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    csv_path = out_dir / f"baselines_{ts}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_OUT_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: -r["strict_f1_optimal"]):
            w.writerow(r)
    print(f"\nCSV → {csv_path}")


if __name__ == "__main__":
    main()
