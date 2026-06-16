#!/usr/bin/env python3
"""run_new_summarization_sweeps.py — dependency-ordered MAP calibration sweep.

Thin orchestrator over the existing replay engine: it reuses ``run_sweep`` +
``_load_map_context`` + the CSV / table helpers and only defines the corrected
stage order, grids, and pin-as-you-go ``BEST_*`` constants. No scorer or
alignment logic is reimplemented here.

WHY A NEW FILE — fixes the old circular order
---------------------------------------------
``run_summarization_sweeps.py`` calibrated ``theta`` once at *default* weights
and then tuned weights afterwards (circular: weights change the agreement-score
distribution, which moves the optimal ``theta``). It also gated ``map_weights``
on ``scorer_kind == "hybrid"`` even though the soft-alignment weights drive the
*embedding* scorer too. This harness instead **re-calibrates the thresholds
LAST** (Stage 4), after every score-function change (weights in Stage 2,
alignment in Stage 3), and tunes the soft-alignment weights for *whichever*
scorer won. The old file is left untouched (it carries the historical gate; see
BUGS.md) so scorer semantics and harness orchestration stay easy to bisect.

STAGES (run one → read its CSV → edit the matching ``BEST_*`` here + the field
in ``configs/run.yaml`` → run the next stage, which picks up the new pins):

  1. map_coarse     embedder × scorer × theta × reject_theta  (default weights+alignment)
                    → pin BEST_EMBEDDER, BEST_SCORER
  2. map_weights    soft-align weights for the winning scorer (ANY scorer)
                    + hybrid blend weights ONLY if BEST_SCORER == "hybrid"
                    → pin BEST_TAU/COUNT_ALPHA/REUSE_WEIGHT/CONTRADICTION_WEIGHT (+ BEST_W_* if hybrid)
  3. map_alignment  alignment_strategy {soft_max, greedy, hungarian} × theta × reject_theta (joint)
                    → pin BEST_ALIGNMENT
  4. map_theta      re-confirm theta × reject_theta at the FINAL scorer/embedder/weights/alignment
                    → pin BEST_THETA, BEST_REJECT_THETA
  5. map_gates      single_voter_policy {keep, escalate} × force_escalate_on_polarity_conflict {T, F}
                    → pin run.yaml routing.legacy_single_voter_policy + agreement.force_escalate_*

The held-out final evaluation is a SEPARATE runbook step, NOT a sweep cell here:
pin the winners in ``configs/run.yaml`` → ``run_paper.py --from-selection
heldout15.yaml --sync`` → ``export_pipeline`` → ``evaluate``.

Prereqs (same as map_theta_sweep sweep): a primed voter cache
(``eval/data/map_primer/voter_cache.json``) and silver labels
(``eval/data/silver_findings_related15.jsonl``). Offline replay — no LLM calls;
embedding misses are the only (cheap, cached) API cost.

Usage:
  python -m eval.silver.run_new_summarization_sweeps --stage map_coarse --list-variants
  python -m eval.silver.run_new_summarization_sweeps --stage map_coarse
  python -m eval.silver.run_new_summarization_sweeps --stage map_alignment --metric strict_f1
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(str(_REPO_ROOT / ".env"))

from eval.silver.map_theta_sweep import (  # reused engine + helpers
    REJECT_THETA_GRID,
    REPORTS_DIR,
    THETA_GRID,
    ScorerSpec,
    _print_table,
    _write_csv,
    run_sweep,
)
from eval.silver.matcher import SIMILARITY_THRESHOLD
from eval.silver.run_summarization_sweeps import _load_map_context  # reused loader
from pipeline.stages.summarization.config import AgreementConfig, HybridConfig

# ─────────────────────────────────────────────────────────────────────────────
# Pin-as-you-go winners. Edit after each stage; defaults = current production.
# ─────────────────────────────────────────────────────────────────────────────
BEST_EMBEDDER = "gemini"            # Stage 1 — "gemini" | "openai"
BEST_SCORER = "embedding"           # Stage 1 — "embedding" | "hybrid"

BEST_TAU = 0.15                     # Stage 2 — soft-alignment weights (apply to EITHER scorer)
BEST_COUNT_ALPHA = 0.25
BEST_REUSE_WEIGHT = 0.15
BEST_CONTRADICTION_WEIGHT = 0.20

BEST_W_CATEGORY = 0.25              # Stage 2 — hybrid blend weights (used only if BEST_SCORER=="hybrid")
BEST_W_EMBEDDING = 0.40
BEST_W_ENTITY = 0.25
BEST_W_EVIDENCE = 0.10

BEST_ALIGNMENT = "soft_max"         # Stage 3 — "soft_max" | "greedy" | "hungarian"

BEST_THETA = 0.80                   # Stage 4 — final theta (re-confirmed at the full score function)
BEST_REJECT_THETA = 0.20            # Stage 4 — must be < BEST_THETA

# ─────────────────────────────────────────────────────────────────────────────
# Grids. (theta/reject reuse the validated map_theta_sweep engine grids.)
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDER_GRID = ("gemini", "openai")
SCORER_GRID = ("embedding", "hybrid")
ALIGNMENT_GRID = ("soft_max", "greedy", "hungarian")

# Stage 2 — H-EMB-01 soft-alignment weights, one axis at a time around BEST_*.
TAU_GRID = [0.10, 0.15, 0.20, 0.30]
COUNT_ALPHA_GRID = [0.0, 0.25, 0.50]
REUSE_WEIGHT_GRID = [0.0, 0.15, 0.30]
CONTRADICTION_WEIGHT_GRID = [0.0, 0.20, 0.40]

# Stage 2 (hybrid only) — blend weights; six sum-to-1 hypotheses.
HYBRID_BLEND_GRID: dict[str, tuple[float, float, float, float]] = {
    # (w_category, w_embedding, w_entity, w_evidence)
    "hybrid_default":         (0.25, 0.40, 0.25, 0.10),
    "hybrid_balanced":        (0.25, 0.25, 0.25, 0.25),
    "hybrid_embedding_heavy": (0.15, 0.65, 0.15, 0.05),
    "hybrid_category_heavy":  (0.50, 0.30, 0.15, 0.05),
    "hybrid_entity_heavy":    (0.15, 0.30, 0.50, 0.05),
    "hybrid_evidence_heavy":  (0.15, 0.30, 0.15, 0.40),
}

STAGES = ("map_coarse", "map_weights", "map_alignment", "map_theta", "map_gates")

PIN_HINTS: dict[str, str] = {
    "map_coarse": (
        "PIN after map_coarse → BEST_EMBEDDER, BEST_SCORER (+ run.yaml "
        "summarization.agreement.embedder, summarization.agreement.scorer_kind). "
        "The coarse theta/reject here are NOT final — Stage 4 re-confirms them."
    ),
    "map_weights": (
        "PIN after map_weights → BEST_TAU / BEST_COUNT_ALPHA / BEST_REUSE_WEIGHT / "
        "BEST_CONTRADICTION_WEIGHT (+ run.yaml agreement.{tau,count_alpha,reuse_weight,"
        "contradiction_weight}); if BEST_SCORER=='hybrid' also BEST_W_* (+ run.yaml "
        "agreement.hybrid.{w_category,w_embedding,w_entity,w_evidence})."
    ),
    "map_alignment": (
        "PIN after map_alignment → BEST_ALIGNMENT (+ run.yaml agreement.alignment_strategy). "
        "NOTE: the soft-align weights only bite under soft_max; if greedy/hungarian wins, "
        "only tau (min valid-pair sim) applies — the other three are inert."
    ),
    "map_theta": (
        "PIN after map_theta → BEST_THETA, BEST_REJECT_THETA (+ run.yaml map.theta, "
        "map.reject_theta). This is the FINAL theta at the chosen scorer/embedder/weights/alignment."
    ),
    "map_gates": (
        "PIN after map_gates → run.yaml routing.legacy_single_voter_policy + "
        "agreement.force_escalate_on_polarity_conflict. Then run the held-out evaluation "
        "(separate runbook: run_paper.py --sync → export_pipeline → evaluate)."
    ),
}

_CSV_FIELDS = [
    "embedder", "alignment_strategy", "scorer",
    "tau", "count_alpha", "reuse_weight", "contradiction_weight",
    "w_category", "w_embedding", "w_entity", "w_evidence", "weights_sum",
    "theta", "reject_theta",
    "legacy_single_voter_policy", "force_escalate_on_polarity_conflict", "cascade_path",
    "precision", "recall", "f1", "strict_f1",
    "early_accept_rate", "escalate_rate", "early_accept_precision",
    "n_polarity_conflict_chunks", "polarity_conflict_rate",
    "n_matched", "n_silver", "n_pipeline",
    "split", "seed", "dev_fraction", "sim_threshold",
]


# ── ScorerSpec builders ──────────────────────────────────────────────────────

def _pinned_agreement(alignment: str) -> AgreementConfig:
    """AgreementConfig at the pinned BEST_* weights, with the given alignment."""
    return AgreementConfig(
        scorer_kind=BEST_SCORER,
        tau=BEST_TAU,
        count_alpha=BEST_COUNT_ALPHA,
        reuse_weight=BEST_REUSE_WEIGHT,
        contradiction_weight=BEST_CONTRADICTION_WEIGHT,
        alignment_strategy=alignment,
        hybrid=HybridConfig(
            w_category=BEST_W_CATEGORY, w_embedding=BEST_W_EMBEDDING,
            w_entity=BEST_W_ENTITY, w_evidence=BEST_W_EVIDENCE,
        ),
    )


def _weight_specs() -> list[ScorerSpec]:
    """Stage-2 specs: soft-align weights for the WINNING scorer (the gate fix),
    plus hybrid blend weights only when the winner is hybrid."""
    base = dict(
        scorer_kind=BEST_SCORER, tau=BEST_TAU, count_alpha=BEST_COUNT_ALPHA,
        reuse_weight=BEST_REUSE_WEIGHT, contradiction_weight=BEST_CONTRADICTION_WEIGHT,
    )
    specs = [ScorerSpec(f"{BEST_SCORER}_baseline", BEST_SCORER, AgreementConfig(**base))]
    # Soft-alignment weights — consumed by both EmbeddingSimilarityStrategy and
    # HybridStructuredSimilarity (via _align), so they are tuned for EITHER scorer.
    for knob, grid in (
        ("tau", TAU_GRID), ("count_alpha", COUNT_ALPHA_GRID),
        ("reuse_weight", REUSE_WEIGHT_GRID), ("contradiction_weight", CONTRADICTION_WEIGHT_GRID),
    ):
        for v in grid:
            if v == base[knob]:
                continue
            w = dict(base)
            w[knob] = v
            specs.append(ScorerSpec(f"{BEST_SCORER}_{knob}_{v}", BEST_SCORER, AgreementConfig(**w)))
    # Hybrid blend weights — only meaningful for the hybrid scorer.
    if BEST_SCORER == "hybrid":
        for name, (wc, we, wn, wv) in HYBRID_BLEND_GRID.items():
            specs.append(ScorerSpec(
                name, "hybrid",
                AgreementConfig(**base, hybrid=HybridConfig(
                    w_category=wc, w_embedding=we, w_entity=wn, w_evidence=wv)),
            ))
    return specs


# ── Per-stage plan (specs + axes) ────────────────────────────────────────────

def _stage_plan(stage: str):
    """Return (embedders, specs, thetas, rejects, policies, polarities, align_of_name)."""
    if stage == "map_coarse":
        specs = [ScorerSpec(s, s, AgreementConfig(scorer_kind=s)) for s in SCORER_GRID]
        return (list(EMBEDDER_GRID), specs, list(THETA_GRID), list(REJECT_THETA_GRID),
                ("keep",), (True,), {s.name: "soft_max" for s in specs})
    if stage == "map_weights":
        specs = _weight_specs()
        return ([BEST_EMBEDDER], specs, [BEST_THETA], [BEST_REJECT_THETA],
                ("keep",), (True,), {s.name: "soft_max" for s in specs})
    if stage == "map_alignment":
        specs = [ScorerSpec(f"{BEST_SCORER}_{a}", BEST_SCORER, _pinned_agreement(a))
                 for a in ALIGNMENT_GRID]
        return ([BEST_EMBEDDER], specs, list(THETA_GRID), list(REJECT_THETA_GRID),
                ("keep",), (True,), {f"{BEST_SCORER}_{a}": a for a in ALIGNMENT_GRID})
    if stage == "map_theta":
        spec = ScorerSpec(BEST_SCORER, BEST_SCORER, _pinned_agreement(BEST_ALIGNMENT))
        return ([BEST_EMBEDDER], [spec], list(THETA_GRID), list(REJECT_THETA_GRID),
                ("keep",), (True,), {spec.name: BEST_ALIGNMENT})
    if stage == "map_gates":
        spec = ScorerSpec(BEST_SCORER, BEST_SCORER, _pinned_agreement(BEST_ALIGNMENT))
        return ([BEST_EMBEDDER], [spec], [BEST_THETA], [BEST_REJECT_THETA],
                ("keep", "escalate"), (True, False), {spec.name: BEST_ALIGNMENT})
    raise SystemExit(f"unknown stage {stage!r} (choices: {', '.join(STAGES)})")


def _list_variants(stage: str) -> list[str]:
    """Human-readable per-cell names (no API, no cache needed)."""
    embedders, specs, thetas, rejects, policies, polarities, align_of = _stage_plan(stage)
    cells: list[str] = []
    for emb in embedders:
        for s in specs:
            for t in thetas:
                for rj in rejects:
                    if rj >= t:               # reject_theta must be < theta
                        continue
                    for pol in policies:
                        for fe in polarities:
                            cells.append(
                                f"embedder={emb}  scorer={s.name}  "
                                f"align={align_of.get(s.name)}  theta={t:.2f}  reject={rj:.2f}  "
                                f"single_voter={pol}  polarity_fail={fe}"
                            )
    return cells


def _run_stage(stage: str, args) -> list[dict]:
    embedders, specs, thetas, rejects, policies, polarities, align_of = _stage_plan(stage)
    all_rows: list[dict] = []
    for emb in embedders:
        ctx = _load_map_context(emb, embed_cache_path=args.embed_cache)
        rows = run_sweep(
            voter_cache=ctx.voter_cache,
            silver_by_case=ctx.silver_by_case,
            embedder=ctx.embedder,
            embed_cache=ctx.embed_cache,
            sim_threshold=args.sim_threshold,
            scorer_specs=specs,
            thetas=thetas,
            reject_thetas=rejects,
            split=args.split,
            seed=args.seed,
            dev_fraction=args.dev_fraction,
            agreement_embed_fn=ctx.agreement_embed_fn,
            single_voter_policies=policies,
            force_escalate_on_polarity_conflict_grid=polarities,
        )
        for r in rows:
            r["embedder"] = emb
            r["alignment_strategy"] = align_of.get(r.get("scorer"), "soft_max")
        all_rows.extend(rows)
    return all_rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dependency-ordered MAP calibration sweep (thresholds re-confirmed last)."
    )
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--list-variants", action="store_true",
                    help="Enumerate the stage's cells and exit (no API, no cache).")
    ap.add_argument("--metric", default="f1", choices=["f1", "strict_f1"],
                    help="Metric for the 'Best' line (CSV always carries both).")
    ap.add_argument("--sim-threshold", type=float, default=SIMILARITY_THRESHOLD)
    ap.add_argument("--split", default="all", choices=["dev", "test", "all"])
    ap.add_argument("--dev-fraction", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--embed-cache", default=None,
                    help="Override embedding cache path (default: per-embedder default).")
    args = ap.parse_args()

    print(PIN_HINTS[args.stage])

    if args.list_variants:
        cells = _list_variants(args.stage)
        print(f"\n{args.stage}: {len(cells)} variant(s)")
        for c in cells:
            print("  " + c)
        return

    rows = _run_stage(args.stage, args)
    if not rows:
        print("No rows produced (empty grid or no silver overlap).")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    csv_path = REPORTS_DIR / f"new_sweep_{args.stage}_{timestamp}.csv"
    _write_csv(csv_path, rows, _CSV_FIELDS)
    _print_table(rows)

    best = max(rows, key=lambda r: float(r[args.metric]))
    print(
        f"\nBest {args.metric}: embedder={best['embedder']} scorer={best['scorer']} "
        f"alignment={best['alignment_strategy']} theta={best['theta']} "
        f"reject_theta={best['reject_theta']}  {args.metric}={float(best[args.metric]):.4f}"
    )
    print(f"CSV → {csv_path}")
    print("\n" + PIN_HINTS[args.stage])


if __name__ == "__main__":
    main()
