#!/usr/bin/env python3
"""run_new_summarization_sweeps.py — screen→refine MAP calibration sweep.

Thin orchestrator over the existing replay engine: reuses ``run_sweep`` +
``_load_map_context`` + the CSV helper and only defines the stage order, grids,
and pin-as-you-go ``BEST_*`` constants. No scorer/alignment logic is
reimplemented here, and the old ``run_summarization_sweeps.py`` is untouched.

DESIGN — screen→refine (dependency-aware)
-----------------------------------------
``alignment_strategy`` is a *structural* knob: under one-to-one alignments
(``greedy``/``hungarian``) three of the four soft-align weights go inert (only
``tau`` survives), and the best embedder/scorer can differ by alignment. So the
structure block ``{embedder, scorer, alignment}`` is decided JOINTLY first — at
default weights, on a cheap coarse threshold grid — and only the surviving
finalist structures are weight-tuned. This avoids tuning weights that a later-
chosen alignment would discard, while a finalist set (not just the single
winner) guards against eliminating a structure that only shines after tuning.

STAGES (run one → read its summary → edit the matching ``BEST_*`` /
``FINALIST_STRUCTURES`` / ``VOTER_SUBSET_*_CONFIGS`` here + the field in
``configs/run.yaml`` → next stage). Each stage's output lands in its own
E-numbered subfolder ``eval/reports/E##_<stage>/`` (see docs/EXPERIMENTS.md):

  1. structure_screen      embedder × scorer × alignment at DEFAULT weights × a
                           COARSE theta/reject grid. Identifies promising structures,
                           not final configs (top-K ∪ within --keep-within ∪ Pareto
                           ∪ economy ∪ knee, + a diversity add). → paste FINALIST_STRUCTURES.
  2. family_refine         Refines ONLY the finalist structures: applicable weights ×
                           the FULL theta/reject grid (weight × theta swept jointly).
                           soft_max → tau/count_alpha/reuse/contradiction (+ blend);
                           greedy/hungarian → tau only (+ blend, pre-alignment signal).
                           → pin BEST_EMBEDDER, BEST_SCORER, BEST_ALIGNMENT, BEST_TAU,
                           BEST_COUNT_ALPHA/REUSE_WEIGHT/CONTRADICTION_WEIGHT (only if
                           soft_max), BEST_W_* (only if hybrid), provisional BEST_THETA/
                           REJECT. Also prints VOTER_SUBSET_SCREEN_CONFIGS (quality/
                           economy/knee operating points).
  3. voter_subset_screen   [OPTIONAL] VOTER_SUBSET_GRID (leave-one-voter-out, single
                           drops only) on the pasted configs at each config's FIXED
                           theta — no theta sweep. → paste VOTER_SUBSET_REFINE_CONFIGS.
  4. voter_subset_refine   [OPTIONAL] the selected (config, subset) pairs × the FULL
                           theta grid. → pin BEST_VOTER_SUBSET (+ final BEST_THETA/REJECT
                           and the winning config's BEST_* if it changed).
  5. map_theta             Re-confirm theta × reject_theta at the FINAL score function,
                           using BEST_VOTER_SUBSET. → pin final BEST_THETA, BEST_REJECT_THETA.
  6. map_gates             single_voter_policy {keep, escalate} ×
                           force_escalate_on_polarity_conflict {True, False}, using
                           BEST_VOTER_SUBSET. → pin run.yaml routing.legacy_single_voter_policy
                           + agreement.force_escalate_on_polarity_conflict (→ BEST_SINGLE_VOTER_POLICY).
  7. map_theta_shipped     [E08b] θ × reject re-swept at the SHIPPED gate (the one E08 chose,
                           BEST_SINGLE_VOTER_POLICY) — the shipped config's θ-curve. CHARACTERIZATION,
                           not a θ re-selection (θ0.9 is gate-invariant; map_theta/E07 stays the
                           default-gate selection record). → this is the curve E09's cost-quality
                           frontier reads. No new pin.

Stages 3–4 (voter_subset) are OPTIONAL — skip them and BEST_VOTER_SUBSET stays
"all" — unless you are chasing a cost-efficient cascade by dropping a diluting voter
(only meaningful at lower theta, where the cheap voters actually drive decisions).

METRIC / MATCHER — selection uses optimal / Hungarian one-to-one semantic
matching (max embedding-similarity assignment) as the PRIMARY matcher: metric
``strict_f1_optimal``, tie-break lower ``escalate_rate``, then ``f1_optimal``,
then simpler/closer-to-default config. ``greedy`` matching is reported (``*_greedy``
columns) as a sensitivity diagnostic only — it NEVER selects. NOTE: "optimal" is
optimal *by embedding similarity*, NOT a globally optimal strict-F1 assignment.

CALIBRATION SPLIT — runs on ``--split all`` by default because the silver
calibration set is small and a stable point estimate is wanted. ``--split
dev``/``test`` stay available for sensitivity/debugging but are NOT the default.
Sweep scores are CALIBRATION metrics, not unbiased held-out performance. The
held-out final evaluation is a SEPARATE runbook step (pin → ``run_paper.py
--sync`` → ``export_pipeline`` → ``evaluate --matcher optimal``); only that score
is reported as generalization.

Prereqs: a primed voter cache (``eval/data/map_primer/voter_cache.json``) and
silver labels (``eval/data/silver_findings_related15.jsonl``). Offline replay —
no LLM calls; embedding-cache misses are the only (cheap, cached) cost. Generated
CSVs land in per-experiment E-numbered subfolders under ``eval/reports/`` (e.g.
``E06_family_refine/sweep_<ts>.csv``) and are kept untracked.

CHECKPOINT / RESUME — every stage appends each completed cell to
``eval/reports/E##_<stage>/checkpoint.csv`` (flushed per cell), so a Ctrl-C keeps
all finished cells. Re-running the stage resumes automatically: cells already in
the checkpoint are skipped (the embeddings they used are SQLite-cached too, so a
resume pays no API). A changed grid / finalists / weights / split rotates the
stale checkpoint aside and starts fresh; ``--fresh`` forces that. The checkpoint
is deleted once the stage completes and the final timestamped CSV is written.
The voter-subset axis is backward-compatible in the cell key (subset "all" → the
original 6-tuple), so adding it never invalidated an in-flight checkpoint.

Usage:
  python -m eval.silver.analysis.run_new_summarization_sweeps --stage structure_screen --list-variants
  python -m eval.silver.analysis.run_new_summarization_sweeps --stage structure_screen
  python -m eval.silver.analysis.run_new_summarization_sweeps --stage family_refine
  python -m eval.silver.analysis.run_new_summarization_sweeps --stage family_refine --from-csv <csv>   # re-print configs, no rerun
  python -m eval.silver.analysis.run_new_summarization_sweeps --stage voter_subset_screen
  python -m eval.silver.analysis.run_new_summarization_sweeps --stage map_theta
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from eval.paths import REPO_ROOT as _REPO_ROOT

load_dotenv(str(_REPO_ROOT / ".env"))

from eval.silver.analysis.map_theta_sweep import (  # reused engine + helpers
    REJECT_THETA_GRID,
    REPORTS_DIR,
    THETA_GRID,
    ScorerSpec,
    _make_voters,
    _write_csv,
    run_sweep,
)
from nlp_histo.evaluation.matching.matcher import SIMILARITY_THRESHOLD
from eval.silver.analysis.map_context import _load_map_context  # reused loader
from nlp_histo.pipeline.stages.knowledge_extraction.config import AgreementConfig, HybridConfig

# Pin-as-you-go winners. Edit after each stage. Current values = the E06 family_refine
# QUALITY winner (gemini/hybrid/greedy, entity-heavy blend, θ=0.9): strict_f1_optimal
# 0.7133 at escalate_rate 0.962 — CONFIRMED by map_theta (E07) at the full score fn.
# The cheap "economy" operating point is NOT pinned here; it lives in the E06 frontier
# CSV / VOTER_SUBSET_SCREEN_CONFIGS (run voter_subset stages to pursue it).
FINALIST_STRUCTURES: list[tuple[str, str, str]] = [   # (embedder, scorer_kind, alignment)
    # E05 structure_screen 2026-06-21 (sweep_20260621T134042) — 12 finalists,
    # ranked by strict_f1_optimal at each structure's max-F1 cell. Annotations =
    # the operating point that earned each finalist's reason (economy/knee show
    # their CHEAP θ, not the max-F1 θ).
    # All 12 structures kept (hungarian RESTORED 2026-06-21): greedy≈hungarian was measured
    # equivalent only at DEFAULT weights (E05, max|ΔstrictF1|=0.0015); family_refine sweeps
    # NON-default weights, so we run both one-to-one aligners here rather than defend that
    # extrapolation. The θ (≥0.5) and reject ({0,0.2}) reductions stay (both defensible).
    # Annotations re-derived 2026-06-21 with the COST-AWARE axis (_cost_frac, price-weighted
    # L1→L2 + L2→L3) — reasons/operating points differ from the old escalate_rate picks
    # (pareto moved greedy→hungarian for gemini/hybrid; gemini/embedding/greedy knee θ0.8→0.85).
    ("gemini", "hybrid",    "greedy"),     # top-k/within-best        @θ0.9  cost=0.95 esc=0.94 sf1=0.712
    ("gemini", "hybrid",    "hungarian"),  # top-k/within-best/pareto @θ0.7  cost=0.38 esc=0.32 sf1=0.612
    ("gemini", "hybrid",    "soft_max"),   # top-k/within-best/pareto @θ0.7  cost=0.30 esc=0.24 sf1=0.593
    ("openai", "hybrid",    "soft_max"),   # within-best              @θ0.9  cost=0.97 esc=0.96 sf1=0.707
    ("openai", "hybrid",    "greedy"),     # within-best              @θ0.9  cost=0.97 esc=0.97 sf1=0.707
    ("openai", "hybrid",    "hungarian"),  # within-best              @θ0.9  cost=0.97 esc=0.97 sf1=0.707
    ("openai", "embedding", "soft_max"),   # within-best/pareto       @θ0.75 cost=0.38 esc=0.31 sf1=0.617
    ("openai", "embedding", "greedy"),     # within-best/pareto       @θ0.7  cost=0.37 esc=0.30 sf1=0.606
    ("openai", "embedding", "hungarian"),  # within-best/pareto       @θ0.7  cost=0.37 esc=0.30 sf1=0.606
    ("gemini", "embedding", "greedy"),     # within-best/pareto/knee  @θ0.85 cost=0.59 esc=0.51 sf1=0.664
    ("gemini", "embedding", "hungarian"),  # within-best/pareto       @θ0.7  cost=0.20 esc=0.15 sf1=0.565
    ("gemini", "embedding", "soft_max"),   # pareto/economy           @θ0.7  cost=0.09 esc=0.08 sf1=0.538
]

BEST_EMBEDDER = "gemini"            # E06 family_refine — "gemini" | "openai"
BEST_SCORER = "hybrid"              # E06 family_refine — "embedding" | "hybrid"

BEST_TAU = 0.15                     # E06 family_refine — applicable weights of the pinned structure
BEST_COUNT_ALPHA = 0.25             #   count_alpha/reuse/contradiction are INERT under greedy (one-to-one)
BEST_REUSE_WEIGHT = 0.15            #   — kept at defaults; only tau + the hybrid blend apply here.
BEST_CONTRADICTION_WEIGHT = 0.20

BEST_W_CATEGORY = 0.15              # E06 family_refine — hybrid blend (entity_heavy; only if BEST_SCORER=="hybrid")
BEST_W_EMBEDDING = 0.30
BEST_W_ENTITY = 0.50
BEST_W_EVIDENCE = 0.05

BEST_ALIGNMENT = "greedy"           # E06 family_refine — "soft_max" | "greedy" | "hungarian"

BEST_THETA = 0.90                   # E07 map_theta (5-voter drop_l2_2, 2026-06-22) — argmax over the full θ×reject grid, strict_f1 0.7147
BEST_REJECT_THETA = 0.20            # E07 (5-voter) — reject=0.2 weakly dominates 0.1 (sf1 0.7147 vs 0.7140, 1 fewer L3 call).
#                                     The (0.0,0.1] band is now EMPTY (reject 0.0≡0.1) and the lowest-consensus chunk sits in
#                                     (0.1,0.2], so 0.2 rejects it. (The 6-voter config had the empty band at (0.1,0.2] → reject 0.1;
#                                     dropping haiku shifted the agreement-score distribution.) Must be < BEST_THETA. Noise-level.

BEST_VOTER_SUBSET = "drop_l2_2"     # SELECTED on calibration (related15, E06c, 2026-06-22): drop Claude-Haiku
#                                     from L2 → 5-voter cascade. E06c (full θ×reject grid): quality/drop_l2_2
#                                     owns the max-F1 operating point, 0.7147 vs all 0.7131 (+0.0016 = NOISE,
#                                     within E11's ±0.004 CI) at −18% per-chunk cost. So this is an
#                                     F1-EQUIVALENT, cheaper config; selected for (1) cost and (2) evaluator-
#                                     independence — the 5-voter L1/L2 ensemble is entirely non-Anthropic, so
#                                     its agreement with the Opus-silver labeler is not a same-provider artifact
#                                     (L3 is still Sonnet). NOT an F1 claim.
#                                     Decision made ENTIRELY on calibration. E14 REPORTS the 5-voter held-out
#                                     generalization once (descriptive) — it does NOT gate this pin; choosing
#                                     the config on the held-out set would be test-set selection.
#                                     map_theta/map_gates (E07/E08) replay this drop_l2_2 SELECTION over the
#                                     6-voter primer. _make_voters() and E12 (LOO) stay on the full 6 voters
#                                     (the primer was gathered with 6; re-prime is the avoided billable step).

BEST_SINGLE_VOTER_POLICY = "escalate"   # SHIPPED gate, chosen by E08 map_gates (5-voter): escalate is the gate argmax
#                                         (0.7160 vs keep 0.7147). With 2 L2 voters the N=1 case is more common, and a lone
#                                         voter has no agreement signal → escalate it to L3/Sonnet (escalate-when-uncertain).
#                                         Gain over keep is noise-level; chosen as the E08 argmax + principled N=1 handling.
#                                         The SHIPPED config — used by ALL downstream stages: the map_theta re-run (so E09's
#                                         frontier is the shipped curve), E09/E10/E11/E14, + run.yaml routing. The θ DECISION
#                                         (θ0.9) was made at the default gate (the original keep E07) and is gate-invariant,
#                                         so the escalate map_theta re-run regenerates the shipped curve WITHOUT re-selecting θ.
BEST_FORCE_ESCALATE_POLARITY = True     # E08 — force-escalate on polarity conflict (unchanged default; ~no effect at θ0.9).

# Grids. (theta/reject reuse the validated map_theta_sweep engine grids.)
EMBEDDER_GRID = ("gemini", "openai")
SCORER_GRID = ("embedding", "hybrid")
ALIGNMENT_GRID = ("soft_max", "greedy", "hungarian")

# family_refine — applicable soft-alignment weights, one axis at a time around the
# AgreementConfig defaults (each finalist tuned from scratch, not from a BEST_*).
_DEF = AgreementConfig()
_DEFAULTS = {
    "tau": _DEF.tau,
    "count_alpha": _DEF.count_alpha,
    "reuse_weight": _DEF.reuse_weight,
    "contradiction_weight": _DEF.contradiction_weight,
}
TAU_GRID = [0.10, 0.15, 0.20, 0.30]
COUNT_ALPHA_GRID = [0.0, 0.25, 0.50]
REUSE_WEIGHT_GRID = [0.0, 0.15, 0.30]
CONTRADICTION_WEIGHT_GRID = [0.0, 0.20, 0.40]

# family_refine (hybrid only) — blend weights; six sum-to-1 hypotheses.
HYBRID_BLEND_GRID: dict[str, tuple[float, float, float, float]] = {
    # (w_category, w_embedding, w_entity, w_evidence)
    "default":         (0.25, 0.40, 0.25, 0.10),
    "balanced":        (0.25, 0.25, 0.25, 0.25),
    "embedding_heavy": (0.15, 0.65, 0.15, 0.05),
    "category_heavy":  (0.50, 0.30, 0.15, 0.05),
    "entity_heavy":    (0.15, 0.30, 0.50, 0.05),
    "evidence_heavy":  (0.15, 0.30, 0.15, 0.40),
}
_DEFAULT_BLEND = HYBRID_BLEND_GRID["default"]   # == HybridConfig() default; the base spec covers it

# structure_screen — coarse threshold grid (cheap; the full grid is re-confirmed in map_theta).
COARSE_THETA_GRID = [0.70, 0.75, 0.80, 0.85, 0.90]
COARSE_REJECT_GRID = [0.10, 0.20]

# family_refine — reject ENDPOINTS {0.0, 0.2}: the no-rejection baseline and the grid's most
# aggressive setting, swept jointly with EVERY weight config so reject sensitivity is tested
# off-default HERE (not extrapolated from default-weight screening). Semantics: reject rejects
# primary <= reject_theta. reject=0.0 → only primary==0.0, and EXP_1 showed rej=-1.0 ≡ rej=0.0
# (2026-05-26) → no gate chunk scores exactly 0, so 0.0 = reject NOTHING. The only score mass in
# [0,0.2] is a ~0 hard-disagreement cluster; (0.1,0.2] is empty (score_distribution.py: 0/3704),
# so reject=0.2 reproduces 0.1's rejections. The two endpoints bracket the whole effect — no
# third regime between them. Pinned reject=0.1 is interior to the tested range; map_theta keeps
# the full REJECT_THETA_GRID for the final re-confirmation.
FAMILY_REFINE_REJECT_GRID = [0.0, 0.2]

# family_refine — local theta grid starting at 0.5 (drops 0.30, 0.40). The quality winner is at
# θ=0.90 (monotone: more L3 escalation → higher strict_f1; post-gemini E05 structure_screen had
# ALL structures max at θ=0.9), and family_refine ranks structures by their max-F1 cell, so a
# 0.5-start grid selects the SAME winner — only the cheap (low-escalation) end of the cost/quality
# frontier below 0.5 is dropped here. map_theta KEEPS the full THETA_GRID, so the winning config's
# θ (incl. the sub-0.5 economy region) is re-confirmed there. Trims cells 14→10/spec (−29%).
FAMILY_REFINE_THETA_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]

# structure_screen finalist selection.
DEFAULT_TOP_K = 3
DEFAULT_KEEP_WITHIN = 0.02          # also keep structures within this gap of the best primary metric
# Pareto-aware finalist selection over (strict_f1_optimal ↑, cost_frac ↓):
PARETO_CAP = 8                      # max non-dominated frontier structures kept as finalists
MIN_ECONOMY_F1 = 0.50               # quality floor an economy (low-cost) finalist must clear
COST_LAMBDA = 0.20                  # knee = argmax(strict_f1_optimal − COST_LAMBDA · cost_frac)

# Cost axis for pareto/economy/knee. Blended in+out $/1M per tier, summed over the tier's
# voters (configs/model_prices.json × batch/voter_configs.py): L1 = flash-lite(0.5) +
# gpt-4o-mini(0.75) + gpt-4.1-nano(0.5); L2 = flash(2.8) + gpt-4.1-mini(2.0) + haiku(4.8);
# L3 = sonnet(18.0). NOTE: assumes the default "all" voter subset and ~equal tokens/call; a
# voter-subset run (dropping a voter) would lower a tier's weight — not modelled here yet.
TIER_COST_BLENDED = {"l1": 1.75, "l2": 9.60, "l3": 18.00}


def _cost_frac(row: dict) -> float:
    """Price-weighted escalation cost in [0,1] — the cost axis for pareto/economy/knee.

    ``(w2·n_l2_invoked + w3·n_l3_invoked) / ((w2+w3)·n_chunks)``. The L1 term is constant
    (``n_l1_invoked == n_chunks``, every chunk runs L1) so it is omitted — this is
    rank-equivalent to total per-chunk cost. Unlike ``escalate_rate`` (= n_l3/n_chunks),
    this charges the L1→L2 escalation too (L2 ≈ 5.5× L1 per call), so an L2-heavy variant
    is no longer mis-scored as cheap. Falls back to 0.0 on missing counts."""
    n = float(row.get("n_chunks") or 0.0)
    if n <= 0:
        return 0.0
    w2, w3 = TIER_COST_BLENDED["l2"], TIER_COST_BLENDED["l3"]
    l2 = float(row.get("n_l2_invoked") or 0.0)
    l3 = float(row.get("n_l3_invoked") or 0.0)
    return (w2 * l2 + w3 * l3) / ((w2 + w3) * n)

_FINALIST_EMPTY_MSG = (
    "FINALIST_STRUCTURES is empty — family_refine has nothing to refine.\n"
    "  1. Run:  python -m eval.silver.analysis.run_new_summarization_sweeps --stage structure_screen\n"
    "  2. Copy the printed `FINALIST_STRUCTURES = [...]` block into the constant near the top\n"
    "     of eval/silver/analysis/run_new_summarization_sweeps.py.\n"
    "  3. Re-run: python -m eval.silver.analysis.run_new_summarization_sweeps --stage family_refine"
)

# Voter-subset / leave-one-voter-out (targeted, AFTER family_refine)
# Single-voter drops only (no combined L1+L2 drops yet). drop_lN_i removes the
# i-th voter of level N (stable index → model; see docs/EXPERIMENTS.md).
VOTER_SUBSET_GRID = (
    "all",
    "drop_l1_0", "drop_l1_1", "drop_l1_2",
    "drop_l2_0", "drop_l2_1", "drop_l2_2",
)

# voter_subset_screen input — paste from family_refine's printed block. Each dict is a
# full operating point (config) to test under VOTER_SUBSET_GRID at its OWN fixed theta.
# Re-derived 2026-06-22 from E06 family_refine sweep_20260622T013444 under the COST-AWARE
# selection (cost_frac = price-weighted L1→L2 + L2→L3; escalation_breakdown.py). quality =
# max strict_f1; economy = min cost_frac ≥ 0.50 floor; knee = argmax(strict_f1 − 0.20·cost_frac).
# NOTE the knee fix: the script's first print (run without the cost_frac column) picked the
# CHEAPER pure-embedding cell (gemini/embedding θ0.8, sf1 0.630 @ cost 0.42); the cost_frac
# rule's true argmax is hybrid/embedding_heavy θ0.8 (sf1 0.674 @ cost 0.63). Frontier plot:
# eval/reports/E06_family_refine/frontier_20260622.png.
VOTER_SUBSET_SCREEN_CONFIGS: list[dict] = [
    {"label": "quality", "embedder": "gemini", "scorer_kind": "hybrid", "alignment_strategy": "greedy",
     "tau": 0.15, "count_alpha": 0.25, "reuse_weight": 0.15, "contradiction_weight": 0.20,
     "w_category": 0.15, "w_embedding": 0.30, "w_entity": 0.50, "w_evidence": 0.05,   # entity_heavy
     "theta": 0.9, "reject_theta": 0.2},   # sf1 0.7131 @ cost 0.977 (reject 0.2 ≡ 0.1: empty (0.1,0.2] band)
    {"label": "economy", "embedder": "gemini", "scorer_kind": "hybrid", "alignment_strategy": "greedy",
     "tau": 0.15, "count_alpha": 0.25, "reuse_weight": 0.15, "contradiction_weight": 0.20,
     "w_category": 0.15, "w_embedding": 0.30, "w_entity": 0.15, "w_evidence": 0.40,   # evidence_heavy
     "theta": 0.5, "reject_theta": 0.0},   # sf1 0.5419 @ cost 0.0741 (cheapest cell ≥ 0.50 floor)
    {"label": "knee", "embedder": "gemini", "scorer_kind": "hybrid", "alignment_strategy": "greedy",
     "tau": 0.15, "count_alpha": 0.25, "reuse_weight": 0.15, "contradiction_weight": 0.20,
     "w_category": 0.15, "w_embedding": 0.65, "w_entity": 0.15, "w_evidence": 0.05,   # embedding_heavy
     "theta": 0.8, "reject_theta": 0.0},   # sf1 0.6744 @ cost 0.6324 (argmax sf1 − 0.20·cost_frac)
]

# voter_subset_refine input — paste from voter_subset_screen's printed block. Each dict
# is a (config, voter_subset) pair to re-sweep over the FULL theta grid.
VOTER_SUBSET_REFINE_CONFIGS: list[dict] = [
    # E06c — full-θ frontier test of the drops that DOMINATE 'all' at fixed θ on the
    # 2026-06-22 corrected configs (E06b screen + price-weighted per-chunk cost; supersedes
    # the old "every drop loses → keep all" reading, which was measured on the prior
    # openai/θ0.4 economy). Fixed-θ domination ≠ frontier domination ('all' at a different θ
    # may match a drop), so each dominating drop is re-swept over the FULL θ/reject grid
    # against its 'all' baseline. Selected drops (fixed-θ, in-sample):
    #   quality θ0.9  drop haiku (L2_2): Δsf1 +0.0016 (noise) @ -18% cost  → near-free cost cut
    #   economy θ0.5  drop gpt-4.1-nano (L1_2): Δsf1 +0.0333 @ -2.5% cost   → dominates
    #   economy θ0.5  drop gpt-4o-mini (L1_1):  Δsf1 +0.0321 @ -7.6% cost   → dominates
    # (knee drops help sf1 but COST MORE — a trade, not domination — so excluded here.)
    # Weights mirror VOTER_SUBSET_SCREEN_CONFIGS; theta omitted (full grid swept).
    {'label': 'quality', 'base_config': 'quality', 'voter_subset': 'all',
     'embedder': 'gemini', 'scorer_kind': 'hybrid', 'alignment_strategy': 'greedy',
     'tau': 0.15, 'count_alpha': 0.25, 'reuse_weight': 0.15, 'contradiction_weight': 0.2,
     'w_category': 0.15, 'w_embedding': 0.3, 'w_entity': 0.5, 'w_evidence': 0.05},   # entity_heavy
    {'label': 'quality', 'base_config': 'quality', 'voter_subset': 'drop_l2_2',
     'embedder': 'gemini', 'scorer_kind': 'hybrid', 'alignment_strategy': 'greedy',
     'tau': 0.15, 'count_alpha': 0.25, 'reuse_weight': 0.15, 'contradiction_weight': 0.2,
     'w_category': 0.15, 'w_embedding': 0.3, 'w_entity': 0.5, 'w_evidence': 0.05},   # drop haiku
    {'label': 'economy', 'base_config': 'economy', 'voter_subset': 'all',
     'embedder': 'gemini', 'scorer_kind': 'hybrid', 'alignment_strategy': 'greedy',
     'tau': 0.15, 'count_alpha': 0.25, 'reuse_weight': 0.15, 'contradiction_weight': 0.2,
     'w_category': 0.15, 'w_embedding': 0.3, 'w_entity': 0.15, 'w_evidence': 0.4},   # evidence_heavy
    {'label': 'economy', 'base_config': 'economy', 'voter_subset': 'drop_l1_2',
     'embedder': 'gemini', 'scorer_kind': 'hybrid', 'alignment_strategy': 'greedy',
     'tau': 0.15, 'count_alpha': 0.25, 'reuse_weight': 0.15, 'contradiction_weight': 0.2,
     'w_category': 0.15, 'w_embedding': 0.3, 'w_entity': 0.15, 'w_evidence': 0.4},   # drop gpt-4.1-nano
    {'label': 'economy', 'base_config': 'economy', 'voter_subset': 'drop_l1_1',
     'embedder': 'gemini', 'scorer_kind': 'hybrid', 'alignment_strategy': 'greedy',
     'tau': 0.15, 'count_alpha': 0.25, 'reuse_weight': 0.15, 'contradiction_weight': 0.2,
     'w_category': 0.15, 'w_embedding': 0.3, 'w_entity': 0.15, 'w_evidence': 0.4},   # drop gpt-4o-mini
]

_VS_SCREEN_EMPTY_MSG = (
    "VOTER_SUBSET_SCREEN_CONFIGS is empty.\n"
    "  Run family_refine, copy its printed `VOTER_SUBSET_SCREEN_CONFIGS = [...]` block into\n"
    "  the constant near the top of this file, then re-run --stage voter_subset_screen."
)
_VS_REFINE_EMPTY_MSG = (
    "VOTER_SUBSET_REFINE_CONFIGS is empty.\n"
    "  Run voter_subset_screen, copy its printed `VOTER_SUBSET_REFINE_CONFIGS = [...]` block\n"
    "  into the constant near the top of this file, then re-run --stage voter_subset_refine."
)

STAGES = (
    "structure_screen",
    "family_refine",
    "voter_subset_screen",
    "voter_subset_refine",
    "map_theta",
    "map_gates",
    "map_theta_shipped",
)

# Group key for the per-stage "Best per …" summary (None → only the overall best).
GROUP_FIELDS: dict[str, tuple[str, ...]] = {
    "structure_screen": ("embedder", "scorer_kind", "alignment_strategy"),
    "family_refine": ("embedder", "scorer_kind", "alignment_strategy"),
}

PIN_HINTS: dict[str, str] = {
    "structure_screen": (
        "structure_screen: broad screen of embedder × scorer × alignment at DEFAULT weights on a "
        "coarse theta grid. Copy the printed `FINALIST_STRUCTURES = [...]` block into the constant "
        "near the top of this file, then run --stage family_refine. Do NOT pin a single winner here."
    ),
    "family_refine": (
        "family_refine: refines ONLY FINALIST_STRUCTURES (applicable weights × full theta grid). "
        "PIN after → BEST_EMBEDDER, BEST_SCORER, BEST_ALIGNMENT, BEST_TAU, "
        "BEST_COUNT_ALPHA/REUSE_WEIGHT/CONTRADICTION_WEIGHT (only if BEST_ALIGNMENT==soft_max), "
        "BEST_W_* (only if BEST_SCORER==hybrid), provisional BEST_THETA/BEST_REJECT_THETA. "
        "Also prints VOTER_SUBSET_SCREEN_CONFIGS — paste it to run --stage voter_subset_screen. "
        "run.yaml: agreement.embedder, scorer_kind, alignment_strategy + the applicable weights."
    ),
    "voter_subset_screen": (
        "voter_subset_screen: tests VOTER_SUBSET_GRID on the pasted VOTER_SUBSET_SCREEN_CONFIGS at "
        "each config's FIXED theta/reject (no theta sweep). Copy the printed "
        "`VOTER_SUBSET_REFINE_CONFIGS = [...]` block, then run --stage voter_subset_refine."
    ),
    "voter_subset_refine": (
        "voter_subset_refine: re-sweeps the pasted (config, voter_subset) pairs over the FULL theta "
        "grid. PIN after → BEST_VOTER_SUBSET (+ final BEST_THETA/BEST_REJECT_THETA and the winning "
        "config's BEST_* if it changed). map_theta / map_gates then run with BEST_VOTER_SUBSET."
    ),
    "map_theta": (
        "PIN after map_theta → final BEST_THETA, BEST_REJECT_THETA (+ run.yaml map.theta, "
        "map.reject_theta), at the chosen scorer/embedder/weights/alignment."
    ),
    "map_gates": (
        "PIN after map_gates → run.yaml routing.legacy_single_voter_policy + "
        "agreement.force_escalate_on_polarity_conflict. Then run --stage map_theta_shipped (E08b)."
    ),
    "map_theta_shipped": (
        "map_theta_shipped (E08b): the SHIPPED config's θ-curve — θ × reject re-swept at the gate E08 "
        "chose (BEST_SINGLE_VOTER_POLICY). This is what E09's cost-quality frontier reads. NO new pin: "
        "θ0.9/reject0.2 are unchanged (gate-invariant) — this only characterizes the shipped config. "
        "Expect θ0.9/reject0.2 → the E08 headline. Then run the held-out battery (E09–E14)."
    ),
}

_CSV_FIELDS = [
    "embedder", "scorer_kind", "variant", "alignment_strategy",
    "tau", "count_alpha", "reuse_weight", "contradiction_weight",
    "w_category", "w_embedding", "w_entity", "w_evidence", "weights_sum",
    "theta", "reject_theta",
    "legacy_single_voter_policy", "force_escalate_on_polarity_conflict", "cascade_path",
    # PRIMARY matcher = optimal / Hungarian (selection runs on strict_f1_optimal):
    "strict_f1_optimal", "f1_optimal", "precision_optimal", "recall_optimal", "n_matched_optimal",
    # DIAGNOSTIC matcher = greedy (reported, never selected on):
    "strict_f1_greedy", "f1_greedy", "precision_greedy", "recall_greedy", "n_matched_greedy",
    "n_silver", "n_pipeline",
    "early_accept_rate", "escalate_rate", "early_accept_precision",
    # Per-tier invocation counts for cost (n_l3_invoked == escalate_rate·n_chunks):
    "n_chunks", "n_l1_invoked", "n_l2_invoked", "n_l3_invoked",
    "n_polarity_conflict_chunks", "polarity_conflict_rate",
    "split", "seed", "dev_fraction", "sim_threshold",
    # Voter-subset columns — END-appended so existing checkpoints stay prefix-aligned
    # (the resumed family_refine wrote its header without these; "all" rows omit them).
    "voter_subset", "dropped_voter_level", "dropped_voter_idx", "dropped_voter_model",
    "n_l1_voters", "n_l2_voters",
]


# ScorerSpec builders

def _pinned_agreement(alignment: str) -> AgreementConfig:
    """AgreementConfig at the pinned BEST_* weights, with the given alignment.
    Used by map_theta / map_gates (the structure is already chosen)."""
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


def _default_spec(scorer: str, alignment: str) -> ScorerSpec:
    """structure_screen: one all-default-weights spec per (scorer, alignment)."""
    return ScorerSpec(
        f"{scorer}__{alignment}__default", scorer,
        AgreementConfig(scorer_kind=scorer, alignment_strategy=alignment),
    )


def _refine_specs(emb: str, scorer: str, alignment: str) -> list[ScorerSpec]:
    """family_refine: APPLICABLE weight variants for one finalist structure.

    - ``tau`` always (min valid-pair similarity, used by every alignment).
    - ``count_alpha``/``reuse_weight``/``contradiction_weight`` ONLY under
      ``soft_max`` — inert for one-to-one, so not swept there.
    - hybrid blend presets under ANY alignment (they shape the pre-alignment
      category/entity/evidence sub-signals). The ``default`` blend equals the
      base config, so it is skipped to avoid a silent duplicate.

    Names are namespaced by ``emb__scorer__alignment`` so finalists never collide.
    """
    prefix = f"{emb}__{scorer}__{alignment}"
    base = dict(scorer_kind=scorer, alignment_strategy=alignment, **_DEFAULTS)
    specs = [ScorerSpec(f"{prefix}__base", scorer, AgreementConfig(**base))]

    knobs = [("tau", TAU_GRID)]
    if alignment == "soft_max":
        knobs += [
            ("count_alpha", COUNT_ALPHA_GRID),
            ("reuse_weight", REUSE_WEIGHT_GRID),
            ("contradiction_weight", CONTRADICTION_WEIGHT_GRID),
        ]
    for knob, grid in knobs:
        for v in grid:
            if v == _DEFAULTS[knob]:
                continue
            w = dict(base)
            w[knob] = v
            specs.append(ScorerSpec(f"{prefix}__{knob}_{v}", scorer, AgreementConfig(**w)))

    if scorer == "hybrid":
        for name, blend in HYBRID_BLEND_GRID.items():
            if blend == _DEFAULT_BLEND:        # base already covers the default blend
                continue
            wc, we, wn, wv = blend
            specs.append(ScorerSpec(
                f"{prefix}__blend_{name}", "hybrid",
                AgreementConfig(**base, hybrid=HybridConfig(
                    w_category=wc, w_embedding=we, w_entity=wn, w_evidence=wv)),
            ))
    return specs


# Voter-subset filtering + identity (leave-one-voter-out)

def _voter_labels() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Canonical index→(provider, model) for L1 and L2, from _make_voters()."""
    L1, L2, _ = _make_voters()
    return ([(v.provider, v.model) for v in L1], [(v.provider, v.model) for v in L2])


def _subset_meta(voter_subset: str) -> tuple[str, object, str]:
    """(level, idx, 'provider/model') for a drop_lN_i subset; ('', '', '') for 'all'."""
    if voter_subset == "all":
        return ("", "", "")
    _, level, idx_str = voter_subset.split("_")        # drop_l1_2 → ('drop','l1','2')
    idx = int(idx_str)
    labels = _voter_labels()[0 if level == "l1" else 1]
    model = f"{labels[idx][0]}/{labels[idx][1]}" if idx < len(labels) else "?"
    return (level, idx, model)


def _subset_voter_counts(voter_subset: str) -> tuple[int, int]:
    n1, n2 = (len(g) for g in _voter_labels())
    if voter_subset.startswith("drop_l1"):
        n1 -= 1
    elif voter_subset.startswith("drop_l2"):
        n2 -= 1
    return n1, n2


_FILTER_CACHE: dict[tuple, dict] = {}


def _filtered_voter_cache(voter_cache: dict, voter_subset: str) -> dict:
    """Cache view with one voter dropped (drop_lN_i), WITHOUT mutating the original.
    ``all`` → the original object. Drops by nulling slot ``idx`` per chunk via
    ``enumerate`` (never indexes), so short/sparse lists and already-None slots are safe
    no-ops (the N=1 caveat applies: single_voter_policy='keep' auto-accepts). Memoised
    per (cache, subset)."""
    if voter_subset == "all":
        return voter_cache
    key = (id(voter_cache), voter_subset)
    cached = _FILTER_CACHE.get(key)
    if cached is not None:
        return cached
    level, idx, _ = _subset_meta(voter_subset)
    out: dict = {}
    for sid, e in voter_cache.items():
        lvl = {cid: ([None if j == idx else d for j, d in enumerate(lst)] if lst else lst)
               for cid, lst in e[level].items()}
        out[sid] = {**e, level: lvl}
    _FILTER_CACHE[key] = out
    return out


def _assert_cache_matches_voters(voter_cache: dict) -> None:
    """Fail loud if the cache's per-level slot count != the current _make_voters set —
    a regenerated cache primed with different voters would mislabel index→model."""
    n_l1, n_l2 = (len(g) for g in _voter_labels())

    def max_slots(level: str) -> int:
        return max((len(lst) for e in voter_cache.values()
                   for lst in e.get(level, {}).values() if lst), default=0)

    ml1, ml2 = max_slots("l1"), max_slots("l2")
    if ml1 != n_l1 or ml2 != n_l2:
        raise SystemExit(
            f"Voter cache slot count ({ml1} L1 / {ml2} L2) != current _make_voters "
            f"({n_l1} L1 / {n_l2} L2) — index→model labels would be wrong. Re-prime first."
        )


def _validate_config(cfg: dict, need: tuple[str, ...]) -> None:
    base = ("embedder", "scorer_kind", "alignment_strategy",
            "tau", "count_alpha", "reuse_weight", "contradiction_weight")
    missing = [k for k in base + need if k not in cfg]
    if missing:
        raise SystemExit(f"voter-subset config {cfg.get('label', cfg)!r} missing keys: {missing}")


def _spec_from_config(cfg: dict) -> ScorerSpec:
    """Build a ScorerSpec from a pasted operating-point config dict."""
    sk = cfg["scorer_kind"]
    blend = (cfg.get("w_category"), cfg.get("w_embedding"),
             cfg.get("w_entity"), cfg.get("w_evidence"))
    hyb = (HybridConfig(w_category=blend[0], w_embedding=blend[1],
                        w_entity=blend[2], w_evidence=blend[3])
           if all(b is not None for b in blend) else HybridConfig())
    weights = AgreementConfig(
        scorer_kind=sk, alignment_strategy=cfg["alignment_strategy"],
        tau=cfg["tau"], count_alpha=cfg["count_alpha"],
        reuse_weight=cfg["reuse_weight"], contradiction_weight=cfg["contradiction_weight"],
        hybrid=hyb,
    )
    name = cfg.get("label") or f"{cfg['embedder']}__{sk}__{cfg['alignment_strategy']}"
    return ScorerSpec(name, sk, weights)


# Per-stage plan (specs + axes + name→{kind,alignment,embedder} maps)

def _stage_plan(stage: str):
    """Return (embedders, specs, thetas, rejects, policies, polarities,
    kind_of, align_of, embedder_of).

    ``kind_of`` / ``align_of`` map spec.name → scorer KIND / alignment.
    ``embedder_of`` is None for embedder-agnostic stages, or maps spec.name →
    the single embedder that spec must run under (family_refine finalists)."""
    if stage == "structure_screen":
        specs, kind_of, align_of = [], {}, {}
        for sc in SCORER_GRID:
            for al in ALIGNMENT_GRID:
                sp = _default_spec(sc, al)
                specs.append(sp)
                kind_of[sp.name] = sc
                align_of[sp.name] = al
        return (list(EMBEDDER_GRID), specs, list(COARSE_THETA_GRID), list(COARSE_REJECT_GRID),
                ("keep",), (True,), kind_of, align_of, None)

    if stage == "family_refine":
        specs, kind_of, align_of, embedder_of = [], {}, {}, {}
        for emb, scorer, alignment in FINALIST_STRUCTURES:
            for sp in _refine_specs(emb, scorer, alignment):
                specs.append(sp)
                kind_of[sp.name] = scorer
                align_of[sp.name] = alignment
                embedder_of[sp.name] = emb
        embedders = sorted({e for e, _, _ in FINALIST_STRUCTURES})
        return (embedders, specs, list(FAMILY_REFINE_THETA_GRID), list(FAMILY_REFINE_REJECT_GRID),
                ("keep",), (True,), kind_of, align_of, embedder_of)

    if stage == "map_theta":
        spec = ScorerSpec(BEST_SCORER, BEST_SCORER, _pinned_agreement(BEST_ALIGNMENT))
        # E07 — θ SELECTION at the DEFAULT gate (keep). The gate is chosen next (E08, map_gates); the
        # SHIPPED config's θ-curve at the chosen gate is E08b (map_theta_shipped). Forward protocol —
        # this stage stays at the default gate so it remains the clean, reproducible θ-selection record.
        return ([BEST_EMBEDDER], [spec], list(THETA_GRID), list(REJECT_THETA_GRID),
                ("keep",), (True,), {spec.name: BEST_SCORER}, {spec.name: BEST_ALIGNMENT}, None)

    if stage == "map_gates":
        spec = ScorerSpec(BEST_SCORER, BEST_SCORER, _pinned_agreement(BEST_ALIGNMENT))
        return ([BEST_EMBEDDER], [spec], [BEST_THETA], [BEST_REJECT_THETA],
                ("keep", "escalate"), (True, False),
                {spec.name: BEST_SCORER}, {spec.name: BEST_ALIGNMENT}, None)

    if stage == "map_theta_shipped":
        spec = ScorerSpec(BEST_SCORER, BEST_SCORER, _pinned_agreement(BEST_ALIGNMENT))
        # E08b — the SHIPPED config's θ-curve: re-sweep θ × reject at the gate E08 chose
        # (BEST_SINGLE_VOTER_POLICY / BEST_FORCE_ESCALATE_POLARITY). CHARACTERIZATION for E09's frontier
        # + the headline — NOT a θ re-selection (θ0.9 was selected by E07 at the default gate and is
        # gate-invariant). Same pins as map_theta; only the gate differs. → E09 reads this stage's CSV.
        return ([BEST_EMBEDDER], [spec], list(THETA_GRID), list(REJECT_THETA_GRID),
                (BEST_SINGLE_VOTER_POLICY,), (BEST_FORCE_ESCALATE_POLARITY,),
                {spec.name: BEST_SCORER}, {spec.name: BEST_ALIGNMENT}, None)

    raise SystemExit(f"unknown stage {stage!r} (choices: {', '.join(STAGES)})")


def _effective_specs(emb: str, specs: list[ScorerSpec], embedder_of) -> list[ScorerSpec]:
    """Specs that should run under embedder ``emb``. For finalist-tagged stages
    (embedder_of set) this filters to the finalist's own embedder so each spec is
    BUILT with the right embed_fn — shared by _run_stage and _list_variants so the
    listed count always matches the run."""
    if embedder_of is None:
        return list(specs)
    return [s for s in specs if embedder_of[s.name] == emb]


# Ranking (primary metric, tie-breaks: escalate, f1_optimal, simpler config)

def _deviation(row: dict) -> int:
    """Count of fields differing from the production defaults (gemini / embedding /
    soft_max / AgreementConfig weights / default blend). Lower = simpler. Inert
    (``None``) weights and missing keys are skipped, so it is robust to partial
    rows and never penalises a one-to-one row for its blanked soft-align weights."""
    d = 0
    if row.get("embedder", "gemini") != "gemini":
        d += 1
    if row.get("scorer_kind", "embedding") != "embedding":
        d += 1
    if row.get("alignment_strategy", "soft_max") != "soft_max":
        d += 1
    for knob, default in _DEFAULTS.items():
        v = row.get(knob)
        if v not in (None, "") and float(v) != float(default):   # "" = blanked/round-tripped inert
            d += 1
    blend = (row.get("w_category"), row.get("w_embedding"),
             row.get("w_entity"), row.get("w_evidence"))
    if all(b not in (None, "") for b in blend):
        if tuple(round(float(b), 4) for b in blend) != tuple(round(float(b), 4) for b in _DEFAULT_BLEND):
            d += 1
    return d


def _rank(row: dict, metric: str) -> tuple[float, float, float, int]:
    """Selection key: higher PRIMARY metric (an optimal-matcher metric), then LOWER
    cost_frac (price-weighted escalation), then higher f1_optimal, then SIMPLER config (fewer deviations
    from defaults). Greedy metrics NEVER enter the key."""
    return (
        float(row[metric]),
        -_cost_frac(row),
        float(row.get("f1_optimal") or 0.0),
        -_deviation(row),
    )


def _best_per_group(rows: list[dict], group_fields: tuple[str, ...], metric: str) -> dict:
    best: dict = {}
    for r in rows:
        key = tuple(r.get(f) for f in group_fields)
        if key not in best or _rank(r, metric) > _rank(best[key], metric):
            best[key] = r
    return best


def _cell_struct(r: dict) -> tuple:
    """(embedder, scorer_kind, alignment) identity of a sweep row/cell."""
    return (r["embedder"], r["scorer_kind"], r["alignment_strategy"])


def _select_finalists(rows: list[dict], metric: str, top_k: int, keep_within: float, *,
                      pareto_cap: int = PARETO_CAP, min_economy_f1: float = MIN_ECONOMY_F1,
                      cost_lambda: float = COST_LAMBDA) -> list[tuple]:
    """PURE, Pareto-aware finalist selection over (``metric`` ↑, ``cost_frac`` ↓),
    so structure_screen keeps BOTH the quality corner (high F1, expensive) AND the
    economy/knee end (low cost).

    COST AXIS is ``_cost_frac`` (price-weighted L1→L2 + L2→L3 escalation), NOT bare
    ``escalate_rate`` — the latter counts only L3 reach and is blind to the L2 cost an
    L2-heavy variant pays (L2 ≈ 5.5× L1/call). ``escalate_rate`` stays a reported column.

    CELL-LEVEL is load-bearing: the cost axis (cost_frac) is driven by THETA, not
    by structure identity, so the Pareto frontier is computed over the FULL cells
    (structure × theta × reject) — NOT per-structure bests, which would collapse every
    structure to its θ=0.9 (most-expensive) cell and discard the entire cheap-θ economy
    frontier. A structure becomes a finalist if it TOUCHES the frontier at any cell.
    ``top-k`` / ``within-best`` stay QUALITY picks (per-structure max-metric).

    Selection uses ONLY ``metric`` (an optimal-matcher metric) + ``cost_frac`` —
    never a greedy metric. Returns ``[(struct, [reasons], {reason: cell})]`` ranked
    best-first by the structure's max-metric. struct = (embedder, scorer_kind, alignment).
    Reasons (additive): top-k, within-best, pareto, economy, knee, diversity:<axis>.
    NOTE: ``--pareto-cap`` trims quality-first (harmless at 12 structures; if the grid
    grows, trim to preserve the escalate-spread — economy/knee already backstop it)."""
    if not rows:
        return []

    def q(r):
        return float(r[metric])

    def c(r):
        return _cost_frac(r)

    # Quality view: per-structure best-by-metric → top-k / within-best / ranking order.
    struct_best = _best_per_group(rows, GROUP_FIELDS["structure_screen"], metric)
    ranked = sorted(struct_best.items(), key=lambda kv: _rank(kv[1], metric), reverse=True)
    top = q(ranked[0][1])

    reasons: dict[tuple, list[str]] = {}
    evidence: dict[tuple, dict] = {}

    def add(struct, why, cell):
        if why not in reasons.setdefault(struct, []):
            reasons[struct].append(why)
            evidence.setdefault(struct, {})[why] = cell

    # 1. top-k + 2. within-best — quality (the structure's max-metric cell)
    for struct, r in ranked[:top_k]:
        add(struct, "top-k", r)
    for struct, r in ranked:
        if top - q(r) <= keep_within:
            add(struct, "within-best", r)

    # 3. CELL-level Pareto frontier (q ↑, c ↓); keep one cheapest frontier cell per structure
    frontier = [
        r for r in rows
        if not any(q(o) >= q(r) and c(o) <= c(r) and (q(o) > q(r) or c(o) < c(r))
                   for o in rows if o is not r)
    ]
    cheapest_front: dict[tuple, dict] = {}
    for r in sorted(frontier, key=c):                      # lowest escalate first
        cheapest_front.setdefault(_cell_struct(r), r)
    # Cap preserving the CHEAP end: prioritise frontier structures NOT already kept by a
    # quality reason (top-k/within-best), cheapest first. Quality-first trimming would drop
    # exactly the cost-efficient structures (e.g. gemini/embedding) this stage exists to keep.
    by_cost = sorted(cheapest_front, key=lambda s: c(cheapest_front[s]))
    ordered_front = [s for s in by_cost if s not in reasons] + [s for s in by_cost if s in reasons]
    for struct in ordered_front[:pareto_cap]:
        add(struct, "pareto", cheapest_front[struct])

    # 4. economy — lowest-escalate CELL clearing the quality floor
    eligible = [r for r in rows if q(r) >= min_economy_f1]
    if eligible:
        ec = min(eligible, key=c)
        add(_cell_struct(ec), "economy", ec)

    # 5. knee — best quality − λ·cost CELL
    kn = max(rows, key=lambda r: q(r) - cost_lambda * c(r))
    add(_cell_struct(kn), "knee", kn)

    # 6. diversity (additive backstop) on the quality ranking
    near = [kv for kv in ranked if top - q(kv[1]) <= 2 * keep_within]
    for ax_i, ax_n in ((0, "embedder"), (1, "scorer_kind"), (2, "alignment")):
        if len({s[ax_i] for s in reasons}) == 1:
            for struct, r in near:
                if struct[ax_i] not in {s[ax_i] for s in reasons}:
                    add(struct, f"diversity:{ax_n}", r)
                    break

    return [(s, reasons[s], evidence[s]) for s, _ in ranked if s in reasons]


# Cells + checkpoint / resume

def _stage_voter_subset(stage: str) -> str:
    """Grid stages run a single voter subset: 'all' (screen/family), or BEST_VOTER_SUBSET
    for map_theta/map_gates (which respect the pinned subset)."""
    return BEST_VOTER_SUBSET if stage in ("map_theta", "map_gates", "map_theta_shipped") else "all"


def _iter_cells(stage: str):
    """Yield one dict per cell — the SINGLE enumeration source shared by --list-variants,
    the run loop, and the checkpoint keys, so the listed count, what actually runs, and the
    resume keys can never drift apart. Every cell carries a ``voter_subset``."""
    if stage == "voter_subset_screen":
        for cfg in VOTER_SUBSET_SCREEN_CONFIGS:
            _validate_config(cfg, ("theta", "reject_theta"))
            spec = _spec_from_config(cfg)
            for subset in VOTER_SUBSET_GRID:          # each config × subsets at its FIXED theta
                yield {"embedder": cfg["embedder"], "spec": spec,
                       "scorer_kind": cfg["scorer_kind"], "alignment": cfg["alignment_strategy"],
                       "theta": cfg["theta"], "reject": cfg["reject_theta"],
                       "policy": "keep", "polarity": True, "voter_subset": subset}
        return
    if stage == "voter_subset_refine":
        for cfg in VOTER_SUBSET_REFINE_CONFIGS:
            _validate_config(cfg, ("voter_subset",))
            spec = _spec_from_config(cfg)
            for t in THETA_GRID:                      # selected (config, subset) × FULL theta grid
                for rj in REJECT_THETA_GRID:
                    if rj >= t:
                        continue
                    yield {"embedder": cfg["embedder"], "spec": spec,
                           "scorer_kind": cfg["scorer_kind"], "alignment": cfg["alignment_strategy"],
                           "theta": t, "reject": rj, "policy": "keep", "polarity": True,
                           "voter_subset": cfg["voter_subset"]}
        return

    embedders, specs, thetas, rejects, policies, polarities, kind_of, align_of, embedder_of = \
        _stage_plan(stage)
    vs = _stage_voter_subset(stage)
    for emb in embedders:
        for s in _effective_specs(emb, specs, embedder_of):
            for t in thetas:
                for rj in rejects:
                    if rj >= t:                       # reject_theta must be < theta
                        continue
                    for pol in policies:
                        for fe in polarities:
                            yield {
                                "embedder": emb, "spec": s,
                                "scorer_kind": kind_of[s.name], "alignment": align_of[s.name],
                                "theta": t, "reject": rj, "policy": pol, "polarity": fe,
                                "voter_subset": vs,
                            }


def _canonical_key(embedder, variant, theta, reject, policy, polarity, voter_subset="all") -> tuple:
    """Canonical cell identity — the ONE place a cell key is built, coercing types
    identically whether the source is a cell dict or a reloaded CSV row (so a resume
    skip can never silently miss and re-append a duplicate row).

    BACKWARD-COMPAT: ``voter_subset="all"`` yields the original 6-tuple, so every
    existing grid stage's keys + signature stay byte-identical to before the
    voter-subset axis existed (a live family_refine checkpoint remains valid). Only a
    non-"all" subset appends the 7th element."""
    base = (str(embedder), str(variant), f"{float(theta):.2f}", f"{float(reject):.2f}",
            str(policy), str(polarity).lower())
    return base if str(voter_subset) == "all" else base + (str(voter_subset),)


def _cell_key_from_cell(c: dict) -> tuple:
    return _canonical_key(c["embedder"], c["spec"].name, c["theta"], c["reject"],
                          c["policy"], c["polarity"], c.get("voter_subset", "all"))


def _cell_key_from_row(r: dict) -> tuple:
    # rows use the engine's column name "reject_theta"; cells use "reject".
    # voter_subset missing OR empty-string (old checkpoint rows / blank CSV cell) → "all"
    # → 6-tuple (back-compat); `or "all"` coerces both None and "" so a blank cell can't
    # become a spurious 7-tuple that fails to match its 6-tuple cell key.
    return _canonical_key(r["embedder"], r["variant"], r["theta"], r["reject_theta"],
                          r["legacy_single_voter_policy"], r["force_escalate_on_polarity_conflict"],
                          r.get("voter_subset") or "all")


def _weights_sig(cfg) -> tuple:
    """Resolved weight VALUES (not the spec name): editing _DEFAULTS or a default
    blend between runs invalidates a stale checkpoint even though names are unchanged."""
    h = getattr(cfg, "hybrid", None)
    return (
        cfg.scorer_kind, round(cfg.tau, 6), round(cfg.count_alpha, 6),
        round(cfg.reuse_weight, 6), round(cfg.contradiction_weight, 6), cfg.alignment_strategy,
        None if h is None else (round(h.w_category, 6), round(h.w_embedding, 6),
                                round(h.w_entity, 6), round(h.w_evidence, 6)),
    )


def _plan_signature(stage: str, cells: list[dict], args) -> str:
    """Hash the intended cell set + resolved weights + run params. A mismatch ⇒ the
    checkpoint is stale (grid / finalists / weights / split / seed changed)."""
    specs_seen = {c["spec"].name: c["spec"].weights for c in cells}
    payload = {
        "stage": stage,
        "cells": sorted(_cell_key_from_cell(c) for c in cells),
        "weights": [[n, _weights_sig(w)] for n, w in sorted(specs_seen.items())],
        "split": args.split, "seed": args.seed,
        "dev_fraction": args.dev_fraction, "sim_threshold": args.sim_threshold,
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# Experiment numbers (match docs/EXPERIMENTS.md) — prefix the output folders
# so eval/reports/ sorts by experiment.
_EXP_PREFIX = {
    "structure_screen": "E05",
    "family_refine": "E06",
    "voter_subset_screen": "E06b",
    "voter_subset_refine": "E06c",
    "map_theta": "E07",
    "map_gates": "E08",
    "map_theta_shipped": "E08b",
}


def _stage_dir(stage: str) -> Path:
    """Per-experiment subfolder under eval/reports/ (e.g. eval/reports/E06_family_refine/).
    Keeps each experiment's checkpoint + outputs together — see docs/EXPERIMENTS.md."""
    d = REPORTS_DIR / f"{_EXP_PREFIX.get(stage, 'E00')}_{stage}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ckpt_csv(stage: str) -> Path:
    return _stage_dir(stage) / "checkpoint.csv"


def _ckpt_meta(stage: str) -> Path:
    return _stage_dir(stage) / "checkpoint.meta.json"


def _rotate_checkpoint(stage: str, reason: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    moved = []
    for p in (_ckpt_csv(stage), _ckpt_meta(stage)):
        if p.exists():
            p.rename(p.with_name(p.name + f".stale-{ts}"))
            moved.append(p.name)
    if moved:
        print(f"Checkpoint reset ({reason}) — moved {', '.join(moved)} aside.")


def _read_checkpoint_rows(path: Path) -> list[dict]:
    """Load completed rows; defensively skip a truncated trailing line so an
    interrupted final write just recomputes that one cell."""
    required = ("embedder", "variant", "theta", "reject_theta",
                "legacy_single_voter_policy", "force_escalate_on_polarity_conflict")
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if all(r.get(k) not in (None, "") for k in required):
                rows.append(r)
    return rows


def _checkpoint_state(stage: str, signature: str, fresh: bool) -> tuple[list[dict], bool]:
    """Return (done_rows, resume). Resume ONLY when both files exist AND the
    signature matches; otherwise rotate any stale files aside and start fresh
    (writing a new meta sidecar)."""
    ckpt, meta = _ckpt_csv(stage), _ckpt_meta(stage)
    if fresh:
        _rotate_checkpoint(stage, "--fresh")
    elif ckpt.exists() and meta.exists():
        try:
            saved = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            saved = {}
        if saved.get("signature") == signature:
            done = _read_checkpoint_rows(ckpt)
            print(f"Resuming {stage}: {len(done)} cell(s) already done in {ckpt.name}.")
            return done, True
        _rotate_checkpoint(stage, "config changed (signature mismatch)")
    elif ckpt.exists() or meta.exists():
        _rotate_checkpoint(stage, "incomplete checkpoint")

    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(
        {"signature": signature, "stage": stage,
         "created": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    return [], False


def _cleanup_checkpoint(stage: str) -> None:
    for p in (_ckpt_csv(stage), _ckpt_meta(stage)):
        if p.exists():
            p.unlink()


# Variant listing (from _iter_cells, the same source the run uses)

def _list_variants(stage: str) -> list[str]:
    """Human-readable per-cell names (no API, no cache needed)."""
    return [
        f"embedder={c['embedder']}  scorer_kind={c['scorer_kind']}  variant={c['spec'].name}  "
        f"align={c['alignment']}  theta={c['theta']:.2f}  reject={c['reject']:.2f}  "
        f"voter_subset={c['voter_subset']}  single_voter={c['policy']}  polarity_fail={c['polarity']}"
        for c in _iter_cells(stage)
    ]


def _stamp_row(r: dict, emb: str, kind_of: dict, align_of: dict, voter_subset: str = "all") -> dict:
    """Stamp embedder / scorer_kind / variant / alignment + voter-subset metadata onto a
    run_sweep row, and blank the soft-align weights that are INERT under one-to-one
    alignment so the CSV / best-line don't report them as meaningful (``tau`` stays)."""
    name = r.get("scorer")                        # run_sweep stamps spec.name here
    r["variant"] = name                           # concrete variant name
    r["scorer_kind"] = kind_of.get(name, name)    # embedding / hybrid
    r["embedder"] = emb
    align = align_of.get(name, "soft_max")
    r["alignment_strategy"] = align
    if align in ("greedy", "hungarian"):
        r["count_alpha"] = None
        r["reuse_weight"] = None
        r["contradiction_weight"] = None
    r["voter_subset"] = voter_subset
    level, idx, model = _subset_meta(voter_subset)
    r["dropped_voter_level"], r["dropped_voter_idx"], r["dropped_voter_model"] = level, idx, model
    r["n_l1_voters"], r["n_l2_voters"] = _subset_voter_counts(voter_subset)
    return r


def _run_stage(stage: str, args) -> list[dict]:
    """Run the stage cell-by-cell, appending each completed cell to the checkpoint
    (flushed) and resuming from it. Each cell is one single-element ``run_sweep``
    call, so the cell key is constructed in exactly one place (the harness)."""
    if stage == "family_refine" and not FINALIST_STRUCTURES:
        raise SystemExit(_FINALIST_EMPTY_MSG)
    if stage == "voter_subset_screen" and not VOTER_SUBSET_SCREEN_CONFIGS:
        raise SystemExit(_VS_SCREEN_EMPTY_MSG)
    if stage == "voter_subset_refine" and not VOTER_SUBSET_REFINE_CONFIGS:
        raise SystemExit(_VS_REFINE_EMPTY_MSG)

    cells = list(_iter_cells(stage))
    signature = _plan_signature(stage, cells, args)
    done_rows, resume = _checkpoint_state(stage, signature, args.fresh)
    done_keys = {_cell_key_from_row(r) for r in done_rows}
    remaining = [c for c in cells if _cell_key_from_cell(c) not in done_keys]

    if not remaining:
        print(f"All {len(cells)} cell(s) already in the checkpoint — nothing to run.")
        return done_rows
    print(f"{stage}: {len(cells)} cell(s) — {len(done_rows)} done, {len(remaining)} to run"
          f"{' (resuming)' if resume else ''}.")

    # one ctx load per embedder that still has work (stable, first-seen order)
    by_emb: dict[str, list[dict]] = {}
    for c in remaining:
        by_emb.setdefault(c["embedder"], []).append(c)

    ckpt = _ckpt_csv(stage)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    new_rows: list[dict] = []
    fh = ckpt.open("a" if resume else "w", newline="", encoding="utf-8")
    try:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        if not resume:
            writer.writeheader()
            fh.flush()
        for emb, emb_cells in by_emb.items():
            ctx = _load_map_context(emb, embed_cache_path=args.embed_cache)
            if stage in ("voter_subset_screen", "voter_subset_refine"):
                _assert_cache_matches_voters(ctx.voter_cache)   # guard index→model labels
            for c in emb_cells:
                cache = _filtered_voter_cache(ctx.voter_cache, c["voter_subset"])
                rows = run_sweep(
                    voter_cache=cache,
                    silver_by_case=ctx.silver_by_case,
                    embedder=ctx.embedder,
                    embed_cache=ctx.embed_cache,
                    sim_threshold=args.sim_threshold,
                    scorer_specs=[c["spec"]],
                    thetas=[c["theta"]],
                    reject_thetas=[c["reject"]],
                    split=args.split,
                    seed=args.seed,
                    dev_fraction=args.dev_fraction,
                    agreement_embed_fn=ctx.agreement_embed_fn,
                    single_voter_policies=[c["policy"]],
                    force_escalate_on_polarity_conflict_grid=[c["polarity"]],
                )
                for r in rows:
                    _stamp_row(r, emb, {c["spec"].name: c["scorer_kind"]},
                               {c["spec"].name: c["alignment"]}, c["voter_subset"])
                    writer.writerow(r)
                    new_rows.append(r)
                fh.flush()   # durable per cell — a Ctrl-C keeps everything written so far
    finally:
        fh.close()
    return done_rows + new_rows


def _fmt_best(r: dict, metric: str) -> str:
    return (
        f"embedder={r['embedder']} scorer_kind={r.get('scorer_kind')} variant={r.get('variant')} "
        f"alignment={r['alignment_strategy']} "
        f"tau={r.get('tau')} count_alpha={r.get('count_alpha')} reuse={r.get('reuse_weight')} "
        f"contradiction={r.get('contradiction_weight')} "
        f"w=({r.get('w_category')},{r.get('w_embedding')},{r.get('w_entity')},{r.get('w_evidence')}) "
        f"theta={r.get('theta')} reject_theta={r.get('reject_theta')}\n"
        f"      PRIMARY {metric}={float(r[metric]):.4f}  f1_optimal={float(r.get('f1_optimal') or 0):.4f}  "
        f"escalate_rate={float(r.get('escalate_rate') or 0.0):.3f}\n"
        f"      diagnostic (greedy): strict_f1_greedy={float(r.get('strict_f1_greedy') or 0):.4f}  "
        f"f1_greedy={float(r.get('f1_greedy') or 0):.4f}"
    )


# Display the operating point that earns a finalist's MOST cost-relevant reason, so
# an "economy" finalist shows its CHEAP θ — not its max-F1 θ.
_REASON_DISPLAY_PRIORITY = ("economy", "knee", "pareto", "within-best", "top-k")


def _display_cell(evidence: dict, fallback: dict) -> dict:
    for why in _REASON_DISPLAY_PRIORITY:
        if why in evidence:
            return evidence[why]
    return fallback


def _report_structure_screen(rows: list[dict], args, csv_path: Path) -> None:
    struct_best = _best_per_group(rows, GROUP_FIELDS["structure_screen"], args.metric)
    ranked = sorted(struct_best.items(), key=lambda kv: _rank(kv[1], args.metric), reverse=True)
    finalists = _select_finalists(rows, args.metric, args.top_k, args.keep_within,
                                  pareto_cap=args.pareto_cap, min_economy_f1=args.min_economy_f1,
                                  cost_lambda=args.cost_lambda)
    reason_of = {s: r for s, r, _ in finalists}

    print(f"\nStructures ranked by {args.metric}  (★ = finalist; Pareto over "
          f"{args.metric} ↑ / escalate_rate ↓).")
    print("  θ/esc in this table = each structure's MAX-F1 (quality) cell.")
    print(f"  {'':2}{'embedder':8} {'scorer':10} {'align':10} {'strictF1':>8} {'f1opt':>7} "
          f"{'cost':>6} {'esc':>6} {'θ':>4} {'rej':>4} {'[greedy]':>9}  reasons")
    for struct, r in ranked:
        emb, sk, al = struct
        mark = "★" if struct in reason_of else " "
        why = "/".join(reason_of.get(struct, []))
        print(f"  {mark} {str(emb):8} {str(sk):10} {str(al):10} "
              f"{float(r[args.metric]):8.4f} {float(r.get('f1_optimal') or 0):7.4f} "
              f"{_cost_frac(r):6.3f} {float(r.get('escalate_rate') or 0):6.3f} {str(r.get('theta')):>4} "
              f"{str(r.get('reject_theta')):>4} {float(r.get('strict_f1_greedy') or 0):9.4f}  {why}")

    print(f"\nCSV → {csv_path}")
    print(f"\n{len(finalists)} finalist(s). Paste into FINALIST_STRUCTURES near the top of this "
          "file, then run --stage family_refine.")
    print("(annotation = the operating point that earns each finalist's reason — "
          "economy/knee show their CHEAP θ, not the max-F1 θ):\n")
    print("FINALIST_STRUCTURES = [")
    for struct, why, ev in finalists:
        emb, sk, al = struct
        cell = _display_cell(ev, struct_best[struct])
        print(f'    ("{emb}", "{sk}", "{al}"),    # {"/".join(why)} '
              f'@θ{cell.get("theta")} cost={_cost_frac(cell):.2f} esc={float(cell.get("escalate_rate") or 0):.2f} '
              f'sf1={float(cell[args.metric]):.3f}')
    print("]")


# Voter-subset reports (config blocks for the next stage + pin hints)

def _num(v):
    if v in (None, ""):
        return None
    try:
        return round(float(v), 6)
    except (TypeError, ValueError):
        return None


def _print_config_block(var_name: str, configs: list[dict]) -> None:
    print(f"{var_name} = [")
    for cfg in configs:
        print("    {")
        for k, v in cfg.items():
            print(f"        {k!r}: {v!r},")
        print("    },")
    print("]")


def _config_from_cell(cell: dict, label: str, *, for_refine: bool = False) -> dict:
    """Turn a result row/cell into a copy-pastable operating-point config dict.
    screen configs carry theta/reject_theta (fixed); refine configs carry
    base_config + voter_subset and omit theta (the full grid is swept)."""
    sk = cell.get("scorer_kind") or cell.get("scorer")

    def w(key, default):
        v = _num(cell.get(key))
        return default if v is None else v   # inert (blanked) one-to-one weights → harmless default

    cfg: dict = {"label": label}
    if for_refine:
        cfg["base_config"] = label
        cfg["voter_subset"] = cell.get("voter_subset") or "all"
    cfg["embedder"] = cell["embedder"]
    cfg["scorer_kind"] = sk
    cfg["alignment_strategy"] = cell["alignment_strategy"]
    cfg["tau"] = w("tau", _DEFAULTS["tau"])
    cfg["count_alpha"] = w("count_alpha", _DEFAULTS["count_alpha"])
    cfg["reuse_weight"] = w("reuse_weight", _DEFAULTS["reuse_weight"])
    cfg["contradiction_weight"] = w("contradiction_weight", _DEFAULTS["contradiction_weight"])
    is_hybrid = sk == "hybrid"
    cfg["w_category"] = _num(cell.get("w_category")) if is_hybrid else None
    cfg["w_embedding"] = _num(cell.get("w_embedding")) if is_hybrid else None
    cfg["w_entity"] = _num(cell.get("w_entity")) if is_hybrid else None
    cfg["w_evidence"] = _num(cell.get("w_evidence")) if is_hybrid else None
    if not for_refine:
        cfg["theta"] = _num(cell.get("theta"))
        cfg["reject_theta"] = _num(cell.get("reject_theta"))
    return cfg


def _pick_operating_points(rows: list[dict], metric: str, min_economy_f1: float,
                           cost_lambda: float) -> dict:
    """{label: cell} for quality (max metric), economy (lowest escalate ≥ floor),
    knee (max metric − λ·escalate) — same picks as _select_finalists' reasons."""
    def q(r):
        return float(r[metric])

    def c(r):
        return _cost_frac(r)

    pts = {"quality": max(rows, key=lambda r: _rank(r, metric))}
    eligible = [r for r in rows if q(r) >= min_economy_f1]
    if eligible:
        pts["economy"] = min(eligible, key=c)
    pts["knee"] = max(rows, key=lambda r: q(r) - cost_lambda * c(r))
    return pts


def _dedup_points(pts: dict, key_fn) -> list[tuple]:
    """Merge points mapping to the same cell, joining labels. [(merged_label, cell)]."""
    seen: dict = {}
    for label, cell in pts.items():
        seen.setdefault(key_fn(cell), [cell, []])[1].append(label)
    return [("_".join(labels), cell) for cell, labels in seen.values()]


def _report_family_refine(rows: list[dict], args, csv_path: Path) -> None:
    group = GROUP_FIELDS["family_refine"]
    print(f"\nBest per {'/'.join(group)}  (PRIMARY {args.metric}; greedy diagnostic):")
    for key, r in sorted(_best_per_group(rows, group, args.metric).items(),
                         key=lambda kv: _rank(kv[1], args.metric), reverse=True):
        label = "  ".join(f"{f}={v}" for f, v in zip(group, key))
        print(f"  {label:46s} {args.metric}={float(r[args.metric]):.4f}  "
              f"f1_optimal={float(r.get('f1_optimal') or 0):.4f}  "
              f"esc={float(r.get('escalate_rate') or 0):.3f}  "
              f"theta={r.get('theta')} reject={r.get('reject_theta')}")

    pts = _pick_operating_points(rows, args.metric, args.min_economy_f1, args.cost_lambda)
    configs = [_config_from_cell(cell, label) for label, cell in _dedup_points(pts, _cell_key_from_row)]
    print(f"\nCSV → {csv_path}")
    print("\nPaste into VOTER_SUBSET_SCREEN_CONFIGS near the top of this file, then run "
          "--stage voter_subset_screen (each config × VOTER_SUBSET_GRID at its fixed theta):\n")
    _print_config_block("VOTER_SUBSET_SCREEN_CONFIGS", configs)


def _report_voter_subset_screen(rows: list[dict], args, csv_path: Path) -> None:
    def q(r):
        return float(r[args.metric])

    def c(r):
        return float(r.get("escalate_rate") or 0.0)

    by_cfg: dict = {}
    for r in rows:
        by_cfg.setdefault(r["variant"], []).append(r)

    print(f"\nVoter-subset screen — Δ vs 'all' per config (selection = {args.metric}; greedy diagnostic):")
    for variant, rs in by_cfg.items():
        base = next((r for r in rs if r.get("voter_subset") == "all"), None)
        bq = q(base) if base else 0.0
        print(f"  {variant}:")
        for r in sorted(rs, key=lambda r: _rank(r, args.metric), reverse=True):
            print(f"    {str(r.get('voter_subset')):11s} {args.metric}={q(r):.4f} (Δ{q(r) - bq:+.4f})  "
                  f"esc={c(r):.3f}  early={float(r.get('early_accept_rate') or 0):.3f}  "
                  f"[{r.get('dropped_voter_model') or '—'}]")

    pts = _pick_operating_points(rows, args.metric, args.min_economy_f1, args.cost_lambda)
    configs = [_config_from_cell(cell, label, for_refine=True)
               for label, cell in _dedup_points(pts, lambda r: (r["variant"], r.get("voter_subset")))]
    print(f"\nCSV → {csv_path}")
    print("\nPaste into VOTER_SUBSET_REFINE_CONFIGS near the top of this file, then run "
          "--stage voter_subset_refine (selected (config, subset) pairs × FULL theta grid):\n")
    _print_config_block("VOTER_SUBSET_REFINE_CONFIGS", configs)


def _report_voter_subset_refine(rows: list[dict], args, csv_path: Path) -> None:
    best = max(rows, key=lambda r: _rank(r, args.metric))
    print(f"\nVoter-subset refine — overall best (PRIMARY {args.metric}; tie-break esc, f1_optimal, "
          f"simpler):\n  {_fmt_best(best, args.metric)}")
    print(f"      voter_subset={best.get('voter_subset')}  dropped={best.get('dropped_voter_model') or '—'}")
    print(f"\nCSV → {csv_path}")
    print(f"\nPIN → BEST_VOTER_SUBSET = {best.get('voter_subset')!r}")
    print(f"      provisional BEST_THETA = {best.get('theta')}, BEST_REJECT_THETA = {best.get('reject_theta')}")
    print("      (+ the winning config's BEST_EMBEDDER/SCORER/ALIGNMENT/weights if it changed). "
          "map_theta / map_gates then run with BEST_VOTER_SUBSET.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Screen→refine MAP calibration sweep (structure first, then weights, thresholds, gates)."
    )
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--list-variants", action="store_true",
                    help="Enumerate the stage's cells and exit (no API, no cache).")
    ap.add_argument("--metric", default="strict_f1_optimal",
                    choices=["strict_f1_optimal", "f1_optimal"],
                    help="PRIMARY selection metric — optimal/Hungarian matcher only; greedy is a "
                         "diagnostic and never selects. Ties: lower cost_frac, then f1_optimal, "
                         "then simpler config.")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                    help="structure_screen: keep this many top structures (by primary metric) as "
                         "quality finalists.")
    ap.add_argument("--keep-within", type=float, default=DEFAULT_KEEP_WITHIN,
                    help="structure_screen: also keep structures within this primary-metric gap of the best.")
    ap.add_argument("--pareto-cap", type=int, default=PARETO_CAP,
                    help="structure_screen: max non-dominated Pareto-frontier structures to keep.")
    ap.add_argument("--min-economy-f1", type=float, default=MIN_ECONOMY_F1,
                    help="structure_screen / family_refine: strict_f1 floor the economy point (the "
                         "min-cost_frac cell) must clear.")
    ap.add_argument("--cost-lambda", type=float, default=COST_LAMBDA,
                    help="structure_screen / family_refine: knee = argmax(strict_f1_optimal − λ·cost_frac) "
                         "(cost_frac = price-weighted L1→L2 + L2→L3 escalation, NOT bare escalate_rate).")
    ap.add_argument("--from-csv", default=None,
                    help="structure_screen: recompute finalists from an existing screen CSV "
                         "(no rerun, no engine).")
    ap.add_argument("--sim-threshold", type=float, default=SIMILARITY_THRESHOLD)
    ap.add_argument("--split", default="all", choices=["dev", "test", "all"],
                    help="Calibration split. Default 'all' (small silver set → stable estimate). "
                         "'dev'/'test' are for sensitivity only — these are calibration scores, "
                         "not held-out performance.")
    ap.add_argument("--dev-fraction", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--embed-cache", default=None,
                    help="Override embedding cache path (default: per-embedder default).")
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore any existing checkpoint for this stage and start over "
                         "(the stale checkpoint is rotated aside, not deleted).")
    args = ap.parse_args()

    print(PIN_HINTS[args.stage])

    if args.list_variants:
        _empty = {"family_refine": (FINALIST_STRUCTURES, _FINALIST_EMPTY_MSG),
                  "voter_subset_screen": (VOTER_SUBSET_SCREEN_CONFIGS, _VS_SCREEN_EMPTY_MSG),
                  "voter_subset_refine": (VOTER_SUBSET_REFINE_CONFIGS, _VS_REFINE_EMPTY_MSG)}
        if args.stage in _empty and not _empty[args.stage][0]:
            print(f"\n{args.stage}: 0 variant(s) — paste the input config first.\n")
            print(_empty[args.stage][1])
            return
        cells = _list_variants(args.stage)
        print(f"\n{args.stage}: {len(cells)} variant(s)")
        for c in cells:
            print("  " + c)
        return

    if args.from_csv:
        recompute = {"structure_screen": _report_structure_screen,
                     "family_refine": _report_family_refine}
        if args.stage not in recompute:
            raise SystemExit("--from-csv applies to --stage structure_screen or family_refine")
        with open(args.from_csv, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise SystemExit(f"no rows in {args.from_csv}")
        print(f"Recomputing from {args.from_csv} ({len(rows)} rows) — no rerun, no engine.")
        recompute[args.stage](rows, args, Path(args.from_csv))
        return

    rows = _run_stage(args.stage, args)   # raises SystemExit for an empty family_refine
    if not rows:
        print("No rows produced (empty grid or no silver overlap).")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    csv_path = _stage_dir(args.stage) / f"sweep_{timestamp}.csv"
    _write_csv(csv_path, rows, _CSV_FIELDS)
    _cleanup_checkpoint(args.stage)   # stage completed — discard the resume checkpoint

    _CUSTOM_REPORT = {
        "structure_screen": _report_structure_screen,
        "family_refine": _report_family_refine,
        "voter_subset_screen": _report_voter_subset_screen,
        "voter_subset_refine": _report_voter_subset_refine,
    }
    if args.stage in _CUSTOM_REPORT:
        _CUSTOM_REPORT[args.stage](rows, args, csv_path)
        print("\n" + PIN_HINTS[args.stage])
        return

    # map_theta / map_gates → generic best-per (GROUP_FIELDS none) + overall best
    group_fields = GROUP_FIELDS.get(args.stage)
    if group_fields:
        print(f"\nBest per {'/'.join(group_fields)}  (PRIMARY {args.metric}; greedy shown as diagnostic):")
        for key, r in sorted(_best_per_group(rows, group_fields, args.metric).items(),
                             key=lambda kv: _rank(kv[1], args.metric), reverse=True):
            label = "  ".join(f"{f}={v}" for f, v in zip(group_fields, key))
            print(f"  {label:46s} {args.metric}={float(r[args.metric]):.4f}  "
                  f"f1_optimal={float(r.get('f1_optimal') or 0):.4f}  "
                  f"esc={float(r.get('escalate_rate') or 0.0):.3f}  "
                  f"[greedy strict_f1={float(r.get('strict_f1_greedy') or 0):.4f}]  "
                  f"theta={r.get('theta')} reject={r.get('reject_theta')}")

    best = max(rows, key=lambda r: _rank(r, args.metric))
    print(f"\nOverall best — selected by PRIMARY {args.metric} "
          f"(tie-break: lower cost_frac, then f1_optimal, then simpler config):\n  "
          f"{_fmt_best(best, args.metric)}")
    print(f"\nCSV → {csv_path}")
    print("\n" + PIN_HINTS[args.stage])


if __name__ == "__main__":
    main()
