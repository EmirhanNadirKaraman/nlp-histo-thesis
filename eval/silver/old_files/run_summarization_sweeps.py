#!/usr/bin/env python3
"""run_summarization_sweeps.py — staged, greedy calibration sweeps for the
summarisation pipeline, in the style of ``scripts/eval/run_all_sweeps.py``.

Each stage sweeps ONE axis while holding earlier stages at their pinned
``BEST_*`` winner. Workflow: run a stage → read its CSV → edit the matching
``BEST_*`` constant in this file → run the next stage (it picks up the new
winner). Same greedy "select the best and move forward" loop as the PDF
pipeline's ``run_all_sweeps.py``.

────────────────────────────────────────────────────────────────────────────
EXECUTABLE NOW — MAP cascade (offline replay of the voter cache vs silver
labels; NO LLM calls, gemini embeddings only and cached):

  Stage 1  map_scorer   EmbeddingSimilarityStrategy vs HybridStructuredSimilarity
  Stage 2  map_theta     theta × reject_theta (reject_theta < theta) on BEST_SCORER
  Stage 3  map_weights   H-EMB-01 weights (hybrid only) on BEST_SCORER/theta

  Selection metric: silver F1 / strict F1 (per-finding, vs Opus silver),
  with escalation rate reported and optionally constrained (--max-escalate).

────────────────────────────────────────────────────────────────────────────
SCAFFOLD ONLY — defined but not executable. The eval backend / a
silver-comparable selection metric does not exist for these yet; running one
errors out. They are here so the staged structure is complete and the knobs
are catalogued.

  Stage 4  grounding       grounding.threshold
                           metric (today): retention / rejection curve — not silver F1
  Stage 5  relate          entailment_threshold × contradiction_threshold
                           metric (today): relation-count / pair stats — not finding-level F1
  Stage 6  resolve         RESOLVE weights
                           metric (today): TBD — RESOLVE output is FinalRule-level,
                                           no silver eval exists
  Stage 7  contradiction   contradiction_similarity_threshold
                           metric (today): candidate-pair count — not silver F1

  *** THE METRIC MISMATCH IS DELIBERATE. *** Only the MAP stages are scored
  against silver per-finding F1. Grounding / relate / resolve / contradiction
  are not comparable to MAP silver F1 — do not rank them on the same axis.
  See docs/readmes/other_readmes/CALIBRATION_INVENTORY.md and docs/readmes/other_readmes/CALIBRATION_EVAL.md.

────────────────────────────────────────────────────────────────────────────
Prerequisites for the MAP stages:
  * primed voter cache  eval/data/map_primer/voter_cache.json
        (from `python -m eval.silver.analysis.map_theta_sweep prime --split all` + `collect`)
  * silver labels       eval/data/silver_findings.jsonl
        (from `python -m eval.silver.generation.generate --batch`)

Usage:
  python -m eval.silver.run_summarization_sweeps                          # stage menu
  python -m eval.silver.run_summarization_sweeps --stage map_scorer --list-variants
  python -m eval.silver.run_summarization_sweeps --stage map_scorer
  python -m eval.silver.run_summarization_sweeps --stage map_theta --metric strict_f1
  python -m eval.silver.run_summarization_sweeps --stage map_theta --max-escalate 0.30
  python -m eval.silver.run_summarization_sweeps --stage grounding --list-variants   # scaffold: lists, won't run
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Make ``eval.*`` / ``pipeline.*`` importable when run as a file.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(str(_REPO_ROOT / ".env"))

from nlp_histo.pipeline.stages.knowledge_extraction.config import AgreementConfig

from nlp_histo.evaluation.matching.matcher import (
    SIMILARITY_THRESHOLD,
)
from eval.silver.analysis.map_theta_sweep import (
    REJECT_THETA_GRID,
    REPORTS_DIR,
    THETA_GRID,
    ScorerSpec,
    _print_table,
    _write_csv,
    run_sweep,
)


# BEST_* — winners pinned after each stage. Edit in place as each stage's CSV
# lands; later stages dereference these (greedy stage-wise selection).

BEST_SCORER = "embedding"        # Stage 1 winner: "embedding" | "hybrid"
BEST_THETA = 0.80                # Stage 2 winner
BEST_REJECT_THETA = 0.20         # Stage 2 winner (must be < BEST_THETA)

# Stage 3 winners — H-EMB-01 agreement weights (only swept when BEST_SCORER="hybrid").
BEST_TAU = 0.15
BEST_COUNT_ALPHA = 0.25
BEST_REUSE_WEIGHT = 0.15
BEST_CONTRADICTION_WEIGHT = 0.20

# Scaffold-stage winners (placeholders — stages 4-7 are not executable yet).
BEST_GROUNDING_THRESHOLD = 0.50
BEST_RELATE_ENTAILMENT = 0.50
BEST_RELATE_CONTRADICTION = 0.50
BEST_CONTRADICTION_SIM = 0.70

# Grids.

SCORER_CHOICES = ("embedding", "hybrid")

# Stage 1 isolates the scorer: hold theta/reject at a neutral mid default.
STAGE1_THETA = 0.80
STAGE1_REJECT = 0.20

# Stage 2 reuses the validated map_theta_sweep grids (THETA_GRID, REJECT_THETA_GRID).

# Stage 3 — per-knob H-EMB-01 grids, swept one axis at a time around BEST_*.
TAU_GRID = [0.10, 0.15, 0.20, 0.30]
COUNT_ALPHA_GRID = [0.0, 0.25, 0.50]
REUSE_WEIGHT_GRID = [0.0, 0.15, 0.30]
CONTRADICTION_WEIGHT_GRID = [0.0, 0.20, 0.40]

# Stage 3a — legacy_single_voter_policy. Two-cell categorical sweep at the
# pinned BEST_SCORER / BEST_THETA / BEST_REJECT_THETA. Production default is
# "keep" (the pre-existing implicit AgreementChecker behaviour); the
# experiment asks whether escalating single-survivor chunks improves
# precision at acceptable cost.
LEGACY_SINGLE_VOTER_POLICY_GRID = ("keep", "escalate")

# Stage 1b — hybrid blend weights. Six labelled, sum-to-1.0 variants, each a
# distinct hypothesis about which signal dominates. Gated on
# BEST_SCORER == "hybrid" (Stage 1's winner); the existing map_weights stage
# does the same conditional. Convention: weights sum to 1.0 for a calibrated
# score in [0, 1] — not enforced in code (HybridStructuredSimilarity accepts
# any tuple), but every variant here is sum-to-1 by construction.
HYBRID_BLEND_GRID: dict[str, tuple[float, float, float, float]] = {
    # (w_category, w_embedding, w_entity, w_evidence)
    "hybrid_default":         (0.25, 0.40, 0.25, 0.10),
    "hybrid_balanced":        (0.25, 0.25, 0.25, 0.25),
    "hybrid_embedding_heavy": (0.15, 0.65, 0.15, 0.05),
    "hybrid_category_heavy":  (0.50, 0.30, 0.15, 0.05),
    "hybrid_entity_heavy":    (0.15, 0.30, 0.50, 0.05),
    "hybrid_evidence_heavy":  (0.15, 0.30, 0.15, 0.40),
}

# Stage 3b — force_escalate_on_polarity_conflict. Two-cell categorical
# ablation of the B-051 hard-fail safety guard. Production default is True
# (override scorer decision to ESCALATE on comparable positive/negative
# pairs); the experiment asks how often that override fires and whether
# disabling it changes the silver F1 / escalation cost picture. Default-flip
# requires explicit evidence: polarity overrides exist for safety, not cost.
FORCE_ESCALATE_ON_POLARITY_CONFLICT_GRID: tuple[bool, ...] = (True, False)

# Scaffold grids (catalogued; not executed).
GROUNDING_THRESHOLD_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
RELATE_ENTAILMENT_GRID = [0.40, 0.50, 0.60, 0.70]
RELATE_CONTRADICTION_GRID = [0.40, 0.50, 0.60, 0.70]
RESOLVE_GROUNDING_WEIGHT_GRID = [0.50, 0.60, 0.70, 0.80]
CONTRADICTION_SIM_GRID = [0.50, 0.60, 0.70, 0.80]

CSV_FIELDNAMES = [
    "scorer", "tau", "count_alpha", "reuse_weight", "contradiction_weight",
    "w_category", "w_embedding", "w_entity", "w_evidence", "weights_sum",
    "theta", "reject_theta",
    "legacy_single_voter_policy", "force_escalate_on_polarity_conflict",
    "cascade_path",
    "precision", "recall", "f1", "strict_f1",
    "early_accept_rate", "escalate_rate", "early_accept_precision",
    "n_polarity_conflict_chunks", "polarity_conflict_rate",
    "n_matched", "n_silver", "n_pipeline",
    "split", "seed", "dev_fraction", "sim_threshold",
]


# Stage registry (mirrors run_all_sweeps.py's SweepSpec + STAGE_ORDER).

@dataclass
class SweepSpec:
    name: str
    blurb: str
    executable: bool
    metric: str                       # MAP: "silver_f1"; scaffold: its native (non-silver) metric
    scaffold_reason: str = ""


STAGE_ORDER = (
    "map_scorer",          # Stage 1
    "map_hybrid_blend",    # Stage 1b — hybrid blend weights (gated on BEST_SCORER='hybrid')
    "map_theta",           # Stage 2
    "map_weights",         # Stage 3
    "map_routing_policy",  # Stage 3a — legacy_single_voter_policy {keep, escalate}
    "map_polarity_flag",   # Stage 3b — force_escalate_on_polarity_conflict {True, False}
    "grounding",           # Stage 4  (scaffold)
    "relate",              # Stage 5  (scaffold)
    "resolve",             # Stage 6  (scaffold)
    "contradiction",       # Stage 7  (scaffold)
)
STAGE_CHOICES = STAGE_ORDER  # no "all": stages have heterogeneous, non-comparable metrics

STAGES: dict[str, SweepSpec] = {
    "map_scorer": SweepSpec(
        "map_scorer", "Stage 1 — EmbeddingSimilarityStrategy vs HybridStructuredSimilarity",
        executable=True, metric="silver_f1"),
    "map_theta": SweepSpec(
        "map_theta", "Stage 2 — theta × reject_theta on BEST_SCORER (reject < theta)",
        executable=True, metric="silver_f1"),
    "map_weights": SweepSpec(
        "map_weights", "Stage 3 — H-EMB-01 weights (hybrid only) on BEST_SCORER/theta",
        executable=True, metric="silver_f1"),
    "map_hybrid_blend": SweepSpec(
        "map_hybrid_blend",
        "Stage 1b — HybridStructuredSimilarity blend weights "
        "{default, balanced, embedding_heavy, category_heavy, entity_heavy, evidence_heavy} "
        "at BEST_THETA/reject (gated on BEST_SCORER='hybrid')",
        executable=True, metric="silver_f1"),
    "map_routing_policy": SweepSpec(
        "map_routing_policy",
        "Stage 3a — legacy_single_voter_policy ∈ {keep, escalate} at BEST_SCORER/theta/reject",
        executable=True, metric="silver_f1"),
    "map_polarity_flag": SweepSpec(
        "map_polarity_flag",
        "Stage 3b — force_escalate_on_polarity_conflict ∈ {true, false} at BEST_SCORER/theta/reject (B-051 ablation)",
        executable=True, metric="silver_f1"),
    "grounding": SweepSpec(
        "grounding", "Stage 4 — grounding.threshold  [SCAFFOLD]",
        executable=False, metric="retention/rejection curve (NOT silver F1)",
        scaffold_reason=(
            "needs MAP findings + their NLI grounding scores. Either a full-pipeline "
            "prime (eval.silver.analysis.pipeline_sweep) or the Layer-A artifact sweep "
            "(eval/sweeps/grounding.py) — and the latter is retention-only, not "
            "silver-F1-comparable to the MAP stages.")),
    "relate": SweepSpec(
        "relate", "Stage 5 — entailment_threshold × contradiction_threshold  [SCAFFOLD]",
        executable=False, metric="relation-count / pair-distribution stats (NOT finding-level F1)",
        scaffold_reason=(
            "RELATE output is at the FinalRule level; silver labels are per-finding, "
            "so there is no silver-F1 metric for relation pairs. eval.silver.analysis.pipeline_sweep "
            "relate emits relation-count stats only.")),
    "resolve": SweepSpec(
        "resolve", "Stage 6 — RESOLVE weights  [SCAFFOLD]",
        executable=False, metric="TBD — no FinalRule-level silver eval exists",
        scaffold_reason=(
            "RESOLVE scores FinalRules; no silver-comparable metric exists yet. "
            "Needs a FinalRule-level gold or a resolve-score replay before selection "
            "is meaningful.")),
    "contradiction": SweepSpec(
        "contradiction", "Stage 7 — contradiction_similarity_threshold  [SCAFFOLD]",
        executable=False, metric="candidate-pair count (NOT silver F1)",
        scaffold_reason=(
            "ContradictionDetector emits candidate-pair counts; there is no "
            "silver-F1 metric for it yet.")),
}


# MAP stage grids (executable) — each returns (scorer_specs, thetas, reject_thetas).

def _spec_for(scorer: str) -> ScorerSpec:
    if scorer not in SCORER_CHOICES:
        raise ValueError(f"scorer={scorer!r} must be one of {SCORER_CHOICES}")
    return ScorerSpec(f"{scorer}_default", scorer, AgreementConfig())


def _hybrid_blend_specs() -> list[ScorerSpec]:
    """One ScorerSpec per labelled variant in ``HYBRID_BLEND_GRID``.

    Each spec carries an ``AgreementConfig`` with the chosen ``HybridConfig``
    block (default soft-alignment weights everywhere — the soft-alignment
    quartet is held constant in Stage 1b so only the blend axis varies).
    Stage 1b is gated on ``BEST_SCORER == "hybrid"`` in ``_map_grid``.
    """
    from nlp_histo.pipeline.stages.knowledge_extraction.config import HybridConfig
    specs: list[ScorerSpec] = []
    for name, (w_cat, w_emb, w_ent, w_evid) in HYBRID_BLEND_GRID.items():
        cfg = AgreementConfig(
            hybrid=HybridConfig(
                w_category=w_cat, w_embedding=w_emb,
                w_entity=w_ent, w_evidence=w_evid,
            ),
        )
        specs.append(ScorerSpec(name, "hybrid", cfg))
    return specs


def _weight_variant_specs() -> list[ScorerSpec]:
    """One-axis-at-a-time H-EMB-01 variants around the pinned BEST_* weights."""
    base = dict(tau=BEST_TAU, count_alpha=BEST_COUNT_ALPHA,
                reuse_weight=BEST_REUSE_WEIGHT, contradiction_weight=BEST_CONTRADICTION_WEIGHT)
    specs = [ScorerSpec("hybrid_baseline", "hybrid", AgreementConfig(**base))]
    grids = {
        "tau": TAU_GRID, "count_alpha": COUNT_ALPHA_GRID,
        "reuse_weight": REUSE_WEIGHT_GRID, "contradiction_weight": CONTRADICTION_WEIGHT_GRID,
    }
    for knob, grid in grids.items():
        for v in grid:
            if v == base[knob]:
                continue
            w = dict(base)
            w[knob] = v
            specs.append(ScorerSpec(f"hybrid_{knob}_{v}", "hybrid", AgreementConfig(**w)))
    return specs


def _map_grid(stage: str) -> tuple[list[ScorerSpec], list[float], list[float]]:
    if stage == "map_scorer":
        specs = [_spec_for("embedding"), _spec_for("hybrid")]
        return specs, [STAGE1_THETA], [STAGE1_REJECT]
    if stage == "map_theta":
        return [_spec_for(BEST_SCORER)], list(THETA_GRID), list(REJECT_THETA_GRID)
    if stage == "map_weights":
        if BEST_SCORER != "hybrid":
            raise SystemExit(
                "Stage 3 (map_weights) is only swept when BEST_SCORER='hybrid' "
                f"(current BEST_SCORER={BEST_SCORER!r}). The H-EMB-01 weights are "
                "the hybrid scorer's structural-blend knobs; if Stage 1 picked "
                "'embedding', skip Stage 3. (Edit BEST_SCORER to override.)")
        return _weight_variant_specs(), [BEST_THETA], [BEST_REJECT_THETA]
    if stage == "map_hybrid_blend":
        if BEST_SCORER != "hybrid":
            raise SystemExit(
                "Stage 1b (map_hybrid_blend) is only swept when BEST_SCORER='hybrid' "
                f"(current BEST_SCORER={BEST_SCORER!r}). The blend weights "
                "(w_category/w_embedding/w_entity/w_evidence) are HybridStructuredSimilarity's "
                "signal-mix knobs; if Stage 1 picked 'embedding', they're irrelevant — "
                "skip Stage 1b. (Edit BEST_SCORER='hybrid' to override.)")
        return _hybrid_blend_specs(), [BEST_THETA], [BEST_REJECT_THETA]
    if stage == "map_routing_policy":
        # Scorer / theta / reject_theta all pinned to BEST_*; the only axis is
        # the legacy_single_voter_policy enum (handled separately in
        # _policy_grid_for / _run_map_stage).
        return [_spec_for(BEST_SCORER)], [BEST_THETA], [BEST_REJECT_THETA]
    if stage == "map_polarity_flag":
        # Same shape as map_routing_policy — pinned at BEST_*; only axis is
        # force_escalate_on_polarity_conflict (handled in _polarity_grid_for).
        return [_spec_for(BEST_SCORER)], [BEST_THETA], [BEST_REJECT_THETA]
    raise ValueError(f"{stage!r} is not a MAP stage")


def _policy_grid_for(stage: str) -> tuple[str, ...]:
    """legacy_single_voter_policy axis per MAP stage.

    Only ``map_routing_policy`` sweeps the policy; every other MAP stage holds
    it at the production default ("keep") so its results are comparable to
    historical sweep CSVs.
    """
    if stage == "map_routing_policy":
        return LEGACY_SINGLE_VOTER_POLICY_GRID
    return ("keep",)


def _polarity_grid_for(stage: str) -> tuple[bool, ...]:
    """force_escalate_on_polarity_conflict axis per MAP stage.

    Only ``map_polarity_flag`` sweeps the flag; every other MAP stage holds it
    at the production default (True) so the B-051 safety guard is preserved
    and per-stage results stay comparable to historical sweep CSVs.
    """
    if stage == "map_polarity_flag":
        return FORCE_ESCALATE_ON_POLARITY_CONFLICT_GRID
    return (True,)


def _enumerate_cells(stage: str) -> list[str]:
    """Human-readable per-cell names for --list-variants."""
    if stage in ("map_scorer", "map_theta", "map_weights", "map_hybrid_blend",
                 "map_routing_policy", "map_polarity_flag"):
        specs, thetas, rejects = _map_grid(stage)
        policies = _policy_grid_for(stage)
        polarities = _polarity_grid_for(stage)
        cells = []
        for s in specs:
            for t in thetas:
                for r in rejects:
                    if r >= t:
                        continue
                    for pol in policies:
                        for pol_flag in polarities:
                            if stage == "map_routing_policy":
                                # Per the user spec, the visible cell name uses
                                # `legacy_single_voter_<policy>` so the policy
                                # is the salient label.
                                cells.append(
                                    f"legacy_single_voter_{pol}  "
                                    f"({s.name}, theta={t:.2f}, rej={r:.2f})"
                                )
                            elif stage == "map_polarity_flag":
                                # Cell name leads with the flag value so the
                                # B-051 ablation pair (true vs false) is the
                                # salient axis.
                                flag_label = "true" if pol_flag else "false"
                                cells.append(
                                    f"polarity_force_escalate_{flag_label}  "
                                    f"({s.name}, theta={t:.2f}, rej={r:.2f})"
                                )
                            elif stage == "map_hybrid_blend":
                                # Cell name = the variant name from
                                # HYBRID_BLEND_GRID + the (w_*, sum) tuple.
                                h = s.weights.hybrid
                                wt = (h.w_category, h.w_embedding,
                                      h.w_entity, h.w_evidence)
                                wsum = sum(wt)
                                cells.append(
                                    f"{s.name}  (theta={t:.2f}, rej={r:.2f}, "
                                    f"w=({wt[0]:.2f}, {wt[1]:.2f}, "
                                    f"{wt[2]:.2f}, {wt[3]:.2f}), Σ={wsum:.2f})"
                                )
                            elif (len(policies) == 1 and policies[0] == "keep"
                                  and len(polarities) == 1 and polarities[0] is True):
                                # Historical stages — omit the policy suffix
                                # to keep diffs against prior --list-variants
                                # output readable.
                                cells.append(
                                    f"{s.name}  theta={t:.2f} rej={r:.2f}"
                                )
                            else:
                                cells.append(
                                    f"{s.name}  theta={t:.2f} rej={r:.2f}  "
                                    f"pol={pol}  pol_fail={pol_flag}"
                                )
        return cells
    if stage == "grounding":
        return [f"grounding_threshold={t:.2f}" for t in GROUNDING_THRESHOLD_GRID]
    if stage == "relate":
        return [f"ent={e:.2f} con={c:.2f}"
                for e in RELATE_ENTAILMENT_GRID for c in RELATE_CONTRADICTION_GRID]
    if stage == "resolve":
        return [f"grounding_weight={w:.2f}" for w in RESOLVE_GROUNDING_WEIGHT_GRID]
    if stage == "contradiction":
        return [f"contradiction_similarity_threshold={t:.2f}" for t in CONTRADICTION_SIM_GRID]
    return []


# Execution (MAP stages only).

# _MapContext / _load_map_context were EXTRACTED to eval/silver/analysis/map_context.py
# (2026-06-17) so the live loader is not stranded in this legacy module. Re-imported
# here for backward compatibility and for this module's own _run_map_stage (line ~468).
from eval.silver.analysis.map_context import _MapContext, _load_map_context  # noqa: E402,F401


def _select_winner(rows: list[dict], metric: str, max_escalate: Optional[float]) -> Optional[dict]:
    if not rows:
        return None
    cands = rows
    if max_escalate is not None:
        filtered = [r for r in rows if r.get("escalate_rate", 1.0) <= max_escalate]
        if filtered:
            cands = filtered
        else:
            print(f"  [warn] no cell with escalate_rate <= {max_escalate}; ignoring the constraint")
    # Highest metric; tie-break on lower escalation.
    return max(cands, key=lambda r: (r.get(metric, 0.0), -r.get("escalate_rate", 1.0)))


def _run_map_stage(stage: str, *, metric: str, max_escalate: Optional[float],
                   embedder_kind: str, embed_cache_path: Optional[str],
                   split: str, seed: int, dev_fraction: float,
                   only: Optional[set]) -> int:
    specs, thetas, rejects = _map_grid(stage)
    if only:
        specs = [s for s in specs if s.name in only]
        if not specs:
            print(f"error: --only {sorted(only)} matched no scorer spec in {stage}", file=sys.stderr)
            return 2

    ctx = _load_map_context(embedder_kind, embed_cache_path=embed_cache_path)
    policies = _policy_grid_for(stage)
    polarities = _polarity_grid_for(stage)
    rows = run_sweep(
        voter_cache=ctx.voter_cache,
        silver_by_case=ctx.silver_by_case,
        embedder=ctx.embedder,
        embed_cache=ctx.embed_cache,
        sim_threshold=SIMILARITY_THRESHOLD,
        scorer_specs=specs,
        thetas=thetas,
        reject_thetas=rejects,
        split=split,
        seed=seed,
        dev_fraction=dev_fraction,
        agreement_embed_fn=ctx.agreement_embed_fn,
        single_voter_policies=policies,
        force_escalate_on_polarity_conflict_grid=polarities,
    )
    if not rows:
        print("no rows produced — check the voter cache / silver overlap.", file=sys.stderr)
        return 1

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    csv_path = REPORTS_DIR / f"summ_sweep_{stage}_{ts}.csv"
    _write_csv(csv_path, rows, fieldnames=CSV_FIELDNAMES)
    _print_table(rows)

    winner = _select_winner(rows, metric, max_escalate)
    print(f"\nCSV: {csv_path}")
    if winner:
        print(f"WINNER (by {metric}{f', escalate<= {max_escalate}' if max_escalate is not None else ''}): "
              f"scorer={winner['scorer']} theta={winner['theta']} reject={winner['reject_theta']} "
              f"f1={winner['f1']:.3f} strict_f1={winner['strict_f1']:.3f} "
              f"escalate_rate={winner['escalate_rate']:.2f}")
        print(_pin_hint(stage, winner))
    return 0


def _pin_hint(stage: str, w: dict) -> str:
    if stage == "map_scorer":
        scorer = w["scorer"].replace("_default", "")
        return (
            f"→ pin: BEST_SCORER = {scorer!r}  "
            f"(update configs/run.yaml summarization.agreement.scorer_kind: {scorer})"
        )
    if stage == "map_theta":
        return f"→ pin: BEST_THETA = {w['theta']} ; BEST_REJECT_THETA = {w['reject_theta']}"
    if stage == "map_weights":
        return ("→ pin: BEST_TAU / BEST_COUNT_ALPHA / BEST_REUSE_WEIGHT / "
                f"BEST_CONTRADICTION_WEIGHT = ({w['tau']}, {w['count_alpha']}, "
                f"{w['reuse_weight']}, {w['contradiction_weight']})")
    if stage == "map_hybrid_blend":
        return (
            f"→ pin: AgreementConfig.hybrid = "
            f"(w_category={w['w_category']}, w_embedding={w['w_embedding']}, "
            f"w_entity={w['w_entity']}, w_evidence={w['w_evidence']})  "
            f"(update configs/run.yaml summarization.agreement.hybrid)"
        )
    if stage == "map_routing_policy":
        pol = w.get("legacy_single_voter_policy", "keep")
        return (
            f"→ pin: RoutingConfig.legacy_single_voter_policy = {pol!r}  "
            f"(update configs/run.yaml summarization.routing.legacy_single_voter_policy)"
        )
    if stage == "map_polarity_flag":
        # CSV stores the value as lowercase string ("true"/"false") to be
        # CSV-clean; convert back to a bool for the pin hint.
        flag_str = w.get("force_escalate_on_polarity_conflict", "true")
        flag_bool = (flag_str == "true") if isinstance(flag_str, str) else bool(flag_str)
        return (
            f"→ pin: AgreementConfig.force_escalate_on_polarity_conflict = {flag_bool}  "
            f"(update configs/run.yaml summarization.agreement.force_escalate_on_polarity_conflict)"
        )
    return ""


# Listing / menu.

def _print_stage_menu() -> None:
    print("Summarisation calibration — staged sweeps (greedy; pin BEST_* after each stage)\n")
    print("Available --stage values:\n")
    for name in STAGE_ORDER:
        spec = STAGES[name]
        tag = "" if spec.executable else "   [SCAFFOLD — not executable]"
        print(f"  {name}{tag}")
        print(f"      {spec.blurb}")
        print(f"      metric: {spec.metric}")
        print()
    print("Examples:")
    print("  python -m eval.silver.run_summarization_sweeps --stage map_scorer --list-variants")
    print("  python -m eval.silver.run_summarization_sweeps --stage map_scorer")
    print("  python -m eval.silver.run_summarization_sweeps --stage map_theta --metric strict_f1")
    print("\nNOTE: only the MAP stages are scored against silver F1. The scaffold stages")
    print("(grounding/relate/resolve/contradiction) use DIFFERENT, non-comparable metrics")
    print("and are not executable yet — see each stage's note.")


def _list_variants(stage: str, only: Optional[set]) -> None:
    spec = STAGES[stage]
    cells = _enumerate_cells(stage)
    if only:
        cells = [c for c in cells if any(o in c for o in only)]
    tag = "executable, metric=silver F1" if spec.executable else f"SCAFFOLD, metric={spec.metric}"
    print(f"=== {stage} ({tag}) ===")
    if not spec.executable:
        print(f"!! NOT EXECUTABLE: {spec.scaffold_reason}")
        print("!! Metric is NOT comparable to the MAP stages' silver F1.\n")
    for c in cells:
        print(f"  {c}")
    print(f"\n{len(cells)} variant(s).")
    if stage in ("map_theta", "map_weights", "map_hybrid_blend",
                 "map_routing_policy", "map_polarity_flag"):
        print(f"(holding earlier stages at BEST_SCORER={BEST_SCORER!r}, "
              f"BEST_THETA={BEST_THETA}, BEST_REJECT_THETA={BEST_REJECT_THETA})")
    if stage == "map_hybrid_blend":
        print("(sweep axis: AgreementConfig.hybrid blend weights — "
              f"{len(HYBRID_BLEND_GRID)} labelled sum-to-1 variants; "
              "gated on BEST_SCORER='hybrid')")
    if stage == "map_routing_policy":
        print("(sweep axis: RoutingConfig.legacy_single_voter_policy ∈ "
              f"{list(LEGACY_SINGLE_VOTER_POLICY_GRID)} — production default is 'keep')")
    if stage == "map_polarity_flag":
        print("(sweep axis: AgreementConfig.force_escalate_on_polarity_conflict ∈ "
              f"{list(FORCE_ESCALATE_ON_POLARITY_CONFLICT_GRID)} — "
              "production default is True; this is the B-051 hard-fail ablation)")


# CLI.

def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default=None, choices=STAGE_CHOICES,
                   help="Which stage to run. Omit to print the stage menu.")
    p.add_argument("--only", nargs="+", default=None,
                   help="Restrict to variants whose name contains any of these tokens "
                        "(filters --list-variants; for MAP runs, filters scorer specs by name).")
    p.add_argument("--list-variants", "--dry-run", dest="list_variants", action="store_true",
                   help="Print the stage's variants and exit (no run).")
    p.add_argument("--metric", default="f1", choices=("f1", "strict_f1"),
                   help="MAP winner selection metric (default f1).")
    p.add_argument("--max-escalate", type=float, default=None,
                   help="MAP only: secondary constraint — pick the best metric among "
                        "cells with escalate_rate <= this (e.g. 0.30).")
    p.add_argument("--embedder", default="gemini", choices=("gemini", "openai"),
                   help="MAP embedding backend (default gemini = batch-production-faithful).")
    p.add_argument("--embed-cache", default=None, help="Override embedding cache path.")
    p.add_argument("--split", default="all", choices=("dev", "test", "all"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dev-fraction", type=float, default=0.8)
    args = p.parse_args(argv)

    if args.stage is None:
        _print_stage_menu()
        return 0

    only = set(args.only) if args.only else None

    if args.list_variants:
        _list_variants(args.stage, only)
        return 0

    spec = STAGES[args.stage]
    if not spec.executable:
        print(f"Stage not executable yet: no validated selection metric.\n"
              f"  stage : {args.stage}\n"
              f"  metric: {spec.metric}\n"
              f"  reason: {spec.scaffold_reason}\n"
              f"Use --list-variants to see its catalogued knobs.", file=sys.stderr)
        return 2

    return _run_map_stage(
        args.stage, metric=args.metric, max_escalate=args.max_escalate,
        embedder_kind=args.embedder, embed_cache_path=args.embed_cache,
        split=args.split, seed=args.seed, dev_fraction=args.dev_fraction, only=only)


if __name__ == "__main__":
    raise SystemExit(main())
