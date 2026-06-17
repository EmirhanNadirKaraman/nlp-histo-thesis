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
``FINALIST_STRUCTURES`` here + the field in ``configs/run.yaml`` → next stage):

  1. structure_screen  embedder × scorer × alignment at DEFAULT weights × a
                       COARSE theta/reject grid. Identifies promising structures,
                       not final configs. Prints a copy-pastable FINALIST set
                       (top-K ∪ within --keep-within, + a diversity safety add).
                       → paste FINALIST_STRUCTURES.
  2. family_refine     Refines ONLY the finalist structures: applicable weights ×
                       the FULL theta/reject grid (weight choice and theta
                       interact, so they are swept jointly — not at a fixed theta).
                       Applicable weights: soft_max → tau/count_alpha/reuse/
                       contradiction (+ hybrid blend); greedy/hungarian → tau only
                       (+ hybrid blend, which shapes the pre-alignment similarity).
                       → pin BEST_EMBEDDER, BEST_SCORER, BEST_ALIGNMENT, BEST_TAU,
                         BEST_COUNT_ALPHA/REUSE_WEIGHT/CONTRADICTION_WEIGHT (only if
                         soft_max), BEST_W_* (only if hybrid), provisional
                         BEST_THETA / BEST_REJECT_THETA.
  3. map_theta         Re-confirm theta × reject_theta at the FINAL score function.
                       → pin final BEST_THETA, BEST_REJECT_THETA.
  4. map_gates         single_voter_policy {keep, escalate} ×
                       force_escalate_on_polarity_conflict {True, False}.
                       → pin run.yaml routing.legacy_single_voter_policy +
                         agreement.force_escalate_on_polarity_conflict.

METRIC / MATCHER — selection uses **optimal / Hungarian** one-to-one semantic
matching (max embedding-similarity assignment) as the PRIMARY matcher: metric
``strict_f1_optimal``, tie-break lower ``escalate_rate``, then ``f1_optimal``,
then simpler/closer-to-default config. ``greedy`` matching is reported (``*_greedy``
columns) as a sensitivity diagnostic only — it NEVER selects. NOTE: "optimal" is
optimal *by embedding similarity*, NOT a globally optimal strict-F1 assignment.

CALIBRATION SPLIT — runs on ``--split all`` by default because the silver
calibration set is small and a stable point estimate is wanted. ``--split
dev``/``test`` stay available for sensitivity/debugging but are NOT the default.
**Sweep scores are CALIBRATION metrics, not unbiased held-out performance.** The
held-out final evaluation is a SEPARATE runbook step (pin → ``run_paper.py
--sync`` → ``export_pipeline`` → ``evaluate --matcher optimal``); only that score
is reported as generalization.

Prereqs: a primed voter cache (``eval/data/map_primer/voter_cache.json``) and
silver labels (``eval/data/silver_findings_related15.jsonl``). Offline replay —
no LLM calls; embedding-cache misses are the only (cheap, cached) cost. Generated
CSVs land in ``eval/reports/`` and are kept untracked.

CHECKPOINT / RESUME — every stage appends each completed cell to
``eval/reports/checkpoint_<stage>.csv`` (flushed per cell), so a Ctrl-C keeps all
finished cells. Re-running the stage resumes automatically: cells already in the
checkpoint are skipped (the embeddings they used are SQLite-cached too, so a
resume pays no API). A changed grid / finalists / weights / split rotates the
stale checkpoint aside and starts fresh; ``--fresh`` forces that. The checkpoint
is deleted once the stage completes and the final timestamped CSV is written.

Usage:
  python -m eval.silver.run_new_summarization_sweeps --stage structure_screen --list-variants
  python -m eval.silver.run_new_summarization_sweeps --stage structure_screen
  python -m eval.silver.run_new_summarization_sweeps --stage family_refine
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
    _write_csv,
    run_sweep,
)
from eval.silver.matcher import SIMILARITY_THRESHOLD
from eval.silver.run_summarization_sweeps import _load_map_context  # reused loader
from pipeline.stages.summarization.config import AgreementConfig, HybridConfig

# ─────────────────────────────────────────────────────────────────────────────
# Pin-as-you-go winners. Edit after each stage; defaults = current production.
# ─────────────────────────────────────────────────────────────────────────────
FINALIST_STRUCTURES = [                                                        
    ("gemini", "hybrid", "soft_max"),    # top-k                               
    ("gemini", "hybrid", "greedy"),    # top-k                                 
    ("gemini", "hybrid", "hungarian"),    # top-k                              
    ("openai", "hybrid", "soft_max"),    # top-k                               
    ("openai", "hybrid", "greedy"),    # keep-within                           
    ("openai", "hybrid", "hungarian"),    # keep-within                        
]  

BEST_EMBEDDER = "gemini"            # family_refine — "gemini" | "openai"
BEST_SCORER = "embedding"           # family_refine — "embedding" | "hybrid"

BEST_TAU = 0.15                     # family_refine — applicable weights of the pinned structure
BEST_COUNT_ALPHA = 0.25             #   (count_alpha/reuse/contradiction apply only under soft_max)
BEST_REUSE_WEIGHT = 0.15
BEST_CONTRADICTION_WEIGHT = 0.20

BEST_W_CATEGORY = 0.25              # family_refine — hybrid blend weights (only if BEST_SCORER=="hybrid")
BEST_W_EMBEDDING = 0.40
BEST_W_ENTITY = 0.25
BEST_W_EVIDENCE = 0.10

BEST_ALIGNMENT = "soft_max"         # structure_screen/family_refine — "soft_max" | "greedy" | "hungarian"

BEST_THETA = 0.80                   # map_theta — final theta (re-confirmed at the full score function)
BEST_REJECT_THETA = 0.20            # map_theta — must be < BEST_THETA

# ─────────────────────────────────────────────────────────────────────────────
# Grids. (theta/reject reuse the validated map_theta_sweep engine grids.)
# ─────────────────────────────────────────────────────────────────────────────
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

# structure_screen finalist selection.
DEFAULT_TOP_K = 3
DEFAULT_KEEP_WITHIN = 0.02          # also keep structures within this gap of the best primary metric
# Pareto-aware finalist selection over (strict_f1_optimal ↑, escalate_rate ↓):
PARETO_CAP = 8                      # max non-dominated frontier structures kept as finalists
MIN_ECONOMY_F1 = 0.50               # quality floor an economy (low-escalate) finalist must clear
COST_LAMBDA = 0.20                  # knee = argmax(strict_f1_optimal − COST_LAMBDA · escalate_rate)

# Paste the structure_screen output here, then run --stage family_refine.
FINALIST_STRUCTURES: list[tuple[str, str, str]] = []   # (embedder, scorer_kind, alignment)

_FINALIST_EMPTY_MSG = (
    "FINALIST_STRUCTURES is empty — family_refine has nothing to refine.\n"
    "  1. Run:  python -m eval.silver.run_new_summarization_sweeps --stage structure_screen\n"
    "  2. Copy the printed `FINALIST_STRUCTURES = [...]` block into the constant near the top\n"
    "     of eval/silver/run_new_summarization_sweeps.py.\n"
    "  3. Re-run: python -m eval.silver.run_new_summarization_sweeps --stage family_refine"
)

STAGES = ("structure_screen", "family_refine", "map_theta", "map_gates")

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
        "run.yaml: agreement.embedder, scorer_kind, alignment_strategy + the applicable weights."
    ),
    "map_theta": (
        "PIN after map_theta → final BEST_THETA, BEST_REJECT_THETA (+ run.yaml map.theta, "
        "map.reject_theta), at the chosen scorer/embedder/weights/alignment."
    ),
    "map_gates": (
        "PIN after map_gates → run.yaml routing.legacy_single_voter_policy + "
        "agreement.force_escalate_on_polarity_conflict. Then run the held-out evaluation "
        "(separate runbook: run_paper.py --sync → export_pipeline → evaluate --matcher optimal)."
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
    "n_polarity_conflict_chunks", "polarity_conflict_rate",
    "split", "seed", "dev_fraction", "sim_threshold",
]


# ── ScorerSpec builders ──────────────────────────────────────────────────────

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


# ── Per-stage plan (specs + axes + name→{kind,alignment,embedder} maps) ───────

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
        return (embedders, specs, list(THETA_GRID), list(REJECT_THETA_GRID),
                ("keep",), (True,), kind_of, align_of, embedder_of)

    if stage == "map_theta":
        spec = ScorerSpec(BEST_SCORER, BEST_SCORER, _pinned_agreement(BEST_ALIGNMENT))
        return ([BEST_EMBEDDER], [spec], list(THETA_GRID), list(REJECT_THETA_GRID),
                ("keep",), (True,), {spec.name: BEST_SCORER}, {spec.name: BEST_ALIGNMENT}, None)

    if stage == "map_gates":
        spec = ScorerSpec(BEST_SCORER, BEST_SCORER, _pinned_agreement(BEST_ALIGNMENT))
        return ([BEST_EMBEDDER], [spec], [BEST_THETA], [BEST_REJECT_THETA],
                ("keep", "escalate"), (True, False),
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


# ── Ranking (primary metric, tie-breaks: escalate, f1_optimal, simpler config) ─

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
    escalate_rate, then higher f1_optimal, then SIMPLER config (fewer deviations
    from defaults). Greedy metrics NEVER enter the key."""
    return (
        float(row[metric]),
        -float(row.get("escalate_rate") or 0.0),
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
    """PURE, Pareto-aware finalist selection over (``metric`` ↑, ``escalate_rate`` ↓),
    so structure_screen keeps BOTH the quality corner (high F1, expensive) AND the
    economy/knee end (low escalate).

    CELL-LEVEL is load-bearing: the cost axis (escalate_rate) is driven by THETA, not
    by structure identity, so the Pareto frontier is computed over the FULL cells
    (structure × theta × reject) — NOT per-structure bests, which would collapse every
    structure to its θ=0.9 (most-expensive) cell and discard the entire cheap-θ economy
    frontier. A structure becomes a finalist if it TOUCHES the frontier at any cell.
    ``top-k`` / ``within-best`` stay QUALITY picks (per-structure max-metric).

    Selection uses ONLY ``metric`` (an optimal-matcher metric) + ``escalate_rate`` —
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
        return float(r.get("escalate_rate") or 0.0)

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


# ── Cells + checkpoint / resume ──────────────────────────────────────────────

def _iter_cells(stage: str):
    """Yield one dict per grid cell — the SINGLE enumeration source shared by
    --list-variants, the run loop, and the checkpoint keys, so the listed count,
    what actually runs, and the resume keys can never drift apart."""
    embedders, specs, thetas, rejects, policies, polarities, kind_of, align_of, embedder_of = \
        _stage_plan(stage)
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
                            }


def _canonical_key(embedder, variant, theta, reject, policy, polarity) -> tuple:
    """Canonical cell identity — the ONE place a cell key is built, coercing types
    identically whether the source is a cell dict or a reloaded CSV row (so a resume
    skip can never silently miss and re-append a duplicate row)."""
    return (str(embedder), str(variant), f"{float(theta):.2f}", f"{float(reject):.2f}",
            str(policy), str(polarity).lower())


def _cell_key_from_cell(c: dict) -> tuple:
    return _canonical_key(c["embedder"], c["spec"].name, c["theta"], c["reject"],
                          c["policy"], c["polarity"])


def _cell_key_from_row(r: dict) -> tuple:
    # rows use the engine's column name "reject_theta"; cells use "reject"
    return _canonical_key(r["embedder"], r["variant"], r["theta"], r["reject_theta"],
                          r["legacy_single_voter_policy"], r["force_escalate_on_polarity_conflict"])


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


def _ckpt_csv(stage: str) -> Path:
    return REPORTS_DIR / f"checkpoint_{stage}.csv"


def _ckpt_meta(stage: str) -> Path:
    return REPORTS_DIR / f"checkpoint_{stage}.meta.json"


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


# ── Variant listing (from _iter_cells, the same source the run uses) ─────────

def _list_variants(stage: str) -> list[str]:
    """Human-readable per-cell names (no API, no cache needed)."""
    return [
        f"embedder={c['embedder']}  scorer_kind={c['scorer_kind']}  variant={c['spec'].name}  "
        f"align={c['alignment']}  theta={c['theta']:.2f}  reject={c['reject']:.2f}  "
        f"single_voter={c['policy']}  polarity_fail={c['polarity']}"
        for c in _iter_cells(stage)
    ]


def _stamp_row(r: dict, emb: str, kind_of: dict, align_of: dict) -> dict:
    """Stamp embedder / scorer_kind / variant / alignment onto a run_sweep row, and
    blank the soft-align weights that are INERT under one-to-one alignment so the CSV
    / best-line don't report them as meaningful (``tau`` still applies and stays)."""
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
    return r


def _run_stage(stage: str, args) -> list[dict]:
    """Run the stage cell-by-cell, appending each completed cell to the checkpoint
    (flushed) and resuming from it. Each cell is one single-element ``run_sweep``
    call, so the cell key is constructed in exactly one place (the harness)."""
    if stage == "family_refine" and not FINALIST_STRUCTURES:
        raise SystemExit(_FINALIST_EMPTY_MSG)

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
            for c in emb_cells:
                rows = run_sweep(
                    voter_cache=ctx.voter_cache,
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
                               {c["spec"].name: c["alignment"]})
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
          f"{'esc':>6} {'θ':>4} {'rej':>4} {'[greedy]':>9}  reasons")
    for struct, r in ranked:
        emb, sk, al = struct
        mark = "★" if struct in reason_of else " "
        why = "/".join(reason_of.get(struct, []))
        print(f"  {mark} {str(emb):8} {str(sk):10} {str(al):10} "
              f"{float(r[args.metric]):8.4f} {float(r.get('f1_optimal') or 0):7.4f} "
              f"{float(r.get('escalate_rate') or 0):6.3f} {str(r.get('theta')):>4} "
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
              f'@θ{cell.get("theta")} esc={float(cell.get("escalate_rate") or 0):.2f} '
              f'f1={float(cell[args.metric]):.3f}')
    print("]")


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
                         "diagnostic and never selects. Ties: lower escalate_rate, then f1_optimal, "
                         "then simpler config.")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                    help="structure_screen: keep this many top structures (by primary metric) as "
                         "quality finalists.")
    ap.add_argument("--keep-within", type=float, default=DEFAULT_KEEP_WITHIN,
                    help="structure_screen: also keep structures within this primary-metric gap of the best.")
    ap.add_argument("--pareto-cap", type=int, default=PARETO_CAP,
                    help="structure_screen: max non-dominated Pareto-frontier structures to keep.")
    ap.add_argument("--min-economy-f1", type=float, default=MIN_ECONOMY_F1,
                    help="structure_screen: quality floor an economy (low-escalate) finalist must clear.")
    ap.add_argument("--cost-lambda", type=float, default=COST_LAMBDA,
                    help="structure_screen: knee = argmax(strict_f1_optimal − λ·escalate_rate).")
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
        if args.stage == "family_refine" and not FINALIST_STRUCTURES:
            print("\nfamily_refine: 0 variant(s) — FINALIST_STRUCTURES is empty.\n")
            print(_FINALIST_EMPTY_MSG)
            return
        cells = _list_variants(args.stage)
        print(f"\n{args.stage}: {len(cells)} variant(s)")
        for c in cells:
            print("  " + c)
        return

    if args.from_csv:
        if args.stage != "structure_screen":
            raise SystemExit("--from-csv only applies to --stage structure_screen")
        with open(args.from_csv, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise SystemExit(f"no rows in {args.from_csv}")
        print(f"Recomputing finalists from {args.from_csv} ({len(rows)} rows) — no rerun, no engine.")
        _report_structure_screen(rows, args, Path(args.from_csv))
        return

    rows = _run_stage(args.stage, args)   # raises SystemExit for an empty family_refine
    if not rows:
        print("No rows produced (empty grid or no silver overlap).")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    csv_path = REPORTS_DIR / f"new_sweep_{args.stage}_{timestamp}.csv"
    _write_csv(csv_path, rows, _CSV_FIELDS)
    _cleanup_checkpoint(args.stage)   # stage completed — discard the resume checkpoint

    if args.stage == "structure_screen":
        _report_structure_screen(rows, args, csv_path)
        print("\n" + PIN_HINTS[args.stage])
        return

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
          f"(tie-break: lower escalate_rate, then f1_optimal, then simpler config):\n  "
          f"{_fmt_best(best, args.metric)}")
    print(f"\nCSV → {csv_path}")
    print("\n" + PIN_HINTS[args.stage])


if __name__ == "__main__":
    main()
