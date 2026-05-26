#!/usr/bin/env python3
"""
run_summarization_experiments.py — orchestrator for the MAP-stage
experiments described in ``docs/FINAL_SUMMARIZATION_EXPERIMENTS.md``.

Composes the per-stage MAP sweep harness in
``eval/silver/run_summarization_sweeps.py`` into the branched recipe from
the FINAL doc:

  Gemini branch:  EXP 1 → EXP 2 → EXP 3 → FINAL_GEMINI_MAP_CONFIG
  OpenAI branch:  EXP 4 → EXP 5 → EXP 6 → FINAL_OPENAI_MAP_CONFIG
  Comparison:     EXP 7 (FINAL_GEMINI vs FINAL_OPENAI) → FINAL_MAP_CONFIG
  Confirmation:   EXP F (held-out test pass on FINAL_MAP_CONFIG)
  Validation:     EXP A–E (bootstrap CIs, ABC ablation, agreement→accuracy,
                            matcher sensitivity, recall-gap audit)

The orchestrator reuses existing per-stage code (``run_sweep`` from
``eval/silver/map_theta_sweep.py``; helpers from
``eval/silver/run_summarization_sweeps.py``); it does not duplicate sweep
mechanics. Each EXP is an ``ExperimentSpec`` with a ``run(ctx)`` callable
and an explicit ``depends_on`` list so the dependency graph between EXPs
is inspectable from ``--list-variants``.

Run mode is opt-in; the default is dry-run / list. Calibration EXPs are
$0 (offline replay over the existing primer + embedding caches); EXP F is
also $0 (different split, same caches). Validation EXPs A–E require either
new code or external setup (manual audit, separate Sonnet-only primer)
and are shipped as documented stubs that point at the FINAL doc.

Usage::

    python scripts/eval/run_summarization_experiments.py                       # stage menu
    python scripts/eval/run_summarization_experiments.py --list-variants
    python scripts/eval/run_summarization_experiments.py --stage calibration --list-variants
    python scripts/eval/run_summarization_experiments.py --exp exp_1_gemini_scorer
    python scripts/eval/run_summarization_experiments.py --branch gemini
    python scripts/eval/run_summarization_experiments.py --stage all
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

# Make ``eval.*`` and ``pipeline.*`` importable when this script is run directly
# (``scripts/`` is not a package — same pattern as run_all_sweeps.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_DEFAULT_REPORTS_DIR = _REPO_ROOT / "eval" / "reports"
_DEFAULT_SIM_THRESHOLD = 0.55
_DEFAULT_BOOTSTRAP_N = 50


# ─────────────────────────────────────────────────────────────────────────────
# Stage / experiment registry
# ─────────────────────────────────────────────────────────────────────────────

STAGE_ORDER = (
    "calibration",   # EXP 1-7 + F
    "validation",    # EXP A-E
    "all",
)

STAGE_BLURBS: dict[str, str] = {
    "calibration": (
        "Picks the production MAP config. Each calibration EXP wraps existing "
        "per-stage sweeps from eval/silver/run_summarization_sweeps.py and "
        "writes its own CSV. Greedy stage-wise: EXP n+1 reads BEST_* state "
        "produced by EXP n."
    ),
    "validation": (
        "Validates / stress-tests / explains FINAL_MAP_CONFIG. Bootstrap CIs "
        "(EXP A), ABC mechanism ablation (EXP B), agreement→accuracy "
        "calibration (EXP C), matcher-threshold sensitivity (EXP D), and a "
        "manual recall-gap audit (EXP E). Most are documented stubs today."
    ),
    "all": "Every experiment from calibration + validation, in dependency order.",
}


@dataclass
class ExperimentContext:
    """Per-run state passed to each experiment's ``run`` callable.

    ``state`` carries BEST_* picks across EXPs (greedy stage-wise selection).
    EXP n writes into ``state`` via the returned ``ExperimentResult``; EXP n+1
    reads what it needs.
    """
    output_dir: Path
    embedder_hint: Optional[str]    # restrict to "gemini" / "openai" / None=both branches
    split: str = "dev"
    seed: int = 42
    dev_fraction: float = 0.8
    sim_threshold: float = _DEFAULT_SIM_THRESHOLD
    state: dict = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Result returned by an experiment's ``run`` callable.

    ``state_updates`` are merged into ``ExperimentContext.state`` so downstream
    EXPs see the freshly-picked BEST_* values.
    """
    csv_path: Optional[Path]
    winner: Optional[dict]
    notes: str
    state_updates: dict = field(default_factory=dict)


@dataclass
class ExperimentSpec:
    """One experiment from FINAL_SUMMARIZATION_EXPERIMENTS.md.

    ``status``:
      ``"executable"`` — wired to run end-to-end against the existing caches
      ``"stub"``       — recipe printed via --list-variants; needs new code
      ``"manual"``     — needs human labelling / external setup (e.g. EXP E)
    """
    name: str
    exp_id: str                       # "EXP_1", "EXP_A", etc. (for cross-ref to doc)
    blurb: str
    stage: str                        # "calibration" | "validation"
    branch: str                       # "gemini" | "openai" | "shared"
    depends_on: list[str] = field(default_factory=list)
    status: str = "executable"
    run: Callable[[ExperimentContext], ExperimentResult] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers: per-scorer-best aggregation, CSV writer, BEST_* parsers
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("(no rows)\n", encoding="utf-8")
        return
    # Stable column order: union of keys, sorted, with known leaders first.
    leaders = [
        "scorer", "theta", "reject_theta",
        "w_category", "w_embedding", "w_entity", "w_evidence", "weights_sum",
        "tau", "count_alpha", "reuse_weight", "contradiction_weight",
        "legacy_single_voter_policy", "force_escalate_on_polarity_conflict",
        "strict_f1", "f1", "precision", "recall",
        "escalate_rate", "early_accept_rate", "early_accept_precision",
        "n_polarity_conflict_chunks", "polarity_conflict_rate",
        "n_matched", "n_silver", "n_pipeline",
        "split", "seed", "dev_fraction", "sim_threshold", "cascade_path",
    ]
    seen = set(leaders)
    extras = sorted({k for r in rows for k in r.keys()} - seen)
    fieldnames = [c for c in leaders if any(c in r for r in rows)] + extras
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _per_scorer_best(rows: list[dict], metric: str = "strict_f1") -> list[dict]:
    """Group rows by ``scorer`` and keep the max-``metric`` row per group.

    Implements the "fair scorer comparison" rule from FINAL EXP 1 / EXP 4 —
    each scorer config gets its own best θ / reject_θ pair before scorer-to-
    scorer comparison. Tie-break: lower escalation rate.
    """
    best: dict[str, dict] = {}
    for r in rows:
        s = r.get("scorer", "?")
        cur = best.get(s)
        if cur is None:
            best[s] = r
            continue
        a = (r.get(metric, 0.0), -r.get("escalate_rate", 1.0))
        b = (cur.get(metric, 0.0), -cur.get("escalate_rate", 1.0))
        if a > b:
            best[s] = r
    return list(best.values())


def _build_exp1_scorer_specs():
    """Construct the 7 scorer ScorerSpecs for EXP 1 / EXP 4.

    1 embedding (default `AgreementConfig`) + 6 hybrid variants from
    ``run_summarization_sweeps.HYBRID_BLEND_GRID``. The hybrid variants vary
    only the blend weights; soft-alignment weights stay at defaults so the
    comparison isolates the scorer-family axis.
    """
    from eval.silver.map_theta_sweep import ScorerSpec
    from eval.silver.run_summarization_sweeps import HYBRID_BLEND_GRID
    from pipeline.stages.summarization.config import AgreementConfig, HybridConfig

    specs = [ScorerSpec("embedding_default", "embedding", AgreementConfig())]
    for label, (wc, we, wn, wv) in HYBRID_BLEND_GRID.items():
        cfg = AgreementConfig(
            scorer_kind="hybrid",
            hybrid=HybridConfig(
                w_category=wc, w_embedding=we, w_entity=wn, w_evidence=wv,
            ),
        )
        specs.append(ScorerSpec(label, "hybrid", cfg))
    return specs


def _agreement_weight_grid_around(base_cfg) -> list:
    """One-axis-at-a-time variants of soft-alignment weights around the
    selected scorer's defaults. Mirrors ``_weight_variant_specs`` for the
    hybrid-on-hybrid case but built from the EXP 1 / 4 winner.
    """
    from eval.silver.map_theta_sweep import ScorerSpec
    from eval.silver.run_summarization_sweeps import (
        TAU_GRID, COUNT_ALPHA_GRID, REUSE_WEIGHT_GRID, CONTRADICTION_WEIGHT_GRID,
    )
    from pipeline.stages.summarization.config import AgreementConfig

    base_tau = base_cfg.tau
    base_count = base_cfg.count_alpha
    base_reuse = base_cfg.reuse_weight
    base_contra = base_cfg.contradiction_weight

    specs = [ScorerSpec("baseline", base_cfg.scorer_kind, base_cfg)]
    grids = {
        "tau": TAU_GRID, "count_alpha": COUNT_ALPHA_GRID,
        "reuse_weight": REUSE_WEIGHT_GRID, "contradiction_weight": CONTRADICTION_WEIGHT_GRID,
    }
    base_map = {"tau": base_tau, "count_alpha": base_count,
                "reuse_weight": base_reuse, "contradiction_weight": base_contra}
    for knob, grid in grids.items():
        for v in grid:
            if v == base_map[knob]:
                continue
            kwargs = {
                "tau": base_tau, "count_alpha": base_count,
                "reuse_weight": base_reuse, "contradiction_weight": base_contra,
                "scorer_kind": base_cfg.scorer_kind,
                "hybrid": base_cfg.hybrid,
            }
            kwargs[knob] = v
            specs.append(ScorerSpec(
                f"{base_cfg.scorer_kind}_{knob}_{v}",
                base_cfg.scorer_kind,
                AgreementConfig(**kwargs),
            ))
    return specs


# ─────────────────────────────────────────────────────────────────────────────
# Calibration experiments (executable)
# ─────────────────────────────────────────────────────────────────────────────

def _run_branch_scorer_comparison(ctx: ExperimentContext, embedder: str) -> ExperimentResult:
    """Shared body for EXP 1 (Gemini) and EXP 4 (OpenAI) — tuned scorer
    comparison: full θ × reject_θ × 7-scorer sweep, then per-scorer best-of.
    """
    from eval.silver.map_theta_sweep import (
        run_sweep, THETA_GRID, REJECT_THETA_GRID,
    )
    from eval.silver.run_summarization_sweeps import _load_map_context

    map_ctx = _load_map_context(embedder, embed_cache_path=None)
    specs = _build_exp1_scorer_specs()
    rows = run_sweep(
        voter_cache=map_ctx.voter_cache,
        silver_by_case=map_ctx.silver_by_case,
        embedder=map_ctx.embedder,
        embed_cache=map_ctx.embed_cache,
        sim_threshold=ctx.sim_threshold,
        scorer_specs=specs,
        thetas=THETA_GRID,
        reject_thetas=REJECT_THETA_GRID,
        split=ctx.split,
        seed=ctx.seed,
        dev_fraction=ctx.dev_fraction,
        agreement_embed_fn=map_ctx.agreement_embed_fn,
        # Stage 1/4 holds policy + polarity at production defaults — only
        # the scorer + θ/reject axes vary.
        single_voter_policies=("keep",),
        force_escalate_on_polarity_conflict_grid=(True,),
    )
    per_scorer = _per_scorer_best(rows, metric="strict_f1")
    winner = max(per_scorer, key=lambda r: r["strict_f1"]) if per_scorer else None

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    full_csv = ctx.output_dir / f"exp_{embedder}_scorer_full_{ts}.csv"
    best_csv = ctx.output_dir / f"exp_{embedder}_scorer_best_per_{ts}.csv"
    _write_csv_rows(full_csv, rows)
    _write_csv_rows(best_csv, per_scorer)

    notes = (
        f"{len(rows)} cells across 7 scorers × {len(THETA_GRID)*len(REJECT_THETA_GRID)} "
        f"(θ, rej) (reject<θ filter); per-scorer best-of in {best_csv.name}."
    )
    if winner is None:
        return ExperimentResult(csv_path=full_csv, winner=None, notes=notes)
    prefix = embedder.upper()
    state_updates = {
        f"BEST_{prefix}_SCORER": winner["scorer"],
        f"BEST_{prefix}_THETA": winner["theta"],
        f"BEST_{prefix}_REJECT_THETA": winner["reject_theta"],
        f"BEST_{prefix}_HYBRID_WEIGHTS": (
            (winner.get("w_category"), winner.get("w_embedding"),
             winner.get("w_entity"), winner.get("w_evidence"))
            if str(winner["scorer"]).startswith("hybrid") else None
        ),
        f"BEST_{prefix}_STRICT_F1": winner["strict_f1"],
    }
    notes += (
        f"\n  Winner: {winner['scorer']} at θ={winner['theta']} rej={winner['reject_theta']}  "
        f"strict_f1={winner['strict_f1']:.3f}  f1={winner['f1']:.3f}  "
        f"esc={winner['escalate_rate']:.2f}"
    )
    return ExperimentResult(
        csv_path=best_csv, winner=winner, notes=notes, state_updates=state_updates,
    )


def _run_branch_agreement_weights(ctx: ExperimentContext, embedder: str) -> ExperimentResult:
    """Shared body for EXP 2 (Gemini) / EXP 5 (OpenAI) — soft-alignment
    weight sweep around the EXP 1/4 winner, holding scorer + θ + reject_θ
    pinned. One-axis-at-a-time grid; baseline-included.
    """
    prefix = embedder.upper()
    s = ctx.state
    required = (f"BEST_{prefix}_SCORER", f"BEST_{prefix}_THETA",
                f"BEST_{prefix}_REJECT_THETA")
    missing = [k for k in required if k not in s]
    if missing:
        return ExperimentResult(
            csv_path=None, winner=None,
            notes=f"Skipped — missing state from EXP 1/{prefix.lower()}: {missing}",
        )

    from eval.silver.map_theta_sweep import run_sweep
    from eval.silver.run_summarization_sweeps import _load_map_context
    from pipeline.stages.summarization.config import AgreementConfig, HybridConfig

    winner_scorer = s[f"BEST_{prefix}_SCORER"]
    scorer_kind = "hybrid" if winner_scorer.startswith("hybrid") else "embedding"
    hybrid_cfg = HybridConfig()
    if scorer_kind == "hybrid" and s.get(f"BEST_{prefix}_HYBRID_WEIGHTS"):
        wc, we, wn, wv = s[f"BEST_{prefix}_HYBRID_WEIGHTS"]
        hybrid_cfg = HybridConfig(
            w_category=wc, w_embedding=we, w_entity=wn, w_evidence=wv,
        )
    base_cfg = AgreementConfig(scorer_kind=scorer_kind, hybrid=hybrid_cfg)
    specs = _agreement_weight_grid_around(base_cfg)

    map_ctx = _load_map_context(embedder, embed_cache_path=None)
    rows = run_sweep(
        voter_cache=map_ctx.voter_cache,
        silver_by_case=map_ctx.silver_by_case,
        embedder=map_ctx.embedder,
        embed_cache=map_ctx.embed_cache,
        sim_threshold=ctx.sim_threshold,
        scorer_specs=specs,
        thetas=[s[f"BEST_{prefix}_THETA"]],
        reject_thetas=[s[f"BEST_{prefix}_REJECT_THETA"]],
        split=ctx.split,
        seed=ctx.seed,
        dev_fraction=ctx.dev_fraction,
        agreement_embed_fn=map_ctx.agreement_embed_fn,
        single_voter_policies=("keep",),
        force_escalate_on_polarity_conflict_grid=(True,),
    )
    winner = max(rows, key=lambda r: r["strict_f1"]) if rows else None

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    csv_path = ctx.output_dir / f"exp_{embedder}_agreement_weights_{ts}.csv"
    _write_csv_rows(csv_path, rows)

    if winner is None:
        return ExperimentResult(csv_path=csv_path, winner=None,
                                 notes="No rows produced.")
    state_updates = {
        f"BEST_{prefix}_AGREEMENT_WEIGHTS": (
            winner.get("tau"), winner.get("count_alpha"),
            winner.get("reuse_weight"), winner.get("contradiction_weight"),
        ),
    }
    notes = (
        f"{len(rows)} cells (one-axis-at-a-time around BEST_{prefix}_SCORER).\n"
        f"  Winner soft-alignment: tau={winner.get('tau')} "
        f"count_alpha={winner.get('count_alpha')} "
        f"reuse_weight={winner.get('reuse_weight')} "
        f"contradiction_weight={winner.get('contradiction_weight')}  "
        f"strict_f1={winner['strict_f1']:.3f}"
    )
    return ExperimentResult(
        csv_path=csv_path, winner=winner, notes=notes, state_updates=state_updates,
    )


def _run_branch_polarity_flag(ctx: ExperimentContext, embedder: str) -> ExperimentResult:
    """Shared body for EXP 3 (Gemini) / EXP 6 (OpenAI) — polarity-flag
    ablation at the EXP 2/5-pinned config (scorer + θ + reject_θ +
    agreement weights). 2 cells: True vs False.
    """
    prefix = embedder.upper()
    s = ctx.state
    required = (f"BEST_{prefix}_SCORER", f"BEST_{prefix}_THETA",
                f"BEST_{prefix}_REJECT_THETA")
    missing = [k for k in required if k not in s]
    if missing:
        return ExperimentResult(
            csv_path=None, winner=None,
            notes=f"Skipped — missing state from EXP 1-2/{prefix.lower()}: {missing}",
        )

    from eval.silver.map_theta_sweep import run_sweep, ScorerSpec
    from eval.silver.run_summarization_sweeps import _load_map_context
    from pipeline.stages.summarization.config import AgreementConfig, HybridConfig

    winner_scorer = s[f"BEST_{prefix}_SCORER"]
    scorer_kind = "hybrid" if winner_scorer.startswith("hybrid") else "embedding"
    hybrid_cfg = HybridConfig()
    if scorer_kind == "hybrid" and s.get(f"BEST_{prefix}_HYBRID_WEIGHTS"):
        wc, we, wn, wv = s[f"BEST_{prefix}_HYBRID_WEIGHTS"]
        hybrid_cfg = HybridConfig(
            w_category=wc, w_embedding=we, w_entity=wn, w_evidence=wv,
        )
    aw = s.get(f"BEST_{prefix}_AGREEMENT_WEIGHTS")
    base_kwargs = dict(scorer_kind=scorer_kind, hybrid=hybrid_cfg)
    if aw is not None:
        tau, ca, rw, cw = aw
        base_kwargs.update(tau=tau, count_alpha=ca, reuse_weight=rw,
                           contradiction_weight=cw)
    base_cfg = AgreementConfig(**base_kwargs)
    spec = ScorerSpec(f"best_{prefix.lower()}", scorer_kind, base_cfg)

    map_ctx = _load_map_context(embedder, embed_cache_path=None)
    rows = run_sweep(
        voter_cache=map_ctx.voter_cache,
        silver_by_case=map_ctx.silver_by_case,
        embedder=map_ctx.embedder,
        embed_cache=map_ctx.embed_cache,
        sim_threshold=ctx.sim_threshold,
        scorer_specs=[spec],
        thetas=[s[f"BEST_{prefix}_THETA"]],
        reject_thetas=[s[f"BEST_{prefix}_REJECT_THETA"]],
        split=ctx.split,
        seed=ctx.seed,
        dev_fraction=ctx.dev_fraction,
        agreement_embed_fn=map_ctx.agreement_embed_fn,
        single_voter_policies=("keep",),
        force_escalate_on_polarity_conflict_grid=(True, False),
    )
    winner = max(rows, key=lambda r: r["strict_f1"]) if rows else None

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    csv_path = ctx.output_dir / f"exp_{embedder}_polarity_flag_{ts}.csv"
    _write_csv_rows(csv_path, rows)

    if winner is None:
        return ExperimentResult(csv_path=csv_path, winner=None,
                                 notes="No rows produced.")
    state_updates = {
        f"BEST_{prefix}_POLARITY_FLAG": winner.get("force_escalate_on_polarity_conflict"),
        f"FINAL_{prefix}_MAP_CONFIG": {
            "scorer": s[f"BEST_{prefix}_SCORER"],
            "theta": s[f"BEST_{prefix}_THETA"],
            "reject_theta": s[f"BEST_{prefix}_REJECT_THETA"],
            "hybrid_weights": s.get(f"BEST_{prefix}_HYBRID_WEIGHTS"),
            "agreement_weights": s.get(f"BEST_{prefix}_AGREEMENT_WEIGHTS"),
            "polarity_flag": winner.get("force_escalate_on_polarity_conflict"),
            "strict_f1": winner["strict_f1"],
        },
    }
    notes = (
        f"2 cells (True vs False). Winner: "
        f"force_escalate_on_polarity_conflict={winner.get('force_escalate_on_polarity_conflict')!r}  "
        f"strict_f1={winner['strict_f1']:.3f}  "
        f"n_polarity_conflicts={winner.get('n_polarity_conflict_chunks')}  "
        f"polarity_rate={winner.get('polarity_conflict_rate')}"
    )
    return ExperimentResult(
        csv_path=csv_path, winner=winner, notes=notes, state_updates=state_updates,
    )


def _run_exp_1(ctx): return _run_branch_scorer_comparison(ctx, "gemini")
def _run_exp_4(ctx): return _run_branch_scorer_comparison(ctx, "openai")
def _run_exp_2(ctx): return _run_branch_agreement_weights(ctx, "gemini")
def _run_exp_5(ctx): return _run_branch_agreement_weights(ctx, "openai")
def _run_exp_3(ctx): return _run_branch_polarity_flag(ctx, "gemini")
def _run_exp_6(ctx): return _run_branch_polarity_flag(ctx, "openai")


def _run_exp_7(ctx: ExperimentContext) -> ExperimentResult:
    """EXP 7 — embedder branch comparison: FINAL_GEMINI_MAP_CONFIG vs
    FINAL_OPENAI_MAP_CONFIG. Reads the two FINAL_* picks from state and
    emits a side-by-side comparison row.
    """
    s = ctx.state
    g = s.get("FINAL_GEMINI_MAP_CONFIG")
    o = s.get("FINAL_OPENAI_MAP_CONFIG")
    if g is None or o is None:
        return ExperimentResult(
            csv_path=None, winner=None,
            notes=f"Skipped — missing FINAL config: gemini={g is not None}, openai={o is not None}",
        )

    def _row(embedder: str, cfg: dict) -> dict:
        return {
            "embedder": embedder,
            "scorer": cfg["scorer"],
            "theta": cfg["theta"],
            "reject_theta": cfg["reject_theta"],
            "polarity_flag": cfg["polarity_flag"],
            "strict_f1": cfg["strict_f1"],
            "hybrid_weights": json.dumps(cfg["hybrid_weights"]),
            "agreement_weights": json.dumps(cfg["agreement_weights"]),
        }
    rows = [_row("gemini", g), _row("openai", o)]

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    csv_path = ctx.output_dir / f"exp_7_embedder_comparison_{ts}.csv"
    _write_csv_rows(csv_path, rows)

    winner = max(rows, key=lambda r: r["strict_f1"])
    notes = (
        f"Gemini strict_f1={g['strict_f1']:.3f} vs OpenAI strict_f1={o['strict_f1']:.3f}.\n"
        f"  Winner: {winner['embedder']} (Δstrict_f1={abs(g['strict_f1']-o['strict_f1']):.3f}).\n"
        f"  NOTE: bootstrap CI (EXP A) is required to tell whether the gap is "
        f"statistically meaningful. If overlap → pick cheaper/faster embedder."
    )
    return ExperimentResult(
        csv_path=csv_path, winner=winner, notes=notes,
        state_updates={
            "BEST_EMBEDDER": winner["embedder"],
            "FINAL_MAP_CONFIG": winner,
        },
    )


def _run_exp_f(ctx: ExperimentContext) -> ExperimentResult:
    """EXP F — held-out test confirmation of FINAL_MAP_CONFIG.

    Re-runs the chosen embedder + scorer + θ/reject + agreement weights +
    polarity flag on ``--split test`` (48 cases) to confirm dev-tuned picks
    generalise. Single row out. $0 (different split, same caches).
    """
    final = ctx.state.get("FINAL_MAP_CONFIG")
    if final is None:
        return ExperimentResult(
            csv_path=None, winner=None,
            notes="Skipped — FINAL_MAP_CONFIG not set (run EXP 7 first).",
        )

    embedder = final["embedder"]
    prefix = embedder.upper()
    cfg = ctx.state.get(f"FINAL_{prefix}_MAP_CONFIG", final)

    from eval.silver.map_theta_sweep import run_sweep, ScorerSpec
    from eval.silver.run_summarization_sweeps import _load_map_context
    from pipeline.stages.summarization.config import AgreementConfig, HybridConfig

    scorer_kind = "hybrid" if str(cfg["scorer"]).startswith("hybrid") else "embedding"
    hybrid_cfg = HybridConfig()
    if scorer_kind == "hybrid" and cfg.get("hybrid_weights"):
        wc, we, wn, wv = cfg["hybrid_weights"]
        hybrid_cfg = HybridConfig(
            w_category=wc, w_embedding=we, w_entity=wn, w_evidence=wv,
        )
    base_kwargs = dict(scorer_kind=scorer_kind, hybrid=hybrid_cfg)
    if cfg.get("agreement_weights"):
        tau, ca, rw, cw = cfg["agreement_weights"]
        base_kwargs.update(tau=tau, count_alpha=ca, reuse_weight=rw,
                           contradiction_weight=cw)
    base_cfg = AgreementConfig(**base_kwargs)
    spec = ScorerSpec(f"final_{embedder}", scorer_kind, base_cfg)

    map_ctx = _load_map_context(embedder, embed_cache_path=None)
    rows = run_sweep(
        voter_cache=map_ctx.voter_cache,
        silver_by_case=map_ctx.silver_by_case,
        embedder=map_ctx.embedder,
        embed_cache=map_ctx.embed_cache,
        sim_threshold=ctx.sim_threshold,
        scorer_specs=[spec],
        thetas=[cfg["theta"]],
        reject_thetas=[cfg["reject_theta"]],
        split="test",                       # ← the whole point of EXP F
        seed=ctx.seed,
        dev_fraction=ctx.dev_fraction,
        agreement_embed_fn=map_ctx.agreement_embed_fn,
        single_voter_policies=("keep",),
        force_escalate_on_polarity_conflict_grid=(cfg["polarity_flag"],),
    )
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    csv_path = ctx.output_dir / f"exp_f_test_confirmation_{ts}.csv"
    _write_csv_rows(csv_path, rows)

    if not rows:
        return ExperimentResult(csv_path=csv_path, winner=None,
                                 notes="No test rows — verify --split test has cases.")
    r = rows[0]
    dev_strict = cfg.get("strict_f1")
    delta = (r["strict_f1"] - dev_strict) if dev_strict is not None else None
    notes = (
        f"Test-split strict_f1={r['strict_f1']:.3f} (dev was {dev_strict:.3f}); "
        f"Δ={delta:+.3f}." if delta is not None else
        f"Test-split strict_f1={r['strict_f1']:.3f}."
    )
    notes += "\n  If Δ << 0 the dev-tuned picks may be overfit to the silver dev split."
    return ExperimentResult(csv_path=csv_path, winner=r, notes=notes)


# ─────────────────────────────────────────────────────────────────────────────
# Validation experiments (stubs — recipes printed; code lives in next PRs)
# ─────────────────────────────────────────────────────────────────────────────

def _stub_result(exp_id: str, recipe: str) -> ExperimentResult:
    """Render a stub experiment's recipe + the "would do" message."""
    return ExperimentResult(
        csv_path=None, winner=None,
        notes=(
            f"{exp_id} not implemented in this orchestrator yet. Recipe:\n"
            f"{recipe}\n"
            f"  See docs/FINAL_SUMMARIZATION_EXPERIMENTS.md §{exp_id} for full spec."
        ),
    )


def _run_exp_a(ctx):
    """EXP A — bootstrap confidence intervals on the winner + nearest competitors."""
    return _stub_result("EXP A — bootstrap CI", recipe=(
        "  For each sweep CSV produced by EXP 1–7 + F:\n"
        "    1. Identify the winner row + 3 nearest competitors (top-4 by strict_f1).\n"
        "    2. For n=50 random resamples of case_ids (seeded), recompute strict_f1 from\n"
        "       the per-case match outputs for each of the 4 candidates.\n"
        "    3. Emit (lo, hi) 95% bootstrap CI per candidate + tied-candidate list\n"
        "       (overlap-of-CIs == tie).\n"
        "  Implementation needs the per-case match outputs (case_id → matched flag)\n"
        "  to be retained from run_sweep — today they're discarded after _evaluate_outputs.\n"
        "  Add `--keep-per-case` to run_sweep, then implement here."
    ))


def _run_exp_b1(ctx):
    """EXP B.1 — routing usefulness: ABC vs matched-random escalation."""
    return _stub_result("EXP B.1 — matched random escalation", recipe=(
        "  Replay the FINAL_MAP_CONFIG cascade twice:\n"
        "    A. Real ABC (current _replay path) → escalate_rate=E\n"
        "    B. Matched random: route ⌊E·N_chunks⌋ chunks to L3 uniformly at random\n"
        "       (n=20 seeds). For each chunk in the random-escalation set, take L3\n"
        "       output; for the rest, take L1 voter's best (no agreement gate).\n"
        "    Compare strict_f1 mean ± 95% CI across seeds.\n"
        "  Implementation: subclass _replay with a `routing_mode={abc,random}` arg + seed.\n"
        "  No API; uses the same voter_cache + L3 cache."
    ))


def _run_exp_b2(ctx):
    """EXP B.2 — cost-quality: cascade vs cheap-only vs strong-model-only."""
    return _stub_result("EXP B.2 — cost-quality comparison", recipe=(
        "  Compare three baselines on FINAL_MAP_CONFIG papers:\n"
        "    1. cheap-only: L1 voter best, never escalate.\n"
        "    2. ABC cascade (FINAL_MAP_CONFIG).\n"
        "    3. strong-only: L3 (Sonnet) on every chunk.\n"
        "  CAVEAT: voter_cache.json has L3 output only for chunks ABC escalated.\n"
        "  To compute (3) honestly we need a separate all-Sonnet primer batch —\n"
        "  ~$30 estimated per the SUMMARIZATION_EXPERIMENT_PLAN cost notes.\n"
        "  Implementation needs a new prime mode + cost-table (configs/model_prices.json)."
    ))


def _run_exp_c(ctx):
    """EXP C — agreement-score vs accuracy calibration."""
    return _stub_result("EXP C — agreement→accuracy calibration", recipe=(
        "  For every chunk in the FINAL_MAP_CONFIG replay:\n"
        "    1. Record the agreement score (bundle.confidence from AgreementChecker).\n"
        "    2. Record the per-chunk silver-match flag (matched / unmatched).\n"
        "    3. Bin chunks by agreement: [0–0.5], (0.5–0.7], (0.7–0.85], (0.85–1.0].\n"
        "    4. Report per-bin: count, strict_f1, precision, recall.\n"
        "  Output: agreement_calibration_curve.csv + a sparkline-style summary.\n"
        "  Implementation: extend _replay to retain per-chunk agreement scores;\n"
        "  the AgreementChecker already returns them via bundle.confidence."
    ))


def _run_exp_d(ctx):
    """EXP D — matcher-threshold sensitivity."""
    return _stub_result("EXP D — matcher-threshold sensitivity", recipe=(
        "  Re-run the EXP 1 sweep at sim_threshold ∈ {0.50, 0.55, 0.60}.\n"
        "  For each threshold, recompute per-scorer best and re-rank the candidates.\n"
        "  Report:\n"
        "    - Kendall τ of candidate rankings between threshold pairs (stability)\n"
        "    - Identity of the winner under each threshold\n"
        "  Implementation: this orchestrator can drive it directly — iterate\n"
        "  sim_thresholds, call _run_branch_scorer_comparison with sim_threshold= override.\n"
        "  Could be promoted to executable in a follow-up if desired."
    ))


def _run_exp_e(ctx):
    """EXP E — recall-gap audit (manual classification)."""
    return _stub_result("EXP E — recall-gap audit", recipe=(
        "  Manual labelling task. For ~50 sampled false negatives from FINAL_MAP_CONFIG:\n"
        "    1. Auto-extract case_id, silver_finding, closest_pipeline_finding,\n"
        "       closest_similarity, all_pipeline_findings → eval/results/exp_e_audit.csv\n"
        "    2. Human reviewer fills `miss_category` ∈ {true_miss, atomicity_mismatch,\n"
        "       unsupported_silver, matcher_failure, category/entity/polarity_mismatch,\n"
        "       too_broad_pipeline, too_specific_silver, unclear}.\n"
        "    3. Auto-aggregate the category histogram and write final report.\n"
        "  Implementation needed: (a) sampler script that extracts the 50 rows from\n"
        "  per-case match outputs (waits on EXP A's --keep-per-case), (b) post-fill\n"
        "  aggregator. The human labelling itself is ~1-2 hours of reviewer time."
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Experiment registry
# ─────────────────────────────────────────────────────────────────────────────

ALL_EXPERIMENTS: list[ExperimentSpec] = [
    # ── Gemini branch (calibration) ────────────────────────────────────────
    ExperimentSpec(
        name="exp_1_gemini_scorer", exp_id="EXP_1", stage="calibration",
        branch="gemini", depends_on=[],
        blurb="Tuned scorer comparison (Gemini): 7 scorers × full θ/reject grid; per-scorer best-of.",
        run=_run_exp_1,
    ),
    ExperimentSpec(
        name="exp_2_gemini_weights", exp_id="EXP_2", stage="calibration",
        branch="gemini", depends_on=["EXP_1"],
        blurb="Soft-alignment weight sweep (Gemini) around EXP 1 winner.",
        run=_run_exp_2,
    ),
    ExperimentSpec(
        name="exp_3_gemini_polarity", exp_id="EXP_3", stage="calibration",
        branch="gemini", depends_on=["EXP_2"],
        blurb="Polarity-flag ablation (Gemini) at EXP 1+2 config.",
        run=_run_exp_3,
    ),
    # ── OpenAI branch (calibration) ────────────────────────────────────────
    ExperimentSpec(
        name="exp_4_openai_scorer", exp_id="EXP_4", stage="calibration",
        branch="openai", depends_on=[],
        blurb="Tuned scorer comparison (OpenAI): 7 scorers × full θ/reject grid; per-scorer best-of.",
        run=_run_exp_4,
    ),
    ExperimentSpec(
        name="exp_5_openai_weights", exp_id="EXP_5", stage="calibration",
        branch="openai", depends_on=["EXP_4"],
        blurb="Soft-alignment weight sweep (OpenAI) around EXP 4 winner.",
        run=_run_exp_5,
    ),
    ExperimentSpec(
        name="exp_6_openai_polarity", exp_id="EXP_6", stage="calibration",
        branch="openai", depends_on=["EXP_5"],
        blurb="Polarity-flag ablation (OpenAI) at EXP 4+5 config.",
        run=_run_exp_6,
    ),
    # ── Final embedder comparison ──────────────────────────────────────────
    ExperimentSpec(
        name="exp_7_embedder_comparison", exp_id="EXP_7", stage="calibration",
        branch="shared", depends_on=["EXP_3", "EXP_6"],
        blurb="Final embedder comparison: FINAL_GEMINI_MAP_CONFIG vs FINAL_OPENAI_MAP_CONFIG.",
        run=_run_exp_7,
    ),
    # ── Held-out test confirmation (new — recommended addition) ───────────
    ExperimentSpec(
        name="exp_f_test_confirmation", exp_id="EXP_F", stage="calibration",
        branch="shared", depends_on=["EXP_7"],
        blurb=("Held-out test pass on FINAL_MAP_CONFIG (--split test). "
               "Closes the dev-overfit gap; not in original FINAL doc."),
        run=_run_exp_f,
    ),
    # ── Validation block (stubs) ──────────────────────────────────────────
    ExperimentSpec(
        name="exp_a_bootstrap_ci", exp_id="EXP_A", stage="validation",
        branch="shared", depends_on=["EXP_1", "EXP_4", "EXP_7"],
        status="stub", run=_run_exp_a,
        blurb="Bootstrap 95% CIs on the winner + 3 nearest competitors for every sweep CSV.",
    ),
    ExperimentSpec(
        name="exp_b1_routing_usefulness", exp_id="EXP_B.1", stage="validation",
        branch="shared", depends_on=["EXP_7"],
        status="stub", run=_run_exp_b1,
        blurb="ABC vs matched-random escalation: does agreement beat chance?",
    ),
    ExperimentSpec(
        name="exp_b2_cost_quality", exp_id="EXP_B.2", stage="validation",
        branch="shared", depends_on=["EXP_7"],
        status="stub", run=_run_exp_b2,
        blurb=("Cost-quality: cascade vs cheap-only vs strong-model-only. "
               "Needs separate all-Sonnet primer."),
    ),
    ExperimentSpec(
        name="exp_c_agreement_accuracy", exp_id="EXP_C", stage="validation",
        branch="shared", depends_on=["EXP_7"],
        status="stub", run=_run_exp_c,
        blurb="Agreement-score → accuracy calibration curve (validates ABC's premise).",
    ),
    ExperimentSpec(
        name="exp_d_matcher_sensitivity", exp_id="EXP_D", stage="validation",
        branch="shared", depends_on=["EXP_1"],
        status="stub", run=_run_exp_d,
        blurb="Sensitivity of EXP 1 ranking to matcher sim_threshold ∈ {0.50, 0.55, 0.60}.",
    ),
    ExperimentSpec(
        name="exp_e_recall_gap_audit", exp_id="EXP_E", stage="validation",
        branch="shared", depends_on=["EXP_7"],
        status="manual", run=_run_exp_e,
        blurb=("Manual recall-gap audit on FINAL_MAP_CONFIG: 50 false negatives × "
               "10-category classification. ~1-2 hrs reviewer time."),
    ),
]


EXP_BY_NAME = {e.name: e for e in ALL_EXPERIMENTS}
EXP_BY_ID = {e.exp_id: e for e in ALL_EXPERIMENTS}


# ─────────────────────────────────────────────────────────────────────────────
# Listing / menu / orchestration
# ─────────────────────────────────────────────────────────────────────────────

def _print_stage_menu() -> None:
    print("Summarisation experiment orchestrator — calibration + validation.\n")
    print("Available --stage values:\n")
    for st in STAGE_ORDER:
        print(f"  {st}")
        print(f"      {STAGE_BLURBS.get(st, '')}")
    print()
    print("Experiments in dependency order:\n")
    for e in ALL_EXPERIMENTS:
        tag = ("            " if e.status == "executable"
               else "  [stub]   " if e.status == "stub"
               else "  [manual] ")
        deps = f"  ← {', '.join(e.depends_on)}" if e.depends_on else ""
        print(f"  {tag}{e.exp_id:<8} {e.name:<30} ({e.branch}, {e.stage}){deps}")
        print(f"            {e.blurb}")
    print()
    print("Examples:")
    print("  python scripts/eval/run_summarization_experiments.py --list-variants")
    print("  python scripts/eval/run_summarization_experiments.py --exp exp_1_gemini_scorer")
    print("  python scripts/eval/run_summarization_experiments.py --branch gemini")
    print("  python scripts/eval/run_summarization_experiments.py --stage all")
    print()
    print("NOTE: orchestrator defaults to --dry-run unless --run is passed.")


def _print_variants_table(experiments: list[ExperimentSpec]) -> None:
    header = ("exp_id", "name", "branch", "stage", "status", "depends_on", "blurb")
    rows = []
    for e in experiments:
        rows.append((
            e.exp_id, e.name, e.branch, e.stage, e.status,
            ",".join(e.depends_on) or "-",
            e.blurb,
        ))
    widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]

    def _fmt(row):
        return "  ".join(str(c).ljust(w) for c, w in zip(row, widths))

    print(_fmt(header))
    print(_fmt(tuple("-" * w for w in widths)))
    for r in rows:
        print(_fmt(r))


def _filter_experiments(
    *,
    stage: Optional[str],
    branch: Optional[str],
    only: Optional[set],
    exp: Optional[str],
) -> list[ExperimentSpec]:
    out = list(ALL_EXPERIMENTS)
    if exp is not None:
        if exp not in EXP_BY_NAME:
            raise SystemExit(f"unknown --exp: {exp!r} (known: {sorted(EXP_BY_NAME)})")
        return [EXP_BY_NAME[exp]]
    if stage and stage != "all":
        out = [e for e in out if e.stage == stage]
    if branch and branch != "all":
        out = [e for e in out if e.branch in (branch, "shared")]
    if only:
        wanted = set(only)
        unknown = wanted - {e.name for e in out}
        if unknown:
            raise SystemExit(f"unknown experiment name(s): {sorted(unknown)}")
        out = [e for e in out if e.name in wanted]
    return out


def _topological_order(experiments: list[ExperimentSpec]) -> list[ExperimentSpec]:
    """Sort experiments so dependencies come before dependents.

    Respects ``depends_on`` (exp_id references). Stable: input order preserved
    where the dependency graph allows. Cycles raise SystemExit.
    """
    by_id = {e.exp_id: e for e in experiments}
    seen: set[str] = set()
    out: list[ExperimentSpec] = []

    def visit(exp_id: str, stack: tuple = ()):
        if exp_id in seen:
            return
        if exp_id in stack:
            raise SystemExit(f"cyclic dependency involving {exp_id}")
        e = by_id.get(exp_id)
        if e is None:
            return  # depends on an experiment outside the filter set
        for dep in e.depends_on:
            visit(dep, stack + (exp_id,))
        seen.add(exp_id)
        out.append(e)

    for e in experiments:
        visit(e.exp_id)
    return out


def _orchestrate(
    experiments: list[ExperimentSpec],
    ctx: ExperimentContext,
    *,
    dry_run: bool,
) -> int:
    print(f"=== running {len(experiments)} experiment(s) ===")
    print(f"  output_dir:    {ctx.output_dir}")
    print(f"  split:         {ctx.split}")
    print(f"  sim_threshold: {ctx.sim_threshold}")
    print(f"  dry_run:       {dry_run}")
    print()

    failed = 0
    for i, exp in enumerate(experiments, 1):
        print(f"--- [{i}/{len(experiments)}] {exp.exp_id} {exp.name} ({exp.branch}, {exp.stage}) ---")
        print(f"    blurb: {exp.blurb}")
        if exp.depends_on:
            print(f"    depends_on: {exp.depends_on}")
        if dry_run:
            print("    (dry-run: not executing)")
            print()
            continue
        if exp.status == "stub" or exp.status == "manual":
            print(f"    status={exp.status} — stub will print recipe only.")
        t0 = time.perf_counter()
        try:
            result = exp.run(ctx) if exp.run else _stub_result(exp.exp_id, "  (no run callable)")
        except Exception as ex:
            failed += 1
            elapsed = time.perf_counter() - t0
            print(f"    FAILED in {elapsed:.1f}s: {ex}")
            logging.exception("experiment %s failed", exp.exp_id)
            print()
            continue
        elapsed = time.perf_counter() - t0
        print(f"    elapsed: {elapsed:.1f}s")
        if result.csv_path:
            print(f"    csv:     {result.csv_path}")
        if result.winner:
            wstr = ", ".join(f"{k}={v}" for k, v in list(result.winner.items())[:6])
            print(f"    winner:  {wstr}")
        for line in result.notes.splitlines():
            print(f"    {line}")
        ctx.state.update(result.state_updates)
        print()

    print("=== summary ===")
    print(f"  state keys after run: {sorted(ctx.state.keys())}")
    if "FINAL_MAP_CONFIG" in ctx.state:
        print(f"  FINAL_MAP_CONFIG: {json.dumps(ctx.state['FINAL_MAP_CONFIG'], indent=2)}")
    return 0 if failed == 0 else 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--stage", default=None, choices=STAGE_ORDER,
                   help="Which group to run (calibration, validation, all). "
                        "Omit to print the experiment menu.")
    p.add_argument("--branch", default=None, choices=("gemini", "openai", "all"),
                   help="Filter to one embedder branch (or 'all'). "
                        "'shared' experiments (EXP 7, F, A–E) are always included.")
    p.add_argument("--exp", default=None,
                   help="Run a single experiment by name (e.g. exp_1_gemini_scorer). "
                        "Overrides --stage / --branch.")
    p.add_argument("--only", nargs="+", default=None,
                   help="Intersect with --stage / --branch: restrict to these names.")
    p.add_argument("--list-variants", "--dry-run", dest="list_variants",
                   action="store_true",
                   help="Print the resolved experiment list and exit.")
    p.add_argument("--run", action="store_true",
                   help="Actually execute experiments. Default is dry-run (recipe only). "
                        "Required because most calibration EXPs touch caches and write CSVs.")
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_REPORTS_DIR,
                   help="Where to write per-experiment CSVs.")
    p.add_argument("--split", default="dev", choices=("dev", "test", "all"),
                   help="silver_findings split for sweeps (EXP F overrides to 'test').")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dev-fraction", type=float, default=0.8)
    p.add_argument("--sim-threshold", type=float, default=_DEFAULT_SIM_THRESHOLD,
                   help="Matcher sim_threshold for silver matching (EXP D varies it).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.stage is None and args.exp is None:
        _print_stage_menu()
        return 0

    experiments = _filter_experiments(
        stage=args.stage, branch=args.branch,
        only=set(args.only) if args.only else None,
        exp=args.exp,
    )
    if not experiments:
        print("error: no experiments matched the filters", file=sys.stderr)
        return 2
    experiments = _topological_order(experiments)

    if args.list_variants:
        _print_variants_table(experiments)
        return 0

    ctx = ExperimentContext(
        output_dir=args.output_dir,
        embedder_hint=args.branch if args.branch in ("gemini", "openai") else None,
        split=args.split,
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        sim_threshold=args.sim_threshold,
        state={},
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return _orchestrate(experiments, ctx, dry_run=not args.run)


if __name__ == "__main__":
    raise SystemExit(main())
