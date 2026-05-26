#!/usr/bin/env python3
"""
run_summarization_experiments.py — phase-based orchestrator for the MAP-stage
experiments described in ``docs/FINAL_SUMMARIZATION_EXPERIMENTS.md``.

Composes the per-stage MAP sweep harness in
``eval/silver/run_summarization_sweeps.py`` into a phased recipe::

    Phase 1A  gemini_branch    EXP 1 → 2 → 3   → FINAL_GEMINI_MAP_CONFIG
    Phase 1B  openai_branch    EXP 4 → 5 → 6   → FINAL_OPENAI_MAP_CONFIG
    Phase 2   compare          EXP 7           → PROVISIONAL_FINAL_MAP_CONFIG
    Phase 3   confirm          EXP F           → test-split confirmation CSV
    Phase 4   validation       EXP A–E         (bootstrap CIs, ABC ablation, ...)

State (BEST_*, FINAL_*, PROVISIONAL_*) is persisted between invocations to
``eval/reports/summarization_experiment_state.json`` so phases can be run in
separate terminal sessions without losing prior picks. Every invocation also
writes a run manifest under ``output_dir/manifests/`` capturing the phase /
experiments run / output paths / git commit / final state keys.

Test-split protection: EXP F always uses ``split=test``. All other experiments
default to ``split=dev`` and refuse ``--split test`` unless
``--allow-test-tuning`` is passed — protects against accidental held-out tuning.

The orchestrator reuses existing per-stage code (``run_sweep`` from
``eval/silver/map_theta_sweep.py``; ``_load_map_context`` from
``run_summarization_sweeps.py``); it does not duplicate sweep mechanics.

Usage::

    python scripts/eval/run_summarization_experiments.py                              # menu
    python scripts/eval/run_summarization_experiments.py --list-variants              # full table
    python scripts/eval/run_summarization_experiments.py --phase gemini_branch        # dry-run
    python scripts/eval/run_summarization_experiments.py --phase gemini_branch --run
    python scripts/eval/run_summarization_experiments.py --phase openai_branch --run
    python scripts/eval/run_summarization_experiments.py --phase compare --run
    python scripts/eval/run_summarization_experiments.py --phase confirm --run
    python scripts/eval/run_summarization_experiments.py --phase validation --run
    python scripts/eval/run_summarization_experiments.py --phase all --run
    python scripts/eval/run_summarization_experiments.py --exp exp_1_gemini_scorer --run
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
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
_DEFAULT_STATE_PATH = _DEFAULT_REPORTS_DIR / "summarization_experiment_state.json"
_DEFAULT_SIM_THRESHOLD = 0.55
_STATE_SCHEMA_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
# Phase / experiment registry
# ─────────────────────────────────────────────────────────────────────────────

PHASE_ORDER = (
    "gemini_branch",   # Phase 1A — EXP 1 → 2 → 3
    "openai_branch",   # Phase 1B — EXP 4 → 5 → 6
    "branches",        # Phase 1  — both branches
    "compare",         # Phase 2  — EXP 7
    "confirm",         # Phase 3  — EXP F
    "validation",      # Phase 4  — EXP A–E
    "all",             # branches + compare + confirm + validation
)

PHASE_BLURBS: dict[str, str] = {
    "gemini_branch": "Phase 1A — Gemini calibration: EXP 1 → 2 → 3, pins FINAL_GEMINI_MAP_CONFIG.",
    "openai_branch": "Phase 1B — OpenAI calibration: EXP 4 → 5 → 6, pins FINAL_OPENAI_MAP_CONFIG.",
    "branches":      "Phase 1  — runs both gemini_branch and openai_branch in sequence.",
    "compare":       "Phase 2  — EXP 7 embedder comparison, pins PROVISIONAL_FINAL_MAP_CONFIG.",
    "confirm":       "Phase 3  — EXP F held-out test confirmation on PROVISIONAL_FINAL_MAP_CONFIG.",
    "validation":    "Phase 4  — EXP A–E validation / explanation experiments (mostly stubs today).",
    "all":           "Every phase in dependency order: branches → compare → confirm → validation.",
}

# Phase → ordered list of exp_ids. Each entry is the exp_id (matches
# ExperimentSpec.exp_id). 'all' is composed at filter time.
PHASE_TO_EXPS: dict[str, tuple[str, ...]] = {
    "gemini_branch": ("EXP_1", "EXP_2", "EXP_3"),
    "openai_branch": ("EXP_4", "EXP_5", "EXP_6"),
    "branches":      ("EXP_1", "EXP_2", "EXP_3", "EXP_4", "EXP_5", "EXP_6"),
    "compare":       ("EXP_7",),
    "confirm":       ("EXP_F",),
    "validation":    ("EXP_A", "EXP_B.1", "EXP_B.2", "EXP_C", "EXP_D", "EXP_E"),
    "all":           (
        # branches
        "EXP_1", "EXP_2", "EXP_3", "EXP_4", "EXP_5", "EXP_6",
        # compare
        "EXP_7",
        # confirm
        "EXP_F",
        # validation
        "EXP_A", "EXP_B.1", "EXP_B.2", "EXP_C", "EXP_D", "EXP_E",
    ),
}


@dataclass
class ExperimentContext:
    """Per-run state passed to each experiment's ``run`` callable."""
    output_dir: Path
    split: str = "dev"
    seed: int = 42
    dev_fraction: float = 0.8
    sim_threshold: float = _DEFAULT_SIM_THRESHOLD
    state: dict = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Result returned by an experiment's ``run`` callable.

    ``state_updates`` are merged into ``ExperimentContext.state`` AND persisted
    to the state file so downstream EXPs (possibly in a later invocation) see
    the freshly-picked BEST_* / FINAL_* / PROVISIONAL_* values.
    """
    csv_path: Optional[Path]
    winner: Optional[dict]
    notes: str
    state_updates: dict = field(default_factory=dict)
    status: str = "ok"            # "ok" | "no_rows" | "skipped" | "failed"


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
    branch: str                       # "gemini" | "openai" | "shared"
    depends_on: list[str] = field(default_factory=list)
    status: str = "executable"
    run: Callable[[ExperimentContext], ExperimentResult] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# State persistence (durable across invocations)
# ─────────────────────────────────────────────────────────────────────────────

def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT),
            capture_output=True, text=True, timeout=2, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None


def _load_state(path: Path) -> tuple[dict, dict, list[dict]]:
    """Return ``(state, last_metadata, history)``. Empty if file missing."""
    if not path.exists():
        return {}, {}, []
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"state file {path} is corrupt ({exc}); fix or pass --reset-state"
        )
    if blob.get("schema_version") != _STATE_SCHEMA_VERSION:
        logging.warning(
            "state file schema_version=%r (expected %d); loading best-effort",
            blob.get("schema_version"), _STATE_SCHEMA_VERSION,
        )
    return (
        blob.get("state", {}),
        blob.get("last_invocation", {}),
        blob.get("history", []),
    )


def _save_state(
    path: Path, state: dict, metadata: dict, history: list[dict],
) -> None:
    """Atomic write of state + metadata + history."""
    payload = {
        "schema_version": _STATE_SCHEMA_VERSION,
        "last_updated":   datetime.now(tz=timezone.utc).isoformat(),
        "last_invocation": metadata,
        "state":          state,
        "history":        history,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


# ─────────────────────────────────────────────────────────────────────────────
# CSV writer (header-only when empty + sibling status.json)
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv_rows(path: Path, rows: list[dict]) -> str:
    """Write rows to CSV. If ``rows`` is empty, write only a status.json marker.

    Returns ``"ok"`` if rows were written, ``"no_rows"`` if a status.json was
    written instead. Keeps downstream consumers honest — they fail loudly when
    a CSV is missing rather than parsing a "(no rows)" sentinel.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        status_path = path.with_suffix(".status.json")
        status_path.write_text(json.dumps({
            "status": "no_rows",
            "csv_path": str(path),
            "written_at": datetime.now(tz=timezone.utc).isoformat(),
            "note": "Experiment produced zero rows; see orchestrator logs for cause.",
        }, indent=2), encoding="utf-8")
        return "no_rows"

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
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return "ok"


# ─────────────────────────────────────────────────────────────────────────────
# Sweep helpers (shared across EXPs 1-6, F)
# ─────────────────────────────────────────────────────────────────────────────

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
    """The 7 scorer ScorerSpecs for EXP 1 / EXP 4 (1 embedding + 6 hybrid blends)."""
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

    base_map = {
        "tau":                  base_cfg.tau,
        "count_alpha":          base_cfg.count_alpha,
        "reuse_weight":         base_cfg.reuse_weight,
        "contradiction_weight": base_cfg.contradiction_weight,
    }
    specs = [ScorerSpec("baseline", base_cfg.scorer_kind, base_cfg)]
    grids = {
        "tau":                  TAU_GRID,
        "count_alpha":          COUNT_ALPHA_GRID,
        "reuse_weight":         REUSE_WEIGHT_GRID,
        "contradiction_weight": CONTRADICTION_WEIGHT_GRID,
    }
    for knob, grid in grids.items():
        for v in grid:
            if v == base_map[knob]:
                continue
            kwargs = dict(base_map)
            kwargs["scorer_kind"] = base_cfg.scorer_kind
            kwargs["hybrid"] = base_cfg.hybrid
            kwargs[knob] = v
            specs.append(ScorerSpec(
                f"{base_cfg.scorer_kind}_{knob}_{v}", base_cfg.scorer_kind,
                AgreementConfig(**kwargs),
            ))
    return specs


# ─────────────────────────────────────────────────────────────────────────────
# Calibration EXPs (executable)
# ─────────────────────────────────────────────────────────────────────────────

def _run_branch_scorer_comparison(
    ctx: ExperimentContext, embedder: str, exp_label: str,
) -> ExperimentResult:
    """Shared body for EXP 1 (Gemini) and EXP 4 (OpenAI)."""
    from eval.silver.map_theta_sweep import run_sweep, THETA_GRID, REJECT_THETA_GRID
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
        single_voter_policies=("keep",),
        force_escalate_on_polarity_conflict_grid=(True,),
    )
    per_scorer = _per_scorer_best(rows, metric="strict_f1")
    winner = max(per_scorer, key=lambda r: r["strict_f1"]) if per_scorer else None

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    full_csv = ctx.output_dir / f"{exp_label}_{embedder}_scorer_full_{ts}.csv"
    best_csv = ctx.output_dir / f"{exp_label}_{embedder}_scorer_best_per_{ts}.csv"
    _write_csv_rows(full_csv, rows)
    status = _write_csv_rows(best_csv, per_scorer)

    notes = (
        f"{len(rows)} cells across 7 scorers × {len(THETA_GRID)*len(REJECT_THETA_GRID)} "
        f"(θ, rej) (reject<θ filter); per-scorer best-of in {best_csv.name}."
    )
    if winner is None:
        return ExperimentResult(csv_path=full_csv, winner=None, notes=notes,
                                 status="no_rows")
    prefix = embedder.upper()
    state_updates = {
        f"BEST_{prefix}_SCORER":         winner["scorer"],
        f"BEST_{prefix}_THETA":          winner["theta"],
        f"BEST_{prefix}_REJECT_THETA":   winner["reject_theta"],
        f"BEST_{prefix}_HYBRID_WEIGHTS": (
            (winner.get("w_category"), winner.get("w_embedding"),
             winner.get("w_entity"), winner.get("w_evidence"))
            if str(winner["scorer"]).startswith("hybrid") else None
        ),
        f"BEST_{prefix}_STRICT_F1":      winner["strict_f1"],
    }
    notes += (
        f"\n  Winner: {winner['scorer']} at θ={winner['theta']} rej={winner['reject_theta']}  "
        f"strict_f1={winner['strict_f1']:.3f}  f1={winner['f1']:.3f}  "
        f"esc={winner['escalate_rate']:.2f}"
    )
    return ExperimentResult(
        csv_path=best_csv, winner=winner, notes=notes,
        state_updates=state_updates, status=status,
    )


def _run_branch_agreement_weights(
    ctx: ExperimentContext, embedder: str, exp_label: str,
) -> ExperimentResult:
    """Shared body for EXP 2 (Gemini) / EXP 5 (OpenAI)."""
    prefix = embedder.upper()
    s = ctx.state
    required = (f"BEST_{prefix}_SCORER", f"BEST_{prefix}_THETA",
                f"BEST_{prefix}_REJECT_THETA")
    missing = [k for k in required if k not in s]
    if missing:
        raise SystemExit(
            f"{exp_label} requires state keys {missing} from EXP 1/{prefix.lower()}. "
            f"Run the {embedder}_branch phase first, or pass --include-deps."
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
    csv_path = ctx.output_dir / f"{exp_label}_{embedder}_agreement_weights_{ts}.csv"
    status = _write_csv_rows(csv_path, rows)
    if winner is None:
        return ExperimentResult(csv_path=csv_path, winner=None,
                                 notes="No rows produced.", status="no_rows")
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
        csv_path=csv_path, winner=winner, notes=notes,
        state_updates=state_updates, status=status,
    )


def _run_branch_polarity_flag(
    ctx: ExperimentContext, embedder: str, exp_label: str,
) -> ExperimentResult:
    """Shared body for EXP 3 (Gemini) / EXP 6 (OpenAI)."""
    prefix = embedder.upper()
    s = ctx.state
    required = (f"BEST_{prefix}_SCORER", f"BEST_{prefix}_THETA",
                f"BEST_{prefix}_REJECT_THETA")
    missing = [k for k in required if k not in s]
    if missing:
        raise SystemExit(
            f"{exp_label} requires state keys {missing} from EXP 1-2/{prefix.lower()}. "
            f"Run the {embedder}_branch phase first, or pass --include-deps."
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
    csv_path = ctx.output_dir / f"{exp_label}_{embedder}_polarity_flag_{ts}.csv"
    status = _write_csv_rows(csv_path, rows)
    if winner is None:
        return ExperimentResult(csv_path=csv_path, winner=None,
                                 notes="No rows produced.", status="no_rows")
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
        csv_path=csv_path, winner=winner, notes=notes,
        state_updates=state_updates, status=status,
    )


def _run_exp_1(ctx): return _run_branch_scorer_comparison(ctx, "gemini", "exp_1")
def _run_exp_4(ctx): return _run_branch_scorer_comparison(ctx, "openai", "exp_4")
def _run_exp_2(ctx): return _run_branch_agreement_weights(ctx, "gemini", "exp_2")
def _run_exp_5(ctx): return _run_branch_agreement_weights(ctx, "openai", "exp_5")
def _run_exp_3(ctx): return _run_branch_polarity_flag(ctx, "gemini", "exp_3")
def _run_exp_6(ctx): return _run_branch_polarity_flag(ctx, "openai", "exp_6")


def _run_exp_7(ctx: ExperimentContext) -> ExperimentResult:
    """EXP 7 — embedder branch comparison.

    Selects ``PROVISIONAL_FINAL_MAP_CONFIG`` (not ``FINAL_MAP_CONFIG``) because
    final-final selection requires bootstrap CIs / tie logic (EXP A), which
    isn't implemented yet. Hard-fails if either FINAL_*_MAP_CONFIG is missing.
    """
    s = ctx.state
    g = s.get("FINAL_GEMINI_MAP_CONFIG")
    o = s.get("FINAL_OPENAI_MAP_CONFIG")
    missing = []
    if g is None: missing.append("FINAL_GEMINI_MAP_CONFIG")
    if o is None: missing.append("FINAL_OPENAI_MAP_CONFIG")
    if missing:
        raise SystemExit(
            f"EXP 7 (compare) requires state keys {missing}. Run gemini_branch + "
            f"openai_branch phases first (or pass --include-deps)."
        )

    def _row(embedder: str, cfg: dict) -> dict:
        return {
            "embedder":          embedder,
            "scorer":            cfg["scorer"],
            "theta":             cfg["theta"],
            "reject_theta":      cfg["reject_theta"],
            "polarity_flag":     cfg["polarity_flag"],
            "strict_f1":         cfg["strict_f1"],
            "hybrid_weights":    json.dumps(cfg["hybrid_weights"]),
            "agreement_weights": json.dumps(cfg["agreement_weights"]),
        }
    rows = [_row("gemini", g), _row("openai", o)]

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    csv_path = ctx.output_dir / f"exp_7_embedder_comparison_{ts}.csv"
    status = _write_csv_rows(csv_path, rows)

    winner = max(rows, key=lambda r: r["strict_f1"])
    notes = (
        f"Gemini strict_f1={g['strict_f1']:.3f} vs OpenAI strict_f1={o['strict_f1']:.3f}.\n"
        f"  Provisional winner: {winner['embedder']} "
        f"(Δstrict_f1={abs(g['strict_f1']-o['strict_f1']):.3f}).\n"
        f"  NOTE: bootstrap CI (EXP A) is required to promote to FINAL_MAP_CONFIG. "
        f"If CIs overlap → pick cheaper/faster embedder."
    )
    return ExperimentResult(
        csv_path=csv_path, winner=winner, notes=notes, status=status,
        state_updates={
            "PROVISIONAL_FINAL_EMBEDDER":   winner["embedder"],
            "PROVISIONAL_FINAL_MAP_CONFIG": winner,
        },
    )


def _run_exp_f(ctx: ExperimentContext) -> ExperimentResult:
    """EXP F — held-out test confirmation of PROVISIONAL_FINAL_MAP_CONFIG.

    Always uses ``split=test`` regardless of the orchestrator's --split arg.
    Does not promote provisional → final; just records the test-split metric
    for thesis citation. Hard-fails if PROVISIONAL_FINAL_MAP_CONFIG is missing.
    """
    final = ctx.state.get("PROVISIONAL_FINAL_MAP_CONFIG")
    if final is None:
        raise SystemExit(
            "EXP F (confirm) requires PROVISIONAL_FINAL_MAP_CONFIG. "
            "Run the 'compare' phase first (or pass --include-deps)."
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
    status = _write_csv_rows(csv_path, rows)
    if not rows:
        return ExperimentResult(csv_path=csv_path, winner=None,
                                 notes="No test rows produced.", status="no_rows")
    r = rows[0]
    dev_strict = cfg.get("strict_f1")
    delta = (r["strict_f1"] - dev_strict) if dev_strict is not None else None
    notes = (
        f"Test-split strict_f1={r['strict_f1']:.3f} (dev was {dev_strict:.3f}); "
        f"Δ={delta:+.3f}." if delta is not None else
        f"Test-split strict_f1={r['strict_f1']:.3f}."
    )
    notes += "\n  If Δ << 0 the dev-tuned picks may be overfit to the silver dev split."
    return ExperimentResult(
        csv_path=csv_path, winner=r, notes=notes, status=status,
        state_updates={
            "TEST_CONFIRMATION_STRICT_F1":   r["strict_f1"],
            "TEST_CONFIRMATION_DEV_DELTA":   delta,
            "TEST_CONFIRMATION_EMBEDDER":    embedder,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation EXPs (stubs)
# ─────────────────────────────────────────────────────────────────────────────

def _stub_result(exp_id: str, recipe: str) -> ExperimentResult:
    return ExperimentResult(
        csv_path=None, winner=None, status="skipped",
        notes=(
            f"{exp_id} not implemented in this orchestrator yet. Recipe:\n"
            f"{recipe}\n"
            f"  See docs/FINAL_SUMMARIZATION_EXPERIMENTS.md §{exp_id} for full spec."
        ),
    )


def _run_exp_a(ctx):
    return _stub_result("EXP A — bootstrap CI", recipe=(
        "  For each sweep CSV produced by EXP 1–7 + F:\n"
        "    1. Identify the winner row + 3 nearest competitors (top-4 by strict_f1).\n"
        "    2. For n=50 random resamples of case_ids (seeded), recompute strict_f1\n"
        "       from the per-case match outputs for each of the 4 candidates.\n"
        "    3. Emit (lo, hi) 95% bootstrap CI per candidate + tied-candidate list.\n"
        "  Implementation needs the per-case match outputs to be retained from\n"
        "  run_sweep — today they're discarded. Add `--keep-per-case` first.\n"
        "  Once EXP A runs, PROVISIONAL_FINAL_MAP_CONFIG can be promoted to\n"
        "  FINAL_MAP_CONFIG (tie-broken on cost/speed if CIs overlap)."
    ))


def _run_exp_b1(ctx):
    return _stub_result("EXP B.1 — matched random escalation", recipe=(
        "  Replay PROVISIONAL_FINAL_MAP_CONFIG cascade twice:\n"
        "    A. Real ABC (current _replay path) → escalate_rate=E\n"
        "    B. Matched random: route ⌊E·N_chunks⌋ chunks to L3 uniformly\n"
        "       (n=20 seeds). Compare strict_f1 mean ± 95% CI across seeds.\n"
        "  Implementation: subclass _replay with `routing_mode={abc,random}` + seed."
    ))


def _run_exp_b2(ctx):
    return _stub_result("EXP B.2 — cost-quality comparison", recipe=(
        "  Compare three baselines on PROVISIONAL_FINAL_MAP_CONFIG papers:\n"
        "    1. cheap-only: L1 voter best, never escalate.\n"
        "    2. ABC cascade (current).\n"
        "    3. strong-only: L3 (Sonnet) on every chunk.\n"
        "  CAVEAT: voter_cache.json has L3 output only for chunks ABC escalated;\n"
        "  baseline (3) needs a separate all-Sonnet primer batch (~$30)."
    ))


def _run_exp_c(ctx):
    return _stub_result("EXP C — agreement→accuracy calibration", recipe=(
        "  For every chunk in the PROVISIONAL_FINAL_MAP_CONFIG replay:\n"
        "    1. Record agreement score (bundle.confidence).\n"
        "    2. Record per-chunk silver-match flag.\n"
        "    3. Bin chunks by agreement: [0–0.5], (0.5–0.7], (0.7–0.85], (0.85–1.0].\n"
        "    4. Report per-bin: count, strict_f1, precision, recall.\n"
        "  Implementation: extend _replay to retain per-chunk agreement scores."
    ))


def _run_exp_d(ctx):
    return _stub_result("EXP D — matcher-threshold sensitivity", recipe=(
        "  Re-run EXP 1 at sim_threshold ∈ {0.50, 0.55, 0.60}.\n"
        "  For each threshold, recompute per-scorer best and re-rank candidates.\n"
        "  Report Kendall τ of candidate rankings between threshold pairs."
    ))


def _run_exp_e(ctx):
    return _stub_result("EXP E — recall-gap audit", recipe=(
        "  Manual labelling. For ~50 sampled false negatives:\n"
        "    1. Auto-extract case_id, silver_finding, closest_pipeline_finding,\n"
        "       closest_similarity, all_pipeline_findings.\n"
        "    2. Human reviewer fills `miss_category` per 10-category schema.\n"
        "    3. Aggregate histogram + final report.\n"
        "  ~1-2 hours reviewer time per audit pass."
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Experiment registry
# ─────────────────────────────────────────────────────────────────────────────

ALL_EXPERIMENTS: list[ExperimentSpec] = [
    ExperimentSpec(
        name="exp_1_gemini_scorer", exp_id="EXP_1", branch="gemini",
        depends_on=[], run=_run_exp_1,
        blurb="Tuned scorer comparison (Gemini): 7 scorers × full θ/reject grid; per-scorer best-of.",
    ),
    ExperimentSpec(
        name="exp_2_gemini_weights", exp_id="EXP_2", branch="gemini",
        depends_on=["EXP_1"], run=_run_exp_2,
        blurb="Soft-alignment weight sweep (Gemini) around EXP 1 winner.",
    ),
    ExperimentSpec(
        name="exp_3_gemini_polarity", exp_id="EXP_3", branch="gemini",
        depends_on=["EXP_2"], run=_run_exp_3,
        blurb="Polarity-flag ablation (Gemini) at EXP 1+2 config.",
    ),
    ExperimentSpec(
        name="exp_4_openai_scorer", exp_id="EXP_4", branch="openai",
        depends_on=[], run=_run_exp_4,
        blurb="Tuned scorer comparison (OpenAI): 7 scorers × full θ/reject grid; per-scorer best-of.",
    ),
    ExperimentSpec(
        name="exp_5_openai_weights", exp_id="EXP_5", branch="openai",
        depends_on=["EXP_4"], run=_run_exp_5,
        blurb="Soft-alignment weight sweep (OpenAI) around EXP 4 winner.",
    ),
    ExperimentSpec(
        name="exp_6_openai_polarity", exp_id="EXP_6", branch="openai",
        depends_on=["EXP_5"], run=_run_exp_6,
        blurb="Polarity-flag ablation (OpenAI) at EXP 4+5 config.",
    ),
    ExperimentSpec(
        name="exp_7_embedder_comparison", exp_id="EXP_7", branch="shared",
        depends_on=["EXP_3", "EXP_6"], run=_run_exp_7,
        blurb="Final embedder comparison → PROVISIONAL_FINAL_MAP_CONFIG (provisional until EXP A bootstrap CI).",
    ),
    ExperimentSpec(
        name="exp_f_test_confirmation", exp_id="EXP_F", branch="shared",
        depends_on=["EXP_7"], run=_run_exp_f,
        blurb="Held-out test pass on PROVISIONAL_FINAL_MAP_CONFIG (always uses --split test).",
    ),
    ExperimentSpec(
        name="exp_a_bootstrap_ci", exp_id="EXP_A", branch="shared",
        depends_on=["EXP_1", "EXP_4", "EXP_7"],
        status="stub", run=_run_exp_a,
        blurb="Bootstrap 95% CIs on the winner + 3 nearest competitors for every sweep CSV.",
    ),
    ExperimentSpec(
        name="exp_b1_routing_usefulness", exp_id="EXP_B.1", branch="shared",
        depends_on=["EXP_7"], status="stub", run=_run_exp_b1,
        blurb="ABC vs matched-random escalation: does agreement beat chance?",
    ),
    ExperimentSpec(
        name="exp_b2_cost_quality", exp_id="EXP_B.2", branch="shared",
        depends_on=["EXP_7"], status="stub", run=_run_exp_b2,
        blurb="Cost-quality: cascade vs cheap-only vs strong-model-only. Needs separate all-Sonnet primer.",
    ),
    ExperimentSpec(
        name="exp_c_agreement_accuracy", exp_id="EXP_C", branch="shared",
        depends_on=["EXP_7"], status="stub", run=_run_exp_c,
        blurb="Agreement-score → accuracy calibration curve (validates ABC's premise).",
    ),
    ExperimentSpec(
        name="exp_d_matcher_sensitivity", exp_id="EXP_D", branch="shared",
        depends_on=["EXP_1"], status="stub", run=_run_exp_d,
        blurb="Sensitivity of EXP 1 ranking to matcher sim_threshold ∈ {0.50, 0.55, 0.60}.",
    ),
    ExperimentSpec(
        name="exp_e_recall_gap_audit", exp_id="EXP_E", branch="shared",
        depends_on=["EXP_7"], status="manual", run=_run_exp_e,
        blurb="Manual recall-gap audit: 50 false negatives × 10-category classification.",
    ),
]


EXP_BY_NAME = {e.name: e for e in ALL_EXPERIMENTS}
EXP_BY_ID = {e.exp_id: e for e in ALL_EXPERIMENTS}


# ─────────────────────────────────────────────────────────────────────────────
# Listing / menu
# ─────────────────────────────────────────────────────────────────────────────

def _print_phase_menu() -> None:
    print("Summarisation experiment orchestrator — phase-based execution.\n")
    print("Available --phase values:\n")
    for ph in PHASE_ORDER:
        exps = PHASE_TO_EXPS.get(ph, ())
        print(f"  {ph:<14} ({len(exps)} exp{'s' if len(exps) != 1 else ''})")
        print(f"      {PHASE_BLURBS.get(ph, '')}")
    print()
    print("Available experiments (in dependency order):\n")
    for e in ALL_EXPERIMENTS:
        tag = ("            " if e.status == "executable"
               else "  [stub]   " if e.status == "stub"
               else "  [manual] ")
        deps = f"  ← {', '.join(e.depends_on)}" if e.depends_on else ""
        print(f"  {tag}{e.exp_id:<8} {e.name:<30} ({e.branch}){deps}")
        print(f"            {e.blurb}")
    print()
    print("Examples:")
    print("  python scripts/eval/run_summarization_experiments.py --list-variants")
    print("  python scripts/eval/run_summarization_experiments.py --phase gemini_branch --run")
    print("  python scripts/eval/run_summarization_experiments.py --phase compare --run")
    print("  python scripts/eval/run_summarization_experiments.py --phase confirm --run")
    print("  python scripts/eval/run_summarization_experiments.py --phase all --run")
    print()
    print("NOTE: default is dry-run. Pass --run to actually execute.")
    print("State persists across invocations at:")
    print(f"  {_DEFAULT_STATE_PATH}")


def _print_variants_table(experiments: list[ExperimentSpec]) -> None:
    header = ("exp_id", "name", "branch", "status", "depends_on", "blurb")
    rows = []
    for e in experiments:
        rows.append((
            e.exp_id, e.name, e.branch, e.status,
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


# ─────────────────────────────────────────────────────────────────────────────
# Experiment selection (phase / exp / only filters + dependency handling)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_experiments(
    *,
    phase: Optional[str],
    exp: Optional[str],
    only: Optional[set],
    include_deps: bool,
) -> list[ExperimentSpec]:
    """Resolve the user's filters into a final ordered experiment list.

    Precedence:
      1. --exp wins (single experiment)
      2. --phase filters by phase membership
      3. otherwise all experiments
      4. --only intersects with the selection
      5. --include-deps recursively pulls in dependencies (else: hard-fail later
         in the orchestrator if state is missing)
    """
    if exp is not None:
        if exp not in EXP_BY_NAME:
            raise SystemExit(
                f"unknown --exp: {exp!r}\n"
                f"  known: {sorted(EXP_BY_NAME)}"
            )
        selected = [EXP_BY_NAME[exp]]
    elif phase is not None:
        if phase not in PHASE_TO_EXPS:
            raise SystemExit(
                f"unknown --phase: {phase!r}\n"
                f"  known: {list(PHASE_ORDER)}"
            )
        wanted_ids = PHASE_TO_EXPS[phase]
        selected = [EXP_BY_ID[i] for i in wanted_ids if i in EXP_BY_ID]
    else:
        selected = list(ALL_EXPERIMENTS)

    if only:
        wanted = set(only)
        all_in_selection = {e.name for e in selected}
        unknown = wanted - {e.name for e in ALL_EXPERIMENTS}
        if unknown:
            raise SystemExit(f"unknown experiment name(s): {sorted(unknown)}")
        missing_in_phase = wanted - all_in_selection
        if missing_in_phase:
            raise SystemExit(
                f"--only names {sorted(missing_in_phase)} not in selected --phase / --exp scope. "
                f"Drop --phase / --exp, or remove these from --only."
            )
        selected = [e for e in selected if e.name in wanted]

    if include_deps:
        selected = _expand_with_dependencies(selected)
    return _topological_order(selected)


def _expand_with_dependencies(experiments: list[ExperimentSpec]) -> list[ExperimentSpec]:
    seen = {e.exp_id for e in experiments}
    out = list(experiments)
    queue = list(experiments)
    while queue:
        e = queue.pop()
        for dep_id in e.depends_on:
            if dep_id in seen:
                continue
            dep = EXP_BY_ID.get(dep_id)
            if dep is None:
                continue
            seen.add(dep_id)
            out.append(dep)
            queue.append(dep)
    return out


def _topological_order(experiments: list[ExperimentSpec]) -> list[ExperimentSpec]:
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
            return  # dependency outside the filter set — orchestrator will hard-fail at run time
        for dep in e.depends_on:
            visit(dep, stack + (exp_id,))
        seen.add(exp_id)
        out.append(e)

    for e in experiments:
        visit(e.exp_id)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration: run + state persistence + manifest
# ─────────────────────────────────────────────────────────────────────────────

def _orchestrate(
    experiments: list[ExperimentSpec],
    ctx: ExperimentContext,
    *,
    dry_run: bool,
    state_path: Path,
    history: list[dict],
    invocation_meta: dict,
    manifest_path: Path,
    fail_fast: bool = False,
) -> int:
    """Run the resolved experiment list.

    ``fail_fast=True`` stops after the first experiment that raises ``SystemExit``
    (hard-fail on missing state) or any other exception (failed run). Stubs and
    manual experiments do **not** trigger fail-fast — they return normally with
    ``status="skipped"`` and never increment the failure counter. State and the
    per-invocation manifest are still written before the function returns,
    regardless of whether fail-fast tripped.
    """
    print(f"=== running {len(experiments)} experiment(s) ===")
    print(f"  output_dir:    {ctx.output_dir}")
    print(f"  state_path:    {state_path}")
    print(f"  split:         {ctx.split}")
    print(f"  sim_threshold: {ctx.sim_threshold}")
    print(f"  dry_run:       {dry_run}")
    print(f"  fail_fast:     {fail_fast}")
    print()

    manifest_runs: list[dict] = []
    failed = 0
    for i, exp in enumerate(experiments, 1):
        print(f"--- [{i}/{len(experiments)}] {exp.exp_id} {exp.name} ({exp.branch}) ---")
        print(f"    blurb: {exp.blurb}")
        if exp.depends_on:
            print(f"    depends_on: {exp.depends_on}")
        if dry_run:
            print("    (dry-run: not executing)")
            manifest_runs.append({
                "exp_id": exp.exp_id, "name": exp.name, "status": "dry_run",
                "csv_path": None, "elapsed_s": 0.0,
            })
            print()
            continue
        if exp.status in ("stub", "manual"):
            print(f"    status={exp.status} — stub will print recipe only.")
        t0 = time.perf_counter()
        try:
            result = exp.run(ctx) if exp.run else _stub_result(
                exp.exp_id, "  (no run callable)",
            )
        except SystemExit as ex:
            failed += 1
            elapsed = time.perf_counter() - t0
            print(f"    HARD FAIL in {elapsed:.1f}s: {ex}")
            manifest_runs.append({
                "exp_id": exp.exp_id, "name": exp.name, "status": "hard_fail",
                "error": str(ex), "elapsed_s": elapsed, "csv_path": None,
            })
            print()
            if fail_fast:
                print(f"--- fail-fast: stopping after {exp.exp_id} hard-fail ---\n")
                break
            continue
        except Exception as ex:  # noqa: BLE001
            failed += 1
            elapsed = time.perf_counter() - t0
            print(f"    FAILED in {elapsed:.1f}s: {ex}")
            logging.exception("experiment %s failed", exp.exp_id)
            manifest_runs.append({
                "exp_id": exp.exp_id, "name": exp.name, "status": "failed",
                "error": str(ex), "elapsed_s": elapsed, "csv_path": None,
            })
            print()
            if fail_fast:
                print(f"--- fail-fast: stopping after {exp.exp_id} failure ---\n")
                break
            continue
        elapsed = time.perf_counter() - t0
        print(f"    elapsed: {elapsed:.1f}s   status: {result.status}")
        if result.csv_path:
            print(f"    csv:     {result.csv_path}")
        if result.winner:
            wstr = ", ".join(f"{k}={v}" for k, v in list(result.winner.items())[:6])
            print(f"    winner:  {wstr}")
        for line in result.notes.splitlines():
            print(f"    {line}")

        # Merge state updates and persist immediately after each EXP so a
        # later phase / a separate invocation sees the BEST_* picks.
        if result.state_updates:
            ctx.state.update(result.state_updates)
            history.append({
                "exp_id": exp.exp_id, "ran_at": datetime.now(tz=timezone.utc).isoformat(),
                "csv_path": str(result.csv_path) if result.csv_path else None,
                "winner_summary": {
                    k: v for k, v in (result.winner or {}).items()
                    if k in ("scorer", "theta", "reject_theta", "strict_f1",
                             "embedder", "polarity_flag",
                             "force_escalate_on_polarity_conflict")
                },
                "state_keys_updated": sorted(result.state_updates.keys()),
            })
            _save_state(state_path, ctx.state, invocation_meta, history)
            print(f"    state: updated keys {sorted(result.state_updates.keys())} → {state_path}")

        manifest_runs.append({
            "exp_id": exp.exp_id, "name": exp.name, "status": result.status,
            "elapsed_s": elapsed,
            "csv_path": str(result.csv_path) if result.csv_path else None,
            "state_keys_updated": sorted(result.state_updates.keys()),
        })
        print()

    # Write the per-invocation run manifest
    manifest = {
        **invocation_meta,
        "experiments_run": manifest_runs,
        "state_path":      str(state_path),
        "state_keys_after": sorted(ctx.state.keys()),
        "failed_count":    failed,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print("=== summary ===")
    print(f"  state keys after run: {sorted(ctx.state.keys())}")
    print(f"  state file:           {state_path}")
    print(f"  run manifest:         {manifest_path}")
    if "PROVISIONAL_FINAL_MAP_CONFIG" in ctx.state:
        print("  PROVISIONAL_FINAL_MAP_CONFIG:")
        for k, v in ctx.state["PROVISIONAL_FINAL_MAP_CONFIG"].items():
            print(f"    {k}: {v}")
    return 0 if failed == 0 else 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Selection axes
    p.add_argument("--phase", default=None, choices=PHASE_ORDER,
                   help="Which phase to run (omit to print the menu).")
    p.add_argument("--exp", default=None,
                   help="Run a single experiment by name (overrides --phase).")
    p.add_argument("--only", nargs="+", default=None,
                   help="Restrict the --phase / --exp selection to these names.")
    p.add_argument("--include-deps", action="store_true",
                   help="Auto-include transitive dependencies (default: fail loudly if "
                        "a required state key is missing).")
    # Listing / dry-run vs run
    p.add_argument("--list-variants", "--dry-run", dest="list_variants",
                   action="store_true",
                   help="Print the resolved experiment list and exit. With no filter, "
                        "lists every experiment.")
    p.add_argument("--run", action="store_true",
                   help="Actually execute experiments. Default is dry-run (recipe only).")
    p.add_argument("--fail-fast", action="store_true",
                   help="Stop after the first experiment failure or hard-fail. State and "
                        "manifest are still written. Stub/manual experiments don't trigger "
                        "fail-fast (they return status='skipped' without raising).")
    # State
    p.add_argument("--state-path", type=Path, default=_DEFAULT_STATE_PATH,
                   help=f"State JSON path. Default: {_DEFAULT_STATE_PATH}")
    p.add_argument("--reset-state", action="store_true",
                   help="Delete the state file at startup before loading.")
    # Output
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_REPORTS_DIR,
                   help="Where to write per-experiment CSVs + run manifests.")
    # Sweep params
    p.add_argument("--split", default="dev", choices=("dev", "test", "all"),
                   help="silver_findings split (EXP F always overrides to 'test').")
    p.add_argument("--allow-test-tuning", action="store_true",
                   help="Required to pass --split test for any experiment other than EXP F. "
                        "Default: refuse, to protect against accidental held-out tuning.")
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

    # No filter + no --list-variants → menu
    if args.phase is None and args.exp is None and not args.list_variants:
        _print_phase_menu()
        return 0

    # Test-split protection: refuse --split test unless explicitly authorised.
    # EXP F overrides to 'test' internally regardless of this.
    if args.split == "test" and not args.allow_test_tuning:
        # Allow when the user explicitly asks for ONLY EXP F.
        only_exp_f = args.exp == "exp_f_test_confirmation"
        if not only_exp_f:
            print(
                "error: --split test rejected. EXP 1-7 should not be tuned on the "
                "held-out test set. Pass --allow-test-tuning to override, or run "
                "EXP F (which always uses split=test internally).",
                file=sys.stderr,
            )
            return 2

    # State reset
    if args.reset_state and args.state_path.exists():
        args.state_path.unlink()
        print(f"reset-state: removed {args.state_path}")

    # Load state for both listing and running (so the table can hint at deps)
    state, _last_meta, history = _load_state(args.state_path)

    experiments = _resolve_experiments(
        phase=args.phase, exp=args.exp,
        only=set(args.only) if args.only else None,
        include_deps=args.include_deps,
    )
    if not experiments:
        print("error: no experiments matched the filters", file=sys.stderr)
        return 2

    if args.list_variants:
        _print_variants_table(experiments)
        print()
        print(f"state file: {args.state_path}  ({'exists' if args.state_path.exists() else 'absent'})")
        print(f"state keys currently set: {sorted(state.keys()) or '(none)'}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    invocation_ts = datetime.now(tz=timezone.utc)
    invocation_meta = {
        "timestamp_utc": invocation_ts.isoformat(),
        "script_path":   str(Path(__file__).relative_to(_REPO_ROOT)),
        "git_commit":    _git_commit(),
        "args": {
            "phase": args.phase, "exp": args.exp, "only": args.only,
            "include_deps": args.include_deps,
            "split": args.split, "seed": args.seed,
            "dev_fraction": args.dev_fraction,
            "sim_threshold": args.sim_threshold,
            "allow_test_tuning": args.allow_test_tuning,
        },
        "experiments_selected": [e.exp_id for e in experiments],
    }
    manifest_path = (
        args.output_dir / "manifests" /
        f"manifest_{invocation_ts.strftime('%Y%m%dT%H%M%S')}.json"
    )

    ctx = ExperimentContext(
        output_dir=args.output_dir,
        split=args.split,
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        sim_threshold=args.sim_threshold,
        state=state,
    )

    return _orchestrate(
        experiments, ctx,
        dry_run=not args.run,
        state_path=args.state_path,
        history=history,
        invocation_meta=invocation_meta,
        manifest_path=manifest_path,
        fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    raise SystemExit(main())
