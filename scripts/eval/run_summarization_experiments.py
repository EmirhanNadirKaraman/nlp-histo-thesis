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
    Phase 3b  baselines        EXP B.2         → cost-quality baseline CSV
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
    python scripts/eval/run_summarization_experiments.py --phase baselines --run
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

# Load .env so OPENAI_API_KEY / GOOGLE_API_KEY are available to every EXP.
# EXPs 1-7 worked transitively via ``run_summarization_sweeps.py:load_dotenv``;
# EXP B.2 does not import that module, so the orchestrator loads .env itself.
try:
    from dotenv import load_dotenv
    load_dotenv(str(_REPO_ROOT / ".env"))
except ImportError:  # pragma: no cover — dotenv is a hard dep elsewhere
    pass


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
    "baselines",       # Phase 3b — EXP B.2 (cost-quality replay baselines)
    "validation",      # Phase 4  — EXP A–E
    "all",             # branches + compare + confirm + baselines + validation
)

PHASE_BLURBS: dict[str, str] = {
    "gemini_branch": "Phase 1A — Gemini calibration: EXP 1 → 2 → 3, pins FINAL_GEMINI_MAP_CONFIG.",
    "openai_branch": "Phase 1B — OpenAI calibration: EXP 4 → 5 → 6, pins FINAL_OPENAI_MAP_CONFIG.",
    "branches":      "Phase 1  — runs both gemini_branch and openai_branch in sequence.",
    "compare":       "Phase 2  — EXP 7 embedder comparison, pins PROVISIONAL_FINAL_MAP_CONFIG.",
    "confirm":       "Phase 3  — EXP F held-out test confirmation on PROVISIONAL_FINAL_MAP_CONFIG.",
    "baselines":     "Phase 3b — EXP B.2 cost-quality replay: cascade vs L1-only vs L3-only (no state mutation).",
    "validation":    "Phase 4  — EXP A–E validation / explanation experiments (mostly stubs today).",
    "all":           "Every phase in dependency order: branches → compare → confirm → baselines → validation.",
}

# Phase → ordered list of exp_ids. Each entry is the exp_id (matches
# ExperimentSpec.exp_id). 'all' is composed at filter time.
PHASE_TO_EXPS: dict[str, tuple[str, ...]] = {
    "gemini_branch": ("EXP_1", "EXP_2", "EXP_3"),
    "openai_branch": ("EXP_4", "EXP_5", "EXP_6"),
    "branches":      ("EXP_1", "EXP_2", "EXP_3", "EXP_4", "EXP_5", "EXP_6"),
    "compare":       ("EXP_7",),
    "confirm":       ("EXP_F",),
    "baselines":     ("EXP_B.2",),
    "validation":    ("EXP_A", "EXP_B.1", "EXP_B.2", "EXP_C", "EXP_D", "EXP_E"),
    "all":           (
        # branches
        "EXP_1", "EXP_2", "EXP_3", "EXP_4", "EXP_5", "EXP_6",
        # compare
        "EXP_7",
        # confirm
        "EXP_F",
        # baselines (Phase 3b — non-tuning cost-quality replay)
        "EXP_B.2",
        # validation
        "EXP_A", "EXP_B.1", "EXP_C", "EXP_D", "EXP_E",
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
    from eval.silver.old_files.run_summarization_sweeps import HYBRID_BLEND_GRID
    from pipeline.stages.knowledge_extraction.config import AgreementConfig, HybridConfig

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
    from eval.silver.old_files.run_summarization_sweeps import (
        TAU_GRID, COUNT_ALPHA_GRID, REUSE_WEIGHT_GRID, CONTRADICTION_WEIGHT_GRID,
    )
    from pipeline.stages.knowledge_extraction.config import AgreementConfig

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
    from eval.silver.map_context import _load_map_context

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
    from eval.silver.map_context import _load_map_context
    from pipeline.stages.knowledge_extraction.config import AgreementConfig, HybridConfig

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
    from eval.silver.map_context import _load_map_context
    from pipeline.stages.knowledge_extraction.config import AgreementConfig, HybridConfig

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
    if g is None:
        missing.append("FINAL_GEMINI_MAP_CONFIG")
    if o is None:
        missing.append("FINAL_OPENAI_MAP_CONFIG")
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
    from eval.silver.map_context import _load_map_context
    from pipeline.stages.knowledge_extraction.config import AgreementConfig, HybridConfig

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
    """EXP B.2 — cost-quality replay baselines vs the calibrated cascade.

    Diagnostic-only. No API calls, no state mutation beyond the
    invocation history (the manifest/state file's ``history`` list).

    Builds replay-only baselines from the existing
    ``eval/data/map_primer/voter_cache.json``:

      1. One row per L1 voter (model-by-model strict_f1 floor).
      2. ``l1_best_oracle`` row — per-case oracle picking the L1 voter whose
         findings match silver best. Clearly labelled as oracle / not
         deployable (uses silver to decide which voter to keep).
      3. ``l3_only`` row — Sonnet replay using cached L3 outputs.
      4. ``cascade_provisional_final`` row — only emitted when
         ``PROVISIONAL_FINAL_MAP_CONFIG`` is present in state. Built by
         re-replaying the cascade at the pinned settings via
         ``map_theta_sweep._replay`` + the same matcher path. Skipped with a
         clear note otherwise.

    CSV columns: split, baseline, model_or_source, strict_f1, f1, precision,
    recall, n_matched, n_silver, n_pipeline, estimated_cost_per_chunk, notes.
    Hard-fails when ``voter_cache.json`` is missing or has zero L1/L3
    outputs (cannot construct any baseline). Returns no state updates.
    """
    cache_path = _B2_CACHE_PATH
    silver_path = _B2_SILVER_PATH

    if not cache_path.exists():
        raise SystemExit(
            f"EXP B.2 (baselines): voter cache not found: {cache_path}\n"
            "  Prime it with `python -m eval.silver.map_theta_sweep prime --split all` "
            "then `collect`."
        )
    if not silver_path.exists():
        raise SystemExit(
            f"EXP B.2 (baselines): silver labels not found: {silver_path}\n"
            "  Generate with `python -m eval.silver.generate --batch`."
        )

    from eval.silver.data.jsonl_utils import read_jsonl
    from eval.silver.data.schemas import SilverCaseResult
    from eval.silver.data.split import assign_split

    voter_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    silver_by_case = {rec.case_id: rec for rec in read_jsonl(silver_path, SilverCaseResult)}

    # Detect cascade shape (n_l1_voters, l3_population) from the cache.
    shape = _b2_cache_shape(voter_cache)
    if shape["n_l1_voters"] == 0:
        raise SystemExit(
            "EXP B.2 (baselines): voter cache has no L1 outputs; cannot build "
            "L1-only baseline. Re-prime with all voters."
        )
    if shape["l3_populated_chunks"] == 0:
        raise SystemExit(
            "EXP B.2 (baselines): voter cache has zero populated L3 outputs; "
            "cannot build L3-only (Sonnet) baseline. Re-prime including L3."
        )

    # Production voter model names (for the model_or_source column + cost).
    try:
        from pipeline.stages.knowledge_extraction.batch.voter_configs import (
            make_l1_voters, make_l2_voters, make_l3_voter,
        )
        l1_models = [v.model for v in make_l1_voters()]
        l2_models = [v.model for v in make_l2_voters()]
        l3_model = make_l3_voter().model
    except Exception:  # pragma: no cover — keep B.2 robust to import drift
        l1_models = [f"l1_voter_{i}" for i in range(shape["n_l1_voters"])]
        l2_models = [f"l2_voter_{i}" for i in range(shape["n_l2_voters"])]
        l3_model = "claude-sonnet-4-6"

    if len(l1_models) < shape["n_l1_voters"]:
        l1_models = l1_models + [
            f"l1_voter_{i}" for i in range(len(l1_models), shape["n_l1_voters"])
        ]
    if len(l2_models) < shape["n_l2_voters"]:
        l2_models = l2_models + [
            f"l2_voter_{i}" for i in range(len(l2_models), shape["n_l2_voters"])
        ]

    # Embedder for matcher (OpenAI text-embedding-3-small). The cache is
    # usually warm from EXP 1-7, but B.2 surfaces single-voter outputs that
    # weren't necessarily embedded by the cascade-only previous runs. Even
    # one cache miss requires a real API key — there is no honest dry-run
    # for a missing embedding, so refuse upfront if the key is absent.
    from eval.silver.matching.embedders import OpenAIEmbedder
    from eval.silver.matching.matcher import (
        DEFAULT_CACHE_PATH as MATCHER_CACHE_PATH,
        EMBEDDING_MODEL as MATCHER_EMBEDDING_MODEL,
        make_embedding_cache,
    )
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "EXP B.2 (baselines): OPENAI_API_KEY not set.\n"
            "  The silver matcher uses OpenAI text-embedding-3-small. The\n"
            "  shared cache covers all EXP 1-7 finding texts, but B.2 surfaces\n"
            "  per-L1-voter outputs the cascade-only sweeps did not embed.\n"
            "  Set OPENAI_API_KEY (any valid key — cache misses cost ~\\$0.0001\n"
            "  total) and re-run."
        )
    embedder = OpenAIEmbedder(api_key)
    embed_cache = make_embedding_cache(MATCHER_CACHE_PATH, MATCHER_EMBEDDING_MODEL)

    def _case_in_split(case_id: str) -> bool:
        if ctx.split == "all":
            return True
        return assign_split(case_id, ctx.dev_fraction, ctx.seed) == ctx.split

    rows: list[dict] = []
    skipped_notes: list[str] = []

    # ── (1) L1 per-voter rows ────────────────────────────────────────────────
    l1_outputs_per_voter = []
    for v_idx, model in enumerate(l1_models[: shape["n_l1_voters"]]):
        outs = _b2_build_voter_outputs(
            voter_cache, level="l1", voter_index=v_idx, case_filter=_case_in_split,
        )
        l1_outputs_per_voter.append(outs)
        metrics = _b2_eval_outputs(
            outs, silver_by_case, embedder, embed_cache, ctx.sim_threshold,
        )
        rows.append({
            **metrics,
            "split":                      ctx.split,
            "baseline":                   "l1_only",
            "model_or_source":            model,
            "estimated_cost_per_chunk":   _B2_PER_VOTER_COST.get(model, 0.0),
            "notes":                      _b2_baseline_note("l1_only", ctx.split),
        })

    # ── (2) L1-best oracle row (uses silver to pick best L1 voter per case) ──
    oracle_outs = _b2_l1_best_oracle(
        l1_outputs_per_voter, silver_by_case, embedder, embed_cache, ctx.sim_threshold,
    )
    oracle_metrics = _b2_eval_outputs(
        oracle_outs, silver_by_case, embedder, embed_cache, ctx.sim_threshold,
    )
    rows.append({
        **oracle_metrics,
        "split":                      ctx.split,
        "baseline":                   "l1_best_oracle",
        "model_or_source":            "oracle(silver-selected L1)",
        # Oracle still runs all 3 L1 voters in production cost terms (only the
        # pick is oracular). Sum of L1 voter unit costs.
        "estimated_cost_per_chunk":   sum(
            _B2_PER_VOTER_COST.get(m, 0.0) for m in l1_models[: shape["n_l1_voters"]]
        ),
        "notes":                      _b2_baseline_note("l1_best_oracle", ctx.split),
    })

    # ── (3) L2 per-voter rows (mid-tier — if populated in cache) ─────────────
    l2_outputs_per_voter: list = []
    if shape["l2_populated_chunks"] > 0 and shape["n_l2_voters"] > 0:
        for v_idx, model in enumerate(l2_models[: shape["n_l2_voters"]]):
            outs = _b2_build_voter_outputs(
                voter_cache, level="l2", voter_index=v_idx, case_filter=_case_in_split,
            )
            l2_outputs_per_voter.append(outs)
            metrics = _b2_eval_outputs(
                outs, silver_by_case, embedder, embed_cache, ctx.sim_threshold,
            )
            rows.append({
                **metrics,
                "split":                      ctx.split,
                "baseline":                   "l2_only",
                "model_or_source":            model,
                "estimated_cost_per_chunk":   _B2_PER_VOTER_COST.get(model, 0.0),
                "notes":                      _b2_baseline_note("l2_only", ctx.split),
            })
    else:
        skipped_notes.append(
            "l2_only skipped: voter cache has no populated L2 outputs "
            "(re-prime with L2 voters if you want the mid-tier baseline)."
        )

    # ── (4) L3-only (Sonnet) row ─────────────────────────────────────────────
    l3_outs = _b2_build_voter_outputs(
        voter_cache, level="l3", voter_index=0, case_filter=_case_in_split,
    )
    l3_metrics = _b2_eval_outputs(
        l3_outs, silver_by_case, embedder, embed_cache, ctx.sim_threshold,
    )
    rows.append({
        **l3_metrics,
        "split":                      ctx.split,
        "baseline":                   "l3_only",
        "model_or_source":            l3_model,
        "estimated_cost_per_chunk":   _B2_PER_VOTER_COST.get(l3_model, 0.0),
        "notes":                      _b2_baseline_note("l3_only", ctx.split),
    })

    # ── (5) Cascade row (if PROVISIONAL_FINAL_MAP_CONFIG present) ────────────
    pfm = ctx.state.get("PROVISIONAL_FINAL_MAP_CONFIG")
    if pfm:
        try:
            cascade_metrics, cascade_esc = _b2_replay_cascade(
                voter_cache, pfm, _case_in_split, silver_by_case,
                embedder, embed_cache, ctx.sim_threshold,
            )
        except Exception as exc:
            skipped_notes.append(
                f"cascade_provisional_final skipped: replay failed ({exc!r})."
            )
        else:
            embedder_kind = pfm.get("embedder", "?")
            scorer = pfm.get("scorer", "?")
            theta = pfm.get("theta", "?")
            unit_cost = _b2_cascade_cost_per_chunk(cascade_esc, l1_models, l3_model)
            rows.append({
                **cascade_metrics,
                "split":                    ctx.split,
                "baseline":                 "cascade_provisional_final",
                "model_or_source":          f"{embedder_kind}/{scorer}@θ={theta}",
                "estimated_cost_per_chunk": round(unit_cost, 5),
                "notes": (
                    "PROVISIONAL_FINAL_MAP_CONFIG replay; cost = L1_fixed + "
                    f"esc_rate({cascade_esc:.3f}) × (L2_sum + L3); "
                    "no state promotion."
                ),
            })
    else:
        skipped_notes.append(
            "cascade_provisional_final skipped: PROVISIONAL_FINAL_MAP_CONFIG "
            "missing from state (run --phase compare first)."
        )

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    csv_path = ctx.output_dir / f"exp_b2_cost_quality_{ts}.csv"
    status = _write_csv_rows(csv_path, rows)

    summary_lines = [
        f"  {r['baseline']:30} ({r['model_or_source']:30}) "
        f"strict_f1={r['strict_f1']:.4f}  $/chunk=${r['estimated_cost_per_chunk']:.5f}"
        for r in rows
    ]
    notes_blob = (
        f"EXP B.2 baselines @ split={ctx.split} ({len(rows)} rows):\n"
        + "\n".join(summary_lines)
    )
    if skipped_notes:
        notes_blob += "\n  Skipped:\n  - " + "\n  - ".join(skipped_notes)
    notes_blob += (
        "\n  Diagnostic only — does not promote any baseline. "
        "PROVISIONAL_FINAL_MAP_CONFIG remains the canonical pick."
    )

    return ExperimentResult(
        csv_path=csv_path,
        winner=None,        # baselines compare; they don't crown a winner
        notes=notes_blob,
        status=status,
        state_updates={},   # intentional: no state mutation
    )


# ── EXP B.2 helpers ──────────────────────────────────────────────────────────

# Module-level constants so tests can monkey-patch without touching env.
_B2_CACHE_PATH  = _REPO_ROOT / "eval" / "data" / "map_primer" / "voter_cache.json"
_B2_SILVER_PATH = _REPO_ROOT / "eval" / "data" / "silver_findings_related15.jsonl"

# Per-voter cost estimate (USD/chunk) — token assumption: 1500 input + 600
# output, prices from configs/model_prices.json. One model = one row.
# Recompute if model_prices.json or the token assumption change.
_B2_TOK_IN, _B2_TOK_OUT = 1500, 600

def _b2_unit_cost(price_in_per_1m: float, price_out_per_1m: float) -> float:
    return _B2_TOK_IN * price_in_per_1m / 1e6 + _B2_TOK_OUT * price_out_per_1m / 1e6

_B2_PER_VOTER_COST: dict[str, float] = {
    "gemini-2.5-flash-lite":           _b2_unit_cost(0.10, 0.40),
    "gpt-4o-mini":                     _b2_unit_cost(0.15, 0.60),
    "gpt-4.1-nano":                    _b2_unit_cost(0.10, 0.40),
    "gemini-2.5-flash":                _b2_unit_cost(0.30, 2.50),
    "gpt-4.1-mini":                    _b2_unit_cost(0.40, 1.60),
    "claude-haiku-4-5-20251001":       _b2_unit_cost(0.80, 4.00),
    "claude-haiku-4-5":                _b2_unit_cost(0.80, 4.00),
    "claude-sonnet-4-6":               _b2_unit_cost(3.00, 15.00),
}


def _b2_cache_shape(voter_cache: dict) -> dict:
    """Inspect cache for (max L1/L2 voters across chunks, populated L3 chunks)."""
    max_l1 = 0
    max_l2 = 0
    l2_pop = 0
    l2_total = 0
    l3_pop = 0
    l3_total = 0
    for entry in voter_cache.values():
        for cid, l1 in (entry.get("l1") or {}).items():
            if l1:
                max_l1 = max(max_l1, len(l1))
        for cid, l2 in (entry.get("l2") or {}).items():
            l2_total += 1
            if l2:
                max_l2 = max(max_l2, len(l2))
                l2_pop += 1
        for cid, l3 in (entry.get("l3") or {}).items():
            l3_total += 1
            if l3 is not None:
                l3_pop += 1
    return {
        "n_l1_voters":            max_l1,
        "n_l2_voters":            max_l2,
        "l2_populated_chunks":    l2_pop,
        "l2_total_chunks":        l2_total,
        "l3_populated_chunks":    l3_pop,
        "l3_total_chunks":        l3_total,
    }


def _b2_finding_to_pipeline(f: dict, pmcid: str, chunk_id: str, run_id: str):
    """Mirror eval.silver.map_theta_sweep._finding_to_pipeline (without θ stamp)."""
    from eval.silver.data.schemas import PipelineFinding
    scope = f.get("scope") or {}
    return PipelineFinding(
        pipeline_run_id=0,
        run_id=run_id,
        pmcid=pmcid,
        chunk_id=chunk_id,
        category=f.get("category", ""),
        claim=f.get("claim", ""),
        subject_entity=f.get("subject_entity"),
        outcome_entity=f.get("outcome_entity"),
        relation_type=str(f.get("relation_type", "unclear")),
        direction=str(f["direction"]) if f.get("direction") else None,
        confidence=str(f.get("confidence", "medium")),
        verbatim_support=f.get("verbatim_support", ""),
        grounding_score=f.get("grounding_score"),
        scope_disease_subtype=scope.get("disease_subtype"),
        scope_cohort_n=scope.get("cohort_n"),
        scope_assay_method=scope.get("assay_method"),
        scope_biomarker_cutoff=scope.get("biomarker_cutoff"),
        scope_tissue_site=scope.get("tissue_site"),
        scope_treatment_context=scope.get("treatment_context"),
        scope_endpoint=scope.get("endpoint"),
        scope_study_design=scope.get("study_design"),
    )


def _b2_build_voter_outputs(
    voter_cache: dict, *, level: str, voter_index: int, case_filter,
):
    """Build PipelineCaseOutputs taking findings from a single voter slot.

    For ``level="l3"``, the cached entry is a single dict (not a list).
    For ``level="l1"`` / ``"l2"``, entries are voter lists indexed by
    ``voter_index``.
    """
    from eval.silver.data.schemas import PipelineCaseOutput
    outputs = []
    for entry in voter_cache.values():
        case_id = entry["case_id"]
        if not case_filter(case_id):
            continue
        pmcid = entry["pmcid"]
        te_id = entry["te_id"]
        findings = []
        chunk_map = entry.get("chunk_map") or {}
        for chunk_id in chunk_map:
            level_data = (entry.get(level) or {}).get(chunk_id)
            if level == "l3":
                voter_output = level_data  # single dict or None
            else:
                if not level_data or len(level_data) <= voter_index:
                    continue
                voter_output = level_data[voter_index]
            if not voter_output:
                continue
            for f in voter_output.get("findings", []) or []:
                findings.append(
                    _b2_finding_to_pipeline(
                        f, pmcid, chunk_id,
                        run_id=f"exp_b2_{level}_v{voter_index}",
                    )
                )
        outputs.append(PipelineCaseOutput(
            case_id=case_id, pmcid=pmcid, te_id=te_id,
            run_id=f"exp_b2_{level}_v{voter_index}",
            pipeline_run_id=0,
            findings=findings,
        ))
    return outputs


def _b2_eval_outputs(
    case_outputs, silver_by_case, embedder, embed_cache, sim_threshold,
) -> dict:
    """Mirror eval.silver.pipeline_sweep._evaluate_outputs.

    Returns ``{strict_f1, f1, precision, recall, n_matched, n_silver, n_pipeline}``.
    """
    from eval.silver.matching.matcher import compute_sim_matrix, match_from_matrix
    STRICT_FIELDS = {"category", "relation_type", "direction"}
    n_silver = n_pipeline = n_matched = 0
    strict_tp = 0.0
    for co in case_outputs:
        silver = silver_by_case.get(co.case_id)
        if silver is None:
            continue
        sim, _, _ = compute_sim_matrix(silver, co, embedder, embed_cache)
        result = match_from_matrix(silver, co, sim, sim_threshold)
        n_silver += len(silver.findings)
        n_pipeline += len(co.findings)
        n_matched += len(result.matched)
        for pair in result.matched:
            has_strict_mismatch = any(
                m.field in STRICT_FIELDS for m in pair.field_mismatches
            )
            strict_tp += 0.5 if has_strict_mismatch else 1.0
    precision = n_matched / n_pipeline if n_pipeline else 0.0
    recall    = n_matched / n_silver   if n_silver   else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    sp = strict_tp / n_pipeline if n_pipeline else 0.0
    sr = strict_tp / n_silver   if n_silver   else 0.0
    strict_f1 = (2 * sp * sr / (sp + sr)) if (sp + sr) else 0.0
    return {
        "strict_f1": round(strict_f1, 4),
        "f1":        round(f1, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "n_matched": n_matched,
        "n_silver":  n_silver,
        "n_pipeline": n_pipeline,
    }


def _b2_l1_best_oracle(
    l1_outputs_per_voter, silver_by_case, embedder, embed_cache, sim_threshold,
):
    """Per-case, pick the L1 voter with the highest case-level strict_f1.

    Oracle — uses silver to choose the voter, so not deployable. The CSV
    labels this row ``l1_best_oracle`` and the column ``model_or_source`` is
    ``oracle(silver-selected L1)`` to keep the framing honest.
    """
    from eval.silver.matching.matcher import compute_sim_matrix, match_from_matrix
    STRICT_FIELDS = {"category", "relation_type", "direction"}
    n_voters = len(l1_outputs_per_voter)
    if n_voters == 0:
        return []
    by_case: dict = {}
    for v_idx, outs in enumerate(l1_outputs_per_voter):
        for co in outs:
            slot = by_case.setdefault(co.case_id, [None] * n_voters)
            slot[v_idx] = co
    chosen = []
    for case_id, slot in by_case.items():
        silver = silver_by_case.get(case_id)
        if silver is None:
            continue
        best_co = None
        best_sf1 = -1.0
        for co in slot:
            if co is None:
                continue
            sim, _, _ = compute_sim_matrix(silver, co, embedder, embed_cache)
            result = match_from_matrix(silver, co, sim, sim_threshold)
            n_p = len(co.findings)
            n_s = len(silver.findings)
            strict_tp = sum(
                0.5 if any(m.field in STRICT_FIELDS for m in pair.field_mismatches) else 1.0
                for pair in result.matched
            )
            sp = strict_tp / n_p if n_p else 0.0
            sr = strict_tp / n_s if n_s else 0.0
            sf1 = 2 * sp * sr / (sp + sr) if (sp + sr) else 0.0
            if sf1 > best_sf1:
                best_sf1 = sf1
                best_co = co
        if best_co is not None:
            chosen.append(best_co)
    return chosen


def _b2_replay_cascade(
    voter_cache, pfm: dict, case_filter, silver_by_case,
    embedder, embed_cache, sim_threshold,
):
    """Replay the cascade at PROVISIONAL_FINAL_MAP_CONFIG settings.

    Returns ``(metrics_dict, escalate_rate)``. Replay-only — needs
    ``GOOGLE_API_KEY`` (gemini) or ``OPENAI_API_KEY`` (openai) only when the
    agreement embedding cache is cold; warm caches → zero API calls.
    """
    from eval.silver.map_theta_sweep import _replay, ScorerSpec, _build_scorer
    from pipeline.stages.knowledge_extraction.config import AgreementConfig, HybridConfig
    from eval.silver.map_context import _load_map_context

    scorer_kind = "hybrid" if str(pfm.get("scorer", "")).startswith("hybrid") else "embedding"

    hw = pfm.get("hybrid_weights")
    if isinstance(hw, str):
        hw = json.loads(hw)
    hybrid_cfg = HybridConfig()
    if scorer_kind == "hybrid" and hw:
        wc, we, wn, wv = hw
        hybrid_cfg = HybridConfig(
            w_category=wc, w_embedding=we, w_entity=wn, w_evidence=wv,
        )

    base_kwargs = dict(scorer_kind=scorer_kind, hybrid=hybrid_cfg)
    aw = pfm.get("agreement_weights")
    if isinstance(aw, str):
        aw = json.loads(aw)
    if aw:
        tau, ca, rw, cw = aw
        base_kwargs.update(
            tau=tau, count_alpha=ca, reuse_weight=rw, contradiction_weight=cw,
        )
    base_cfg = AgreementConfig(**base_kwargs)
    embedder_kind = pfm.get("embedder", "gemini")
    map_ctx = _load_map_context(embedder_kind, embed_cache_path=None)

    spec = ScorerSpec(f"cascade_pfm_{embedder_kind}", scorer_kind, base_cfg)
    scorer = _build_scorer(spec, map_ctx.agreement_embed_fn)

    polarity_flag = pfm.get("polarity_flag", "true")
    if isinstance(polarity_flag, str):
        polarity_flag = polarity_flag.lower() == "true"

    case_outputs, accept, _early, _polconf = _replay(
        voter_cache, scorer,
        theta=float(pfm["theta"]),
        reject_theta=float(pfm["reject_theta"]),
        single_voter_policy="keep",
        force_escalate_on_polarity_conflict=bool(polarity_flag),
    )
    case_outputs = [
        co for co in case_outputs
        if co.case_id in silver_by_case and case_filter(co.case_id)
    ]
    metrics = _b2_eval_outputs(
        case_outputs, silver_by_case, embedder, embed_cache, sim_threshold,
    )
    total = sum(accept.values()) or 1
    esc = accept.get("l3", 0) / total
    return metrics, esc


def _b2_cascade_cost_per_chunk(
    esc_rate: float, l1_models: list[str], l3_model: str,
) -> float:
    """L1 fixed + esc × (L2 + L3), with L2 from the production profile."""
    try:
        from pipeline.stages.knowledge_extraction.batch.voter_configs import make_l2_voters
        l2_sum = sum(_B2_PER_VOTER_COST.get(v.model, 0.0) for v in make_l2_voters())
    except Exception:
        l2_sum = 0.0
    l1_sum = sum(_B2_PER_VOTER_COST.get(m, 0.0) for m in l1_models)
    l3 = _B2_PER_VOTER_COST.get(l3_model, 0.0)
    return l1_sum + esc_rate * (l2_sum + l3)


def _b2_baseline_note(baseline_id: str, split: str) -> str:
    base = {
        "l1_only": (
            "Single L1 voter, no agreement, no escalation. Floor accuracy at "
            "this voter's cost."
        ),
        "l1_best_oracle": (
            "ORACLE — per-case best L1 voter chosen by silver match. Not "
            "deployable; bounds the L1-tier upside."
        ),
        "l2_only": (
            "Single L2 voter — mid-tier strong-model baseline; ignores L1 + L3 "
            "entirely. Useful for sizing the L1→L2 vs L2→L3 accuracy gaps."
        ),
        "l3_only": (
            "Always-escalate to Sonnet — single-model strong baseline; ignores "
            "L1+L2 entirely."
        ),
    }.get(baseline_id, "")
    if split == "test":
        base += " [TEST split — evaluation-only; not used for tuning.]"
    return base


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
        depends_on=["EXP_7"], status="executable", run=_run_exp_b2,
        blurb=(
            "Cost-quality replay: per-L1-voter rows + l1_best_oracle + l3_only + "
            "cascade@PROVISIONAL_FINAL. Replay-only, no state mutation."
        ),
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
    print("  python scripts/eval/run_summarization_experiments.py --phase baselines --run")
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
