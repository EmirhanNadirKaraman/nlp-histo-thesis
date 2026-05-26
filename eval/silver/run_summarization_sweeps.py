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

  Selection metric: **silver F1 / strict F1** (per-finding, vs Opus silver),
  with escalation rate reported and optionally constrained (--max-escalate).

────────────────────────────────────────────────────────────────────────────
SCAFFOLD ONLY — defined but NOT executable. The eval backend / a
silver-comparable selection metric does not exist for these yet; running one
errors out. They are here so the staged structure is complete and the knobs
are catalogued.

  Stage 4  grounding       grounding.threshold
                           metric (today): retention / rejection curve — NOT silver F1
  Stage 5  relate          entailment_threshold × contradiction_threshold
                           metric (today): relation-count / pair stats — NOT finding-level F1
  Stage 6  resolve         RESOLVE weights
                           metric (today): TBD — RESOLVE output is FinalRule-level,
                                           no silver eval exists
  Stage 7  contradiction   contradiction_similarity_threshold
                           metric (today): candidate-pair count — NOT silver F1

  *** THE METRIC MISMATCH IS DELIBERATE. *** Only the MAP stages are scored
  against silver per-finding F1. Grounding / relate / resolve / contradiction
  are NOT comparable to MAP silver F1 — do not rank them on the same axis.
  See docs/CALIBRATION_INVENTORY.md and docs/CALIBRATION_EVAL.md.

────────────────────────────────────────────────────────────────────────────
Prerequisites for the MAP stages:
  * primed voter cache  eval/data/map_primer/voter_cache.json
        (from `python -m eval.silver.map_theta_sweep prime --split all` + `collect`)
  * silver labels       eval/data/silver_findings.jsonl
        (from `python -m eval.silver.generate --batch`)

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
from typing import Callable, Optional

# Make ``eval.*`` / ``pipeline.*`` importable when run as a file.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(str(_REPO_ROOT / ".env"))

from pipeline.stages.summarization.config import AgreementConfig

from eval.silver.jsonl_utils import read_jsonl
from eval.silver.matcher import (
    DEFAULT_GEMINI_CACHE_PATH,
    GEMINI_EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD,
    make_embedding_cache,
)
from eval.silver.schemas import SilverCaseResult
from eval.silver.map_theta_sweep import (
    CACHE_PATH,
    REJECT_THETA_GRID,
    REPORTS_DIR,
    SILVER_PATH,
    THETA_GRID,
    ScorerSpec,
    _make_cached_embed_fn,
    _prewarm_agreement_cache,
    _print_table,
    _write_csv,
    run_sweep,
)

import json

# ─────────────────────────────────────────────────────────────────────────────
# BEST_* — winners pinned after each stage. Edit in place as each stage's CSV
# lands; later stages dereference these (greedy stage-wise selection).
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# Grids.
# ─────────────────────────────────────────────────────────────────────────────

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

# Scaffold grids (catalogued; not executed).
GROUNDING_THRESHOLD_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
RELATE_ENTAILMENT_GRID = [0.40, 0.50, 0.60, 0.70]
RELATE_CONTRADICTION_GRID = [0.40, 0.50, 0.60, 0.70]
RESOLVE_GROUNDING_WEIGHT_GRID = [0.50, 0.60, 0.70, 0.80]
CONTRADICTION_SIM_GRID = [0.50, 0.60, 0.70, 0.80]

CSV_FIELDNAMES = [
    "scorer", "tau", "count_alpha", "reuse_weight", "contradiction_weight",
    "theta", "reject_theta", "legacy_single_voter_policy", "cascade_path",
    "precision", "recall", "f1", "strict_f1",
    "early_accept_rate", "escalate_rate", "early_accept_precision",
    "n_matched", "n_silver", "n_pipeline",
    "split", "seed", "dev_fraction", "sim_threshold",
]


# ─────────────────────────────────────────────────────────────────────────────
# Stage registry (mirrors run_all_sweeps.py's SweepSpec + STAGE_ORDER).
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SweepSpec:
    name: str
    blurb: str
    executable: bool
    metric: str                       # MAP: "silver_f1"; scaffold: its native (non-silver) metric
    scaffold_reason: str = ""


STAGE_ORDER = (
    "map_scorer",          # Stage 1
    "map_theta",           # Stage 2
    "map_weights",         # Stage 3
    "map_routing_policy",  # Stage 3a — legacy_single_voter_policy {keep, escalate}
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
    "map_routing_policy": SweepSpec(
        "map_routing_policy",
        "Stage 3a — legacy_single_voter_policy ∈ {keep, escalate} at BEST_SCORER/theta/reject",
        executable=True, metric="silver_f1"),
    "grounding": SweepSpec(
        "grounding", "Stage 4 — grounding.threshold  [SCAFFOLD]",
        executable=False, metric="retention/rejection curve (NOT silver F1)",
        scaffold_reason=(
            "needs MAP findings + their NLI grounding scores. Either a full-pipeline "
            "prime (eval.silver.pipeline_sweep) or the Layer-A artifact sweep "
            "(eval/sweeps/grounding.py) — and the latter is retention-only, not "
            "silver-F1-comparable to the MAP stages.")),
    "relate": SweepSpec(
        "relate", "Stage 5 — entailment_threshold × contradiction_threshold  [SCAFFOLD]",
        executable=False, metric="relation-count / pair-distribution stats (NOT finding-level F1)",
        scaffold_reason=(
            "RELATE output is at the FinalRule level; silver labels are per-finding, "
            "so there is no silver-F1 metric for relation pairs. eval.silver.pipeline_sweep "
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


# ─────────────────────────────────────────────────────────────────────────────
# MAP stage grids (executable) — each returns (scorer_specs, thetas, reject_thetas).
# ─────────────────────────────────────────────────────────────────────────────

def _spec_for(scorer: str) -> ScorerSpec:
    if scorer not in SCORER_CHOICES:
        raise ValueError(f"scorer={scorer!r} must be one of {SCORER_CHOICES}")
    return ScorerSpec(f"{scorer}_default", scorer, AgreementConfig())


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
    if stage == "map_routing_policy":
        # Scorer / theta / reject_theta all pinned to BEST_*; the only axis is
        # the legacy_single_voter_policy enum (handled separately in
        # _policy_grid_for / _run_map_stage).
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


def _enumerate_cells(stage: str) -> list[str]:
    """Human-readable per-cell names for --list-variants."""
    if stage in ("map_scorer", "map_theta", "map_weights", "map_routing_policy"):
        specs, thetas, rejects = _map_grid(stage)
        policies = _policy_grid_for(stage)
        cells = []
        for s in specs:
            for t in thetas:
                for r in rejects:
                    if r >= t:
                        continue
                    for pol in policies:
                        if stage == "map_routing_policy":
                            # Per the user spec, the visible cell name uses the
                            # `legacy_single_voter_<policy>` form so the policy
                            # is the salient label, not buried in trailing
                            # metadata.
                            cells.append(
                                f"legacy_single_voter_{pol}  "
                                f"({s.name}, theta={t:.2f}, rej={r:.2f})"
                            )
                        elif len(policies) == 1 and policies[0] == "keep":
                            # Historical stages — omit the policy suffix to
                            # keep diffs against prior --list-variants output
                            # readable.
                            cells.append(f"{s.name}  theta={t:.2f} rej={r:.2f}")
                        else:
                            cells.append(
                                f"{s.name}  theta={t:.2f} rej={r:.2f}  pol={pol}"
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


# ─────────────────────────────────────────────────────────────────────────────
# Execution (MAP stages only).
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _MapContext:
    voter_cache: dict
    silver_by_case: dict
    embedder: object
    embed_cache: object
    agreement_embed_fn: object


def _load_map_context(embedder_kind: str, *, embed_cache_path: Optional[str]) -> _MapContext:
    """Load voter cache + silver + the gemini/openai embedder and pre-warm the
    agreement cache. Mirrors map_theta_sweep's sweep-mode setup (no LLM calls)."""
    if not CACHE_PATH.exists():
        raise SystemExit(
            f"voter cache not found: {CACHE_PATH}\n"
            "Run `python -m eval.silver.map_theta_sweep prime --split all` then `collect`.")
    if not SILVER_PATH.exists():
        raise SystemExit(
            f"silver labels not found: {SILVER_PATH}\n"
            "Run `python -m eval.silver.generate --batch`.")

    voter_cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    silver_by_case = {rec.case_id: rec for rec in read_jsonl(SILVER_PATH, SilverCaseResult)}

    import os
    if embedder_kind == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("GOOGLE_API_KEY not set")
        from eval.silver.embedders import GeminiEmbedder
        from pipeline.stages.summarization.agreement.providers import (
            GeminiEmbedder as AgreementGeminiEmbedder,
        )
        embedder = GeminiEmbedder(api_key)
        path = Path(embed_cache_path) if embed_cache_path else DEFAULT_GEMINI_CACHE_PATH
        embed_cache = make_embedding_cache(path, GEMINI_EMBEDDING_MODEL)
        raw_agreement_fn = AgreementGeminiEmbedder()
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY not set")
        from eval.silver.embedders import OpenAIEmbedder
        from eval.silver.matcher import DEFAULT_CACHE_PATH, EMBEDDING_MODEL
        from pipeline.stages.summarization.agreement.providers import (
            OpenAIEmbedder as AgreementOpenAIEmbedder,
        )
        embedder = OpenAIEmbedder(api_key)
        path = Path(embed_cache_path) if embed_cache_path else DEFAULT_CACHE_PATH
        embed_cache = make_embedding_cache(path, EMBEDDING_MODEL)
        raw_agreement_fn = AgreementOpenAIEmbedder()

    _prewarm_agreement_cache(voter_cache, raw_agreement_fn, embed_cache)
    agreement_embed_fn = _make_cached_embed_fn(embed_cache, raw_agreement_fn)
    return _MapContext(voter_cache, silver_by_case, embedder, embed_cache, agreement_embed_fn)


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
        return f"→ pin: BEST_SCORER = {scorer!r}"
    if stage == "map_theta":
        return f"→ pin: BEST_THETA = {w['theta']} ; BEST_REJECT_THETA = {w['reject_theta']}"
    if stage == "map_weights":
        return ("→ pin: BEST_TAU / BEST_COUNT_ALPHA / BEST_REUSE_WEIGHT / "
                f"BEST_CONTRADICTION_WEIGHT = ({w['tau']}, {w['count_alpha']}, "
                f"{w['reuse_weight']}, {w['contradiction_weight']})")
    if stage == "map_routing_policy":
        pol = w.get("legacy_single_voter_policy", "keep")
        return (
            f"→ pin: RoutingConfig.legacy_single_voter_policy = {pol!r}  "
            f"(update configs/run.yaml summarization.routing.legacy_single_voter_policy)"
        )
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Listing / menu.
# ─────────────────────────────────────────────────────────────────────────────

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
    if stage in ("map_theta", "map_weights", "map_routing_policy"):
        print(f"(holding earlier stages at BEST_SCORER={BEST_SCORER!r}, "
              f"BEST_THETA={BEST_THETA}, BEST_REJECT_THETA={BEST_REJECT_THETA})")
    if stage == "map_routing_policy":
        print("(sweep axis: RoutingConfig.legacy_single_voter_policy ∈ "
              f"{list(LEGACY_SINGLE_VOTER_POLICY_GRID)} — production default is 'keep')")


# ─────────────────────────────────────────────────────────────────────────────
# CLI.
# ─────────────────────────────────────────────────────────────────────────────

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
