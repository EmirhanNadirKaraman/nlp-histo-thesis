#!/usr/bin/env python3
"""
Select the best routing policy from a policy_results.jsonl file using PuLP ILP.

Usage
-----
python scripts/select_policy.py \\
    --results out/policy_results.jsonl \\
    --max-false-accept 0.10 \\
    --min-recall 0.70 \\
    [--cost-alpha 0.6] \\
    --out out/selected_policy.json

PuLP ILP formulation
--------------------
Variables:  x_i ∈ {0, 1}   for each policy i with non-null selection_objective

Objective:
  minimize  Σ_i (selection_objective_i * x_i)

Subject to:
  Σ_i x_i = 1                                    (select exactly one)
  Σ_i (false_accept_rate_i * x_i) ≤ max_far      (safety — primary constraint)
  Σ_i (recall_i * x_i) ≥ min_recall              (coverage — secondary constraint)
  x_i ∈ {0, 1}

Equivalence note
----------------
For the current case (select one policy, two scalar constraints), this ILP is
equivalent to:
    feasible = [p for p in policies
                if p.false_accept_rate <= max_far and p.recall >= min_recall]
    selected = min(feasible, key=lambda p: p.selection_objective)

PuLP is used to formalize the optimization structure.  Its practical value is
extensibility: adding per-paper-type fairness constraints, hard budget limits,
multi-dimensional objectives, or selecting k > 1 policies requires only
additional constraints on the existing variable set — no rewrite.

Infeasibility handling
----------------------
If no policy satisfies both constraints:
  1. Try relaxing min_recall (safety-first).
  2. If still infeasible, select minimum false_accept_rate (safety only).
  3. Record infeasibility_reason in PolicySelectionResult.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Bootstrap the repo root onto sys.path so the repository-local imports below resolve
# when this script is run directly (`python scripts/…`) — B-095.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nlp_histo.pipeline.stages.knowledge_extraction.routing.policy import (  # noqa: E402
    PolicyEvaluationResult,
    PolicyEvaluationStore,
    PolicySelectionResult,
)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results", required=True, help="policy_results.jsonl")
    p.add_argument("--max-false-accept", type=float, default=0.10,
                   help="Max allowed false_accept_rate (default: 0.10)")
    p.add_argument("--min-recall", type=float, default=0.70,
                   help="Min required recall (default: 0.70)")
    p.add_argument("--cost-alpha", type=float, default=0.6,
                   help="Alpha for selection_objective if not already set in records "
                        "(default: 0.6). Records with pre-computed selection_objective "
                        "use their stored value; this flag only applies to records "
                        "where selection_objective is None.")
    p.add_argument("--out", default="out/selected_policy.json",
                   help="Output path for PolicySelectionResult JSON")
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _objective(r: PolicyEvaluationResult, alpha: float) -> float | None:
    """Return selection_objective; fall back to computing from components."""
    if r.selection_objective is not None:
        return r.selection_objective
    if r.escalation_rate is not None and r.false_accept_rate is not None:
        return alpha * r.escalation_rate + (1 - alpha) * r.false_accept_rate
    return None


def _compact_winner(r: PolicyEvaluationResult) -> dict:
    return {
        "selected_scorer_name": r.scorer_name,
        "selected_theta": r.theta,
        "selected_false_accept_rate": r.false_accept_rate,
        "selected_recall": r.recall,
        "selected_escalation_rate": r.escalation_rate,
        "selected_selection_objective": r.selection_objective,
    }


# ── PuLP selector ──────────────────────────────────────────────────────────────

def select_with_pulp(
    candidates: list[PolicyEvaluationResult],
    max_false_accept: float,
    min_recall: float,
    alpha: float,
) -> tuple[PolicyEvaluationResult | None, bool, str | None, PolicyEvaluationResult | None]:
    """
    Run the PuLP ILP.

    Returns (selected, feasible, infeasibility_reason, runner_up).
    runner_up is the second-best feasible candidate (or None).
    """
    try:
        import pulp  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "pulp is required for select_policy.py. "
            "Install it with: pip install pulp"
        )

    # Only consider candidates with a computable objective
    eligible = [r for r in candidates if _objective(r, alpha) is not None]
    if not eligible:
        return None, False, "No candidates with computable selection_objective.", None

    # Assign pre-computed or fallback objective
    objectives = {r.policy_id: _objective(r, alpha) for r in eligible}

    # ── Full constraints: FAR + recall ─────────────────────────────────────────
    prob = pulp.LpProblem("select_policy", pulp.LpMinimize)
    x = {r.policy_id: pulp.LpVariable(f"x_{r.policy_id}", cat="Binary") for r in eligible}

    # Objective: minimize selection_objective of chosen policy
    prob += pulp.lpSum(objectives[pid] * x[pid] for pid in x)

    # Exactly one policy selected
    prob += pulp.lpSum(x[pid] for pid in x) == 1

    # Safety constraint: false_accept_rate ≤ max_false_accept
    for r in eligible:
        if r.false_accept_rate is not None:
            prob += x[r.policy_id] * r.false_accept_rate <= max_false_accept * x[r.policy_id]

    # Coverage constraint: recall ≥ min_recall
    for r in eligible:
        if r.recall is not None:
            prob += x[r.policy_id] * r.recall >= min_recall * x[r.policy_id]

    # Force ineligible (no FAR/recall data) to 0
    for r in eligible:
        if r.false_accept_rate is None or r.recall is None:
            prob += x[r.policy_id] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] == "Optimal":
        selected_id = max(x, key=lambda pid: pulp.value(x[pid]) or 0)
        selected = next(r for r in eligible if r.policy_id == selected_id)
        # Runner-up: second best feasible (by objective)
        feasible_sorted = sorted(
            [r for r in eligible
             if r.false_accept_rate is not None and r.false_accept_rate <= max_false_accept
             and r.recall is not None and r.recall >= min_recall],
            key=lambda r: objectives[r.policy_id],
        )
        runner_up = next(
            (r for r in feasible_sorted if r.policy_id != selected_id),
            None,
        )
        return selected, True, None, runner_up

    # ── Infeasible — relax recall constraint ───────────────────────────────────
    infeasibility_reason = (
        f"No policy satisfies both false_accept_rate ≤ {max_false_accept} "
        f"and recall ≥ {min_recall}."
    )

    prob2 = pulp.LpProblem("select_policy_relax", pulp.LpMinimize)
    x2 = {r.policy_id: pulp.LpVariable(f"x_{r.policy_id}", cat="Binary") for r in eligible}
    prob2 += pulp.lpSum(objectives[pid] * x2[pid] for pid in x2)
    prob2 += pulp.lpSum(x2[pid] for pid in x2) == 1
    for r in eligible:
        if r.false_accept_rate is not None:
            prob2 += x2[r.policy_id] * r.false_accept_rate <= max_false_accept * x2[r.policy_id]
    for r in eligible:
        if r.false_accept_rate is None:
            prob2 += x2[r.policy_id] == 0

    prob2.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob2.status] == "Optimal":
        selected_id = max(x2, key=lambda pid: pulp.value(x2[pid]) or 0)
        selected = next(r for r in eligible if r.policy_id == selected_id)
        infeasibility_reason += (
            "  Relaxed min_recall; selected under false_accept_rate constraint only."
        )
        return selected, False, infeasibility_reason, None

    # ── Cannot satisfy FAR — return minimum-FAR candidate ─────────────────────
    infeasibility_reason += "  Cannot satisfy false_accept_rate constraint at any candidate."
    with_far = [r for r in eligible if r.false_accept_rate is not None]
    if with_far:
        selected = min(with_far, key=lambda r: r.false_accept_rate)  # type: ignore[arg-type]
        return selected, False, infeasibility_reason, None

    return None, False, infeasibility_reason + "  No candidates with false_accept_rate.", None


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = PolicyEvaluationStore.load(args.results)
    if not all_results:
        print("[ERROR] No policy evaluation results found in", args.results)
        sys.exit(1)

    print(f"\n=== Candidates: {len(all_results)} ===")
    for r in all_results:
        obj = _objective(r, args.cost_alpha)
        far_str = f"{r.false_accept_rate:.4f}" if r.false_accept_rate is not None else "—"
        rec_str = f"{r.recall:.4f}" if r.recall is not None else "—"
        obj_str = f"{obj:.4f}" if obj is not None else "—"
        print(
            f"  {r.policy_id:<30}  far={far_str}  recall={rec_str}  "
            f"obj={obj_str}  scorer={r.scorer_name}  theta={r.theta}"
        )

    selected, feasible, infeasibility_reason, runner_up = select_with_pulp(
        all_results, args.max_false_accept, args.min_recall, args.cost_alpha
    )

    n_feasible = sum(
        1 for r in all_results
        if r.false_accept_rate is not None and r.false_accept_rate <= args.max_false_accept
        and r.recall is not None and r.recall >= args.min_recall
    )

    winner_fields = _compact_winner(selected) if selected is not None else {
        "selected_scorer_name": None,
        "selected_theta": None,
        "selected_false_accept_rate": None,
        "selected_recall": None,
        "selected_escalation_rate": None,
        "selected_selection_objective": None,
    }

    result = PolicySelectionResult(
        selected_policy_id=selected.policy_id if selected is not None else None,
        feasible=feasible,
        candidates_evaluated=len(all_results),
        candidates_feasible=n_feasible,
        constraints={
            "max_false_accept": args.max_false_accept,
            "min_recall": args.min_recall,
        },
        selection_objective_alpha=args.cost_alpha,
        runner_up_policy_id=runner_up.policy_id if runner_up is not None else None,
        infeasibility_reason=infeasibility_reason,
        **winner_fields,
    )

    print("\n=== Result ===")
    print(f"  selected          : {result.selected_policy_id}")
    print(f"  feasible          : {result.feasible}")
    print(f"  scorer            : {result.selected_scorer_name}")
    print(f"  theta             : {result.selected_theta}")
    print(f"  false_accept_rate : {result.selected_false_accept_rate}")
    print(f"  recall            : {result.selected_recall}")
    print(f"  escalation_rate   : {result.selected_escalation_rate}")
    print(f"  selection_obj     : {result.selected_selection_objective}")
    print(f"  runner_up         : {result.runner_up_policy_id}")
    if infeasibility_reason:
        print(f"\n  [WARNING] {infeasibility_reason}")

    out_path.write_text(result.model_dump_json(indent=2))
    print(f"\n[output]  {out_path}")


if __name__ == "__main__":
    main()
