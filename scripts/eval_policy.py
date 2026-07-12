#!/usr/bin/env python3
"""
Evaluate a routing policy spec against a labeled routing records file.

Computes PolicyEvaluationResult metrics at the given theta and appends to a
policy_results.jsonl file.

Usage
-----
python scripts/eval_policy.py \\
    --records out/routing_records_labeled.jsonl \\
    --policy-id "sem_theta074" \\
    --scorer-name "SemanticAgreementScorer" \\
    --theta 0.74 \\
    --voter-models gpt-4o-mini claude-haiku-4-5-20251001 \\
    --escalation-model gpt-4o \\
    [--single-voter-policy escalate] \\
    [--reject-on-low-agreement] \\
    [--fabricated-threshold 0.25] \\
    [--weak-threshold 0.60] \\
    [--cost-alpha 0.6] \\
    [--results out/policy_results.jsonl]

Scorer compatibility
--------------------
``deferral_score`` values in the records file were computed by a specific
scorer.  Evaluating a policy at theta against records from a different scorer
is unsound — the score distributions are incommensurable.

The check compares ``scorer_name`` (from this CLI) against the
``agreement_scorer`` field in each record.  This is the primary compatibility
dimension, but not the only one: voter model changes can shift score
distributions even for the same scorer class.  A warning is printed and
recorded in PolicyEvaluationResult.warnings if the scorer names do not match.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Bootstrap the repo root onto sys.path so the repository-local imports below resolve
# when this script is run directly (`python scripts/…`) — B-095.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.stages.knowledge_extraction.routing.policy import (  # noqa: E402
    PolicyEvaluationResult,
    PolicyEvaluationStore,
    RoutingPolicySpec,
)
from pipeline.stages.knowledge_extraction.routing.routing_dataset import RoutingDataset  # noqa: E402


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--records", required=True, help="Labeled routing_records.jsonl")
    p.add_argument("--policy-id", required=True)
    p.add_argument("--scorer-name", required=True,
                   help="Scorer class name: EmbeddingScorer | SemanticAgreementScorer | ...")
    p.add_argument("--theta", required=True, type=float,
                   help="Deferral score threshold for the agreement gate")
    p.add_argument("--voter-models", nargs="+", required=True)
    p.add_argument("--escalation-model", required=True)
    p.add_argument("--single-voter-policy", default="escalate",
                   choices=["escalate", "keep"])
    p.add_argument("--reject-on-low-agreement", action="store_true")
    p.add_argument("--fabricated-threshold", type=float, default=0.25)
    p.add_argument("--weak-threshold", type=float, default=0.60)
    p.add_argument("--cost-alpha", type=float, default=0.6,
                   help="Weight for escalation_rate in selection_objective "
                        "(selection_objective = alpha*esc + (1-alpha)*far). Default: 0.6")
    p.add_argument("--results", default="out/policy_results.jsonl",
                   help="JSONL file to append PolicyEvaluationResult to")
    return p.parse_args()


# ── Metric computation ─────────────────────────────────────────────────────────

def compute_at_theta(
    ag_records,
    labeled,
    acceptable,
    theta: float,
) -> dict:
    """
    Compute metrics at a single theta value.

    Precision/recall/FAR are over labeled records only.
    escalation_rate is over all AG records (production cost proxy).
    keep_rate is n_kept / n_labeled.
    """
    n_labeled = len(labeled)
    n_acceptable = len(acceptable)
    n_ag = len(ag_records)

    if n_labeled == 0:
        return {
            "keep_precision": None,
            "false_accept_rate": None,
            "recall": None,
            "keep_rate": None,
            "escalation_rate": (
                sum(1 for r in ag_records if (r.deferral_score or 0.0) < theta) / n_ag
                if n_ag > 0 else None
            ),
        }

    kept_l = [r for r in labeled if (r.deferral_score or 0.0) >= theta]
    tp = sum(1 for r in kept_l if r.keep_ok is True)
    fp = sum(1 for r in kept_l if r.keep_ok is False)
    n_kept = len(kept_l)

    keep_precision = tp / n_kept if n_kept > 0 else float("nan")
    false_accept_rate = fp / n_kept if n_kept > 0 else float("nan")
    recall = tp / n_acceptable if n_acceptable > 0 else float("nan")
    keep_rate = n_kept / n_labeled if n_labeled > 0 else float("nan")

    n_escalated_ag = sum(1 for r in ag_records if (r.deferral_score or 0.0) < theta)
    escalation_rate = n_escalated_ag / n_ag if n_ag > 0 else float("nan")

    def _null(v: float) -> float | None:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return v

    return {
        "keep_precision": _null(keep_precision),
        "false_accept_rate": _null(false_accept_rate),
        "recall": _null(recall),
        "keep_rate": _null(keep_rate),
        "escalation_rate": _null(escalation_rate),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    results_path = Path(args.results)

    spec = RoutingPolicySpec(
        policy_id=args.policy_id,
        scorer_name=args.scorer_name,
        theta=args.theta,
        voter_models=args.voter_models,
        escalation_model=args.escalation_model,
        single_voter_policy=args.single_voter_policy,
        reject_on_low_agreement=args.reject_on_low_agreement,
        fabricated_threshold=args.fabricated_threshold,
        weak_threshold=args.weak_threshold,
    )

    all_records = list(RoutingDataset.iter_records(args.records))
    ag_records = [
        r for r in all_records
        if r.gate_origin == "agreement_gate" and r.deferral_score is not None
    ]
    labeled = [r for r in ag_records if r.keep_ok is not None]
    acceptable = [r for r in labeled if r.keep_ok is True]

    print("\n=== Dataset ===")
    print(f"  total records   : {len(all_records)}")
    print(f"  agreement_gate  : {len(ag_records)}")
    print(f"  labeled         : {len(labeled)}")
    print(f"  acceptable      : {len(acceptable)}")
    print(f"  unacceptable    : {len(labeled) - len(acceptable)}")

    # ── Scorer compatibility check ─────────────────────────────────────────────
    warnings: list[str] = []
    scorers_in_file: set[str] = {r.agreement_scorer for r in ag_records if r.agreement_scorer}
    if scorers_in_file and spec.scorer_name not in scorers_in_file:
        msg = (
            f"Policy scorer '{spec.scorer_name}' does not match scorer(s) in records file: "
            f"{scorers_in_file}.  deferral_score values were computed by a different scorer — "
            f"the score distributions may be incommensurable.  scorer_name is the primary "
            f"compatibility dimension, but other factors (voter models, temperature) can also "
            f"shift distributions.  Threshold evaluation at theta={spec.theta} may not be "
            f"meaningful for this policy.  Proceeding — interpret results with caution."
        )
        print(f"\n[WARNING] {msg}")
        warnings.append(msg)

    # ── Compute metrics ────────────────────────────────────────────────────────
    metrics = compute_at_theta(ag_records, labeled, acceptable, spec.theta)

    # ── Selection objective ────────────────────────────────────────────────────
    alpha = args.cost_alpha
    selection_objective: float | None = None
    if metrics["escalation_rate"] is not None and metrics["false_accept_rate"] is not None:
        selection_objective = (
            alpha * metrics["escalation_rate"]
            + (1 - alpha) * metrics["false_accept_rate"]
        )

    result = PolicyEvaluationResult(
        policy_id=spec.policy_id,
        scorer_name=spec.scorer_name,
        theta=spec.theta,
        records_file=str(Path(args.records).resolve()),
        n_records=len(all_records),
        n_agreement_gate=len(ag_records),
        n_labeled=len(labeled),
        n_acceptable=len(acceptable),
        keep_precision=metrics["keep_precision"],
        false_accept_rate=metrics["false_accept_rate"],
        recall=metrics["recall"],
        keep_rate=metrics["keep_rate"],
        escalation_rate=metrics["escalation_rate"],
        selection_objective=selection_objective,
        selection_objective_alpha=alpha,
        warnings=warnings,
    )

    print(f"\n=== Result (theta={spec.theta}) ===")
    print(f"  policy_id         : {result.policy_id}")
    print(f"  scorer_name       : {result.scorer_name}")
    print(f"  keep_precision    : {result.keep_precision}")
    print(f"  false_accept_rate : {result.false_accept_rate}")
    print(f"  recall            : {result.recall}")
    print(f"  keep_rate         : {result.keep_rate}")
    print(f"  escalation_rate   : {result.escalation_rate}")
    print(f"  selection_obj     : {result.selection_objective} (alpha={alpha})")

    PolicyEvaluationStore.append(result, results_path)
    print(f"\n[appended] {results_path}")


if __name__ == "__main__":
    main()
