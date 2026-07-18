"""
Routing policy configuration, evaluation results, and selection output.

Three separate concerns are kept in three separate objects:

RoutingPolicySpec      — pure configuration descriptor (dataclass, no metrics)
PolicyEvaluationResult — metrics from evaluating a spec against a labeled dataset
PolicySelectionResult  — output of the PuLP ILP optimizer

PolicyEvaluationStore  — static JSONL I/O for PolicyEvaluationResult, same pattern
                         as RoutingDataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    pass


# RoutingPolicySpec

@dataclass
class RoutingPolicySpec:
    """
    Pure configuration snapshot for one MAP-stage routing setup.

    Does NOT hold LLM objects (those are instantiated at runtime); captures only
    the parameters needed to reproduce and compare runs.

    All fields are plain scalars / lists so conversion to Pydantic (for
    serialization) is straightforward: replace ``@dataclass`` with
    ``class RoutingPolicySpec(BaseModel)`` and remove the decorator.

    Notes
    -----
    ``scorer_name`` is the primary compatibility dimension when evaluating a
    policy against a routing records file: ``deferral_score`` values are only
    meaningful relative to the scorer that computed them.  Other dimensions
    (voter_models, fabricated_threshold, etc.) may also affect score
    distributions — the compatibility check in ``eval_policy.py`` uses
    ``scorer_name`` as the primary signal, but the full spec should be inspected
    when results are unexpected.
    """

    policy_id: str
    scorer_name: str            # class name: "EmbeddingScorer" | "SemanticAgreementScorer" | ...
    theta: float                # deferral score threshold for the agreement gate
    voter_models: list[str]     # e.g. ["gpt-4o-mini", "claude-haiku-4-5-20251001"]
    escalation_model: str       # e.g. "gpt-4o"
    single_voter_policy: str = "escalate"
    reject_on_low_agreement: bool = False
    fabricated_threshold: float = 0.25
    weak_threshold: float = 0.60
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "policy_id": self.policy_id,
            "scorer_name": self.scorer_name,
            "theta": self.theta,
            "voter_models": list(self.voter_models),
            "escalation_model": self.escalation_model,
            "single_voter_policy": self.single_voter_policy,
            "reject_on_low_agreement": self.reject_on_low_agreement,
            "fabricated_threshold": self.fabricated_threshold,
            "weak_threshold": self.weak_threshold,
            "notes": self.notes,
        }


# PolicyEvaluationResult

class PolicyEvaluationResult(BaseModel):
    """
    Metrics produced by evaluating a RoutingPolicySpec against a labeled dataset.

    Serialized to ``policy_results.jsonl`` by PolicyEvaluationStore.

    ``selection_objective`` is a configurable scalar proxy used by
    ``select_policy.py`` to rank candidates:

        selection_objective = alpha * escalation_rate + (1 - alpha) * false_accept_rate

    This is NOT a principled cost model — it combines two rates with a
    user-specified alpha.  The alpha used is echoed in the record for
    traceability.
    """

    model_config = {"extra": "ignore"}

    # Identity
    policy_id: str
    scorer_name: str
    theta: float
    records_file: str = ""          # source JSONL path (traceability)

    # Dataset counts
    n_records: int = 0
    n_agreement_gate: int = 0
    n_labeled: int = 0
    n_acceptable: int = 0           # keep_ok == True

    # Metrics (None when no labeled data)
    keep_precision: float | None = None
    false_accept_rate: float | None = None
    recall: float | None = None
    keep_rate: float | None = None
    escalation_rate: float | None = None    # computed from all AG records

    # Selection objective proxy
    selection_objective: float | None = None
    selection_objective_alpha: float | None = None  # alpha used when computing

    warnings: list[str] = []


# PolicySelectionResult

class PolicySelectionResult(BaseModel):
    """
    Output of the PuLP ILP optimizer in select_policy.py.

    Includes a compact summary of the winning policy so that the result can be
    fully interpreted without reopening multiple files.
    """

    model_config = {"extra": "ignore"}

    # Selection outcome
    selected_policy_id: str | None          # None if no feasible policy found
    feasible: bool                          # True if constraints were satisfiable
    candidates_evaluated: int
    candidates_feasible: int
    constraints: dict                       # {"max_false_accept": 0.10, "min_recall": 0.70}
    selection_objective_alpha: float        # alpha used for selection_objective

    # Compact winner summary
    selected_scorer_name: str | None = None
    selected_theta: float | None = None
    selected_false_accept_rate: float | None = None
    selected_recall: float | None = None
    selected_escalation_rate: float | None = None
    selected_selection_objective: float | None = None

    # Runner-up and diagnostics
    runner_up_policy_id: str | None = None
    infeasibility_reason: str | None = None
    notes: str = ""


# PolicyEvaluationStore

class PolicyEvaluationStore:
    """
    Static JSONL I/O for PolicyEvaluationResult.

    Mirrors the RoutingDataset pattern: one JSON object per line, append-safe.
    """

    @staticmethod
    def save(results: list[PolicyEvaluationResult], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(r.model_dump_json() + "\n")

    @staticmethod
    def append(result: PolicyEvaluationResult, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(result.model_dump_json() + "\n")

    @staticmethod
    def load(path: str | Path) -> list[PolicyEvaluationResult]:
        return list(PolicyEvaluationStore.iter_results(path))

    @staticmethod
    def iter_results(path: str | Path):
        path = Path(path)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield PolicyEvaluationResult.model_validate_json(line)
