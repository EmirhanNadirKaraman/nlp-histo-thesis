"""ThresholdOptimizer — LP-based threshold selection for the hybrid agreement scorer."""
from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from .dataset import DatasetRecord
from pipeline.stages.knowledge_extraction.interfaces.scoring import ChunkDecision

logger = logging.getLogger(__name__)


@dataclass
class OptimizedThresholds:
    """
    Decision thresholds for CascadedCompositeScorer, produced by ThresholdOptimizer.

    Decision rule
    -------------
    KEEP    : emb >= keep_emb  AND  ner >= keep_ner
    REJECT  : emb <= reject_emb  (and not KEEP)
    ESCALATE: everything else

    Fields
    ------
    keep_emb, keep_ner:
        Minimum embedding / NER scores required to accept a chunk.
    reject_emb:
        Maximum embedding score below which a chunk is rejected outright.
    cost_achieved:
        Expected per-chunk cost under the optimal thresholds on the training set.
    metrics:
        Dict with precision_keep, recall_keep, precision_reject on training set.
    lp_feasible:
        True if the LP found a solution satisfying all accuracy constraints.
    """

    keep_emb: float
    keep_ner: float
    reject_emb: float
    cost_achieved: float
    metrics: dict
    lp_feasible: bool = True

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        logger.info("Saved thresholds to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "OptimizedThresholds":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class ThresholdOptimizer:
    """
    Finds KEEP / REJECT / ESCALATE thresholds by solving a linear program.

    Problem formulation
    -------------------
    Given a labeled dataset of (embedding_score, ner_score, gold_label) tuples,
    enumerate K candidate threshold combinations on a discrete grid.  For each
    combination k = (α, β, γ):

        KEEP    : emb >= α  AND  ner >= β
        REJECT  : emb <= γ  (and not KEEP)
        ESCALATE: otherwise

    Define for each k:
        cost_k            = expected cost per chunk
        precision_keep_k  = |TP_keep| / |pred_keep|
        recall_keep_k     = |TP_keep| / |gold_keep|
        precision_reject_k = |TP_reject| / |pred_reject|

    LP (K variables x_k, convex combination selector):
        minimize   cost · x
        subject to
            −precision_keep · x  ≤ −min_precision_keep
            −recall_keep · x     ≤ −min_recall_keep
            −precision_reject · x ≤ −min_precision_reject
            1ᵀ x = 1,  x ≥ 0

    The LP solution is an extreme point of the feasible simplex, selecting the
    single cheapest threshold combination that satisfies all accuracy constraints.
    argmax(x) identifies that combination.

    If the LP is infeasible (no combination meets all constraints), a fallback
    selects the combination with the best weighted accuracy.

    Parameters
    ----------
    cost_keep:
        Cost assigned to a KEEP decision (no extra LLM call).  Default 1.0.
    cost_escalate:
        Cost assigned to an ESCALATE decision (triggers expensive LLM call).
        Default 10.0.
    cost_reject:
        Cost assigned to a REJECT decision.  Default 1.0.
    min_precision_keep:
        Minimum required precision on KEEP predictions.  Default 0.85.
    min_recall_keep:
        Minimum required recall on KEEP predictions.  Default 0.75.
    min_precision_reject:
        Minimum required precision on REJECT predictions.  Default 0.85.
    n_grid_points:
        Number of candidate values per threshold axis.  Default 15.
    """

    def __init__(
        self,
        cost_keep: float = 1.0,
        cost_escalate: float = 10.0,
        cost_reject: float = 1.0,
        min_precision_keep: float = 0.85,
        min_recall_keep: float = 0.75,
        min_precision_reject: float = 0.85,
        n_grid_points: int = 15,
    ) -> None:
        self.cost_keep = cost_keep
        self.cost_escalate = cost_escalate
        self.cost_reject = cost_reject
        self.min_precision_keep = min_precision_keep
        self.min_recall_keep = min_recall_keep
        self.min_precision_reject = min_precision_reject
        self.n_grid_points = n_grid_points

    def fit(self, records: list[DatasetRecord]) -> OptimizedThresholds:
        """
        Solve the LP and return the optimal thresholds.

        Parameters
        ----------
        records:
            Labeled dataset produced by DatasetCollector.

        Returns
        -------
        OptimizedThresholds with the selected keep_emb, keep_ner, reject_emb,
        achieved cost, per-class metrics, and feasibility flag.
        """
        if not records:
            raise ValueError("Cannot fit on an empty dataset.")

        emb = np.array([r.embedding_score for r in records])
        ner = np.array([r.ner_score for r in records])
        gold_keep = np.array([r.gold_label == ChunkDecision.KEEP for r in records])
        gold_reject = np.array([r.gold_label == ChunkDecision.REJECT for r in records])
        n = len(records)

        grid = np.linspace(0.0, 1.0, self.n_grid_points)
        combos: list[tuple[float, float, float]] = []

        for alpha, beta, gamma in product(grid, grid, grid):
            combos.append((float(alpha), float(beta), float(gamma)))

        K = len(combos)
        logger.info("Evaluating %d threshold combinations on %d records", K, n)

        costs = np.empty(K)
        prec_keep = np.empty(K)
        rec_keep = np.empty(K)
        prec_reject = np.empty(K)

        for k, (alpha, beta, gamma) in enumerate(combos):
            pred_keep = (emb >= alpha) & (ner >= beta)
            pred_reject = (emb <= gamma) & ~pred_keep
            pred_escalate = ~pred_keep & ~pred_reject

            costs[k] = (
                self.cost_keep * pred_keep.sum()
                + self.cost_reject * pred_reject.sum()
                + self.cost_escalate * pred_escalate.sum()
            ) / n

            n_pred_keep = pred_keep.sum()
            n_gold_keep = gold_keep.sum()
            if n_pred_keep > 0:
                prec_keep[k] = float((pred_keep & gold_keep).sum()) / n_pred_keep
            else:
                prec_keep[k] = 1.0  # vacuously precise; recall penalises this

            rec_keep[k] = (
                float((pred_keep & gold_keep).sum()) / n_gold_keep
                if n_gold_keep > 0
                else 1.0
            )

            n_pred_reject = pred_reject.sum()
            n_gold_reject = gold_reject.sum()
            if n_pred_reject > 0:
                prec_reject[k] = float((pred_reject & gold_reject).sum()) / n_pred_reject
            elif n_gold_reject == 0:
                prec_reject[k] = 1.0  # no reject gold; any threshold is vacuously fine
            else:
                prec_reject[k] = 1.0  # no reject predictions → vacuously precise

        # LP: minimize cost · x
        # Inequality constraints (A_ub x ≤ b_ub):
        #   -prec_keep  · x ≤ -min_precision_keep
        #   -rec_keep   · x ≤ -min_recall_keep
        #   -prec_reject· x ≤ -min_precision_reject
        A_ub = np.vstack([-prec_keep, -rec_keep, -prec_reject])
        b_ub = np.array(
            [-self.min_precision_keep, -self.min_recall_keep, -self.min_precision_reject]
        )
        A_eq = np.ones((1, K))
        b_eq = np.array([1.0])
        bounds = [(0.0, 1.0)] * K

        lp_result = linprog(
            costs,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        lp_feasible = lp_result.success
        if lp_feasible:
            best_k = int(np.argmax(lp_result.x))
            logger.info(
                "LP solution: k=%d cost=%.3f (status: %s)",
                best_k,
                lp_result.fun,
                lp_result.message,
            )
        else:
            warnings.warn(
                f"LP infeasible ({lp_result.message}). "
                "Falling back to best achievable accuracy.",
                stacklevel=2,
            )
            # Fallback: maximise weighted accuracy sum ignoring cost constraints
            accuracy_score = (
                prec_keep * self.min_precision_keep
                + rec_keep * self.min_recall_keep
                + prec_reject * self.min_precision_reject
            )
            best_k = int(np.argmax(accuracy_score))
            logger.info("Fallback best_k=%d", best_k)

        alpha_star, beta_star, gamma_star = combos[best_k]
        metrics = {
            "precision_keep": float(prec_keep[best_k]),
            "recall_keep": float(rec_keep[best_k]),
            "precision_reject": float(prec_reject[best_k]),
        }
        logger.info(
            "Optimal thresholds: keep_emb=%.3f keep_ner=%.3f reject_emb=%.3f | "
            "precision_keep=%.3f recall_keep=%.3f precision_reject=%.3f cost=%.3f",
            alpha_star, beta_star, gamma_star,
            metrics["precision_keep"],
            metrics["recall_keep"],
            metrics["precision_reject"],
            float(costs[best_k]),
        )
        return OptimizedThresholds(
            keep_emb=alpha_star,
            keep_ner=beta_star,
            reject_emb=gamma_star,
            cost_achieved=float(costs[best_k]),
            metrics=metrics,
            lp_feasible=lp_feasible,
        )
