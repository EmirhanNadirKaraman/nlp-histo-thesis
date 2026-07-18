"""ILP-based selectors for the related / diverse / hard buckets.

PuLP-backed alternatives to the greedy selectors in :mod:`selectors`. Each
function returns the same ``(papers, rationale)`` shape so the rest of the
pipeline (export, validation) is strategy-agnostic.

PuLP is an optional dependency — call sites can fall back to greedy when it is
not installed; see :func:`pulp_available`.

Design notes
------------
The two main scaling levers are *candidate pruning* (how many papers reach the
solver per bucket) and *edge sparsification* (how many pair variables are
materialised). For the related bucket, dense O(n²) pair variables were the
historical bottleneck; this module keeps only top-M neighbour edges above a
threshold. For the diverse bucket, pair variables are restricted to
strongly-related pairs, and per-bucket concept caps prevent CUI or generic-
entity inventories from dominating the coverage objective.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

from .metrics import Hardness, Relatedness
from .models import HardnessBreakdown, PaperFingerprint, SelectionResult
from .selectors import (
    SelectionConfig,
    _eligible_short,
    select_calibration_set,
)

logger = logging.getLogger(__name__)


# PuLP availability

try:
    import pulp as _pulp  # type: ignore
    _PULP_IMPORT_ERROR: Exception | None = None
except ImportError as _e:  # pragma: no cover — exercised only without PuLP
    _pulp = None  # type: ignore
    _PULP_IMPORT_ERROR = _e


def pulp_available() -> bool:
    """True iff PuLP can be imported in the current environment."""
    return _pulp is not None


def _require_pulp() -> None:
    if _pulp is None:
        raise ImportError(
            "PuLP is required for ILP-based selection. "
            "Install with `pip install pulp`, or pass --strategy greedy."
        ) from _PULP_IMPORT_ERROR


# ILP-specific knobs

@dataclass
class ILPConfig:
    """Numeric weights and limits specific to the ILP selectors.

    These do not replace :class:`SelectionConfig`; they extend it. Weights are
    chosen so each objective is roughly normalised to the same magnitude as
    the corresponding greedy heuristic — small enough to keep solver time
    short, large enough to discriminate tied candidates.
    """
    # Candidate pool sizing
    # Per-bucket caps applied before building the ILP.
    related_candidate_limit: int = 80
    diverse_candidate_limit: int = 100
    hard_candidate_limit:    int = 120
    # Backward-compat override. When non-None, applies to all three buckets
    # (use 0 to disable capping). Prefer the per-bucket fields above.
    candidate_limit: int | None = None

    # Solver knobs
    # Solver time budget (seconds, per-bucket). None = no limit.
    time_limit_seconds: float | None = 30.0
    # Stream CBC's own progress (LP root, cuts, nodes, gap) to stdout.
    solver_verbose: bool = False
    # Accept the best feasible (non-Optimal) solution iff CBC found one and
    # it has exactly k papers selected. Falls back to prescore otherwise.
    accept_feasible: bool = True

    # Related
    related_quality_weight: float = 0.05   # × paper_quality_score(i)
    related_length_penalty: float = 0.02   # × normalised n_sentences
    # Edge sparsification: keep an edge (i,j) iff its relatedness score is
    # ≥ threshold AND it appears in the top-M neighbours of i or j. Caps the
    # number of y variables at roughly |pool|·M.
    related_pair_top_m:    int   = 15
    related_pair_threshold: float = 0.01

    # Diverse
    diverse_concept_weights: dict[str, float] = field(
        default_factory=lambda: {
            "disease":   0.30,
            "biomarker": 0.20,
            "gene":      0.15,
            "tissue":    0.10,
            "method":    0.10,
            "outcome":   0.10,
            "cui":       0.01,   # tiny by design; CUIs are noisy and many
        }
    )
    # Cap concepts per bucket before creating z variables — keeps the model
    # small and prevents one rich bucket (typically CUIs) from drowning others.
    diverse_max_concepts_by_bucket: dict[str, int] = field(
        default_factory=lambda: {
            "disease":   80,
            "biomarker": 80,
            "gene":      80,
            "tissue":    50,
            "method":    50,
            "outcome":   50,
            "cui":       30,
        }
    )
    # Hard upper bound on the total reward a single bucket can contribute to
    # the diversity objective, regardless of concept count.
    diverse_bucket_reward_caps: dict[str, float] = field(
        default_factory=lambda: {
            "disease":   3.0,
            "biomarker": 2.0,
            "gene":      1.5,
            "tissue":    1.0,
            "method":    1.0,
            "outcome":   1.0,
            "cui":       0.3,
        }
    )
    diverse_pairwise_threshold: float = 0.50
    diverse_pairwise_penalty:   float = 0.0
    diverse_near_duplicate_threshold: float = 0.65
    diverse_near_duplicate_penalty:   float = 1.0
    # CUIs are off by default; raw count vastly outpaces structured buckets.
    diverse_include_cuis: bool = False

    # Hard
    hard_top_normalized_pool: int = 30
    hard_top_absolute_pool:   int = 30
    hard_medium_pool_low_pct: float = 0.40
    hard_medium_pool_high_pct: float = 0.65


# Solver helpers

def _solve(prob: "_pulp.LpProblem", time_limit_seconds: float | None,
           tag: str = "", verbose: bool = False) -> int:
    """Run CBC (PuLP's bundled solver) with optional time budget. Returns status."""
    solver = _pulp.PULP_CBC_CMD(
        msg=1 if verbose else 0,
        timeLimit=time_limit_seconds if time_limit_seconds and time_limit_seconds > 0 else None,
    )
    n_vars = len(prob.variables())
    n_cons = len(prob.constraints)
    label = f"[{tag}] " if tag else ""
    logger.info("%sCBC solve: vars=%d constraints=%d time_limit=%s",
                label, n_vars, n_cons,
                f"{time_limit_seconds}s" if time_limit_seconds else "unlimited")
    t0 = time.perf_counter()
    status = prob.solve(solver)
    obj = _pulp.value(prob.objective)
    logger.info("%sCBC solve: done in %.2fs status=%s objective=%s",
                label, time.perf_counter() - t0,
                _pulp.LpStatus[status],
                f"{obj:.4f}" if obj is not None else "None")
    return status


def _selected_pmcids(papers: list[PaperFingerprint],
                     x: dict[str, "_pulp.LpVariable"]) -> list[str]:
    chosen: list[str] = []
    for p in papers:
        v = x.get(p.pmcid)
        if v is None:
            continue
        val = _pulp.value(v)
        if val is not None and val > 0.5:
            chosen.append(p.pmcid)
    return chosen


def _is_optimal(status: int) -> bool:
    """True iff the CBC status code corresponds to a valid optimal solution."""
    return _pulp.LpStatus[status] == "Optimal"


def _extract_valid_solution(
    pool: list[PaperFingerprint],
    x: dict[str, "_pulp.LpVariable"],
    k: int,
) -> list[str] | None:
    """Return the chosen PMCIDs iff CBC selected exactly ``k`` of them.

    Variable values are only meaningful when the solver finished with at least
    a feasible solution. Callers are responsible for checking solver status
    before deciding whether to trust this output.
    """
    chosen = _selected_pmcids(pool, x)
    return chosen if len(chosen) == k else None


def _validate_selection_count(chosen: list, k: int, tag: str) -> bool:
    if len(chosen) != k:
        logger.warning("[%s] selection count mismatch: got %d, need %d",
                       tag, len(chosen), k)
        return False
    return True


def _prescore_fallback(
    pool: list[PaperFingerprint],
    *,
    k: int,
    key,
    tag: str,
    status: int,
    reason: str | None = None,
) -> tuple[list[PaperFingerprint], dict[str, dict]]:
    """Return the top-k by prescore when the ILP didn't yield a usable solution.

    Returns exactly ``k`` papers when ``len(pool) >= k``; otherwise returns
    everything available. Callers cap the pool size earlier; this helper just
    enforces a defensible last-resort ordering.
    """
    status_label = _pulp.LpStatus[status]
    why = reason or f"prescore top-k fallback (ILP status={status_label})"
    logger.warning("[%s] ILP fallback engaged (status=%s) — top-%d by prescore",
                   tag, status_label, k)
    ranked = sorted(pool, key=lambda p: (-key(p), p.pmcid))[:k]
    rationale: dict[str, dict] = {
        p.pmcid: {
            "rank": i + 1,
            "selection_reason": why,
            "ilp_status": status_label,
            "ilp_pool_size": len(pool),
            "ilp_solution_quality": "prescore_fallback",
        }
        for i, p in enumerate(ranked)
    }
    return ranked, rationale


# Candidate pre-scoring

def _related_prescore(p: PaperFingerprint) -> float:
    """Favour short, entity-rich papers with disease/biomarker/gene anchors."""
    bucket_richness = (
        len(p.disease_entities) + len(p.biomarker_entities) + len(p.gene_entities)
    )
    length = max(p.n_sentences, 1)
    return bucket_richness / math.log1p(length + 10)


def _diverse_prescore(p: PaperFingerprint) -> float:
    """Favour entity-rich, manageable-size papers."""
    vocab = (
        len(p.disease_entities) + len(p.method_entities)
        + len(p.biomarker_entities) + len(p.outcome_entities)
        + len(p.tissue_entities)
    )
    length = max(p.n_sentences, 1)
    return vocab / math.log1p(length / 50 + 1)


def _hard_prescore(p: PaperFingerprint, hardness: Hardness) -> float:
    return hardness.breakdown(p).absolute_hardness


def _resolved_limit(ilp: ILPConfig, bucket: str) -> int:
    """Resolve the candidate cap for a bucket, honoring the backward-compat override."""
    if ilp.candidate_limit is not None:
        return ilp.candidate_limit
    per_bucket = {
        "related": ilp.related_candidate_limit,
        "diverse": ilp.diverse_candidate_limit,
        "hard":    ilp.hard_candidate_limit,
    }
    return per_bucket[bucket]


def _limit_candidates(
    papers: list[PaperFingerprint],
    *,
    limit: int,
    key,
    tag: str = "",
) -> list[PaperFingerprint]:
    label = f"[{tag}] " if tag else ""
    if limit <= 0:
        logger.debug("%s_limit_candidates: limit<=0 → keeping all %d papers",
                     label, len(papers))
        return list(papers)
    if len(papers) <= limit:
        logger.debug("%s_limit_candidates: %d ≤ limit %d → keeping all",
                     label, len(papers), limit)
        return list(papers)
    scored = sorted(papers, key=lambda p: (-key(p), p.pmcid))
    logger.info("%sprescore cutoff: %d → top %d", label, len(papers), limit)
    return scored[:limit]


def _paper_quality_score(p: PaperFingerprint) -> float:
    """Crude, deterministic quality signal in roughly [0, 1].

    Rewards papers that have well-rounded structured-entity coverage (each
    bucket contributes at most 0.2). Independent of length.
    """
    buckets = (
        p.disease_entities, p.biomarker_entities, p.gene_entities,
        p.tissue_entities, p.method_entities, p.outcome_entities,
    )
    return sum(min(len(b) / 5.0, 0.2) for b in buckets)


def _length_penalty(p: PaperFingerprint, ref_sentences: int = 350) -> float:
    """Normalised length penalty in [0, 1+]."""
    return min(p.n_sentences / max(ref_sentences, 1), 2.0)


# Related ILP

def _retained_related_edges(
    pool: list[PaperFingerprint],
    rel_metric: Relatedness,
    *,
    top_m: int,
    threshold: float,
) -> tuple[dict[tuple[str, str], float], int]:
    """Return the sparse edge set used by the related ILP.

    Edge ``(a, b)`` is retained iff ``rel(a, b) >= threshold`` AND it is among
    the top-``top_m`` neighbours (by score) of either endpoint. This keeps the
    number of pair variables at roughly ``|pool|·top_m`` instead of ``|pool|²``.

    Returns ``(edges, scanned_pairs)`` where ``scanned_pairs`` is the total
    number of pair scores computed (always n·(n−1)/2).
    """
    n = len(pool)
    scanned = n * (n - 1) // 2
    # Per-node top-M (pmcid → list of (score, neighbour_pmcid), sorted desc).
    top: list[list[tuple[float, str]]] = [[] for _ in range(n)]
    pmcids = [p.pmcid for p in pool]
    for i in range(n):
        for j in range(i + 1, n):
            s = rel_metric.score(pool[i], pool[j])
            if s < threshold:
                continue
            # Keep a bounded heap-style top-M per endpoint.
            for owner, neighbour in ((i, j), (j, i)):
                top[owner].append((s, pmcids[neighbour]))
    for owner_list in top:
        owner_list.sort(key=lambda t: (-t[0], t[1]))
        del owner_list[top_m:]

    # Union of endpoint top-M edges, deduplicated (unordered pair).
    edges: dict[tuple[str, str], float] = {}
    for i, owner_list in enumerate(top):
        for s, neighbour in owner_list:
            a, b = sorted((pmcids[i], neighbour))
            edges[(a, b)] = s
    return edges, scanned


def select_related_papers_ilp(
    fingerprints: list[PaperFingerprint],
    *,
    k: int = 5,
    config: SelectionConfig | None = None,
    ilp_config: ILPConfig | None = None,
    exclude_pmcids: set[str] | None = None,
) -> tuple[list[PaperFingerprint], dict[str, dict]]:
    """ILP version of related-paper selection.

    Decision variables::

        x_i ∈ {0,1}                     — paper i selected
        y_ij ∈ [0,1] (continuous)       — both i and j selected, for retained edges

    Edges are restricted to the union of each node's top-M neighbours with
    score ≥ ``related_pair_threshold``. Pair coefficients are non-negative on
    retained edges, so the two ``y_ij ≤ x_i, x_j`` links suffice for the LP
    to self-push y to its integer vertex.

    Objective (maximise)::

        Σ_{(i,j)∈E} rel(i,j)·y_ij
        + α · Σ paper_quality(i)·x_i
        − β · Σ length_penalty(i)·x_i
    """
    _require_pulp()
    cfg = config or SelectionConfig()
    ilp = ilp_config or ILPConfig()
    excluded = set(exclude_pmcids or [])

    t0 = time.perf_counter()
    eligible = [p for p in _eligible_short(fingerprints, cfg, cfg.max_sentences_related)
                if p.pmcid not in excluded]
    logger.info("[related] ILP: candidates=%d eligible=%d excluded=%d k=%d",
                len(fingerprints), len(eligible), len(excluded), k)
    pool = _limit_candidates(eligible, limit=_resolved_limit(ilp, "related"),
                             key=_related_prescore, tag="related")
    logger.info("[related] pool size after prescore: %d", len(pool))

    if len(pool) < k:
        logger.warning("select_related_ilp: only %d candidates after limiting (need %d)",
                       len(pool), k)
        if not pool:
            return [], {}
        # Cannot enforce =k when |pool| < k — return whatever we have.
        return pool[:k], {p.pmcid: {"rank": i + 1,
                                     "selection_reason": "insufficient ILP pool, returned without optimisation"}
                          for i, p in enumerate(pool[:k])}

    rel_metric = cfg.relatedness if isinstance(cfg.relatedness, Relatedness) else Relatedness()
    pmcids = [p.pmcid for p in pool]
    pidx = {pid: i for i, pid in enumerate(pmcids)}
    quality = {p.pmcid: _paper_quality_score(p) for p in pool}
    length  = {p.pmcid: _length_penalty(p) for p in pool}

    t_pairs = time.perf_counter()
    edges, scanned = _retained_related_edges(
        pool, rel_metric,
        top_m=ilp.related_pair_top_m,
        threshold=ilp.related_pair_threshold,
    )
    logger.info("[related] edge sparsification: scanned %d pairs in %.2fs → "
                "retained %d edges (top_m=%d, threshold=%.3f)",
                scanned, time.perf_counter() - t_pairs, len(edges),
                ilp.related_pair_top_m, ilp.related_pair_threshold)

    prob = _pulp.LpProblem("related_ilp", _pulp.LpMaximize)
    # Index-based variable names — PuLP rejects some characters in PMCIDs.
    x = {pid: _pulp.LpVariable(f"x_{i}", cat=_pulp.LpBinary)
         for i, pid in enumerate(pmcids)}
    edge_list = sorted(edges.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    y = {pair: _pulp.LpVariable(f"y_{i}", lowBound=0, upBound=1)
         for i, (pair, _s) in enumerate(edge_list)}

    pair_term    = _pulp.lpSum(edges[pair] * y[pair] for pair in y)
    quality_term = _pulp.lpSum(quality[pid] * x[pid] for pid in pmcids)
    length_term  = _pulp.lpSum(length[pid] * x[pid] for pid in pmcids)
    prob += (
        pair_term
        + ilp.related_quality_weight * quality_term
        - ilp.related_length_penalty * length_term
    )
    prob += _pulp.lpSum(x[pid] for pid in pmcids) == k

    for (a, b) in y:
        prob += y[(a, b)] <= x[a]
        prob += y[(a, b)] <= x[b]

    status = _solve(prob, ilp.time_limit_seconds, tag="related",
                    verbose=ilp.solver_verbose)
    chosen_ids = _extract_valid_solution(pool, x, k)
    used_feasible = False
    if _is_optimal(status) and chosen_ids is not None:
        quality_label = "optimal"
    elif ilp.accept_feasible and chosen_ids is not None:
        quality_label = "feasible_nonoptimal"
        used_feasible = True
        logger.info("[related] accepting feasible (non-optimal) solution status=%s",
                    _pulp.LpStatus[status])
    else:
        return _prescore_fallback(pool, k=k, key=_related_prescore,
                                  tag="related", status=status)

    by_pid = {p.pmcid: p for p in pool}
    chosen = [by_pid[pid] for pid in chosen_ids]
    chosen.sort(key=lambda p: pidx[p.pmcid])

    obj = _pulp.value(prob.objective)
    pair_reward    = _pulp.value(pair_term)    or 0.0
    quality_reward = _pulp.value(quality_term) or 0.0
    length_reward  = _pulp.value(length_term)  or 0.0
    logger.info("[related] ILP picked %d in %.2fs — pair_reward=%.4f "
                "quality_reward=%.4f length_penalty=%.4f total_obj=%s — %s",
                len(chosen), time.perf_counter() - t0,
                pair_reward, quality_reward, length_reward,
                f"{obj:.4f}" if obj is not None else "None",
                [p.pmcid for p in chosen])
    rationale: dict[str, dict] = {}
    for rank, p in enumerate(chosen, start=1):
        rationale[p.pmcid] = {
            "rank": rank,
            "selection_reason": (
                f"ILP related: quality={quality[p.pmcid]:.3f}, "
                f"length_penalty={length[p.pmcid]:.3f}"
            ),
            "ilp_status": _pulp.LpStatus[status],
            "ilp_objective": float(obj) if obj is not None else None,
            "ilp_pool_size": len(pool),
            "ilp_n_pair_edges": len(edges),
            "ilp_solution_quality": quality_label,
            "ilp_pair_reward":    float(pair_reward),
            "ilp_quality_reward": float(quality_reward),
            "ilp_length_penalty": float(length_reward),
        }
    if used_feasible:
        for entry in rationale.values():
            entry["selection_reason"] = (
                entry["selection_reason"] + " [feasible-non-optimal]"
            )
    return chosen, rationale


# Diverse ILP

# Concepts pulled from each fingerprint, prefixed by bucket. Prefixing keeps
# concept identifiers globally unique and lets us weight per-bucket directly.
_DIVERSE_BUCKETS = (
    ("disease",   "disease_entities"),
    ("biomarker", "biomarker_entities"),
    ("gene",      "gene_entities"),
    ("tissue",    "tissue_entities"),
    ("method",    "method_entities"),
    ("outcome",   "outcome_entities"),
)


def _concepts_for(p: PaperFingerprint, *,
                  include_cuis: bool) -> dict[str, set[str]]:
    """Bucket → set of concept tokens for one paper."""
    out: dict[str, set[str]] = {}
    for bucket, attr in _DIVERSE_BUCKETS:
        out[bucket] = {f"{bucket}::{e.lower()}" for e in getattr(p, attr)}
    if include_cuis:
        out["cui"] = {f"cui::{c}" for c in p.umls_cuis}
    return out


def _rare_concept_bonus(
    papers: list[PaperFingerprint], *, include_cuis: bool,
) -> dict[str, float]:
    """Per-paper bonus: Σ 1 / frequency(c) over concepts the paper contains.

    Rare concepts contribute more than common ones; ensures the diverse pool
    keeps papers that hold uncommon-but-useful concepts even if their overall
    entity vocabulary is small.
    """
    frequency: dict[str, int] = {}
    paper_concepts: dict[str, set[str]] = {}
    for p in papers:
        all_concepts: set[str] = set()
        for s in _concepts_for(p, include_cuis=include_cuis).values():
            all_concepts |= s
        paper_concepts[p.pmcid] = all_concepts
        for c in all_concepts:
            frequency[c] = frequency.get(c, 0) + 1
    bonus: dict[str, float] = {}
    for p in papers:
        bonus[p.pmcid] = sum(
            1.0 / max(frequency.get(c, 1), 1) for c in paper_concepts[p.pmcid]
        )
    return bonus


def _diverse_combined_prescore_key(
    p: PaperFingerprint, rare_bonus: dict[str, float],
) -> float:
    return _diverse_prescore(p) + rare_bonus.get(p.pmcid, 0.0)


def _cap_concepts(
    concept_to_papers: dict[str, list[str]],
    concept_bucket: dict[str, str],
    *,
    max_per_bucket: dict[str, int],
) -> dict[str, list[str]]:
    """Keep at most ``max_per_bucket[bucket]`` concepts per bucket.

    Concepts are ranked by ``(coverage_count_descending, concept_name)`` —
    favours concepts that several pool papers share (otherwise the z variable
    can never activate via more than one paper) while staying deterministic.
    """
    by_bucket: dict[str, list[tuple[int, str]]] = {}
    for c, owners in concept_to_papers.items():
        bucket = concept_bucket[c]
        by_bucket.setdefault(bucket, []).append((len(owners), c))
    kept: dict[str, list[str]] = {}
    for bucket, items in by_bucket.items():
        items.sort(key=lambda t: (-t[0], t[1]))
        cap = max_per_bucket.get(bucket, len(items))
        for _, c in items[:cap]:
            kept[c] = concept_to_papers[c]
    return kept


def _effective_concept_rewards(
    concepts: dict[str, list[str]],
    concept_bucket: dict[str, str],
    *,
    base_weights: dict[str, float],
    reward_caps: dict[str, float],
) -> dict[str, float]:
    """Per-concept reward, capped so each bucket cannot exceed ``reward_caps[b]``.

    For bucket ``b`` with ``n_b`` retained concepts, the per-concept reward is
    ``min(base_weights[b], reward_caps[b] / max(n_b, 1))``. Total bucket reward
    is therefore ≤ ``reward_caps[b]`` regardless of how many concepts survive.
    """
    counts: dict[str, int] = {}
    for c in concepts:
        counts[concept_bucket[c]] = counts.get(concept_bucket[c], 0) + 1
    per_concept: dict[str, float] = {}
    for c in concepts:
        b = concept_bucket[c]
        base = base_weights.get(b, 0.0)
        cap = reward_caps.get(b)
        if cap is None or counts[b] == 0:
            per_concept[c] = base
        else:
            per_concept[c] = min(base, cap / counts[b])
    return per_concept


def select_diverse_papers_ilp(
    fingerprints: list[PaperFingerprint],
    *,
    k: int = 5,
    config: SelectionConfig | None = None,
    ilp_config: ILPConfig | None = None,
    exclude_pmcids: set[str] | None = None,
    include_cuis: bool | None = None,
) -> tuple[list[PaperFingerprint], dict[str, dict]]:
    """ILP version of diverse-paper selection.

    Variables::

        x_i ∈ {0,1}, y_ij ∈ [0,1], z_c ∈ [0,1]

    Constraints::

        Σ x_i = k
        z_c ≤ Σ_{i: c ∈ p_i} x_i              — coverage activation
        y_ij ≥ x_i + x_j − 1                   — near-duplicate / pair linkage
                                                 (only for retained pairs)

    Objective (maximise)::

        Σ_c eff_reward(c) · z_c
        − pairwise_pen · Σ rel(i,j) · y_ij
        − near_dup_pen · Σ_{rel(i,j) ≥ τ} y_ij

    Per-bucket caps keep the model small and prevent one rich bucket (e.g.
    raw UMLS CUIs) from dominating the coverage objective.
    """
    _require_pulp()
    cfg = config or SelectionConfig()
    ilp = ilp_config or ILPConfig()
    excluded = set(exclude_pmcids or [])
    include_cuis_effective = (
        include_cuis if include_cuis is not None else ilp.diverse_include_cuis
    )

    t0 = time.perf_counter()
    eligible = [p for p in _eligible_short(fingerprints, cfg, cfg.max_sentences_diverse)
                if p.pmcid not in excluded]
    logger.info("[diverse] ILP: candidates=%d eligible=%d excluded=%d k=%d "
                "include_cuis=%s",
                len(fingerprints), len(eligible), len(excluded), k,
                include_cuis_effective)

    rare_bonus = _rare_concept_bonus(eligible, include_cuis=include_cuis_effective)
    pool = _limit_candidates(
        eligible,
        limit=_resolved_limit(ilp, "diverse"),
        key=lambda p: _diverse_combined_prescore_key(p, rare_bonus),
        tag="diverse",
    )
    logger.info("[diverse] pool size after prescore (rare-rescue): %d", len(pool))

    if len(pool) < k:
        logger.warning("select_diverse_ilp: only %d candidates after limiting (need %d)",
                       len(pool), k)
        if not pool:
            return [], {}
        return pool[:k], {p.pmcid: {"rank": i + 1,
                                     "selection_reason": "insufficient ILP pool, returned without optimisation"}
                          for i, p in enumerate(pool[:k])}

    rel_metric = cfg.relatedness if isinstance(cfg.relatedness, Relatedness) else Relatedness()

    # Concept inventory per bucket
    paper_concepts: dict[str, dict[str, set[str]]] = {
        p.pmcid: _concepts_for(p, include_cuis=include_cuis_effective) for p in pool
    }
    concept_to_papers_full: dict[str, list[str]] = {}
    concept_bucket: dict[str, str] = {}
    for pid, by_bucket in paper_concepts.items():
        for bucket, concepts in by_bucket.items():
            for c in concepts:
                concept_to_papers_full.setdefault(c, []).append(pid)
                concept_bucket[c] = bucket

    concept_to_papers = _cap_concepts(
        concept_to_papers_full, concept_bucket,
        max_per_bucket=ilp.diverse_max_concepts_by_bucket,
    )
    n_dropped_concepts = len(concept_to_papers_full) - len(concept_to_papers)
    logger.info("[diverse] concept inventory: %d → %d after per-bucket caps "
                "(dropped %d)",
                len(concept_to_papers_full), len(concept_to_papers),
                n_dropped_concepts)

    concept_rewards = _effective_concept_rewards(
        concept_to_papers, concept_bucket,
        base_weights=ilp.diverse_concept_weights,
        reward_caps=ilp.diverse_bucket_reward_caps,
    )

    # Pair scores — only build a y_ij when the pair actually shows up in the
    # objective: above the pairwise threshold (always penalised) or above the
    # near-duplicate threshold (extra penalty).
    n_pairs_total = len(pool) * (len(pool) - 1) // 2
    logger.info("[diverse] scanning %d candidate pairs for retained edges",
                n_pairs_total)
    t_pairs = time.perf_counter()
    pair_rel: dict[tuple[str, str], float] = {}
    # If the pairwise penalty is zero, retaining pairs in
    # [pairwise_threshold, near_duplicate_threshold) would create y variables
    # with a 0 objective coefficient — dead vars and dead constraints. In that
    # case, skip them and only keep pairs above the near-duplicate threshold.
    if ilp.diverse_pairwise_penalty == 0.0:
        edge_cutoff = ilp.diverse_near_duplicate_threshold
    else:
        edge_cutoff = min(
            ilp.diverse_pairwise_threshold,
            ilp.diverse_near_duplicate_threshold,
        )
    for i, a in enumerate(pool):
        for j in range(i + 1, len(pool)):
            b = pool[j]
            s = rel_metric.score(a, b)
            if s >= edge_cutoff:
                pair_rel[(a.pmcid, b.pmcid)] = s
    near_dupes = [pair for pair, s in pair_rel.items()
                  if s >= ilp.diverse_near_duplicate_threshold]
    logger.info("[diverse] pair scan done in %.2fs — kept %d/%d retained pairs "
                "(%d near-duplicates, edge_cutoff=%.2f)",
                time.perf_counter() - t_pairs, len(pair_rel), n_pairs_total,
                len(near_dupes), edge_cutoff)

    prob = _pulp.LpProblem("diverse_ilp", _pulp.LpMaximize)
    pmcids = [p.pmcid for p in pool]
    x = {pid: _pulp.LpVariable(f"x_{i}", cat=_pulp.LpBinary)
         for i, pid in enumerate(pmcids)}
    concept_list = sorted(concept_to_papers.keys())
    z = {c: _pulp.LpVariable(f"z_{i}", lowBound=0, upBound=1)
         for i, c in enumerate(concept_list)}
    pair_list = sorted(pair_rel.keys())
    y = {pair: _pulp.LpVariable(f"y_{i}", lowBound=0, upBound=1)
         for i, pair in enumerate(pair_list)}

    coverage_term = _pulp.lpSum(concept_rewards[c] * z[c] for c in concept_list)
    pair_penalty_term = _pulp.lpSum(
        ilp.diverse_pairwise_penalty * pair_rel[pair] * y[pair] for pair in pair_list
    )
    near_dup_penalty_term = _pulp.lpSum(
        ilp.diverse_near_duplicate_penalty * y[pair] for pair in near_dupes
    )
    prob += coverage_term - pair_penalty_term - near_dup_penalty_term

    prob += _pulp.lpSum(x[pid] for pid in pmcids) == k

    for c in concept_list:
        prob += z[c] <= _pulp.lpSum(x[pid] for pid in concept_to_papers[c])

    for (a, b) in pair_list:
        prob += y[(a, b)] >= x[a] + x[b] - 1

    status = _solve(prob, ilp.time_limit_seconds, tag="diverse",
                    verbose=ilp.solver_verbose)
    chosen_ids = _extract_valid_solution(pool, x, k)
    used_feasible = False
    if _is_optimal(status) and chosen_ids is not None:
        quality_label = "optimal"
    elif ilp.accept_feasible and chosen_ids is not None:
        quality_label = "feasible_nonoptimal"
        used_feasible = True
        logger.info("[diverse] accepting feasible (non-optimal) solution status=%s",
                    _pulp.LpStatus[status])
    else:
        return _prescore_fallback(
            pool, k=k,
            key=lambda p: _diverse_combined_prescore_key(p, rare_bonus),
            tag="diverse", status=status,
        )

    by_pid = {p.pmcid: p for p in pool}
    chosen = [by_pid[pid] for pid in chosen_ids]
    chosen.sort(key=lambda p: p.pmcid)

    obj = _pulp.value(prob.objective)
    coverage_reward      = _pulp.value(coverage_term)         or 0.0
    pair_penalty_value   = _pulp.value(pair_penalty_term)     or 0.0
    near_dup_penalty_value = _pulp.value(near_dup_penalty_term) or 0.0
    chosen_set = set(chosen_ids)

    # Per-bucket coverage + reward diagnostics for the chosen set.
    covered_by_bucket: dict[str, list[str]] = {}
    reward_by_bucket:  dict[str, float] = {}
    for c, owners in concept_to_papers.items():
        if any(pid in chosen_set for pid in owners):
            b = concept_bucket[c]
            covered_by_bucket.setdefault(b, []).append(c)
            reward_by_bucket[b] = reward_by_bucket.get(b, 0.0) + concept_rewards[c]
    near_dup_in_chosen = sum(
        1 for (a, b) in near_dupes if a in chosen_set and b in chosen_set
    )

    logger.info("[diverse] ILP picked %d in %.2fs — coverage=%.4f "
                "pair_penalty=%.4f near_dup_penalty=%.4f total_obj=%s — %s",
                len(chosen), time.perf_counter() - t0,
                coverage_reward, pair_penalty_value, near_dup_penalty_value,
                f"{obj:.4f}" if obj is not None else "None",
                [p.pmcid for p in chosen])

    rationale: dict[str, dict] = {}
    for rank, p in enumerate(chosen, start=1):
        contributed = sum(1 for c, owners in concept_to_papers.items()
                          if p.pmcid in owners)
        rationale[p.pmcid] = {
            "rank": rank,
            "selection_reason": f"ILP diverse: contributes {contributed} concepts to coverage",
            "ilp_status": _pulp.LpStatus[status],
            "ilp_objective": float(obj) if obj is not None else None,
            "ilp_pool_size": len(pool),
            "ilp_n_concepts": len(concept_to_papers),
            "ilp_n_pair_edges": len(pair_rel),
            "ilp_n_near_dup_pairs": len(near_dupes),
            "ilp_solution_quality": quality_label,
            "ilp_coverage_reward":    float(coverage_reward),
            "ilp_pair_penalty":       float(pair_penalty_value),
            "ilp_near_duplicate_penalty": float(near_dup_penalty_value),
            "covered_concepts_by_bucket": {b: sorted(v) for b, v in covered_by_bucket.items()},
            "reward_by_bucket":           dict(reward_by_bucket),
            "near_duplicate_pair_count":  near_dup_in_chosen,
            "include_cuis":               include_cuis_effective,
        }
    if used_feasible:
        for entry in rationale.values():
            entry["selection_reason"] = (
                entry["selection_reason"] + " [feasible-non-optimal]"
            )
    return chosen, rationale


# Hard ILP

def select_hard_papers_ilp(
    fingerprints: list[PaperFingerprint],
    *,
    k: int = 5,
    config: SelectionConfig | None = None,
    ilp_config: ILPConfig | None = None,
    exclude_pmcids: set[str] | None = None,
    allow_overlap: bool = False,
) -> tuple[list[PaperFingerprint], dict[str, dict]]:
    """ILP version of hard-paper selection.

    Composition constraints::

        Σ x_i = k
        Σ_{i ∈ TopNorm}    x_i ≥ effective_hard_high_normalized
        Σ_{i ∈ TopAbs}     x_i ≥ effective_hard_high_absolute
        Σ_{i ∈ MediumBand} x_i ≥ effective_hard_medium_control

    The *effective* values are clamped to each sub-pool's available size and
    surfaced in the rationale so a too-small sub-pool is auditable.

    Objective: maximise Σ absolute_hardness(i) · x_i. Empty / near-empty
    papers are filtered before constraints are built. No pair variables.
    """
    _require_pulp()
    cfg = config or SelectionConfig()
    ilp = ilp_config or ILPConfig()
    excluded = set() if allow_overlap else set(exclude_pmcids or [])
    hardness = cfg.hardness

    t0 = time.perf_counter()
    candidates = [p for p in fingerprints
                  if p.pmcid not in excluded and not p.is_empty()]
    logger.info("[hard] ILP: candidates=%d excluded=%d non_empty=%d k=%d",
                len(fingerprints), len(excluded), len(candidates), k)
    if not candidates:
        return [], {}

    # Pre-prune by absolute hardness so the ILP is small even on 980-paper inputs.
    pool = _limit_candidates(
        candidates,
        limit=_resolved_limit(ilp, "hard"),
        key=lambda p: _hard_prescore(p, hardness),
        tag="hard",
    )
    logger.info("[hard] pool size after prescore: %d", len(pool))

    breakdowns: dict[str, HardnessBreakdown] = {p.pmcid: hardness.breakdown(p) for p in pool}
    by_norm = sorted(pool, key=lambda p: -breakdowns[p.pmcid].normalized_hardness)
    by_abs  = sorted(pool, key=lambda p: -breakdowns[p.pmcid].absolute_hardness)
    by_med  = sorted(pool, key=lambda p:  breakdowns[p.pmcid].normalized_hardness)

    top_norm = {p.pmcid for p in by_norm[: ilp.hard_top_normalized_pool]}
    top_abs  = {p.pmcid for p in by_abs[:  ilp.hard_top_absolute_pool]}

    n = len(by_med)
    lo = int(n * ilp.hard_medium_pool_low_pct)
    hi = max(int(n * ilp.hard_medium_pool_high_pct), lo + 1)
    medium_band = {p.pmcid for p in by_med[lo:hi]}
    logger.info("[hard] sub-pools: |top_norm|=%d |top_abs|=%d |medium_band|=%d",
                len(top_norm), len(top_abs), len(medium_band))

    if len(pool) < k:
        logger.warning("select_hard_ilp: only %d candidates after limiting (need %d)",
                       len(pool), k)
        return pool[:k], {p.pmcid: {"rank": i + 1,
                                     "selection_reason": "insufficient ILP pool, returned without optimisation"}
                          for i, p in enumerate(pool[:k])}

    pmcids = [p.pmcid for p in pool]
    prob = _pulp.LpProblem("hard_ilp", _pulp.LpMaximize)
    x = {pid: _pulp.LpVariable(f"x_{i}", cat=_pulp.LpBinary)
         for i, pid in enumerate(pmcids)}

    prob += _pulp.lpSum(breakdowns[pid].absolute_hardness * x[pid] for pid in pmcids)
    prob += _pulp.lpSum(x[pid] for pid in pmcids) == k

    composition_sum = (
        cfg.hard_high_normalized + cfg.hard_high_absolute + cfg.hard_medium_control
    )
    effective_norm = effective_abs = effective_med = 0
    if composition_sum > k:
        logger.info(
            "[hard] composition sum (%d=%d+%d+%d) > k=%d — skipping "
            "min_top_normalized / min_top_absolute / min_medium_band constraints",
            composition_sum, cfg.hard_high_normalized, cfg.hard_high_absolute,
            cfg.hard_medium_control, k,
        )
    else:
        if cfg.hard_high_normalized > 0 and top_norm:
            effective_norm = min(cfg.hard_high_normalized, len(top_norm))
            prob += _pulp.lpSum(x[pid] for pid in top_norm) >= effective_norm, \
                    "min_top_normalized"
            if effective_norm < cfg.hard_high_normalized:
                logger.warning("[hard] top_norm pool too small (%d < requested %d) — "
                               "effective_hard_high_normalized=%d",
                               len(top_norm), cfg.hard_high_normalized, effective_norm)
        if cfg.hard_high_absolute > 0 and top_abs:
            effective_abs = min(cfg.hard_high_absolute, len(top_abs))
            prob += _pulp.lpSum(x[pid] for pid in top_abs) >= effective_abs, \
                    "min_top_absolute"
            if effective_abs < cfg.hard_high_absolute:
                logger.warning("[hard] top_abs pool too small (%d < requested %d) — "
                               "effective_hard_high_absolute=%d",
                               len(top_abs), cfg.hard_high_absolute, effective_abs)
        if cfg.hard_medium_control > 0 and medium_band:
            effective_med = min(cfg.hard_medium_control, len(medium_band))
            prob += _pulp.lpSum(x[pid] for pid in medium_band) >= effective_med, \
                    "min_medium_band"
            if effective_med < cfg.hard_medium_control:
                logger.warning("[hard] medium_band pool too small (%d < requested %d) — "
                               "effective_hard_medium_control=%d",
                               len(medium_band), cfg.hard_medium_control, effective_med)

    status = _solve(prob, ilp.time_limit_seconds, tag="hard",
                    verbose=ilp.solver_verbose)
    chosen_ids = _extract_valid_solution(pool, x, k)
    used_feasible = False
    if _is_optimal(status) and chosen_ids is not None:
        quality_label = "optimal"
    elif ilp.accept_feasible and chosen_ids is not None:
        quality_label = "feasible_nonoptimal"
        used_feasible = True
        logger.info("[hard] accepting feasible (non-optimal) solution status=%s",
                    _pulp.LpStatus[status])
    else:
        return _prescore_fallback(pool, k=k,
                                  key=lambda p: _hard_prescore(p, hardness),
                                  tag="hard", status=status)

    by_pid = {p.pmcid: p for p in pool}
    chosen = [by_pid[pid] for pid in chosen_ids]
    chosen.sort(key=lambda p: -breakdowns[p.pmcid].absolute_hardness)

    obj = _pulp.value(prob.objective)
    logger.info("[hard] ILP picked %d in %.2fs — total_obj=%s — %s",
                len(chosen), time.perf_counter() - t0,
                f"{obj:.4f}" if obj is not None else "None",
                [p.pmcid for p in chosen])
    rationale: dict[str, dict] = {}
    for rank, p in enumerate(chosen, start=1):
        b = breakdowns[p.pmcid]
        bucket_tags: list[str] = []
        if p.pmcid in top_norm:
            bucket_tags.append("top_normalized")
        if p.pmcid in top_abs:
            bucket_tags.append("top_absolute")
        if p.pmcid in medium_band:
            bucket_tags.append("medium_band")
        rationale[p.pmcid] = {
            "rank": rank,
            "selection_reason": (
                f"ILP hard: absolute={b.absolute_hardness:.3f}, "
                f"normalized={b.normalized_hardness:.3f}, "
                f"sub-pools=[{','.join(bucket_tags) or 'none'}]"
            ),
            "ilp_status": _pulp.LpStatus[status],
            "ilp_objective": float(obj) if obj is not None else None,
            "ilp_pool_size": len(pool),
            "ilp_solution_quality": quality_label,
            "effective_hard_high_normalized": effective_norm,
            "effective_hard_high_absolute":   effective_abs,
            "effective_hard_medium_control":  effective_med,
            "hardness_breakdown": {
                "normalized_hardness": b.normalized_hardness,
                "absolute_hardness":   b.absolute_hardness,
                "content_complexity":  b.content_complexity,
                "layout_complexity":   b.layout_complexity,
                "relation_complexity": b.relation_complexity,
                "workload_factor":     b.workload_factor,
                "reasons":             list(b.reasons),
            },
        }
    if used_feasible:
        for entry in rationale.values():
            entry["selection_reason"] = (
                entry["selection_reason"] + " [feasible-non-optimal]"
            )
    return chosen, rationale


# Calibration set (ILP-end-to-end)

def select_calibration_set_ilp(
    fingerprints: list[PaperFingerprint],
    *,
    config: SelectionConfig | None = None,
    ilp_config: ILPConfig | None = None,
    allow_overlap: bool = False,
    fallback_to_greedy: bool = False,
) -> SelectionResult:
    """ILP-driven 15-paper calibration set.

    Order: related → diverse (excluding related) → hard (excluding both),
    matching the greedy version. With ``fallback_to_greedy=True``, falls back
    silently if PuLP is missing.
    """
    if not pulp_available():
        if fallback_to_greedy:
            logger.warning("PuLP not installed — falling back to greedy selectors.")
            return select_calibration_set(fingerprints, config=config,
                                          allow_overlap=allow_overlap)
        _require_pulp()  # raises with the install hint

    cfg = config or SelectionConfig()
    ilp = ilp_config or ILPConfig()
    t_total = time.perf_counter()

    logger.info("=== Calibration set (ILP) — stage 1/3: related ===")
    related, related_r = select_related_papers_ilp(
        fingerprints, k=cfg.k_related, config=cfg, ilp_config=ilp,
    )
    rel_pmcids = {p.pmcid for p in related}

    logger.info("=== Calibration set (ILP) — stage 2/3: diverse ===")
    div_excl = set() if allow_overlap else rel_pmcids
    diverse, diverse_r = select_diverse_papers_ilp(
        fingerprints, k=cfg.k_diverse, config=cfg, ilp_config=ilp,
        exclude_pmcids=div_excl,
    )
    div_pmcids = {p.pmcid for p in diverse}

    logger.info("=== Calibration set (ILP) — stage 3/3: hard ===")
    hard_excl = set() if allow_overlap else (rel_pmcids | div_pmcids)
    hard, hard_r = select_hard_papers_ilp(
        fingerprints, k=cfg.k_hard, config=cfg, ilp_config=ilp,
        exclude_pmcids=hard_excl, allow_overlap=allow_overlap,
    )
    logger.info("=== Calibration set complete in %.2fs ===",
                time.perf_counter() - t_total)

    result = SelectionResult(
        related=[p.pmcid for p in related],
        diverse=[p.pmcid for p in diverse],
        hard=[p.pmcid for p in hard],
        rationale={},
    )
    for pmcid, info in related_r.items():
        result.rationale[pmcid] = {**info, "bucket": "related", "strategy": "ilp"}
    for pmcid, info in diverse_r.items():
        if pmcid in result.rationale and allow_overlap:
            result.rationale[pmcid]["diverse_info"] = info
        else:
            result.rationale[pmcid] = {**info, "bucket": "diverse", "strategy": "ilp"}
    for pmcid, info in hard_r.items():
        if pmcid in result.rationale and allow_overlap:
            result.rationale[pmcid]["hard_info"] = info
        else:
            result.rationale[pmcid] = {**info, "bucket": "hard", "strategy": "ilp"}

    return result


__all__ = [
    "ILPConfig",
    "pulp_available",
    "select_calibration_set_ilp",
    "select_diverse_papers_ilp",
    "select_hard_papers_ilp",
    "select_related_papers_ilp",
]
