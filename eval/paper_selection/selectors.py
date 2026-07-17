"""Greedy selectors for the related / diverse / hard buckets.

The selectors operate on lists of fingerprints and yield PMCID lists. They
take pluggable metrics (relatedness, diversity, hardness) so the formulas can
be swapped without rewriting the selection loops.

All selectors are deterministic for a given fingerprint list + metric config.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

from .metrics import (
    Diversity,
    Hardness,
    PairMetric,
    Relatedness,
)
from .models import HardnessBreakdown, PaperFingerprint, SelectionResult

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class SelectionConfig:
    """Numeric knobs and metric instances for the selectors.

    Sensible defaults; override at run time via ``run_select`` CLI.
    """
    k_related: int = 5
    k_diverse: int = 5
    k_hard:    int = 5

    max_sentences_related: int = 350
    max_sentences_diverse: int = 350
    min_sentences:         int = 20
    min_useful_entities:   int = 3
    min_disease_overlap:   int = 1

    # Hard-bucket composition
    hard_high_normalized: int = 2
    hard_high_absolute:   int = 2
    hard_medium_control:  int = 1
    medium_control_low_pct:  float = 0.40
    medium_control_high_pct: float = 0.65

    # Pluggable metrics
    relatedness: PairMetric = field(default_factory=Relatedness)
    diversity:   Diversity  = field(default_factory=Diversity)
    hardness:    Hardness   = field(default_factory=Hardness)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _by_pmcid(fps: list[PaperFingerprint]) -> dict[str, PaperFingerprint]:
    return {p.pmcid: p for p in fps}


def _eligible_short(fps: list[PaperFingerprint], cfg: SelectionConfig,
                    max_sentences: int) -> list[PaperFingerprint]:
    """Filter to non-empty, useful, short-enough papers."""
    out: list[PaperFingerprint] = []
    for p in fps:
        if p.n_sentences < cfg.min_sentences:
            continue
        if p.n_sentences > max_sentences:
            continue
        if not p.has_useful_entities():
            continue
        if len(p.entity_counts) < cfg.min_useful_entities:
            continue
        out.append(p)
    return out


def _greedy_related_seed(eligible: list[PaperFingerprint],
                         cfg: SelectionConfig) -> PaperFingerprint | None:
    """Pick the paper with the highest mean relatedness to all others."""
    if not eligible:
        return None
    scored: list[tuple[float, PaperFingerprint]] = []
    for p in eligible:
        rels = [cfg.relatedness.score(p, q) for q in eligible if q is not p]
        scored.append((sum(rels) / max(len(rels), 1), p))
    scored.sort(key=lambda t: (-t[0], t[1].pmcid))
    return scored[0][1]


# ── related ───────────────────────────────────────────────────────────────────

def select_related_papers(
    fingerprints: list[PaperFingerprint],
    *,
    k: int = 5,
    config: SelectionConfig | None = None,
) -> tuple[list[PaperFingerprint], dict[str, dict]]:
    """5 highly related, short, well-characterised papers.

    Strategy: pick the paper with the highest mean relatedness to all
    candidates as the seed, then greedily add the candidate with the largest
    mean relatedness to the already-picked set, requiring ≥1 disease overlap
    with any previously-picked paper.
    """
    cfg = config or SelectionConfig()
    t0 = time.perf_counter()
    eligible = _eligible_short(fingerprints, cfg, cfg.max_sentences_related)
    logger.info("[related] greedy: candidates=%d eligible=%d k=%d max_sent=%d",
                len(fingerprints), len(eligible), k, cfg.max_sentences_related)
    if len(eligible) < k:
        logger.warning("select_related: only %d eligible papers (need %d)",
                       len(eligible), k)

    chosen: list[PaperFingerprint] = []
    seed = _greedy_related_seed(eligible, cfg)
    if seed is None:
        return [], {}
    chosen.append(seed)

    rationale: dict[str, dict] = {}
    rationale[seed.pmcid] = {
        "rank": 1,
        "selection_reason": "highest mean pairwise relatedness across the candidate pool",
    }

    while len(chosen) < k:
        best: tuple[float, PaperFingerprint | None] = (-math.inf, None)
        for p in eligible:
            if p in chosen:
                continue
            disease_ok = any(p.disease_entities & q.disease_entities for q in chosen)
            if cfg.min_disease_overlap and not disease_ok:
                continue
            rels = [cfg.relatedness.score(p, q) for q in chosen]
            mean_rel = sum(rels) / len(rels)
            if mean_rel > best[0]:
                best = (mean_rel, p)
        if best[1] is None:
            logger.warning("select_related: ran out of disease-overlapping candidates "
                           "after %d picks", len(chosen))
            # Relax disease constraint to fill out the bucket
            for p in eligible:
                if p in chosen:
                    continue
                rels = [cfg.relatedness.score(p, q) for q in chosen]
                mean_rel = sum(rels) / len(rels)
                if mean_rel > best[0]:
                    best = (mean_rel, p)
        if best[1] is None:
            break
        rel_to_set = best[0]
        chosen.append(best[1])
        rationale[best[1].pmcid] = {
            "rank": len(chosen),
            "selection_reason": (
                f"mean relatedness to set so far = {rel_to_set:.3f}; "
                "greedy pick that preserved disease-family overlap"
            ),
            "mean_relatedness_to_set": rel_to_set,
        }

    logger.info("[related] greedy: picked %d in %.2fs — %s",
                len(chosen), time.perf_counter() - t0,
                [p.pmcid for p in chosen])
    return chosen, rationale


# ── diverse ───────────────────────────────────────────────────────────────────

def select_diverse_papers(
    fingerprints: list[PaperFingerprint],
    *,
    k: int = 5,
    config: SelectionConfig | None = None,
    exclude_pmcids: set[str] | None = None,
) -> tuple[list[PaperFingerprint], dict[str, dict]]:
    """k papers chosen to maximise the :class:`Diversity` set-metric.

    Greedy farthest-point: seed with the paper having the largest entity
    vocabulary, then iteratively add the paper that most increases set
    diversity.
    """
    cfg = config or SelectionConfig()
    t0 = time.perf_counter()
    excluded = set(exclude_pmcids or [])
    eligible = [p for p in _eligible_short(fingerprints, cfg, cfg.max_sentences_diverse)
                if p.pmcid not in excluded]
    logger.info("[diverse] greedy: candidates=%d eligible=%d excluded=%d k=%d",
                len(fingerprints), len(eligible), len(excluded), k)
    if len(eligible) < k:
        logger.warning("select_diverse: only %d eligible papers (need %d)",
                       len(eligible), k)

    if not eligible:
        return [], {}

    # Seed: paper with the largest combined disease/method/biomarker vocab.
    def _vocab_size(p: PaperFingerprint) -> int:
        return (len(p.disease_entities) + len(p.method_entities)
                + len(p.biomarker_entities) + len(p.outcome_entities))
    seed = max(eligible, key=lambda p: (_vocab_size(p), -p.n_sentences))
    chosen = [seed]
    rationale = {seed.pmcid: {
        "rank": 1,
        "selection_reason": (
            f"seed: largest combined disease/method/biomarker vocab "
            f"({_vocab_size(seed)})"
        ),
    }}

    while len(chosen) < k:
        baseline = cfg.diversity.score(chosen)
        best: tuple[float, PaperFingerprint | None, float] = (-math.inf, None, 0.0)
        for p in eligible:
            if p in chosen:
                continue
            new_score = cfg.diversity.score([*chosen, p])
            gain = new_score - baseline
            if gain > best[0]:
                best = (gain, p, new_score)
        if best[1] is None:
            break
        chosen.append(best[1])
        rationale[best[1].pmcid] = {
            "rank": len(chosen),
            "selection_reason": (
                f"diversity gain = {best[0]:+.3f} (set diversity now {best[2]:.3f})"
            ),
            "diversity_gain": best[0],
        }

    logger.info("[diverse] greedy: picked %d in %.2fs — %s",
                len(chosen), time.perf_counter() - t0,
                [p.pmcid for p in chosen])
    return chosen, rationale


# ── hard ─────────────────────────────────────────────────────────────────────

def _classify_hard(
    candidates: list[PaperFingerprint],
    cfg: SelectionConfig,
) -> tuple[list[tuple[PaperFingerprint, HardnessBreakdown]],
           list[tuple[PaperFingerprint, HardnessBreakdown]],
           list[tuple[PaperFingerprint, HardnessBreakdown]]]:
    scored = [(p, cfg.hardness.breakdown(p)) for p in candidates]
    by_norm = sorted(scored, key=lambda t: -t[1].normalized_hardness)
    by_abs  = sorted(scored, key=lambda t: -t[1].absolute_hardness)
    by_med  = sorted(scored, key=lambda t: t[1].normalized_hardness)
    return by_norm, by_abs, by_med


def select_hard_papers(
    fingerprints: list[PaperFingerprint],
    *,
    k: int = 5,
    config: SelectionConfig | None = None,
    exclude_pmcids: set[str] | None = None,
) -> tuple[list[PaperFingerprint], dict[str, dict]]:
    """Compose the hard bucket: 2 high-normalized + 2 high-absolute + 1 medium control."""
    cfg = config or SelectionConfig()
    t0 = time.perf_counter()
    excluded = set(exclude_pmcids or [])

    candidates = [p for p in fingerprints
                  if p.pmcid not in excluded and not p.is_empty()]
    logger.info("[hard] greedy: candidates=%d excluded=%d non_empty=%d k=%d",
                len(fingerprints), len(excluded), len(candidates), k)
    if not candidates:
        return [], {}

    by_norm, by_abs, by_med = _classify_hard(candidates, cfg)

    chosen: list[PaperFingerprint] = []
    rationale: dict[str, dict] = {}
    seen: set[str] = set()

    def _add(p: PaperFingerprint, breakdown: HardnessBreakdown, why: str) -> None:
        if p.pmcid in seen:
            return
        chosen.append(p)
        seen.add(p.pmcid)
        rationale[p.pmcid] = {
            "rank": len(chosen),
            "selection_reason": why,
            "hardness_breakdown": {
                "normalized_hardness": breakdown.normalized_hardness,
                "absolute_hardness":   breakdown.absolute_hardness,
                "content_complexity":  breakdown.content_complexity,
                "layout_complexity":   breakdown.layout_complexity,
                "relation_complexity": breakdown.relation_complexity,
                "workload_factor":     breakdown.workload_factor,
                "reasons":             list(breakdown.reasons),
            },
        }

    # 2 high normalized
    for p, b in by_norm:
        if len([x for x in rationale.values() if "high normalized" in x["selection_reason"]]) >= cfg.hard_high_normalized:
            break
        _add(p, b, f"high normalized hardness (score={b.normalized_hardness:.3f})")
        if len(chosen) >= cfg.hard_high_normalized:
            break

    # 2 high absolute (filter out already chosen)
    abs_added = 0
    for p, b in by_abs:
        if abs_added >= cfg.hard_high_absolute:
            break
        if p.pmcid in seen:
            continue
        _add(p, b, f"high absolute hardness (score={b.absolute_hardness:.3f}, n_chunks={p.n_chunks})")
        abs_added += 1

    # 1 medium control: pick from the middle band of normalized hardness.
    if cfg.hard_medium_control > 0:
        ordered = [(p, b) for p, b in by_med if p.pmcid not in seen]
        if ordered:
            n = len(ordered)
            lo = int(n * cfg.medium_control_low_pct)
            hi = max(int(n * cfg.medium_control_high_pct), lo + 1)
            band = ordered[lo:hi] or ordered[max(n // 2, 0):max(n // 2, 0) + 1]
            p, b = band[len(band) // 2]
            _add(p, b, (f"medium hardness control (normalized={b.normalized_hardness:.3f}, "
                        f"middle of the {cfg.medium_control_low_pct:.0%}-"
                        f"{cfg.medium_control_high_pct:.0%} band)"))

    # Pad with any remaining hardest candidates if we still need more
    if len(chosen) < k:
        for p, b in by_norm:
            if len(chosen) >= k:
                break
            if p.pmcid in seen:
                continue
            _add(p, b, "padding: next-hardest available candidate")

    logger.info("[hard] greedy: picked %d in %.2fs — %s",
                len(chosen[:k]), time.perf_counter() - t0,
                [p.pmcid for p in chosen[:k]])
    return chosen[:k], rationale


# ── calibration set (15 papers) ──────────────────────────────────────────────

def select_calibration_set(
    fingerprints: list[PaperFingerprint],
    *,
    config: SelectionConfig | None = None,
    allow_overlap: bool = False,
) -> SelectionResult:
    """Compose the 15-paper calibration set.

    By default the three buckets are mutually exclusive — already-picked
    PMCIDs are excluded from later buckets. With ``allow_overlap=True`` the
    same paper may appear in more than one bucket (mostly useful when the
    pool is too small for strict disjointness).
    """
    cfg = config or SelectionConfig()

    related, related_r = select_related_papers(fingerprints, k=cfg.k_related, config=cfg)
    rel_pmcids = {p.pmcid for p in related}

    diverse_excl = set() if allow_overlap else rel_pmcids
    diverse, diverse_r = select_diverse_papers(
        fingerprints, k=cfg.k_diverse, config=cfg, exclude_pmcids=diverse_excl,
    )
    div_pmcids = {p.pmcid for p in diverse}

    hard_excl = set() if allow_overlap else (rel_pmcids | div_pmcids)
    hard, hard_r = select_hard_papers(
        fingerprints, k=cfg.k_hard, config=cfg, exclude_pmcids=hard_excl,
    )

    result = SelectionResult(
        related=[p.pmcid for p in related],
        diverse=[p.pmcid for p in diverse],
        hard=[p.pmcid for p in hard],
        rationale={},
    )
    for pmcid, info in related_r.items():
        result.rationale[pmcid] = {**info, "bucket": "related"}
    for pmcid, info in diverse_r.items():
        result.rationale.setdefault(pmcid, {}).update({**info, "bucket": "diverse"}
                                                      if not allow_overlap else
                                                      {"diverse_info": info, "bucket": result.rationale.get(pmcid, {}).get("bucket", "diverse")})
    for pmcid, info in hard_r.items():
        result.rationale.setdefault(pmcid, {}).update({**info, "bucket": "hard"}
                                                      if not allow_overlap else
                                                      {"hard_info": info, "bucket": result.rationale.get(pmcid, {}).get("bucket", "hard")})
    return result
