"""EmbeddingScorer — soft-alignment cosine similarity on claim embeddings."""
from __future__ import annotations

import re
from itertools import combinations

import numpy as np

from nlp_histo.pipeline.stages.knowledge_extraction.interfaces.scoring import AgreementContext, ScoreBundle
from nlp_histo.pipeline.stages.knowledge_extraction.models import AuditableSummary
from .providers import EmbedFn, OpenAIEmbedder


# Polarity vocabulary for contradiction heuristic

_POSITIVE = frozenset({
    "positive", "present", "increased", "detected", "high",
    "expressed", "elevated", "overexpressed", "upregulated",
})
_NEGATIVE = frozenset({
    "negative", "absent", "decreased", "low",
    "reduced", "downregulated", "lost", "undetected",
})
# Words that negate the polarity of a following keyword within _NEG_WINDOW tokens.
_NEGATIONS = frozenset({"not", "no", "without", "lack", "lacking", "never", "neither", "nor"})
_NEG_WINDOW = 3   # tokens before a polarity keyword to check for negation

# Numeric contradiction heuristic

_NUM_RE = re.compile(r"\b(\d+(?:\.\d+)?)")
_NUMERIC_RATIO_THRESHOLD = 2.0   # max/min ratio above which numbers are "contradictory"


def _polarity(claim: str) -> int:
    """
    Returns +1 (positive polarity), -1 (negative), 0 (neutral/ambiguous).

    Token-window negation: if a negation word appears in the _NEG_WINDOW tokens
    before a polarity keyword, the keyword's polarity is flipped.  This catches
    patterns like "not positive", "not detected", "lack of expression".

    ``non-`` prefix: tokens starting with "non-" (e.g. "non-expressed") are
    treated as the suffix keyword with pre-flipped polarity.

    All polarity keywords found are accumulated; the final sign is determined
    by whether effective positives or effective negatives dominate.
    """
    tokens = claim.lower().split()
    effective_pos = 0
    effective_neg = 0
    for i, tok in enumerate(tokens):
        non_prefix = tok.startswith("non-")
        check = tok[4:] if non_prefix else tok
        if check in _POSITIVE:
            base = -1 if non_prefix else 1
        elif check in _NEGATIVE:
            base = 1 if non_prefix else -1
        else:
            continue
        # Negation window only applies to bare keywords (not to non- prefix).
        if not non_prefix:
            window = tokens[max(0, i - _NEG_WINDOW) : i]
            if any(n in window for n in _NEGATIONS):
                base = -base
        if base > 0:
            effective_pos += 1
        else:
            effective_neg += 1
    if effective_pos > 0 and effective_neg == 0:
        return 1
    if effective_neg > 0 and effective_pos == 0:
        return -1
    return 0


def _reuse_factor(best_match_indices: np.ndarray, n_target: int, weight: float) -> float:
    """
    Soft reuse-concentration penalty factor in [1 - weight, 1.0].

    Penalizes cases where multiple source claims all route to the same target
    claim as their best match.  When there is only one target claim, concentration
    is expected and no penalty is applied.
    """
    n_source = len(best_match_indices)
    if n_target <= 1 or n_source == 0:
        return 1.0
    counts = np.bincount(best_match_indices, minlength=n_target)
    max_reuse_frac = float(counts.max()) / n_source
    expected_frac = 1.0 / n_target
    max_excess = 1.0 - expected_frac
    if max_excess <= 0.0:
        return 1.0
    excess = max(0.0, max_reuse_frac - expected_frac)
    normalized = excess / max_excess          # in [0, 1]
    return 1.0 - weight * normalized          # floor = 1 - weight


def _polarity_contradiction_ratio(
    a_claims: list[str],
    b_claims: list[str],
    sim_t: np.ndarray,
    tau: float,
) -> float:
    """
    Fraction of strong best-matches with opposing polarity signals.

    Only counts pairs where both claims carry a clear unambiguous polarity.
    Returns 0.0 when no strong pairs exist.
    """
    if sim_t.shape[0] == 0 or sim_t.shape[1] == 0:
        return 0.0
    best_j = sim_t.argmax(axis=1)
    strong_pairs = [
        (i, int(best_j[i]))
        for i in range(sim_t.shape[0])
        if sim_t[i, best_j[i]] >= tau
    ]
    if not strong_pairs:
        return 0.0
    contradictions = sum(
        1 for i, j in strong_pairs
        if _polarity(a_claims[i]) != 0
        and _polarity(b_claims[j]) != 0
        and _polarity(a_claims[i]) != _polarity(b_claims[j])
    )
    return contradictions / len(strong_pairs)


def _numeric_contradiction_ratio(
    a_claims: list[str],
    b_claims: list[str],
    sim_t: np.ndarray,
    tau: float,
) -> float:
    """
    Fraction of strong best-matches with substantially different single numeric values.

    Only considers pairs where each claim contains exactly one number; flags the
    pair as contradictory when max / min >= _NUMERIC_RATIO_THRESHOLD.  This catches
    disagreements like "tumor 5mm" vs "tumor 15mm" without requiring NER.
    """
    if sim_t.shape[0] == 0 or sim_t.shape[1] == 0:
        return 0.0
    best_j = sim_t.argmax(axis=1)
    strong_pairs = [
        (i, int(best_j[i]))
        for i in range(sim_t.shape[0])
        if sim_t[i, best_j[i]] >= tau
    ]
    if not strong_pairs:
        return 0.0
    contradictions = 0
    for i, j in strong_pairs:
        nums_a = _NUM_RE.findall(a_claims[i])
        nums_b = _NUM_RE.findall(b_claims[j])
        if len(nums_a) == len(nums_b) == 1:
            va, vb = float(nums_a[0]), float(nums_b[0])
            if va > 0 and vb > 0:
                ratio = max(va, vb) / min(va, vb)
                if ratio >= _NUMERIC_RATIO_THRESHOLD:
                    contradictions += 1
    return contradictions / len(strong_pairs)


def _contradiction_ratio(
    a_claims: list[str],
    b_claims: list[str],
    sim_t: np.ndarray,
    tau: float,
) -> float:
    """
    Combined contradiction signal: max of polarity and numeric contradiction ratios.

    Using max() rather than a weighted sum keeps both signals bounded in [0, 1]
    and avoids double-counting when both heuristics fire on the same pair.
    """
    return max(
        _polarity_contradiction_ratio(a_claims, b_claims, sim_t, tau),
        _numeric_contradiction_ratio(a_claims, b_claims, sim_t, tau),
    )


def _align_precomputed(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    a_claims: list[str],
    b_claims: list[str],
    tau: float,
    count_alpha: float,
    reuse_weight: float,
    contradiction_weight: float,
) -> float:
    """
    Core bidirectional soft-coverage alignment on pre-normalised embeddings.

    Applies four adjustments on top of the raw bidirectional soft-alignment:

    1. Weak-match threshold (tau): similarities below tau are zeroed before
       computing coverage — weak incidental matches don't inflate the score.
    2. Count mismatch penalty: mild exponential factor based on |A|/|B| ratio.
    3. Reuse concentration penalty: applied symmetrically in both directions;
       penalises a broad claim absorbing all best-matches from the other set.
    4. Contradiction penalty: combined polarity + numeric heuristics.

    Grounding penalty is not applied here — it is caller-level (EmbeddingScorer).
    """
    score, _ = _align_precomputed_with_breakdown(
        emb_a, emb_b, a_claims, b_claims,
        tau, count_alpha, reuse_weight, contradiction_weight,
    )
    return score


def _align_precomputed_with_breakdown(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    a_claims: list[str],
    b_claims: list[str],
    tau: float,
    count_alpha: float,
    reuse_weight: float,
    contradiction_weight: float,
) -> tuple[float, dict]:
    """
    Same computation as ``_align_precomputed`` but returns ``(score, breakdown)``.

    The breakdown dict contains every intermediate component so callers can
    store or display the derivation without re-running the computation:

    ``claim_count_a``, ``claim_count_b``,
    ``coverage_a_to_b``, ``coverage_b_to_a``, ``base``,
    ``count_factor``, ``reuse_factor``,
    ``polarity_contradiction_ratio``, ``numeric_contradiction_ratio``,
    ``contradiction_ratio``, ``contradiction_factor``,
    ``pre_grounding_score``  (= base × count_factor × reuse_factor × contradiction_factor)

    Grounding penalty is applied by the caller; ``pre_grounding_score`` equals
    the final score when no grounding penalty is applied.
    """
    if emb_a.shape[0] == 0 or emb_b.shape[0] == 0:
        bd: dict = {
            "claim_count_a": len(a_claims),
            "claim_count_b": len(b_claims),
            "coverage_a_to_b": 0.0,
            "coverage_b_to_a": 0.0,
            "base": 0.0,
            "count_factor": 0.0,
            "reuse_factor": 0.0,
            "polarity_contradiction_ratio": 0.0,
            "numeric_contradiction_ratio": 0.0,
            "contradiction_ratio": 0.0,
            "contradiction_factor": 0.0,
            "pre_grounding_score": 0.0,
        }
        return 0.0, bd

    sim = np.clip(emb_a @ emb_b.T, 0.0, 1.0)      # (na, nb), clipped to [0, 1]
    sim_t = np.where(sim < tau, 0.0, sim)            # zero out weak matches

    a_to_b = float(sim_t.max(axis=1).mean())
    b_to_a = float(sim_t.max(axis=0).mean())
    base = (a_to_b + b_to_a) / 2.0

    # 1. Count mismatch — mild exponent keeps large splits from being harshly penalised
    len_a, len_b = len(a_claims), len(b_claims)
    count_factor = (min(len_a, len_b) / max(len_a, len_b)) ** count_alpha

    # 2. Reuse concentration — symmetric check in both directions
    best_b = sim_t.argmax(axis=1)                   # for each a_i, best b_j
    best_a = sim_t.argmax(axis=0)                   # for each b_j, best a_i
    rf_a2b = _reuse_factor(best_b, len_b, reuse_weight)
    rf_b2a = _reuse_factor(best_a, len_a, reuse_weight)
    reuse = (rf_a2b + rf_b2a) / 2.0

    # 3. Contradiction — polarity + numeric heuristics
    polarity_cr = _polarity_contradiction_ratio(a_claims, b_claims, sim_t, tau)
    numeric_cr = _numeric_contradiction_ratio(a_claims, b_claims, sim_t, tau)
    contra_ratio = max(polarity_cr, numeric_cr)
    contra_factor = 1.0 - contradiction_weight * contra_ratio

    score = base * count_factor * reuse * contra_factor

    breakdown: dict = {
        "claim_count_a": len_a,
        "claim_count_b": len_b,
        "coverage_a_to_b": round(a_to_b, 6),
        "coverage_b_to_a": round(b_to_a, 6),
        "base": round(base, 6),
        "count_factor": round(count_factor, 6),
        "reuse_factor": round(reuse, 6),
        "polarity_contradiction_ratio": round(polarity_cr, 6),
        "numeric_contradiction_ratio": round(numeric_cr, 6),
        "contradiction_ratio": round(contra_ratio, 6),
        "contradiction_factor": round(contra_factor, 6),
        "pre_grounding_score": round(score, 6),
    }
    return score, breakdown


def _align_one_to_one(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    tau: float,
    strategy: str,
) -> float:
    """One-to-one alignment agreement score (greedy / hungarian modes).

    Each finding is matched to at most ONE finding on the other side, so extra
    unmatched findings on either side lower the score. ``tau`` is the minimum
    valid-pair similarity — pairs below it cannot be matched. The score is a
    precision/recall-style F1 over the matched similarities:

        precision = matched_score / |A|,  recall = matched_score / |B|

    so over-production (extra A) lowers precision and under-production (extra B)
    lowers recall. Operates on pre-normalised embeddings; returns a value in
    ``[0, 1]``. Does NOT use count_alpha / reuse_weight / contradiction_weight
    (those are soft_max-only).
    """
    na, nb = emb_a.shape[0], emb_b.shape[0]
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0

    sim = np.clip(emb_a @ emb_b.T, 0.0, 1.0)        # (na, nb)
    sim = np.where(sim < tau, 0.0, sim)              # tau = min valid-pair similarity

    if strategy == "greedy":
        used_a: set[int] = set()
        used_b: set[int] = set()
        matched = 0.0
        for flat in np.argsort(sim, axis=None)[::-1]:   # candidate pairs, sim desc
            i, j = divmod(int(flat), nb)
            s = float(sim[i, j])
            if s <= 0.0:
                break                                    # remaining pairs are below tau
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            matched += s
    elif strategy == "hungarian":
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-sim)         # maximise total similarity
        matched = float(sum(sim[r, c] for r, c in zip(rows, cols) if sim[r, c] > 0.0))
    else:
        raise ValueError(
            f"alignment_strategy must be 'greedy' or 'hungarian', got {strategy!r}"
        )

    precision = matched / na
    recall = matched / nb
    if precision + recall > 0.0:
        return 2.0 * precision * recall / (precision + recall)
    return 0.0


def _align(
    embed: EmbedFn,
    a_claims: list[str],
    b_claims: list[str],
    tau: float = 0.15,
    count_alpha: float = 0.25,
    reuse_weight: float = 0.15,
    contradiction_weight: float = 0.20,
    alignment_strategy: str = "soft_max",
) -> float:
    """Embed and align two claim lists.  Issues a single embed() call.

    ``alignment_strategy`` selects ``soft_max`` (default, many-to-one coverage
    with the full penalty set) or one-to-one ``greedy`` / ``hungarian`` (see
    ``_align_one_to_one``)."""
    if not a_claims and not b_claims:
        return 1.0
    if not a_claims or not b_claims:
        return 0.0

    raw = np.array(embed(a_claims + b_claims), dtype=float)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    normalized = raw / np.where(norms == 0.0, 1.0, norms)
    emb_a = normalized[: len(a_claims)]
    emb_b = normalized[len(a_claims) :]

    if alignment_strategy == "soft_max":
        return _align_precomputed(
            emb_a, emb_b, a_claims, b_claims,
            tau, count_alpha, reuse_weight, contradiction_weight,
        )
    return _align_one_to_one(emb_a, emb_b, tau, alignment_strategy)


def _claims(output: AuditableSummary) -> list[str]:
    return [f.claim for f in output.findings]


class EmbeddingScorer:
    """
    Mean pairwise soft-alignment cosine similarity on claim embeddings.

    Extends the original bidirectional soft-alignment with:

    - tau               : weak-match threshold; similarities below tau are zeroed
                          before computing coverage.  Default 0.15.
    - count_alpha       : exponent for the soft count-mismatch penalty.
                          0.25 → a 4:1 claim-count ratio applies a ~0.71× factor.
    - reuse_weight      : concentration penalty weight; 0.15 gives at most a 15%
                          reduction when one broad claim absorbs all matches.
    - contradiction_weight: polarity-flip + numeric penalty weight; 0.20 reduces
                          the score by up to 20% when all strong matches contradict.
    - grounding_floor   : minimum grounding factor applied per voter pair.
                          Prevents a fully-ungrounded pair from zeroing the score.
                          Default 0.50.

    The score is stored in ScoreBundle.embedding_agreement.
    AgreementChecker applies the theta threshold after this scorer returns.

    Parameters
    ----------
    embed_fn:
        Callable mapping a list of strings to embedding vectors.
        Defaults to OpenAIEmbedder (text-embedding-3-small).
    """

    def __init__(
        self,
        embed_fn: EmbedFn | None = None,
        tau: float = 0.15,
        count_alpha: float = 0.25,
        reuse_weight: float = 0.15,
        contradiction_weight: float = 0.20,
        grounding_floor: float = 0.50,
        alignment_strategy: str = "soft_max",
    ) -> None:
        self._embed: EmbedFn = embed_fn or OpenAIEmbedder()
        self.tau = tau
        self.count_alpha = count_alpha
        self.reuse_weight = reuse_weight
        self.contradiction_weight = contradiction_weight
        self.grounding_floor = grounding_floor
        self.alignment_strategy = alignment_strategy

    def compute(
        self,
        outputs: list[AuditableSummary],
        source_text: str | None = None,  # noqa: ARG002
        context: AgreementContext | None = None,
    ) -> ScoreBundle:
        claim_lists = [_claims(o) for o in outputs]

        # Apply grounding penalty per voter pair using min(pf_i, pf_j).
        # This is stricter than mean: one bad voter pulls down every pair it is in.
        pair_scores: list[float] = []
        for i, j in combinations(range(len(outputs)), 2):
            raw = _align(
                self._embed, claim_lists[i], claim_lists[j],
                tau=self.tau,
                count_alpha=self.count_alpha,
                reuse_weight=self.reuse_weight,
                contradiction_weight=self.contradiction_weight,
                alignment_strategy=self.alignment_strategy,
            )
            if context is not None and context.voter_contexts:
                pf_i = context.voter_contexts[i].grounding_pass_fraction
                pf_j = context.voter_contexts[j].grounding_pass_fraction
                g = min(pf_i, pf_j)
                raw *= self.grounding_floor + (1.0 - self.grounding_floor) * g
            pair_scores.append(raw)

        score = sum(pair_scores) / len(pair_scores) if pair_scores else 1.0
        return ScoreBundle(embedding_agreement=score)
