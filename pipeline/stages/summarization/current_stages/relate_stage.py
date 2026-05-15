"""
RELATE stage: CanonicalRule[] → Relation[]

NLI-based pairwise comparison between canonical rules.  Each pair is classified
as one of: SUPPORT, CONTRADICT, UNRELATED.

Only non-UNRELATED pairs are returned.  (Note: ``RelationTypeLabel.SCOPE_QUALIFY``
existed as a placeholder for an asymmetric-entailment branch that was never
implemented; the plumbing was removed in B-006.  Reinstate the branch + log
columns + RESOLVE filter if the signal is wanted in the future.)

Comparability gate (ALL four must hold for NLI to run):
  1. exact category match
  2. exact relation_type match
  3. exact normalized subject_entity match
  4. exact normalized outcome_entity match

  Rationale for condition 4: two has_feature rules sharing the same subject but
  describing different histological features (e.g. "giant cells" vs "open vascular
  channels") are coexisting attributes — not contradiction candidates.  Comparing
  them produces spurious CONTRADICT labels from NLI.  Only rules about the SAME
  outcome can meaningfully SUPPORT or CONTRADICT each other.

  Special rule: for has_feature relation_type, CONTRADICT is only emitted when
  outcome_entity also matches.  Since condition 4 already enforces outcome match
  before NLI runs, this is automatically satisfied.  The explicit check in
  _classify_pair is a safety belt for edge cases.

The NLI pipeline is shared with GroundingFilter via a module-level singleton
so the model is loaded only once per process.
"""
from __future__ import annotations

import itertools
import logging

from ..artifact_models import SkippedPair
from ..models import (
    CanonicalRule,
    NON_POLARITY_DIRS,
    RawNLIPair,
    Relation,
    RelationTypeLabel,
    direction_value,
)
from ..nli_config import get_active_spec

logger = logging.getLogger(__name__)

# ── Module-level NLI singleton ─────────────────────────────────────────────────
# Delegates to the grounding_filter module-level cache so the same model
# instance is shared with GroundingFilter (loaded once per process).

_ACTIVE_SPEC = get_active_spec()
_NLI_MODEL = _ACTIVE_SPEC.hf_id
_DEFAULT_BATCH_SIZE = _ACTIVE_SPEC.batch_size


def _get_nli_pipe(model_name: str = _NLI_MODEL, batch_size: int = _DEFAULT_BATCH_SIZE, device: int | str | None = None):
    """Return cached NLI pipeline, sharing the grounding_filter singleton."""
    from ..helpers.grounding_filter import _NLI_PIPE_CACHE, _get_device
    resolved_device = device if device is not None else _get_device()
    cache_key = (model_name, resolved_device, batch_size)
    if cache_key not in _NLI_PIPE_CACHE:
        from transformers import pipeline  # type: ignore
        logger.info(
            "RELATE: loading NLI model %r on device=%r batch_size=%d",
            model_name, resolved_device, batch_size,
        )
        _NLI_PIPE_CACHE[cache_key] = pipeline(
            "text-classification", model=model_name, top_k=None,
            device=resolved_device, batch_size=batch_size,
        )
    return _NLI_PIPE_CACHE[cache_key]



# ── NLI helpers ────────────────────────────────────────────────────────────────

def _nli_scores(pairs: list[tuple[str, str]], pipe) -> list[dict[str, float]]:
    """
    Run NLI on (premise, hypothesis) pairs with sliding window support.
    Returns a list of dicts {entailment, contradiction, neutral} per pair.

    Long premises are split into overlapping windows (budget computed from the
    hypothesis length so the joint sequence never exceeds 512 tokens).
    Per-label scores are max-pooled across windows so a supporting or
    contradicting sentence near a window boundary is not silently truncated.
    """
    from ..helpers.grounding_filter import _split_windows

    _empty: dict[str, float] = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
    out: list[dict[str, float]] = [dict(_empty) for _ in pairs]

    flat_inputs: list[dict] = []
    flat_indices: list[int] = []

    for i, (premise, hyp) in enumerate(pairs):
        if not premise.strip():
            continue
        for window in _split_windows(premise, hyp, pipe.tokenizer):
            flat_inputs.append({"text": window, "text_pair": hyp})
            flat_indices.append(i)

    if flat_inputs:
        from tqdm.auto import tqdm  # noqa: PLC0415
        _bs = getattr(pipe, "_batch_size", 16)
        batch_results = []
        with tqdm(total=len(flat_inputs), desc="NLI [relate]", unit="pair", leave=False) as pbar:
            for start in range(0, len(flat_inputs), _bs):
                batch_results.extend(pipe(flat_inputs[start:start + _bs], truncation=True))
                pbar.update(min(_bs, len(flat_inputs) - start))
        for pair_idx, result in zip(flat_indices, batch_results):
            for item in result:
                label = item["label"].lower()
                if item["score"] > out[pair_idx].get(label, 0.0):
                    out[pair_idx][label] = item["score"]

    return out


def _classify_pair(
    rule_a: CanonicalRule,
    rule_b: CanonicalRule,
    scores_ab: dict[str, float],
    scores_ba: dict[str, float],
    entailment_threshold: float,
    contradiction_threshold: float,
) -> RelationTypeLabel | None:
    """
    Decide the relation type for a (rule_a, rule_b) pair using bidirectional
    NLI scores.

    Logic:
      - CONTRADICT   : both directions score high on 'contradiction',
                       AND the two rules do not share the same direction
                       (two positive rules or two negative rules cannot
                       contradict each other — they are coexisting findings).
                       The comparability gate already ensures same
                       (subject, outcome, relation_type, category), so a high
                       mutual contradiction score with opposite directions is a
                       genuine conflict.
      - SUPPORT      : both directions score high on 'entailment'
      - None         : UNRELATED — caller skips this pair. Asymmetric
                       entailment (one direction only) collapses here so the
                       output schema mirrors the 3-class NLI gold.
    """
    from ..models import DirectionEnum

    ent_ab = scores_ab.get("entailment", 0.0)
    ent_ba = scores_ba.get("entailment", 0.0)
    con_ab = scores_ab.get("contradiction", 0.0)
    con_ba = scores_ba.get("contradiction", 0.0)

    # CONTRADICT guard: block when both rules point in the same direction.
    # Two positive (or two negative/absent) findings about the same entity
    # pair are coexisting observations — not contradictions — even if the NLI
    # model scores them high on 'contradiction' due to lexical overlap.
    # We use the structured direction field (set by the LLM) rather than
    # re-inferring polarity from text, which was unreliable and blocked almost
    # all genuine contradictions.
    _NEGATIVE_DIRECTIONS = {DirectionEnum.negative, DirectionEnum.absent}
    _POSITIVE_DIRECTIONS = {DirectionEnum.positive, DirectionEnum.partial}

    dir_a = rule_a.direction
    dir_b = rule_b.direction

    same_polarity = (
        # both clearly positive
        (dir_a in _POSITIVE_DIRECTIONS and dir_b in _POSITIVE_DIRECTIONS)
        # both clearly negative / absent
        or (dir_a in _NEGATIVE_DIRECTIONS and dir_b in _NEGATIVE_DIRECTIONS)
    )
    contradict_allowed = not same_polarity

    # Mutual contradiction
    if contradict_allowed and con_ab >= contradiction_threshold and con_ba >= contradiction_threshold:
        return RelationTypeLabel.CONTRADICT

    # Mutual entailment → support
    if ent_ab >= entailment_threshold and ent_ba >= entailment_threshold:
        return RelationTypeLabel.SUPPORT

    return None  # UNRELATED


# ── Pruning ────────────────────────────────────────────────────────────────────

def _norm_subject(entity: str) -> str:
    """
    Synonym-dict normalization + lowercase for subject gate comparison.
    Uses the same synonym resolution as NormalizeStage so that surface variants
    that survived NORMALIZE (e.g. "ALK-positive ALCL" vs "ALK+ ALCL") still match.
    """
    from .normalize_stage import normalize_entity
    return (normalize_entity(entity) or "").strip().lower()


def _norm_outcome(entity: str) -> str:
    """
    Lowercase + strip for outcome gate comparison.
    Used for has_feature and all other relation types except expression.
    """
    return entity.strip().lower()


def _norm_outcome_expression(entity: str) -> str:
    """
    For expression relation_type: strip marker suffixes before comparing.
    "CD31 expression" → "cd31", "CD34 expression" → "cd34".
    This ensures different markers are never considered the same outcome.
    """
    s = entity.strip().lower()
    for suffix in (" expression", " positivity", " staining", " immunoreactivity"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def _should_compare(a: CanonicalRule, b: CanonicalRule) -> tuple[bool, str]:
    """
    Strict 4-condition gate. A pair reaches NLI only when ALL hold:
      1. category matches exactly
      2. relation_type matches exactly
      3. subject_entity matches exactly
      4. normalized outcome_entity matches exactly
         (for expression: marker name after stripping suffixes must match)

    Returns (eligible, rejection_reason).
    rejection_reason is "" when eligible=True.

    Debug rejection reasons:
      category_mismatch
      relation_type_mismatch
      subject_mismatch
      outcome_incompatible
    """
    from ..models import RelationTypeEnum  # local import avoids circular at module level
    # B-049: skip any pair where either side carries a non-polarity direction
    # (unclear / no_direction). NLI on those is meaningless — there's nothing
    # to SUPPORT or CONTRADICT. `direction_value` normalises enum / string /
    # None inputs uniformly, including legacy rows where `direction is None`.
    if (
        direction_value(a.direction) in NON_POLARITY_DIRS
        or direction_value(b.direction) in NON_POLARITY_DIRS
    ):
        return False, "non_polarity_direction"
    if a.category != b.category:
        return False, "category_mismatch"
    if a.relation_type != b.relation_type:
        return False, "relation_type_mismatch"
    if _norm_subject(a.subject_entity) != _norm_subject(b.subject_entity):
        return False, "subject_mismatch"
    # Outcome comparison — stricter normalization for expression rules
    if a.relation_type == RelationTypeEnum.expression:
        if _norm_outcome_expression(a.outcome_entity) != _norm_outcome_expression(b.outcome_entity):
            return False, "outcome_incompatible"
    else:
        if _norm_outcome(a.outcome_entity) != _norm_outcome(b.outcome_entity):
            return False, "outcome_incompatible"
    return True, ""


# ── Public API ─────────────────────────────────────────────────────────────────

class RelateStage:
    """
    RELATE stage: NLI-based pairwise comparison of CanonicalRules.

    Parameters
    ----------
    model_name:
        HuggingFace NLI model name.  Defaults to the same model used by
        GroundingFilter.  The pipeline is cached at module level.
    entailment_threshold:
        Minimum entailment score to classify as SUPPORT.
    contradiction_threshold:
        Minimum contradiction score (both directions required) to classify
        as CONTRADICT.
    """

    def __init__(
        self,
        model_name: str = _NLI_MODEL,
        entailment_threshold: float = 0.55,
        contradiction_threshold: float = 0.65,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        device: int | str | None = None,
    ) -> None:
        self._model_name = model_name
        self._entailment_threshold = entailment_threshold
        self._contradiction_threshold = contradiction_threshold
        self._batch_size = batch_size
        self._device = device

    def relate(
        self,
        rules: list[CanonicalRule],
        pmcid: str = "",
        gate=None,
    ) -> tuple[list[Relation], list[RawNLIPair], list[SkippedPair]]:
        """
        Compare all eligible pairs of CanonicalRules.

        Parameters
        ----------
        rules:
            Output of CanonicalizeStage.canonicalize().
        pmcid:
            For logging only.
        gate:
            Callable(a, b) -> (bool, str) used to filter pairs before NLI.
            Defaults to _should_compare (strict: category + relation_type +
            subject + outcome must all match).  Pass a looser function for
            cross-paper mode where entity names differ across papers.

        Returns
        -------
        (relations, raw_pairs, skipped_pairs)
            relations     — non-UNRELATED Relation objects (SUPPORT | CONTRADICT).
            raw_pairs     — RawNLIPair for every eligible pair including UNRELATED, with all
                            four NLI scores, so thresholds can be swept offline.
            skipped_pairs — one SkippedPair per pre-NLI gate rejection, for offline debug
                            when `raw_pairs` is empty or smaller than expected.
        """
        if len(rules) < 2:
            return [], [], []

        _gate = gate if gate is not None else _should_compare
        pipe = _get_nli_pipe(self._model_name, self._batch_size, self._device)

        # Generate eligible pairs
        eligible: list[tuple[int, int]] = []
        rejection_counts: dict[str, int] = {}
        skipped_pairs: list[SkippedPair] = []
        all_pairs = list(itertools.combinations(range(len(rules)), 2))

        for i, j in all_pairs:
            ok, reason = _gate(rules[i], rules[j])
            if ok:
                eligible.append((i, j))
            else:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                a, b = rules[i], rules[j]
                skipped_pairs.append(SkippedPair(
                    rule_id_a=a.canonical_id,
                    rule_id_b=b.canonical_id,
                    reason=reason,
                    pmcid=pmcid or None,
                    category_a=a.category,
                    category_b=b.category,
                    relation_type_a=a.relation_type.value if a.relation_type else None,
                    relation_type_b=b.relation_type.value if b.relation_type else None,
                    subject_entity_a=a.subject_entity,
                    subject_entity_b=b.subject_entity,
                    outcome_entity_a=a.outcome_entity,
                    outcome_entity_b=b.outcome_entity,
                ))
                logger.debug(
                    "[%s] RELATE skip (%s): [%s/%s] %r  vs  [%s/%s] %r",
                    pmcid, reason,
                    a.category, a.relation_type, a.subject_entity,
                    b.category, b.relation_type, b.subject_entity,
                )

        if rejection_counts:
            logger.info(
                "[%s] RELATE gate rejected %d/%d pairs: %s",
                pmcid,
                sum(rejection_counts.values()),
                len(all_pairs),
                ", ".join(f"{k}={v}" for k, v in sorted(rejection_counts.items())),
            )

        if not eligible:
            return [], [], skipped_pairs

        # Build bidirectional NLI input
        nli_inputs_ab: list[tuple[str, str]] = []
        nli_inputs_ba: list[tuple[str, str]] = []
        for i, j in eligible:
            nli_inputs_ab.append((rules[i].predicate_text, rules[j].predicate_text))
            nli_inputs_ba.append((rules[j].predicate_text, rules[i].predicate_text))

        scores_ab = _nli_scores(nli_inputs_ab, pipe)
        scores_ba = _nli_scores(nli_inputs_ba, pipe)

        relations: list[Relation] = []
        raw_pairs: list[RawNLIPair] = []

        for (i, j), s_ab, s_ba in zip(eligible, scores_ab, scores_ba):
            label = _classify_pair(
                rules[i], rules[j], s_ab, s_ba,
                self._entailment_threshold,
                self._contradiction_threshold,
            )
            classified = label if label is not None else RelationTypeLabel.UNRELATED

            raw_pairs.append(RawNLIPair(
                rule_id_a=rules[i].canonical_id,
                rule_id_b=rules[j].canonical_id,
                ent_a_to_b=round(s_ab.get("entailment", 0.0), 6),
                ent_b_to_a=round(s_ba.get("entailment", 0.0), 6),
                con_a_to_b=round(s_ab.get("contradiction", 0.0), 6),
                con_b_to_a=round(s_ba.get("contradiction", 0.0), 6),
                classified_label=classified,
            ))

            if label is None:
                continue  # UNRELATED — not added to relations

            # Store the score that is semantically meaningful for this label:
            #   SUPPORT     → entailment score (the signal that fired)
            #   CONTRADICT  → contradiction score
            if label == RelationTypeLabel.CONTRADICT:
                score_ab = s_ab.get("contradiction", 0.0)
                score_ba = s_ba.get("contradiction", 0.0)
            else:
                score_ab = s_ab.get("entailment", 0.0)
                score_ba = s_ba.get("entailment", 0.0)

            relations.append(Relation(
                rule_id_a=rules[i].canonical_id,
                rule_id_b=rules[j].canonical_id,
                relation_type=label,
                nli_score_a_to_b=score_ab,
                nli_score_b_to_a=score_ba,
            ))

        logger.info(
            "[%s] RELATE: %d pairs checked → %d non-UNRELATED relations "
            "(SUPPORT=%d, CONTRADICT=%d)",
            pmcid,
            len(eligible),
            len(relations),
            sum(1 for r in relations if r.relation_type == RelationTypeLabel.SUPPORT),
            sum(1 for r in relations if r.relation_type == RelationTypeLabel.CONTRADICT),
        )
        return relations, raw_pairs, skipped_pairs
