"""
Grounding filter for MAP findings.

Uses a cross-encoder NLI model to check whether each claim is actually
entailed by its cited verbatim source text.  Applied after MAP — filters
Finding objects whose verbatim_support does not entail the claim.
"""
from __future__ import annotations

import logging
import re

from ..models import AuditableSummary, Finding
from .nli_config import get_active_spec

logger = logging.getLogger(__name__)

_ACTIVE_SPEC = get_active_spec()
_DEFAULT_MODEL = _ACTIVE_SPEC.hf_id
_DEFAULT_BATCH_SIZE = _ACTIVE_SPEC.batch_size

# Module-level NLI pipeline singleton, shared across GroundingFilter instances
# and reused by RelateStage via relate_stage._get_nli_pipe() — one model load
# per process.
_NLI_PIPE_CACHE: dict[tuple[str, str | int, int], object] = {}


def _get_device() -> int | str:
    """Return the best available device: CUDA GPU, MPS (Apple Silicon), or CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return -1


class GroundingFilter:
    """
    NLI-based grounding filter.

    Parameters
    ----------
    threshold:
        Minimum entailment score (0–1) to consider a claim supported.
        Default 0.5 means entailment must be the dominant class.
    model_name:
        HuggingFace model for text-classification NLI.
        Must expose labels including "entailment".
    batch_size:
        Number of (premise, hypothesis) pairs per model forward pass.
        Larger values improve GPU/MPS throughput. Default 16.
    device:
        Device for inference. None = auto-detect (CUDA → MPS → CPU).
        Pass 0 for CUDA, "mps" for Apple Silicon, -1 to force CPU.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        model_name: str = _DEFAULT_MODEL,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        device: int | str | None = None,
    ) -> None:
        self.threshold = threshold
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device if device is not None else _get_device()

    # Public API

    def filter_findings(self, summary: AuditableSummary) -> AuditableSummary:
        """Delegates to filter_findings_with_scores; discards the dropped list."""
        kept, _ = self.filter_findings_with_scores(summary)
        return kept

    def filter_findings_with_scores(
        self, summary: AuditableSummary
    ) -> tuple[AuditableSummary, list[Finding]]:
        """
        Like filter_findings() but scores ALL findings in one NLI pass,
        writing grounding_score in-place on every finding (kept and dropped),
        and returns (kept_summary, dropped_findings).

        Use this in the runner to replace the separate filter_findings() +
        score_findings() two-step calls — one NLI batch instead of two.
        """
        if not summary.findings:
            return summary, []

        pairs = [(f.verbatim_support, f.claim) for f in summary.findings]
        scores = _score_pairs(pairs, self._pipe)

        kept: list[Finding] = []
        dropped: list[Finding] = []
        for finding, score in zip(summary.findings, scores):
            finding.grounding_score = score
            if score >= self.threshold:
                kept.append(finding)
            else:
                dropped.append(finding)

        return summary.model_copy(update={"findings": kept}), dropped

    # Internals

    @property
    def _pipe(self):
        """Return the NLI pipeline, loading it if not already cached."""
        global _NLI_PIPE_CACHE
        cache_key = (self._model_name, self._device, self._batch_size)
        if cache_key not in _NLI_PIPE_CACHE:
            from transformers import pipeline  # optional dep
            logger.info(
                "GroundingFilter: loading NLI model %r on device=%r batch_size=%d",
                self._model_name, self._device, self._batch_size,
            )
            _NLI_PIPE_CACHE[cache_key] = pipeline(
                "text-classification",
                model=self._model_name,
                top_k=None,
                device=self._device,
                batch_size=self._batch_size,
            )
        return _NLI_PIPE_CACHE[cache_key]

    def _entailment_mask(self, pairs: list[tuple[str, str]]) -> list[bool]:
        """
        Run NLI on all (premise, hypothesis) pairs in one batch.
        Returns a bool list: True if entailment score >= threshold.
        """
        scores = _score_pairs(pairs, self._pipe)
        return [s >= self.threshold for s in scores]


# Helpers

# DeBERTa-v3 shares a 512-token limit between premise and hypothesis, so
# _compute_premise_budget() sizes the premise from the actual hypothesis
# length to keep the joint sequence under the limit.
_MODEL_MAX_TOKENS = 512  # hard cap; model_max_length can be unreliable (1e30)
_PREMISE_BUDGET_FLOOR = 64  # never allocate less than this, even for huge hypotheses

_SENTENCIZER = None  # lazy-loaded spaCy pipeline (module-level singleton)


def _get_sentencizer():
    global _SENTENCIZER
    if _SENTENCIZER is None:
        import spacy  # available via scispacy
        # Rule-based sentencizer; fast but fragile on "et al.", "Fig.", decimals.
        # Mis-splits create smaller windows; max() across windows recovers.
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        _SENTENCIZER = nlp
    return _SENTENCIZER


def _token_len(text: str, tokenizer) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _compute_premise_budget(hyp: str, tokenizer) -> int:
    """
    Derive a safe premise token budget for one hypothesis.

    The HF pipeline tokenizes (premise, hyp) together as:
        [CLS] premise [SEP] hyp [SEP]
    so special tokens consume tokenizer.num_special_tokens_to_add(pair=True)
    slots (3 for DeBERTa-v3).  We subtract hypothesis tokens + specials from
    the model limit and clamp to a minimum floor.
    """
    try:
        model_max = min(tokenizer.model_max_length, _MODEL_MAX_TOKENS)
    except AttributeError:
        model_max = _MODEL_MAX_TOKENS

    try:
        n_special = tokenizer.num_special_tokens_to_add(pair=True)
    except Exception:
        n_special = 3  # [CLS] + 2x [SEP]

    hyp_tokens = _token_len(hyp, tokenizer)
    budget = model_max - hyp_tokens - n_special
    return max(budget, _PREMISE_BUDGET_FLOOR)


def _truncate_to_budget(text: str, budget: int, tokenizer) -> str:
    """Truncate *text* to at most *budget* tokens using the tokenizer vocab."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return text
    return tokenizer.decode(ids[:budget], skip_special_tokens=True)


def _split_on_sep(text: str, sep: str) -> list[str]:
    """Split *text* on *sep*, keeping the separator attached to each chunk."""
    if sep not in text:
        return [text]
    pattern = rf"[^{re.escape(sep)}]+(?:{re.escape(sep)}|$)"
    parts = [m.strip() for m in re.findall(pattern, text) if m.strip()]
    return parts or [text]


def _split_oversized(sent: str, budget: int, tokenizer) -> list[str]:
    """
    Split *sent* by ';' then ',' until every piece is ≤ budget tokens.
    Falls back to tokenizer-level truncation if no separator helps.
    """
    n = _token_len(sent, tokenizer)
    if n <= budget:
        return [sent]

    for sep in (";", ","):
        parts = _split_on_sep(sent, sep)
        if len(parts) > 1:
            result: list[str] = []
            for p in parts:
                result.extend(_split_oversized(p, budget, tokenizer))
            return result

    # No separator helped — truncate via tokenizer as last resort.
    logger.warning(
        "_split_oversized: unsplittable clause (%d tokens > budget %d) — truncating",
        n, budget,
    )
    return [_truncate_to_budget(sent, budget, tokenizer)]


def _split_windows(text: str, hyp: str, tokenizer, overlap: bool = True) -> list[str]:
    """
    Split *text* into windows that fit within the per-hypothesis token budget.

    The budget is derived from *hyp* so the joined (premise, hyp) sequence
    never exceeds the model's 512-token limit.

    Strategy:
    1. If the full text fits, return it as a single window.
    2. Sentencize with spaCy; split oversized sentences by ';' / ',' / truncate.
    3. Greedily pack chunks; flush when the exact joined token count would
       exceed the budget.  With overlap=True, the last chunk of each window is
       repeated as the first chunk of the next so boundary sentences appear with
       context in at least one window.
    """
    budget = _compute_premise_budget(hyp, tokenizer)

    if _token_len(text, tokenizer) <= budget:
        return [text]

    sentences = [s.text.strip() for s in _get_sentencizer()(text).sents if s.text.strip()]
    if not sentences:
        return [_truncate_to_budget(text, budget, tokenizer)]

    chunks: list[str] = []
    for sent in sentences:
        chunks.extend(_split_oversized(sent, budget, tokenizer))

    windows: list[str] = []
    current: list[str] = []

    for chunk in chunks:
        candidate = " ".join(current + [chunk]) if current else chunk
        if current and _token_len(candidate, tokenizer) > budget:
            # Flush the current window.
            windows.append(" ".join(current))

            if overlap:
                # Repeat the last chunk of the previous window for boundary context.
                overlap_chunk = current[-1]
                overlap_candidate = f"{overlap_chunk} {chunk}"
                if _token_len(overlap_candidate, tokenizer) <= budget:
                    current = [overlap_chunk, chunk]
                else:
                    # Overlap alone would exhaust the budget; skip it.
                    current = [chunk]
            else:
                current = [chunk]
        else:
            current.append(chunk)

    if current:
        windows.append(" ".join(current))

    return windows


def _score_pairs(pairs: list[tuple[str, str]], nli_pipe) -> list[float]:
    """
    Run NLI on a batch of (premise, hypothesis) pairs.
    Returns a float list of entailment scores in [0, 1].

    Long premises are split into sentence-boundary windows (budget computed per
    hypothesis so the shared 512-token limit is never exceeded).  The maximum
    entailment score across windows is returned so a supporting sentence at a
    boundary is not missed.  Empty-string premises score 0.0.
    """
    flat_inputs: list[dict] = []
    flat_pair_indices: list[int] = []

    scores: list[float] = [0.0] * len(pairs)

    for i, (premise, hyp) in enumerate(pairs):
        if not premise.strip():
            continue
        for window in _split_windows(premise, hyp, nli_pipe.tokenizer):
            flat_inputs.append({"text": window, "text_pair": hyp})
            flat_pair_indices.append(i)

    if flat_inputs:
        from tqdm.auto import tqdm  # noqa: PLC0415
        _bs = getattr(nli_pipe, "_batch_size", 16)
        batch_results = []
        with tqdm(total=len(flat_inputs), desc="NLI [grounding]", unit="sent", leave=False) as pbar:
            for start in range(0, len(flat_inputs), _bs):
                batch_results.extend(nli_pipe(flat_inputs[start:start + _bs], truncation=True))
                pbar.update(min(_bs, len(flat_inputs) - start))
        for pair_idx, result in zip(flat_pair_indices, batch_results):
            window_score = next(
                (s["score"] for s in result if s["label"].lower() == "entailment"),
                0.0,
            )
            if window_score > scores[pair_idx]:
                scores[pair_idx] = window_score

    return scores


def score_findings(findings: list[Finding], nli_pipe) -> None:
    """
    Write grounding_score in-place on every Finding without filtering any out.
    Use this when you want NLI scores available on all findings but do not want
    to drop anything — e.g. for caching the full scored set before Phase 2
    NORMALIZE decides its own threshold.
    """
    if not findings:
        return
    pairs = [(f.verbatim_support, f.claim) for f in findings]
    for finding, score in zip(findings, _score_pairs(pairs, nli_pipe)):
        finding.grounding_score = score


def filter_atomic_findings(
    findings: list[Finding],
    threshold: float,
    nli_pipe,
) -> list[Finding]:
    """
    Score each Finding's (verbatim_support, claim) pair via NLI, write
    grounding_score in-place, and return only findings at or above threshold.

    Parameters
    ----------
    findings:
        Flat list of Finding objects from one or more chunks.
    threshold:
        Minimum entailment score to retain a finding.
    nli_pipe:
        HuggingFace text-classification pipeline (already loaded).
        Pass GroundingFilter._pipe to reuse the cached model instance.
    """
    if not findings:
        return []

    pairs = [(f.verbatim_support, f.claim) for f in findings]
    scores = _score_pairs(pairs, nli_pipe)

    kept: list[Finding] = []
    for finding, score in zip(findings, scores):
        finding.grounding_score = score
        if score >= threshold:
            kept.append(finding)
        else:
            logger.debug(
                "filter_atomic_findings: dropped finding (score=%.3f < %.3f): %s",
                score, threshold, finding.claim,
            )

    return kept
