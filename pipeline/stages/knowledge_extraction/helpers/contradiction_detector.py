"""
Contradiction detector for extracted rules.

Two-step process:
  1. Embedding filter — embed each rule's condition+action and compute pairwise
     cosine similarity.  Only pairs above `similarity_threshold` are candidate
     contradictions (same topic).
  2. LLM judge — for each candidate pair, ask a cheap LLM whether the two rules
     genuinely contradict each other.

Output is a ContradictionReport grouping rules into contradicting pairs and
non-contradicting rules.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
import numpy as np

from ..agreement.providers import EmbedFn, OpenAIEmbedder
from ..models import ContradictingPair, ContradictionReport, ExtractedRules, Rule
from ..prompts import build_judge_chain

logger = logging.getLogger(__name__)


class ContradictionDetector:
    """
    Detects contradicting rules using embedding similarity + LLM judge.

    Parameters
    ----------
    llm:
        LangChain chat model used as the pairwise judge.  A cheap model
        (e.g. gpt-4o-mini) is sufficient — the task is a simple binary
        judgment with a short prompt.
    similarity_threshold:
        Minimum cosine similarity between two rule embeddings to consider
        them as covering the same topic and worth sending to the LLM judge.
        Lower = more judge calls; higher = may miss some contradictions.
    embed_fn:
        Embedding function.  Defaults to OpenAI text-embedding-3-small.
    """

    def __init__(
        self,
        llm,
        similarity_threshold: float = 0.7,
        embed_fn: EmbedFn | None = None,
    ) -> None:
        self._judge = build_judge_chain(llm)
        self.similarity_threshold = similarity_threshold
        self._embed: EmbedFn = embed_fn or OpenAIEmbedder()

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(self, extracted: ExtractedRules) -> ContradictionReport:
        """
        Run contradiction detection on a set of extracted rules.

        Returns a ContradictionReport with:
          - contradicting_pairs: rules that conflict with each other
          - non_contradicting_rules: rules not involved in any contradiction
        """
        rules = extracted.rules
        if len(rules) < 2:
            return ContradictionReport(
                contradicting_pairs=[],
                non_contradicting_rules=rules,
            )

        candidate_pairs = self._candidate_pairs(rules)
        logger.debug(
            "Contradiction: %d rules → %d candidate pairs (threshold=%.2f)",
            len(rules), len(candidate_pairs), self.similarity_threshold,
        )

        contradicting_pairs = self._judge_pairs(candidate_pairs)

        contradicting_ids = {
            rid
            for pair in contradicting_pairs
            for rid in (pair.rule_id_a, pair.rule_id_b)
        }
        non_contradicting = [r for r in rules if r.rule_id not in contradicting_ids]

        logger.info(
            "Contradiction: %d pair(s) found, %d/%d rules non-contradicting",
            len(contradicting_pairs), len(non_contradicting), len(rules),
        )

        return ContradictionReport(
            contradicting_pairs=contradicting_pairs,
            non_contradicting_rules=non_contradicting,
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    def _candidate_pairs(self, rules: list[Rule]) -> list[tuple[Rule, Rule]]:
        """Return rule pairs whose embeddings exceed the similarity threshold."""
        texts = [f"{r.condition} {r.action}" for r in rules]
        embs = np.array(self._embed(texts))
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        sim = embs @ embs.T  # (n, n)

        pairs = []
        for (i, rule_a), (j, rule_b) in combinations(enumerate(rules), 2):
            if sim[i, j] >= self.similarity_threshold:
                pairs.append((rule_a, rule_b))
        return pairs

    def _judge_pairs(
        self, pairs: list[tuple[Rule, Rule]]
    ) -> list[ContradictingPair]:
        """Call the LLM judge on all candidate pairs concurrently."""
        if not pairs:
            return []

        results: list[ContradictingPair | None] = [None] * len(pairs)

        def _judge_one(idx: int, rule_a: Rule, rule_b: Rule) -> tuple[int, ContradictingPair | None]:
            judgment = self._judge.invoke({
                "condition_a": rule_a.condition,
                "action_a": rule_a.action,
                "condition_b": rule_b.condition,
                "action_b": rule_b.action,
            })
            if judgment.contradicts:
                return idx, ContradictingPair(
                    rule_id_a=rule_a.rule_id,
                    rule_id_b=rule_b.rule_id,
                    rule_a=rule_a,
                    rule_b=rule_b,
                    explanation=judgment.explanation,
                )
            return idx, None

        with ThreadPoolExecutor(max_workers=min(len(pairs), 8)) as pool:
            futures = {
                pool.submit(_judge_one, i, a, b): i
                for i, (a, b) in enumerate(pairs)
            }
            for future in as_completed(futures):
                idx, pair = future.result()
                results[idx] = pair

        return [r for r in results if r is not None]
