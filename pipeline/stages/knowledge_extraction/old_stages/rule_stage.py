"""
RULE EXTRACTION stage: ConsolidatedSummary → ExtractedRules, with caching.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from ..cache import PipelineCache
from ..models import ConsolidatedSummary, ExtractedRules
from ..prompts import build_rule_chain

logger = logging.getLogger(__name__)


class RuleStage:
    """
    Extract IF-THEN clinical rules from a ConsolidatedSummary.

    Parameters
    ----------
    llm:
        LLM for rule extraction (typically the smart model).
    """

    def __init__(self, llm) -> None:
        self._chain = build_rule_chain(llm)

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(
        self,
        summary: ConsolidatedSummary,
        pmcid: str,
        cache: Optional[PipelineCache] = None,
        collector=None,  # TraceCollector | None
    ) -> ExtractedRules:
        cache_hit = False
        if cache:
            hit = cache.get_rule(summary, pmcid)
            if hit:
                cache_hit = True
                result = hit
            else:
                result = self._invoke_and_cache(summary, pmcid, cache)
        else:
            result = self._invoke_and_cache(summary, pmcid, cache)

        if collector is not None:
            # Count input findings across all sections
            s = summary.sections
            finding_count_in = (
                len(s.clinical_significance.findings)
                + len(s.histopathological_features.findings)
                + len(s.management_outcomes.findings)
                + len(s.risk_factors_associations.findings)
            )
            rules_by_type: dict[str, int] = {}
            for rule in result.rules:
                t = rule.type.value if hasattr(rule.type, "value") else str(rule.type)
                rules_by_type[t] = rules_by_type.get(t, 0) + 1
            collector.record_rules(
                finding_count_in=finding_count_in,
                rules_extracted=len(result.rules),
                rules_by_type=rules_by_type,
                cache_hit=cache_hit,
            )

        return result

    # ── Internals ──────────────────────────────────────────────────────────────

    def _invoke_and_cache(
        self,
        summary: ConsolidatedSummary,
        pmcid: str,
        cache: Optional[PipelineCache],
    ) -> ExtractedRules:
        result: ExtractedRules = self._chain.invoke(
            {
                "pmcid": pmcid,
                "summary": json.dumps(
                    summary.model_dump(exclude_none=True), separators=(",", ":")
                ),
            }
        )
        if cache:
            cache.set_rule(summary, pmcid, result)
        return result
