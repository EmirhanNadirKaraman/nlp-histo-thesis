"""Call Opus to generate silver findings from source cases."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from nlp_histo.evaluation.jsonl_utils import exists_in_jsonl, read_jsonl
from .prompts import EXTRACT_FINDINGS_TOOL, PROMPT_VERSION, SYSTEM_PROMPT, make_user_prompt
from nlp_histo.evaluation.schemas import SilverCaseResult, SilverFinding, SilverFindingScope, SourceCase

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_MAX_TOKENS = 8192
# No `temperature` is sent: claude-opus-4-7 rejects it ("temperature is deprecated
# for this model", B-072). The forced extract_findings tool keeps extraction
# variance low; the model uses its default sampling. See docs/BUGS.md B-072.

# Known category aliases the LLM produces that map to canonical values.
# Only documented aliases are listed — unknown invalid values are left for Pydantic to reject.
_CATEGORY_ALIASES: dict[str, str] = {
    "demographics": "demographic",  # canonical is singular (matches pipeline models.py:279, B-016)
}

# relation_type="demographic" is intentionally not repaired: RelationTypeEnum.demographic
# is used throughout the pipeline (group_stage, relate_stage, resolve_stage). Req 2's
# "unless the codebase strongly depends on it" clause applies here.


def _repair_finding_fields(item: dict, case_id: str = "") -> dict:
    """Repair known LLM alias mistakes before Pydantic validation.

    Returns a shallow copy; never mutates the input dict.
    Only aliases in _CATEGORY_ALIASES are touched — genuinely invalid values pass
    through unchanged so Pydantic raises a clear ValidationError.
    """
    item = dict(item)
    cat = item.get("category")
    if isinstance(cat, str) and cat in _CATEGORY_ALIASES:
        repaired = _CATEGORY_ALIASES[cat]
        logger.warning("Repaired category alias %r → %r in case %s", cat, repaired, case_id)
        item["category"] = repaired
    return item


def _parse_findings_from_tool_use(content_blocks: list, case_id: str = "") -> list[SilverFinding]:
    """Extract SilverFinding objects from tool-use response content blocks."""
    tool_use = next((b for b in content_blocks if b.type == "tool_use"), None)
    if tool_use is None:
        logger.warning("No tool_use block in response for %s", case_id)
        return []
    findings = []
    for item in tool_use.input.get("findings", []):
        try:
            item = _repair_finding_fields(item, case_id)
            scope_raw = item.pop("scope", {}) or {}
            scope = SilverFindingScope(**{k: v for k, v in scope_raw.items()
                                         if k in SilverFindingScope.model_fields})
            finding = SilverFinding(scope=scope, **{k: v for k, v in item.items()
                                                    if k in SilverFinding.model_fields})
            findings.append(finding)
        except Exception as exc:
            logger.warning("Skipping malformed finding in %s: %s — %s", case_id, item, exc)
    return findings


def _call_opus(
    text: str,
    path_string: str,
    model: str,
    api_key: str,
) -> list[SilverFinding]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=DEFAULT_MAX_TOKENS,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        tools=[EXTRACT_FINDINGS_TOOL],
        tool_choice={"type": "tool", "name": "extract_findings"},
        messages=[{"role": "user", "content": make_user_prompt(text, path_string)}],
    )
    return _parse_findings_from_tool_use(response.content, path_string)


class SilverGenerator:
    def __init__(
        self,
        source_cases_path: Path,
        output_path: Path,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
    ):
        self.source_cases_path = source_cases_path
        self.output_path = output_path
        self.model = model
        self.api_key = api_key or os.environ["ANTHROPIC_API_KEY"]

    def run(self) -> None:
        cases = list(read_jsonl(self.source_cases_path, SourceCase))
        logger.info("Loaded %d source cases from %s", len(cases), self.source_cases_path)

        skipped = 0
        generated = 0

        for case in cases:
            cache_key = f"{case.case_id}|{PROMPT_VERSION}|{self.model}"
            if exists_in_jsonl(self.output_path, "_cache_key", cache_key):
                logger.debug("Cache hit: %s", cache_key)
                skipped += 1
                continue

            logger.info("Generating silver labels for %s", case.case_id)
            try:
                findings = _call_opus(case.text, case.path_string, self.model, self.api_key)
            except Exception as exc:
                logger.error("Failed for %s: %s", case.case_id, exc)
                continue

            result = SilverCaseResult(
                case_id=case.case_id,
                pmcid=case.pmcid,
                te_id=case.te_id,
                prompt_version=PROMPT_VERSION,
                model=self.model,
                findings=findings,
            )

            raw_dict = result.model_dump()
            raw_dict["_cache_key"] = cache_key

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a") as fh:
                fh.write(json.dumps(raw_dict) + "\n")

            logger.info("  → %d finding(s)", len(findings))
            generated += 1

        logger.info("Done. Generated: %d  Skipped (cached): %d", generated, skipped)

    def run_batch(self, batch_state_path: Path | None = None) -> None:
        """
        Submit all uncached cases to the Anthropic batch API (~50% cheaper).

        Poll-and-resume: persists batch_id to disk. Re-run the same command to
        check status and collect results once the batch completes.

        batch_state_path defaults to output_path with .batch.json suffix.
        """
        import anthropic

        cases = list(read_jsonl(self.source_cases_path, SourceCase))
        logger.info("Loaded %d source cases from %s", len(cases), self.source_cases_path)

        batch_state_path = batch_state_path or self.output_path.with_suffix(".batch.json")
        client = anthropic.Anthropic(api_key=self.api_key)

        # Resume: batch already submitted
        if batch_state_path.exists():
            state = json.loads(batch_state_path.read_text())
            batch_id = state["batch_id"]
            # custom_id → original case_id mapping (pipes replaced with underscores for API)
            id_map: dict[str, str] = state["id_map"]  # safe_id → case_id
            logger.info("Resuming batch %s (%d requests)", batch_id, len(id_map))

            batch = client.beta.messages.batches.retrieve(batch_id)
            counts = batch.request_counts
            logger.info(
                "Status: %s  succeeded=%d  errored=%d  processing=%d  canceled=%d",
                batch.processing_status,
                counts.succeeded, counts.errored, counts.processing, counts.canceled,
            )

            if batch.processing_status != "ended":
                logger.info("Batch not done yet — re-run to check status.")
                return

            # Collect results
            case_by_id = {c.case_id: c for c in cases}
            generated = errored = 0
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            for item in client.beta.messages.batches.results(batch_id):
                safe_id = item.custom_id
                case_id = id_map.get(safe_id)
                case = case_by_id.get(case_id) if case_id else None
                if case is None:
                    logger.warning("Unknown custom_id in batch result: %s", safe_id)
                    continue

                if item.result.type == "succeeded":
                    findings = _parse_findings_from_tool_use(
                        item.result.message.content, case_id
                    )
                    result = SilverCaseResult(
                        case_id=case.case_id,
                        pmcid=case.pmcid,
                        te_id=case.te_id,
                        prompt_version=PROMPT_VERSION,
                        model=self.model,
                        findings=findings,
                    )
                    raw_dict = result.model_dump()
                    raw_dict["_cache_key"] = f"{case.case_id}|{PROMPT_VERSION}|{self.model}"
                    with self.output_path.open("a") as fh:
                        fh.write(json.dumps(raw_dict) + "\n")
                    logger.info("  %s → %d finding(s)", case_id, len(findings))
                    generated += 1
                else:
                    logger.error("Batch request failed for %s: %s",
                                 case_id, getattr(item.result, "error", "unknown"))
                    errored += 1

            batch_state_path.unlink()
            logger.info("Batch complete. Generated: %d  Errored: %d", generated, errored)
            return

        # First run: filter uncached and submit
        uncached = []
        for case in cases:
            cache_key = f"{case.case_id}|{PROMPT_VERSION}|{self.model}"
            if not exists_in_jsonl(self.output_path, "_cache_key", cache_key):
                uncached.append(case)

        if not uncached:
            logger.info("All %d cases already cached — nothing to do.", len(cases))
            return

        # Anthropic custom_id: 1-64 chars, [a-zA-Z0-9_-] only
        id_map: dict[str, str] = {}  # safe_id → case_id
        for case in uncached:
            safe_id = case.case_id.replace("|", "_")
            id_map[safe_id] = case.case_id

        logger.info("Submitting %d uncached cases to Anthropic batch API…", len(uncached))
        requests = [
            {
                "custom_id": case.case_id.replace("|", "_"),
                "params": {
                    "model": self.model,
                    "max_tokens": DEFAULT_MAX_TOKENS,
                    "system": [{
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }],
                    "tools": [EXTRACT_FINDINGS_TOOL],
                    "tool_choice": {"type": "tool", "name": "extract_findings"},
                    "messages": [{"role": "user", "content": make_user_prompt(
                        case.text, case.path_string
                    )}],
                },
            }
            for case in uncached
        ]

        batch = client.beta.messages.batches.create(requests=requests)
        state = {"batch_id": batch.id, "id_map": id_map}
        batch_state_path.write_text(json.dumps(state, indent=2))
        logger.info(
            "Batch %s submitted with %d requests. State saved to %s\n"
            "Re-run this command to check status and collect results.",
            batch.id, len(requests), batch_state_path,
        )
