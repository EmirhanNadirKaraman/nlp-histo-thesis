"""
Normalized LLM usage records + JSONL persistence helpers.

One ``LLMUsageRecord`` is written per LLM call (live, batched, or cache hit).
The JSONL file is the canonical, reproducible source of truth — cost reports
can always be re-derived from it.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


_VALID_STATUS = {"success", "failed", "cached", "skipped"}
_VALID_SOURCE = {"live_call", "batch_result", "cache", "estimate"}
_VALID_ROLE = {
    "voter", "escalator", "judge", "mapper", "normalizer",
    "relation_classifier", "resolver", "reducer", "rule_extractor",
    "contradiction_detector", "unknown",
}


@dataclass
class LLMUsageRecord:
    """One normalized record per LLM call.

    See module docstring for semantics. All optional/None fields are omitted
    from the JSON line when ``to_dict(compact=True)`` is used.
    """

    run_id: str
    stage: str                       # "MAP" | "REDUCE" | "RULES" | "RELATE" | …
    model: str
    provider: str
    role: str                        # one of _VALID_ROLE
    call_status: str = "success"     # one of _VALID_STATUS
    source: str = "live_call"        # one of _VALID_SOURCE

    paper_id: str | None = None
    substage: str | None = None      # free-form, e.g. "l1_voter", "l3_escalation"
    item_id: str | None = None       # chunk_id / finding_id / pair_id

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int | None = None

    batch_used: bool = False
    batch_id: str | None = None
    cache_hit: bool = False

    cascade_level: int | None = None        # 1 / 2 / 3
    cascade_item_id: str | None = None      # cascade-step identifier
    was_escalation: bool = False
    accepted_without_escalation: bool | None = None
    agreement_score: float | None = None

    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    raw_usage_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.role not in _VALID_ROLE:
            self.role = "unknown"
        if self.call_status not in _VALID_STATUS:
            raise ValueError(f"invalid call_status: {self.call_status}")
        if self.source not in _VALID_SOURCE:
            raise ValueError(f"invalid source: {self.source}")
        if self.total_tokens == 0 and (self.input_tokens or self.output_tokens):
            self.total_tokens = self.input_tokens + self.output_tokens

    def to_dict(self, compact: bool = True) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        if compact:
            return {k: v for k, v in d.items() if v not in (None, {}, [])}
        return d


def append_records_jsonl(records: Iterable[LLMUsageRecord], path: Path) -> int:
    """Append records as JSONL. Returns number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def load_records(path: Path) -> list[LLMUsageRecord]:
    """Read a JSONL file back into LLMUsageRecord objects. Missing → []."""
    if not path.exists():
        return []
    out: list[LLMUsageRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        fields = {f.name for f in dataclasses.fields(LLMUsageRecord)}
        kept = {k: v for k, v in d.items() if k in fields}
        out.append(LLMUsageRecord(**kept))
    return out


def normalize_token_usage(raw: Any) -> dict[str, int]:
    """Normalize a token-usage dict from any common shape.

    Recognizes:
      - LangChain ``usage_metadata`` : {input_tokens, output_tokens, total_tokens}
      - OpenAI ``usage``             : {prompt_tokens, completion_tokens, total_tokens}
      - Anthropic                    : {input_tokens, output_tokens}
      - Gemini                       : {prompt_token_count, candidates_token_count, total_token_count}
      - Nested response_metadata.token_usage / llm_output.token_usage

    Returns dict with keys: input_tokens, output_tokens, total_tokens. Missing
    values default to 0. Returns all-zeros when ``raw`` is None / unrecognizable.
    """
    if raw is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    elif hasattr(raw, "__dict__") and not isinstance(raw, dict):
        raw = {k: v for k, v in vars(raw).items() if not k.startswith("_")}

    if not isinstance(raw, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # Unwrap one level of nesting if present.
    for nested_key in ("token_usage", "usage", "usage_metadata"):
        if nested_key in raw and isinstance(raw[nested_key], dict):
            raw = {**raw, **raw[nested_key]}

    input_tokens = (
        raw.get("input_tokens")
        or raw.get("prompt_tokens")
        or raw.get("prompt_token_count")
        or 0
    )
    output_tokens = (
        raw.get("output_tokens")
        or raw.get("completion_tokens")
        or raw.get("candidates_token_count")
        or 0
    )
    total_tokens = (
        raw.get("total_tokens")
        or raw.get("total_token_count")
        or (input_tokens + output_tokens)
    )
    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }
