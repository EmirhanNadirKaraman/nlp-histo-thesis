"""
UsageCollector — thread-safe accumulator of ``LLMUsageRecord``s plus a
LangChain ``BaseCallbackHandler`` that extracts token usage from chat
model responses.

Integration pattern
-------------------
1.  Runner creates one ``UsageCollector(run_id, paper_id)`` per paper.
2.  For each chain.invoke (voter / escalator / judge), call
    ``collector.attach(...)`` which returns a ``config={"callbacks": [...]}`` dict
    pre-bound with the call's metadata (stage, role, cascade_level, …).
    Pass that ``config`` to ``chain.invoke(inp, config=config)``.
3.  Cache hits / batch results / cascade overhead are recorded via the explicit
    ``record_*`` methods (no LLM response is involved).
4.  At the end of the run, flush with ``collector.write_jsonl(path)`` and pass
    the records to ``write_cost_reports``.

The callback is no-op if the underlying response has no usage metadata. We
never fabricate tokens — ``source="estimate"`` records are explicit.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from .models import LLMUsageRecord, append_records_jsonl, normalize_token_usage

logger = logging.getLogger(__name__)

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # langchain-core is a hard dep elsewhere; guard for tests
    BaseCallbackHandler = object  # type: ignore[assignment,misc]


class UsageCollector:
    """In-memory store of LLMUsageRecord. Thread-safe."""

    def __init__(self, run_id: str, paper_id: str | None = None) -> None:
        self.run_id = run_id
        self.paper_id = paper_id
        self._records: list[LLMUsageRecord] = []
        self._lock = threading.Lock()

    # Inspection

    def records(self) -> list[LLMUsageRecord]:
        with self._lock:
            return list(self._records)

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    # Direct recording APIs (used for batch / cache / explicit cases)

    def record(self, **fields: Any) -> LLMUsageRecord:
        fields.setdefault("run_id", self.run_id)
        fields.setdefault("paper_id", self.paper_id)
        rec = LLMUsageRecord(**fields)
        with self._lock:
            self._records.append(rec)
        return rec

    def record_live(
        self,
        *,
        stage: str,
        model: str,
        provider: str,
        role: str,
        input_tokens: int,
        output_tokens: int,
        cascade_level: int | None = None,
        substage: str | None = None,
        item_id: str | None = None,
        was_escalation: bool = False,
        agreement_score: float | None = None,
        raw_usage_metadata: dict[str, Any] | None = None,
    ) -> LLMUsageRecord:
        return self.record(
            stage=stage, model=model, provider=provider, role=role,
            source="live_call", call_status="success",
            input_tokens=input_tokens, output_tokens=output_tokens,
            cascade_level=cascade_level, substage=substage, item_id=item_id,
            was_escalation=was_escalation, agreement_score=agreement_score,
            raw_usage_metadata=raw_usage_metadata,
        )

    def record_batch(
        self,
        *,
        stage: str,
        model: str,
        provider: str,
        role: str,
        input_tokens: int,
        output_tokens: int,
        batch_id: str | None = None,
        cascade_level: int | None = None,
        substage: str | None = None,
        item_id: str | None = None,
    ) -> LLMUsageRecord:
        return self.record(
            stage=stage, model=model, provider=provider, role=role,
            source="batch_result", call_status="success",
            input_tokens=input_tokens, output_tokens=output_tokens,
            batch_used=True, batch_id=batch_id,
            cascade_level=cascade_level, substage=substage, item_id=item_id,
        )

    def record_cache_hit(
        self,
        *,
        stage: str,
        model: str = "",
        provider: str = "",
        role: str = "unknown",
        substage: str | None = None,
        item_id: str | None = None,
    ) -> LLMUsageRecord:
        return self.record(
            stage=stage, model=model, provider=provider, role=role,
            source="cache", call_status="cached",
            cache_hit=True, substage=substage, item_id=item_id,
        )

    def record_failed(
        self,
        *,
        stage: str,
        model: str,
        provider: str,
        role: str,
        substage: str | None = None,
        item_id: str | None = None,
    ) -> LLMUsageRecord:
        return self.record(
            stage=stage, model=model, provider=provider, role=role,
            source="live_call", call_status="failed",
            substage=substage, item_id=item_id,
        )

    # LangChain integration

    def callback(self, **context: Any) -> "UsageCallbackHandler":
        """Build a one-shot callback handler bound to the given call context.

        ``context`` keys are stored on the resulting record (stage / role /
        model / provider / cascade_level / substage / item_id / …).  Reuse the
        same collector across all chains; create a fresh callback per invoke().
        """
        return UsageCallbackHandler(self, context)

    def attach(self, **context: Any) -> dict[str, Any]:
        """Convenience: ``chain.invoke(inp, config=collector.attach(...))``."""
        return {"callbacks": [self.callback(**context)]}

    # Persistence

    def write_jsonl(self, path) -> int:
        with self._lock:
            return append_records_jsonl(self._records, path)


class UsageCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
    """LangChain callback that converts every LLM completion into a record.

    The handler ignores callbacks where usage data is absent (e.g. partial
    streaming events). It never raises — a costing failure must not break the
    pipeline.
    """

    raise_error = False
    run_inline = True

    def __init__(self, collector: UsageCollector, context: dict[str, Any]) -> None:
        self._collector = collector
        self._ctx = context

    # The same handler shape works for chat models and plain LLMs.
    def on_llm_end(self, response, **kwargs) -> None:  # noqa: D401 — langchain hook
        try:
            usage = _extract_usage(response)
            if usage is None:
                return
            self._collector.record_live(
                stage=self._ctx.get("stage", "unknown"),
                model=self._ctx.get("model", "") or _infer_model(response),
                provider=self._ctx.get("provider", "") or "",
                role=self._ctx.get("role", "unknown"),
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cascade_level=self._ctx.get("cascade_level"),
                substage=self._ctx.get("substage"),
                item_id=self._ctx.get("item_id"),
                was_escalation=bool(self._ctx.get("was_escalation", False)),
                agreement_score=self._ctx.get("agreement_score"),
                raw_usage_metadata=usage.get("raw"),
            )
        except Exception as exc:  # never propagate
            logger.debug("UsageCallbackHandler error (ignored): %s", exc)

    on_chat_model_end = on_llm_end  # alias — some versions emit this instead


def _extract_usage(response) -> dict[str, Any] | None:
    """Pull token counts from a LangChain ``LLMResult``.

    Looks at, in order:
      - response.llm_output['token_usage']
      - response.generations[0][0].message.usage_metadata
      - response.generations[0][0].message.response_metadata['token_usage']
      - response.usage_metadata (top-level — rare)
    """
    raw_payload: Any = None

    llm_output = getattr(response, "llm_output", None) or {}
    if isinstance(llm_output, dict):
        raw_payload = llm_output.get("token_usage") or llm_output.get("usage")

    if not raw_payload:
        gens = getattr(response, "generations", None) or []
        if gens and gens[0]:
            gen0 = gens[0][0]
            msg = getattr(gen0, "message", None)
            if msg is not None:
                raw_payload = (
                    getattr(msg, "usage_metadata", None)
                    or (getattr(msg, "response_metadata", None) or {}).get("token_usage")
                    or (getattr(msg, "response_metadata", None) or {}).get("usage")
                )
            if not raw_payload:
                gen_info = getattr(gen0, "generation_info", None) or {}
                raw_payload = gen_info.get("token_usage") or gen_info.get("usage")

    if not raw_payload:
        raw_payload = getattr(response, "usage_metadata", None)

    if not raw_payload:
        return None

    normalized = normalize_token_usage(raw_payload)
    if normalized["input_tokens"] == 0 and normalized["output_tokens"] == 0:
        return None
    normalized["raw"] = raw_payload if isinstance(raw_payload, dict) else None
    return normalized


def _infer_model(response) -> str:
    out = getattr(response, "llm_output", None) or {}
    if isinstance(out, dict):
        return str(out.get("model_name") or out.get("model") or "")
    return ""
