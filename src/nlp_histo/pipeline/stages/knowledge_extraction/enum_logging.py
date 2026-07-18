"""
Lightweight JSONL loggers for enum-coercion observations and malformed findings.

Two sinks:
  logs/enum_observations.jsonl   — one record per coerced/repaired enum value.
  logs/bad_findings.jsonl        — one record per Finding that failed validation.

Logging failures are swallowed, so the pipeline never breaks because of
telemetry I/O.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG_DIR_ENV = "NLP_HISTO_LOG_DIR"
_DEFAULT_LOG_DIR = "logs"

_ENUM_FILE = "enum_observations.jsonl"
_BAD_FINDING_FILE = "bad_findings.jsonl"


def _log_dir() -> Path:
    return Path(os.environ.get(_LOG_DIR_ENV, _DEFAULT_LOG_DIR))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(filename: str, record: dict[str, Any]) -> None:
    try:
        d = _log_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / filename
        line = json.dumps(record, ensure_ascii=False, default=str)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def log_enum_observation(
    *,
    field_name: str,
    raw_value: Any,
    valid_values: list[str],
    coerced_value: Any,
    reason: str,
    context: dict[str, Any] | None = None,
) -> None:
    record = {
        "ts": _now_iso(),
        "field_name": field_name,
        "raw_value": raw_value,
        "valid_values": valid_values,
        "coerced_value": coerced_value,
        "reason": reason,
    }
    if context:
        record["context"] = context
    _append_jsonl(_ENUM_FILE, record)


def log_bad_finding(*, raw: Any, error: str, context: dict[str, Any] | None = None) -> None:
    record = {
        "ts": _now_iso(),
        "error": error,
        "raw": raw,
    }
    if context:
        record["context"] = context
    _append_jsonl(_BAD_FINDING_FILE, record)
