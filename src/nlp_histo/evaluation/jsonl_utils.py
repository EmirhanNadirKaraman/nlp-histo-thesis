"""JSONL read/write utilities with Pydantic validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Generator, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def read_jsonl(path: Path, model: Type[T]) -> Generator[T, None, None]:
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield model.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"{path}:{lineno} — {exc}") from exc


def write_jsonl(path: Path, records: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for rec in records:
            fh.write(rec.model_dump_json() + "\n")


def append_jsonl(path: Path, record: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(record.model_dump_json() + "\n")


def exists_in_jsonl(path: Path, key: str, value: str) -> bool:
    """Return True if any line in the JSONL has json[key] == value."""
    if not path.exists():
        return False
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get(key) == value:
                    return True
            except json.JSONDecodeError:
                continue
    return False
