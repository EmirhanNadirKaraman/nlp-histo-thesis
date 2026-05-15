"""Unified YAML loader for the two pipeline configs.

A single YAML file drives both `PipelineConfig` (PDF extraction) and
`SummarizationConfig` (summarisation). Missing keys fall back to dataclass
defaults; unknown keys raise. Nested dataclasses recurse; Enum fields
accept their `.value` string; `Path` fields accept any string.

Usage::

    from pipeline.config_loader import load_config

    pdf_cfg, sum_cfg = load_config("configs/run.yaml")
    PipelineRunner(pdf_cfg).run_batch("files/organized_pdfs")
    SummarizationRunner(..., config=sum_cfg).process(...)
"""
from __future__ import annotations

import types
import typing
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path

import yaml

from pipeline.stages.pdf_text_extraction.config import PipelineConfig
from pipeline.stages.summarization.config import SummarizationConfig


def load_config(yaml_path: str | Path) -> tuple[PipelineConfig, SummarizationConfig]:
    raw = yaml.safe_load(Path(yaml_path).read_text()) or {}
    pdf_cfg = PipelineConfig()
    sum_cfg = SummarizationConfig()
    if raw.get("pdf_extraction"):
        _apply(pdf_cfg, raw["pdf_extraction"])
    if raw.get("summarization"):
        _apply(sum_cfg, raw["summarization"])
    pdf_cfg.prepare()
    return pdf_cfg, sum_cfg


def _apply(instance, overrides: dict) -> None:
    if not is_dataclass(instance):
        raise TypeError(f"Cannot apply overrides to non-dataclass: {type(instance).__name__}")
    type_hints = _get_type_hints(type(instance))
    field_names = {f.name for f in fields(instance)}
    for key, value in overrides.items():
        if key not in field_names:
            raise ValueError(
                f"Unknown config field: {type(instance).__name__}.{key}. "
                f"Known fields: {sorted(field_names)}"
            )
        field_type = type_hints.get(key)
        current = getattr(instance, key)
        setattr(instance, key, _coerce(current, value, field_type))


def _coerce(current, value, field_type):
    if value is None:
        return None
    if isinstance(value, dict):
        # Plain mapping field (e.g. `dict[str, str]`) — pass through as-is.
        # Without this, every dict YAML node is treated as a nested-dataclass
        # override and `extra_synonyms: {acme: ACME}` would crash.
        if _is_mapping_type(field_type):
            return value
        nested = current if is_dataclass(current) else _instantiate_dataclass(field_type)
        if nested is None:
            raise TypeError(
                f"Field type {field_type!r} does not resolve to a dataclass; "
                f"cannot apply nested override {value!r}"
            )
        _apply(nested, value)
        return nested
    enum_cls = _resolve_enum_type(field_type)
    if enum_cls is not None:
        return enum_cls(value)
    if _accepts_path(field_type):
        return Path(value)
    return value


def _get_type_hints(cls) -> dict:
    try:
        return typing.get_type_hints(cls)
    except Exception:
        return {}


def _unwrap_optional(t):
    # Handle both `typing.Union[X, None]` and PEP-604 `X | None`. The latter
    # has origin `types.UnionType`, not `typing.Union`, so the original check
    # silently failed for `dict[str, str] | None` — extra_synonyms then got
    # routed through the dataclass branch and crashed.
    origin = typing.get_origin(t)
    if origin is typing.Union or origin is types.UnionType:
        non_none = [a for a in typing.get_args(t) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return t


def _resolve_enum_type(t):
    inner = _unwrap_optional(t)
    if isinstance(inner, type) and issubclass(inner, Enum):
        return inner
    return None


def _accepts_path(t) -> bool:
    inner = _unwrap_optional(t)
    return inner is Path


def _instantiate_dataclass(t):
    inner = _unwrap_optional(t)
    if isinstance(inner, type) and is_dataclass(inner):
        return inner()
    return None


def _is_mapping_type(t) -> bool:
    inner = _unwrap_optional(t)
    origin = typing.get_origin(inner)
    if origin in (dict, typing.Dict):
        return True
    return isinstance(inner, type) and issubclass(inner, dict)
