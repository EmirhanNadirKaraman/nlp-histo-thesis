"""Tests for the unified YAML config loader."""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from pipeline.config_loader import load_config
from pipeline.stages.pdf_text_extraction.config import (
    LogLevel,
    TableDetectorType,
)


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "run.yaml"
    p.write_text(dedent(body))
    return p


def test_empty_yaml_uses_defaults(tmp_path: Path):
    p = _write(tmp_path, "")
    pdf, sumcfg = load_config(p)
    assert pdf.runtime.num_workers == 1
    assert pdf.runtime.log_level == LogLevel.INFO
    assert pdf.table_detector == TableDetectorType.HYBRID
    assert sumcfg.map.theta == 0.8


def test_partial_override_keeps_other_defaults(tmp_path: Path):
    p = _write(tmp_path, """
        pdf_extraction:
          runtime:
            num_workers: 8
            log_level: DEBUG
    """)
    pdf, _ = load_config(p)
    assert pdf.runtime.num_workers == 8
    assert pdf.runtime.log_level == LogLevel.DEBUG
    # untouched defaults survive
    assert pdf.runtime.seed == 42
    assert pdf.docling.do_ocr is False


def test_enum_coerced_from_string(tmp_path: Path):
    p = _write(tmp_path, """
        pdf_extraction:
          table_detector: tatr
          runtime:
            log_level: DEBUG
    """)
    pdf, _ = load_config(p)
    assert pdf.table_detector == TableDetectorType.TATR
    assert pdf.runtime.log_level == LogLevel.DEBUG


def test_path_field_wrapped(tmp_path: Path):
    p = _write(tmp_path, """
        pdf_extraction:
          paths:
            output_root: /tmp/custom-out
    """)
    pdf, _ = load_config(p)
    assert isinstance(pdf.paths.output_root, Path)
    assert str(pdf.paths.output_root) == "/tmp/custom-out"


def test_unknown_field_raises(tmp_path: Path):
    p = _write(tmp_path, """
        pdf_extraction:
          runtime:
            nonexistent: 1
    """)
    with pytest.raises(ValueError, match="Unknown config field"):
        load_config(p)


def test_summarization_section_independent(tmp_path: Path):
    p = _write(tmp_path, """
        summarization:
          map:
            theta: 0.65
          resolve:
            grounding_weight: 0.7
    """)
    _, sumcfg = load_config(p)
    assert sumcfg.map.theta == 0.65
    assert sumcfg.resolve.grounding_weight == 0.7
    # untouched defaults survive
    assert sumcfg.map.reject_theta == 0.2


def test_normalize_extra_synonyms_loaded_as_mapping(tmp_path: Path):
    """B-037: NormalizeConfig.extra_synonyms is a `dict[str, str]` and must
    pass through the loader without being mistaken for a nested dataclass."""
    p = _write(tmp_path, """
        summarization:
          normalize:
            extra_synonyms:
              acme corp: ACME
              foo bar: FOO
    """)
    _, sumcfg = load_config(p)
    assert sumcfg.normalize.extra_synonyms == {"acme corp": "ACME", "foo bar": "FOO"}


def test_tatr_render_dpi_overridable(tmp_path: Path):
    """B-034: TATRConfig.render_dpi is now a real config knob (was hardcoded
    `_RENDER_DPI = 150` at module level)."""
    p = _write(tmp_path, """
        pdf_extraction:
          tatr:
            render_dpi: 300
    """)
    pdf, _ = load_config(p)
    assert pdf.tatr.render_dpi == 300
