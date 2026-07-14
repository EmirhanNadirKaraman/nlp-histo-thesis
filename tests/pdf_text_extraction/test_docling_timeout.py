"""Regression test for B-035 — DoclingConfig.timeout_sec now enforced.

Pre-fix: `timeout_sec` was a documented but unused dataclass field; a
hanging Docling conversion blocked the entire batch indefinitely. Post-fix:
`DoclingLayoutExtractor._convert_with_timeout` wraps the convert call in a
single-worker ThreadPoolExecutor with `future.result(timeout=)`.
"""
from __future__ import annotations

import time

import pytest

from nlp_histo.pipeline.stages.pdf_text_extraction.components.layout_extractor import (
    DoclingLayoutExtractor,
)
from nlp_histo.pipeline.stages.pdf_text_extraction.config import DoclingConfig


class _SleepConverter:
    """Stand-in for `docling.DocumentConverter` that just sleeps."""

    def __init__(self, sleep_sec: float):
        self._sleep_sec = sleep_sec
        self.invocations = 0

    def convert(self, _path):
        self.invocations += 1
        time.sleep(self._sleep_sec)
        return object()  # caller checks the timeout path before reading


def _make(timeout_sec: int | None) -> DoclingLayoutExtractor:
    cfg = DoclingConfig()
    cfg.timeout_sec = timeout_sec  # type: ignore[assignment]
    return DoclingLayoutExtractor(config=cfg)


def test_timeout_raises_when_converter_exceeds_budget(tmp_path):
    extractor = _make(timeout_sec=1)
    converter = _SleepConverter(sleep_sec=5.0)
    pdf = tmp_path / "doc.pdf"
    pdf.touch()

    with pytest.raises(TimeoutError, match="exceeded 1s on doc.pdf"):
        extractor._convert_with_timeout(converter, pdf)


def test_no_timeout_when_disabled(tmp_path):
    """timeout_sec=0 disables the guard — work runs in the calling thread."""
    extractor = _make(timeout_sec=0)
    converter = _SleepConverter(sleep_sec=0.05)
    pdf = tmp_path / "doc.pdf"
    pdf.touch()

    result = extractor._convert_with_timeout(converter, pdf)
    assert result is not None
    assert converter.invocations == 1


def test_work_under_budget_succeeds(tmp_path):
    extractor = _make(timeout_sec=5)
    converter = _SleepConverter(sleep_sec=0.05)
    pdf = tmp_path / "doc.pdf"
    pdf.touch()

    result = extractor._convert_with_timeout(converter, pdf)
    assert result is not None
