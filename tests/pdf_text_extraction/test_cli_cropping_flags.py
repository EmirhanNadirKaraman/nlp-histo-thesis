"""
Tests for the three cropping-related CLI flags added to ``runner.py``:

* ``--reconstruct-tables-from-lists`` / ``--no-reconstruct-tables-from-lists``
  → ``DoclingConfig.reconstruct_tables_from_lists``
* ``--merge-tables-by-caption`` / ``--no-merge-tables-by-caption``
  → ``CroppingConfig.merge_tables_by_caption``
* ``--expand-tables-with-footnotes`` / ``--no-expand-tables-with-footnotes``
  → ``CroppingConfig.expand_tables_with_footnotes``

The tests verify:

1. ``_parse_args`` defaults the flags to ``None`` (sentinel: no override).
2. ``--flag`` / ``--no-flag`` parse correctly to ``True`` / ``False``.
3. The runner's ``main()`` honors each flag by mutating the in-process
   ``PipelineConfig`` correctly — exercised by patching the eventual
   ``ParallelBatchRunner`` constructor and capturing the config that
   would have been handed to it.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from nlp_histo.pipeline.stages.pdf_text_extraction import runner as runner_mod


# _parse_args defaults


def test_parse_args_defaults_all_three_to_none() -> None:
    args, _ = runner_mod._parse_args([])
    assert args.reconstruct_tables_from_lists is None
    assert args.merge_tables_by_caption is None
    assert args.expand_tables_with_footnotes is None


# Positive flag forms


@pytest.mark.parametrize("flag,attr", [
    ("--reconstruct-tables-from-lists", "reconstruct_tables_from_lists"),
    ("--merge-tables-by-caption",        "merge_tables_by_caption"),
    ("--expand-tables-with-footnotes",   "expand_tables_with_footnotes"),
])
def test_positive_flag_parses_to_true(flag: str, attr: str) -> None:
    args, _ = runner_mod._parse_args([flag])
    assert getattr(args, attr) is True


@pytest.mark.parametrize("flag,attr", [
    ("--no-reconstruct-tables-from-lists", "reconstruct_tables_from_lists"),
    ("--no-merge-tables-by-caption",        "merge_tables_by_caption"),
    ("--no-expand-tables-with-footnotes",   "expand_tables_with_footnotes"),
])
def test_negative_flag_parses_to_false(flag: str, attr: str) -> None:
    args, _ = runner_mod._parse_args([flag])
    assert getattr(args, attr) is False


def test_all_three_flags_together() -> None:
    args, _ = runner_mod._parse_args([
        "--reconstruct-tables-from-lists",
        "--merge-tables-by-caption",
        "--expand-tables-with-footnotes",
    ])
    assert args.reconstruct_tables_from_lists is True
    assert args.merge_tables_by_caption is True
    assert args.expand_tables_with_footnotes is True


# Config threading via main()


class _CapturedConfig:
    """Captures the cfg passed to ParallelBatchRunner; aborts before .run()."""
    def __init__(self) -> None:
        self.cfg = None

    def __call__(self, cfg, max_workers=None):
        self.cfg = cfg
        return self  # ParallelBatchRunner instance stand-in

    def run(self, *args, **kwargs):
        return {"processed": 0, "failed": 0, "skipped": 0}

    def run_paths(self, *args, **kwargs):
        # main() pre-filters the PDF list and dispatches via run_paths(paths),
        # not run(pdf_dir) — mirror the real ParallelBatchRunner API.
        return {"processed": 0, "failed": 0, "skipped": 0}


def _run_main_and_capture(argv: list[str]) -> _CapturedConfig:
    captured = _CapturedConfig()
    with patch("nlp_histo.pipeline.stages.pdf_text_extraction.batch.ParallelBatchRunner",
               captured):
        runner_mod.main(argv)
    return captured


def test_defaults_preserve_config_values() -> None:
    """No flags → all three cfg fields stay at PipelineConfig defaults."""
    captured = _run_main_and_capture(["--pdf-dir", "/tmp/does_not_exist"])
    cfg = captured.cfg
    assert cfg is not None
    # PipelineConfig defaults: reconstruct/merge default False; expand_tables_with_
    # footnotes was frozen to True on 2026-05-21 (OFF collapses strict F1 by 35pp).
    assert cfg.docling.reconstruct_tables_from_lists is False
    assert cfg.cropping.merge_tables_by_caption is False
    assert cfg.cropping.expand_tables_with_footnotes is True


def test_all_flags_on_sets_cfg_true() -> None:
    captured = _run_main_and_capture([
        "--pdf-dir", "/tmp/does_not_exist",
        "--reconstruct-tables-from-lists",
        "--merge-tables-by-caption",
        "--expand-tables-with-footnotes",
    ])
    cfg = captured.cfg
    assert cfg.docling.reconstruct_tables_from_lists is True
    assert cfg.cropping.merge_tables_by_caption is True
    assert cfg.cropping.expand_tables_with_footnotes is True


def test_individual_flags_set_only_their_field() -> None:
    captured = _run_main_and_capture([
        "--pdf-dir", "/tmp/does_not_exist",
        "--reconstruct-tables-from-lists",
    ])
    cfg = captured.cfg
    assert cfg.docling.reconstruct_tables_from_lists is True
    assert cfg.cropping.merge_tables_by_caption is False
    assert cfg.cropping.expand_tables_with_footnotes is True  # default (frozen True 2026-05-21)


def test_negative_flag_explicitly_sets_false() -> None:
    """--no-reconstruct-tables-from-lists also works even though default is False —
    confirms the flag is wired to the override path."""
    captured = _run_main_and_capture([
        "--pdf-dir", "/tmp/does_not_exist",
        "--no-reconstruct-tables-from-lists",
    ])
    assert captured.cfg.docling.reconstruct_tables_from_lists is False


def test_existing_flags_still_work_alongside_new_ones() -> None:
    """Regression: the new flags don't interfere with existing CLI overrides."""
    captured = _run_main_and_capture([
        "--pdf-dir", "/tmp/does_not_exist",
        "--detector", "docling",
        "--tatr-threshold", "0.85",
        "--render-dpi", "200",
        "--no-two-pass",
        "--reconstruct-tables-from-lists",
        "--merge-tables-by-caption",
    ])
    cfg = captured.cfg
    # Existing knobs honored
    assert str(cfg.table_detector).endswith("DOCLING")
    assert cfg.tatr.threshold == 0.85
    assert cfg.tatr.render_dpi == 200
    assert cfg.two_pass.enabled is False
    # New knobs honored
    assert cfg.docling.reconstruct_tables_from_lists is True
    assert cfg.cropping.merge_tables_by_caption is True
    # The unmentioned new knob stays at default (frozen True 2026-05-21)
    assert cfg.cropping.expand_tables_with_footnotes is True


def test_footnote_thresholds_default_to_config_values() -> None:
    captured = _run_main_and_capture(["--pdf-dir", "/tmp/does_not_exist"])
    cfg = captured.cfg
    assert cfg.cropping.footnote_proximity_pts == 20.0
    assert cfg.cropping.text_footnote_proximity_pts == 8.0
    assert cfg.cropping.footnote_threshold_multiplier == 1.2


def test_footnote_threshold_flags_override_config() -> None:
    captured = _run_main_and_capture([
        "--pdf-dir", "/tmp/does_not_exist",
        "--expand-tables-with-footnotes",
        "--footnote-proximity-pts", "30",
        "--text-footnote-proximity-pts", "12",
        "--footnote-threshold-multiplier", "1.5",
    ])
    cfg = captured.cfg
    assert cfg.cropping.expand_tables_with_footnotes is True
    assert cfg.cropping.footnote_proximity_pts == 30.0
    assert cfg.cropping.text_footnote_proximity_pts == 12.0
    assert cfg.cropping.footnote_threshold_multiplier == 1.5


def test_footnote_threshold_individual_flag_independent() -> None:
    """Setting just one threshold doesn't disturb the others."""
    captured = _run_main_and_capture([
        "--pdf-dir", "/tmp/does_not_exist",
        "--footnote-threshold-multiplier", "1.5",
    ])
    cfg = captured.cfg
    assert cfg.cropping.footnote_proximity_pts == 20.0  # default
    assert cfg.cropping.text_footnote_proximity_pts == 8.0  # default
    assert cfg.cropping.footnote_threshold_multiplier == 1.5  # overridden
