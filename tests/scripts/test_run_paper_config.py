"""Tests for the ``--config`` wiring added to ``scripts/run_paper.py``.

Confirms that:
1. ``_load_summarization_config`` reads ``configs/run.yaml::knowledge_extraction.*``
   into a ``KnowledgeExtractionConfig`` instance.
2. Passing an empty / missing path falls back to dataclass defaults.
3. Different grounding thresholds produce different ``pipeline_config_hash``
   values — this is the load-bearing test for Run A vs Run B comparability.

No LLM / API calls. No DB writes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.paths import REPO_ROOT


@pytest.fixture()
def run_paper_module():
    """The knowledge workflow, imported from the installed package."""
    from nlp_histo.workflows import knowledge

    yield knowledge


def _write_minimal_summarization_yaml(path: Path, *, grounding_threshold) -> None:
    """Write a YAML containing both blocks load_config expects."""
    threshold_str = "null" if grounding_threshold is None else str(grounding_threshold)
    path.write_text(
        f"""
pdf_extraction:
  database:
    enabled: false

knowledge_extraction:
  map:
    theta: 0.8
    reject_theta: 0.2
    chunk_size: 10
    chunk_overlap: 2
    chunk_workers: 5
  grounding:
    threshold: {threshold_str}
  relate:
    entailment_threshold: 0.50
    contradiction_threshold: 0.50
  resolve:
    grounding_weight: 0.60
    grounding_default: 0.50
    finding_bonus_max: 0.10
    finding_bonus_scale: 5
    support_boost_per_rel: 0.08
    support_boost_cap: 0.20
    single_study_pen: 0.10
    contradict_pen_per_rel: 0.15
    contradict_pen_cap: 0.30
    no_rel_grounding_weight: 0.80
    no_rel_finding_bonus_max: 0.15
    no_rel_single_study_pen: 0.05
  cost:
    enable_cost_report: true
    write_usage_jsonl: true
""",
        encoding="utf-8",
    )


def test_load_summarization_config_picks_up_grounding_threshold(
    tmp_path: Path, run_paper_module
) -> None:
    yaml_path = tmp_path / "run.yaml"
    _write_minimal_summarization_yaml(yaml_path, grounding_threshold=0.42)

    cfg = run_paper_module._load_summarization_config(str(yaml_path))

    assert cfg.grounding.threshold == 0.42
    assert cfg.map.theta == 0.8
    assert cfg.relate.entailment_threshold == 0.50


def test_load_summarization_config_threshold_null(
    tmp_path: Path, run_paper_module
) -> None:
    yaml_path = tmp_path / "run.yaml"
    _write_minimal_summarization_yaml(yaml_path, grounding_threshold=None)

    cfg = run_paper_module._load_summarization_config(str(yaml_path))

    assert cfg.grounding.threshold is None


def test_load_summarization_config_missing_file_falls_back(
    tmp_path: Path, run_paper_module, caplog
) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    cfg = run_paper_module._load_summarization_config(str(missing))

    # Default KnowledgeExtractionConfig grounding.threshold is None.
    assert cfg.grounding.threshold is None
    # Default map.theta = 0.8 per KnowledgeExtractionConfig dataclass.
    assert cfg.map.theta == 0.8


def test_load_summarization_config_empty_path_falls_back(run_paper_module) -> None:
    cfg_a = run_paper_module._load_summarization_config(None)
    cfg_b = run_paper_module._load_summarization_config("")
    assert cfg_a.grounding.threshold is None
    assert cfg_b.grounding.threshold is None


def test_grounding_threshold_changes_pipeline_config_hash(
    tmp_path: Path, run_paper_module
) -> None:
    """Run A vs Run B parity: the per-paper pipeline_config_hash must diverge
    when grounding.threshold flips. Otherwise the cached result short-circuit
    would silently serve Run-A summaries to a Run-B invocation."""
    from nlp_histo.pipeline.stages.knowledge_extraction.persistence import (
        compute_pipeline_config_hash,
    )

    yaml_a = tmp_path / "runA.yaml"
    yaml_b = tmp_path / "runB.yaml"
    _write_minimal_summarization_yaml(yaml_a, grounding_threshold=None)
    _write_minimal_summarization_yaml(yaml_b, grounding_threshold=0.5)

    cfg_a = run_paper_module._load_summarization_config(str(yaml_a))
    cfg_b = run_paper_module._load_summarization_config(str(yaml_b))

    hash_a = compute_pipeline_config_hash(
        config=cfg_a,
        thresholds={"grounding_threshold": cfg_a.grounding.threshold},
        models={},
        schema_version="test",
        prompt_version="test",
        cascade_signature="test",
    )
    hash_b = compute_pipeline_config_hash(
        config=cfg_b,
        thresholds={"grounding_threshold": cfg_b.grounding.threshold},
        models={},
        schema_version="test",
        prompt_version="test",
        cascade_signature="test",
    )

    assert hash_a != hash_b, (
        "pipeline_config_hash did not change when grounding.threshold flipped "
        "from None to 0.5 — Run A and Run B would silently produce identical "
        "cached summaries."
    )


def test_default_config_path_points_at_repo_yaml(run_paper_module) -> None:
    """Smoke check: DEFAULT_CONFIG_PATH resolves to a real file in the repo."""
    default = run_paper_module.DEFAULT_CONFIG_PATH
    assert (REPO_ROOT / default).exists(), (
        f"DEFAULT_CONFIG_PATH={default!r} does not exist under {REPO_ROOT}"
    )
