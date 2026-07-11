"""Tests for the costing module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.stages.knowledge_extraction.costing import (
    LLMUsageRecord, PriceBook, UsageCollector,
    load_records, write_cost_reports,
)
from pipeline.stages.knowledge_extraction.costing.models import normalize_token_usage
from pipeline.stages.knowledge_extraction.costing.report import build_report_from_records
from pipeline.stages.knowledge_extraction.costing.pricing import ModelPrice


# ── Token normalization ─────────────────────────────────────────────────────


def test_normalize_openai_shape():
    raw = {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125}
    assert normalize_token_usage(raw) == {
        "input_tokens": 100, "output_tokens": 25, "total_tokens": 125,
    }


def test_normalize_langchain_shape():
    raw = {"input_tokens": 200, "output_tokens": 50, "total_tokens": 250}
    assert normalize_token_usage(raw) == {
        "input_tokens": 200, "output_tokens": 50, "total_tokens": 250,
    }


def test_normalize_gemini_shape():
    raw = {
        "prompt_token_count": 300, "candidates_token_count": 40,
        "total_token_count": 340,
    }
    out = normalize_token_usage(raw)
    assert out["input_tokens"] == 300
    assert out["output_tokens"] == 40
    assert out["total_tokens"] == 340


def test_normalize_nested_token_usage():
    raw = {"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    out = normalize_token_usage(raw)
    assert out == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def test_normalize_handles_none():
    assert normalize_token_usage(None) == {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
    }


# ── Pricing ──────────────────────────────────────────────────────────────────


def _book() -> PriceBook:
    return PriceBook(
        prices={
            "fast":   ModelPrice(input_per_1m=0.10, output_per_1m=0.40),
            "strong": ModelPrice(input_per_1m=3.00, output_per_1m=15.00),
        },
        batch_discount_multiplier=0.5,
    )


def test_cost_basic():
    book = _book()
    # 1M input + 1M output at fast → 0.10 + 0.40
    assert book.cost("fast", 1_000_000, 1_000_000) == pytest.approx(0.50)


def test_cost_batch_discount():
    book = _book()
    normal = book.cost("strong", 1_000_000, 1_000_000)
    batched = book.cost("strong", 1_000_000, 1_000_000, batch=True)
    assert batched == pytest.approx(normal * 0.5)


def test_missing_price_returns_none():
    import logging
    book = _book()
    captured: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    h = _Capture(level=logging.WARNING)
    lg = logging.getLogger("pipeline.stages.knowledge_extraction.costing.pricing")
    lg.addHandler(h)
    try:
        assert book.cost("nope", 1000, 1000) is None
        # Second lookup of the same model must not re-warn.
        book.cost("nope", 1, 1)
        warnings = [m for m in captured if "nope" in m]
        assert len(warnings) == 1
    finally:
        lg.removeHandler(h)


def test_pricebook_load_missing_file(tmp_path):
    p = tmp_path / "does_not_exist.json"
    book = PriceBook.load(p)
    assert book.known_models() == []


def test_pricebook_load_real_file(tmp_path):
    p = tmp_path / "prices.json"
    p.write_text(json.dumps({
        "batch_discount_multiplier": 0.4,
        "models": {"m1": {"input_per_1m": 1, "output_per_1m": 2}},
    }))
    book = PriceBook.load(p)
    assert book.has("m1")
    assert book.batch_discount_multiplier == 0.4


# ── Collector + cache hits ──────────────────────────────────────────────────


def test_cache_hit_is_zero_paid_cost():
    coll = UsageCollector(run_id="r1", paper_id="PMC1")
    coll.record_cache_hit(stage="MAP", role="voter", item_id="C1")
    rec = coll.records()[0]
    assert rec.cache_hit is True
    assert rec.source == "cache"
    assert rec.input_tokens == 0
    assert rec.output_tokens == 0


def test_record_live_and_batch():
    coll = UsageCollector(run_id="r1", paper_id="PMC1")
    coll.record_live(
        stage="MAP", model="fast", provider="openai", role="voter",
        input_tokens=100, output_tokens=20, cascade_level=1, item_id="C1",
    )
    coll.record_batch(
        stage="MAP", model="strong", provider="openai", role="escalator",
        input_tokens=500, output_tokens=80, batch_id="b1",
        cascade_level=3, item_id="C2",
    )
    recs = coll.records()
    assert len(recs) == 2
    assert recs[0].source == "live_call"
    assert recs[1].source == "batch_result"
    assert recs[1].batch_used is True


def test_jsonl_roundtrip(tmp_path):
    coll = UsageCollector(run_id="r1", paper_id="PMC1")
    coll.record_live(
        stage="MAP", model="fast", provider="openai", role="voter",
        input_tokens=10, output_tokens=2,
    )
    path = tmp_path / "u.jsonl"
    coll.write_jsonl(path)
    loaded = load_records(path)
    assert len(loaded) == 1
    assert loaded[0].input_tokens == 10
    assert loaded[0].output_tokens == 2


# ── Report scenarios ─────────────────────────────────────────────────────────


def _sample_records() -> list[LLMUsageRecord]:
    """Two MAP chunks: C1 kept at L1, C2 escalated to L3."""
    return [
        # C1 — three L1 voters agree, no escalation
        LLMUsageRecord(run_id="r", paper_id="PMC1", stage="MAP", model="fast",
                       provider="openai", role="voter", cascade_level=1,
                       item_id="C1", input_tokens=200, output_tokens=20),
        LLMUsageRecord(run_id="r", paper_id="PMC1", stage="MAP", model="fast",
                       provider="openai", role="voter", cascade_level=1,
                       item_id="C1", input_tokens=200, output_tokens=22),
        LLMUsageRecord(run_id="r", paper_id="PMC1", stage="MAP", model="fast",
                       provider="openai", role="voter", cascade_level=1,
                       item_id="C1", input_tokens=200, output_tokens=21),
        # C2 — L1 voters disagreed, escalated to L3
        LLMUsageRecord(run_id="r", paper_id="PMC1", stage="MAP", model="fast",
                       provider="openai", role="voter", cascade_level=1,
                       item_id="C2", input_tokens=210, output_tokens=18),
        LLMUsageRecord(run_id="r", paper_id="PMC1", stage="MAP", model="fast",
                       provider="openai", role="voter", cascade_level=1,
                       item_id="C2", input_tokens=210, output_tokens=20),
        LLMUsageRecord(run_id="r", paper_id="PMC1", stage="MAP", model="strong",
                       provider="openai", role="escalator", cascade_level=3,
                       item_id="C2", input_tokens=210, output_tokens=80,
                       was_escalation=True),
    ]


def test_cascade_savings_arithmetic():
    book = _book()
    records = _sample_records()
    report = build_report_from_records(records, book)
    cas = report["cascade_savings"]
    assert cas["total_items"] == 2
    assert cas["accepted_without_escalation"] == 1
    assert cas["escalated"] == 1
    assert cas["escalation_rate"] == pytest.approx(0.5)


def test_scenario_d_matches_actual_total():
    book = _book()
    records = _sample_records()
    report = build_report_from_records(records, book)
    scenarios = {s["scenario"]: s for s in report["scenarios"]}
    d = scenarios["D"]
    actual = report["actual_usage"]["total"]
    assert d["input_tokens"] == actual["input_tokens"]
    assert d["output_tokens"] == actual["output_tokens"]


def test_baseline_a_is_estimated_and_marked():
    book = _book()
    records = _sample_records()
    report = build_report_from_records(records, book)
    a = next(s for s in report["scenarios"] if s["scenario"] == "A")
    assert a["usage_type"] == "estimated"
    # Strong model "strong" is observed → cost should compute.
    assert "cost_central" in a


def test_baseline_a_no_l3_falls_back_to_stage_average():
    book = _book()
    # Drop the L3 record so the strong model can't be inferred.
    records = [r for r in _sample_records() if r.cascade_level != 3]
    report = build_report_from_records(records, book)
    base_meta = report["counterfactual_baseline"]
    # Without an observed L3 call, the strong_model lookup returns None.
    assert base_meta.get("strong_model") is None


def test_batch_savings_section():
    coll = UsageCollector(run_id="r", paper_id="P")
    coll.record_batch(stage="MAP", model="fast", provider="openai", role="voter",
                      input_tokens=1_000_000, output_tokens=1_000_000)
    report = build_report_from_records(coll.records(), _book())
    bs = report["batch_savings"]
    assert bs["regular_cost_usd"] == pytest.approx(0.50)
    assert bs["discounted_cost_usd"] == pytest.approx(0.25)
    assert bs["absolute_savings_usd"] == pytest.approx(0.25)
    assert bs["percent_savings"] == pytest.approx(50.0)


def test_report_writes_files(tmp_path):
    book = _book()
    write_cost_reports(_sample_records(), book, tmp_path)
    assert (tmp_path / "cost_report.md").exists()
    assert (tmp_path / "cost_report.json").exists()
    data = json.loads((tmp_path / "cost_report.json").read_text())
    assert "scenarios" in data
    assert "actual_usage" in data


def test_missing_price_still_reports_tokens():
    book = PriceBook(prices={}, batch_discount_multiplier=0.5)
    records = _sample_records()
    report = build_report_from_records(records, book)
    total = report["actual_usage"]["total"]
    # Tokens still aggregated even though no prices are configured.
    assert total["input_tokens"] > 0
    assert total["output_tokens"] > 0
    # And every per-stage block keeps the same property.
    for stage, agg in report["actual_usage"]["by_stage"].items():
        assert agg["total_tokens"] >= 0
    # cost should be absent.
    assert "cost_usd" not in total


def test_empty_records_returns_empty_report():
    book = _book()
    report = build_report_from_records([], book)
    assert report["empty"] is True


# ── Runner integration: disabled mode ───────────────────────────────────────


def test_config_default_enables_cost_report():
    from pipeline.stages.knowledge_extraction.config import KnowledgeExtractionConfig
    cfg = KnowledgeExtractionConfig()
    assert cfg.cost.enable_cost_report is True


def test_callback_captures_usage_metadata_shape():
    """LangChain ``usage_metadata`` on the AIMessage — the modern shape."""
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    coll = UsageCollector(run_id="r", paper_id="P")
    handler = coll.callback(
        stage="MAP", role="voter", model="m", provider="openai",
        cascade_level=1, item_id="C1",
    )
    msg = AIMessage(content="x", usage_metadata={
        "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
    })
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]))
    recs = coll.records()
    assert len(recs) == 1
    assert recs[0].input_tokens == 100
    assert recs[0].output_tokens == 20
    assert recs[0].cascade_level == 1
    assert recs[0].item_id == "C1"


def test_callback_captures_llm_output_token_usage_shape():
    """OpenAI-style ``llm_output={'token_usage': ...}`` shape."""
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    coll = UsageCollector(run_id="r", paper_id="P")
    handler = coll.callback(
        stage="MAP", role="escalator", model="m", provider="openai",
        cascade_level=3, item_id="C2",
    )
    msg = AIMessage(content="x")
    result = LLMResult(
        generations=[[ChatGeneration(message=msg)]],
        llm_output={"token_usage": {
            "prompt_tokens": 300, "completion_tokens": 40, "total_tokens": 340,
        }},
    )
    handler.on_llm_end(result)
    recs = coll.records()
    assert len(recs) == 1
    assert recs[0].input_tokens == 300
    assert recs[0].output_tokens == 40
    assert recs[0].cascade_level == 3


def test_callback_no_usage_present_is_silent():
    """When the response has no usage data, the callback records nothing."""
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    coll = UsageCollector(run_id="r", paper_id="P")
    handler = coll.callback(stage="MAP", role="voter", model="m", provider="p")
    msg = AIMessage(content="x")
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]))
    assert len(coll) == 0


def _load_run_paper_module():
    """Load scripts/run_paper.py as a module. scripts/ is not a package."""
    import importlib.util
    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "run_paper_script", repo_root / "scripts" / "run_paper.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_baseline_unavailable_when_no_l1_usage():
    rp = _load_run_paper_module()
    assert rp._baseline_cost({}) is None
    assert rp._baseline_cost({"l2": {"m": {"input": 100, "output": 10}}}) is None


def test_baseline_unavailable_when_l3_price_missing():
    rp = _load_run_paper_module()
    usage = {
        "l1": {"some-model": {"input": 1000, "output": 100}},
        "l3": {"unknown-model-xyz": {"input": 100, "output": 10}},
    }
    assert rp._baseline_cost(usage) is None


def test_total_savings_marked_unavailable_when_any_missing(tmp_path):
    """When any per-paper baseline is None, aggregate must also be None."""
    rp = _load_run_paper_module()
    stats = [
        {
            "pmcid": "PMC1", "mode": "batch", "total_chunks": 1,
            "l1_kept": 1, "l2_escalated": 0, "l2_kept": 0,
            "l3_escalated": 0, "l3_kept": 0, "finalized": 1, "dropped": 0,
            "l1_pct": 100.0, "token_usage": {}, "est_cost_usd": 0.001,
            "est_saved_usd": 0.005,
        },
        {
            "pmcid": "PMC2", "mode": "batch", "total_chunks": 1,
            "l1_kept": 1, "l2_escalated": 0, "l2_kept": 0,
            "l3_escalated": 0, "l3_kept": 0, "finalized": 1, "dropped": 0,
            "l1_pct": 100.0, "token_usage": {}, "est_cost_usd": 0.002,
            "est_saved_usd": None,
        },
    ]
    rp._save_escalation_report(stats, tmp_path)
    report_path = next(tmp_path.glob("escalation_report_*.json"))
    data = json.loads(report_path.read_text())
    assert data["totals"]["est_saved_usd"] is None


# ── Shared UMLS resource ─────────────────────────────────────────────────────


def test_umls_disabled_env_skips_load(monkeypatch):
    """Env var kill-switch must short-circuit before any spacy import."""
    from pipeline.stages.knowledge_extraction import umls_resources
    umls_resources._reset_for_tests()
    monkeypatch.setenv("NLP_HISTO_DISABLE_UMLS", "1")
    nlp = umls_resources.get_nlp()
    assert nlp is None
    assert umls_resources.is_available() is False


def test_get_nlp_caches_load_result(monkeypatch):
    """Second call must hit the cached None without re-probing."""
    from pipeline.stages.knowledge_extraction import umls_resources
    umls_resources._reset_for_tests()
    monkeypatch.setenv("NLP_HISTO_DISABLE_UMLS", "1")
    assert umls_resources.get_nlp() is None
    # Flip the env var; a re-probe would now return something, but we should
    # not re-probe because the first call cached the unavailable verdict.
    monkeypatch.delenv("NLP_HISTO_DISABLE_UMLS", raising=False)
    assert umls_resources.get_nlp() is None  # still cached


def test_skip_enrichment_env_no_ops(monkeypatch):
    """--skip-umls-enrichment env var skips before any model load attempt."""
    from pipeline.stages.knowledge_extraction import umls_resources
    from pipeline.stages.knowledge_extraction.helpers.entity_linker import (
        enrich_rules_with_cuis,
    )
    from pipeline.stages.knowledge_extraction.models import (
        CanonicalRule, DirectionEnum, RelationTypeEnum,
    )

    umls_resources._reset_for_tests()
    monkeypatch.setenv("NLP_HISTO_SKIP_UMLS_ENRICHMENT", "1")

    rule = CanonicalRule(
        canonical_id="cr1", group_id="g1",
        subject_entity="x", outcome_entity="y",
        relation_type=RelationTypeEnum.has_feature,
        direction=DirectionEnum.positive,
        predicate_text="x shows y", is_conflicted=False,
        study_coverage="single_study", category="morphology",
        supporting_pmcids=["P"], member_normal_ids=["nf1"],
        mean_grounding_score=None, finding_count=1,
    )
    enrich_rules_with_cuis([rule])
    # No model was loaded; CUIs remain None.
    assert rule.subject_cui is None
    assert rule.outcome_cui is None
    assert umls_resources.is_available() is False


def test_config_disable_round_trips():
    from dataclasses import replace
    from pipeline.stages.knowledge_extraction.config import CostConfig, KnowledgeExtractionConfig
    cfg = KnowledgeExtractionConfig(cost=CostConfig(enable_cost_report=False))
    cfg2 = replace(cfg, cost=replace(cfg.cost, write_usage_jsonl=False))
    assert cfg2.cost.enable_cost_report is False
    assert cfg2.cost.write_usage_jsonl is False
