"""
Local checks for Task 1 — schema/enums/strict-tool/coercion/profiles.

Run with::

    python tests/knowledge_extraction/integration/test_map_schema.py

Exits non-zero on any failure. Logs are written under ``logs/`` so you can
inspect what was coerced.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def _ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def _fail(name: str, exc: BaseException) -> None:
    print(f"  FAIL  {name}: {exc!r}")
    traceback.print_exc()


def check(name: str, fn) -> bool:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 — this is a test runner
        _fail(name, exc)
        return False
    _ok(name)
    return True


# Tests

def test_direction_enum_has_no_direction():
    from nlp_histo.pipeline.stages.knowledge_extraction.models import DirectionEnum
    values = {e.value for e in DirectionEnum}
    expected = {"positive", "negative", "absent", "partial", "unclear", "no_direction"}
    assert values == expected, f"DirectionEnum values mismatch: {values} vs {expected}"


def test_relation_type_enum():
    from nlp_histo.pipeline.stages.knowledge_extraction.models import RelationTypeEnum
    values = {e.value for e in RelationTypeEnum}
    expected = {
        "has_feature", "expression", "prognostic", "comparative",
        "demographic", "treatment_response", "unclear",
    }
    assert values == expected, f"RelationTypeEnum mismatch: {values} vs {expected}"


def test_category_literal_values():
    from nlp_histo.pipeline.stages.knowledge_extraction.models import Finding
    cat_field = Finding.model_fields["category"]
    cat_values = set(getattr(cat_field.annotation, "__args__", ()))
    expected = {
        "morphology", "IHC", "molecular_genetics", "staging",
        "treatment", "prognosis", "demographic",
    }
    assert cat_values == expected, f"category Literal mismatch: {cat_values} vs {expected}"


def test_prompt_lists_match_enums():
    from nlp_histo.pipeline.stages.knowledge_extraction.llm import prompts
    src = prompts._MAP_SYSTEM
    # Direction values must each appear textually in the prompt
    for v in ("positive", "negative", "absent", "partial", "unclear", "no_direction"):
        assert v in src, f"direction value {v!r} missing from MAP prompt"
    # Relation-type values must each appear
    for v in ("has_feature", "expression", "prognostic", "comparative",
              "demographic", "treatment_response", "unclear"):
        assert v in src, f"relation_type value {v!r} missing from MAP prompt"
    # Category values must each appear
    for v in ("morphology", "IHC", "molecular_genetics", "staging",
              "treatment", "prognosis", "demographic"):
        assert v in src, f"category value {v!r} missing from MAP prompt"
    # Prompt must NOT instruct null direction
    assert "direction: null" not in src, "MAP prompt still uses 'direction: null'"
    assert "or null when not applicable" not in src


def test_openai_strict_tool_builds_and_validates():
    from nlp_histo.pipeline.stages.knowledge_extraction.batch.dispatch import (
        _build_openai_tool, validate_openai_strict_schema,
    )
    tool = _build_openai_tool()
    assert tool["function"]["strict"] is True
    params = tool["function"]["parameters"]
    errors = validate_openai_strict_schema(params)
    assert not errors, "strict schema invariants violated:\n  " + "\n  ".join(errors)


def test_no_default_fields_in_finding_or_scope_schema():
    """Pydantic should not produce JSON `default` keys for Finding/FindingScope."""
    from nlp_histo.pipeline.stages.knowledge_extraction.models import AuditableSummary
    schema = AuditableSummary.model_json_schema()
    bad: list[str] = []

    def walk(node, path):
        if not isinstance(node, dict):
            return
        if "default" in node:
            bad.append(f"{path}: default={node['default']!r}")
        for k, v in node.get("properties", {}).items():
            walk(v, f"{path}.{k}")
        if "items" in node:
            walk(node["items"], f"{path}[]")
        for key in ("anyOf", "oneOf", "allOf"):
            for i, sub in enumerate(node.get(key, [])):
                walk(sub, f"{path}.{key}[{i}]")
        for name, defn in node.get("$defs", {}).items():
            walk(defn, f"$defs.{name}")

    walk(schema, "$")
    # Filter to only Finding/FindingScope-relevant defaults
    assert not bad, "Pydantic schema has defaults that break OpenAI strict:\n  " + "\n  ".join(bad)


def test_minimal_auditable_summary():
    from nlp_histo.pipeline.stages.knowledge_extraction.models import (
        AuditableSummary, Finding, AuditMetadata,
        RelationTypeEnum, DirectionEnum,
    )
    f = Finding(
        category="morphology",
        claim="X has Y",
        evidence=["S1|PMC1|1"],
        confidence="high",
        verbatim_support="X has Y",
        subject_entity="X",
        outcome_entity="Y",
        relation_type=RelationTypeEnum.unclear,
        direction=DirectionEnum.no_direction,
        scope=None,
        grounding_score=None,
    )
    s = AuditableSummary(
        chunk_id="C1",
        findings=[f],
        summary_text="t",
        audit_metadata=AuditMetadata(
            sentences_analyzed=1,
            sentences_cited=["S1"],
            pmcids_referenced=["PMC1"],
            uncited_sentences=[],
        ),
    )
    assert len(s.findings) == 1


def test_legacy_null_direction_coerces_to_no_direction():
    from nlp_histo.pipeline.stages.knowledge_extraction.models import Finding, DirectionEnum
    f = Finding.model_validate({
        "category": "demographic",
        "claim": "MGA presents in seventh decade",
        "evidence": ["S1|PMC1|1"],
        "confidence": "low",
        "verbatim_support": "MGA presents in seventh decade",
        "subject_entity": "MGA",
        "outcome_entity": "age",
        "relation_type": "demographic",
        "direction": None,        # legacy null
        "scope": None,
        "grounding_score": None,
    })
    assert f.direction == DirectionEnum.no_direction


def test_invalid_relation_type_coerces_to_unclear():
    from nlp_histo.pipeline.stages.knowledge_extraction.models import Finding, RelationTypeEnum
    f = Finding.model_validate({
        "category": "morphology",
        "claim": "X has Y",
        "evidence": ["S1|PMC1|1"],
        "confidence": "low",
        "verbatim_support": "X has Y",
        "subject_entity": "X",
        "outcome_entity": "Y",
        "relation_type": "totally-bogus-value",
        "direction": "no_direction",
        "scope": None,
        "grounding_score": None,
    })
    assert f.relation_type == RelationTypeEnum.unclear


def test_demographics_alias_repair():
    from nlp_histo.pipeline.stages.knowledge_extraction.models import Finding
    f = Finding.model_validate({
        "category": "demographics",  # legacy alias → demographic
        "claim": "GA more common in women",
        "evidence": ["S1|PMC1|1"],
        "confidence": "low",
        "verbatim_support": "GA more common in women",
        "subject_entity": "GA",
        "outcome_entity": "sex",
        "relation_type": "demographic",
        "direction": "positive",
        "scope": None,
        "grounding_score": None,
    })
    assert f.category == "demographic"


def test_bad_finding_logged_when_dropped():
    """Malformed item passed inside an AuditableSummary should be dropped *and* logged."""
    from nlp_histo.pipeline.stages.knowledge_extraction.models import AuditableSummary, AuditMetadata

    # Reset log dir to a temp location to make this test self-contained
    import os
    import tempfile
    tmp = tempfile.mkdtemp(prefix="map_schema_test_")
    os.environ["NLP_HISTO_LOG_DIR"] = tmp

    s = AuditableSummary(
        chunk_id="C1",
        findings=[
            {  # plainly malformed — missing required fields
                "category": "not-a-real-category",
                "claim": "x",
            },
        ],
        summary_text="t",
        audit_metadata=AuditMetadata(
            sentences_analyzed=0,
            sentences_cited=[],
            pmcids_referenced=[],
            uncited_sentences=[],
        ),
    )
    assert s.findings == []
    log_path = Path(tmp) / "bad_findings.jsonl"
    assert log_path.exists(), "bad_findings.jsonl was not written"
    rec = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert "raw" in rec and "error" in rec
    # Reset
    os.environ.pop("NLP_HISTO_LOG_DIR", None)


def test_profiles():
    from nlp_histo.pipeline.stages.knowledge_extraction.batch.voter_configs import (
        get_profile, list_profiles,
        GEMINI_L1, OPENAI_L1, OPENAI_L1_B,
        GEMINI_L2, OPENAI_L2, CLAUDE_L2, CLAUDE_L3,
    )
    for name in ("cheap", "real"):
        assert name in list_profiles(), f"profile {name!r} missing"

    cheap = get_profile("cheap")
    # L1: 3 cheapest voters — Gemini + two OpenAI (gpt-4o-mini, gpt-4.1-nano).
    assert [v.model for v in cheap.l1_voters] == [GEMINI_L1, OPENAI_L1, OPENAI_L1_B]
    assert [v.provider for v in cheap.l1_voters] == ["gemini", "openai", "openai"]
    # L2 is a single OpenAI voter (gpt-4.1-mini); L3 mirrors L2 for the 2-tier cascade.
    assert len(cheap.l2_voters) == 1
    assert cheap.l2_voters[0].model == OPENAI_L2
    assert cheap.l3_voter.model == OPENAI_L2

    real = get_profile("real")
    # L1 same as cheap.
    assert [v.model for v in real.l1_voters] == [GEMINI_L1, OPENAI_L1, OPENAI_L1_B]
    # L2: 3 mid-tier voters across Gemini/OpenAI/Claude.
    assert [v.model for v in real.l2_voters] == [GEMINI_L2, OPENAI_L2, CLAUDE_L2]
    assert [v.provider for v in real.l2_voters] == ["gemini", "openai", "claude"]
    # L3: Claude Sonnet.
    assert real.l3_voter.model == CLAUDE_L3
    assert real.l3_voter.provider == "claude"

def test_no_profile_raises_when_unset():
    """get_profile() with no arg and NLP_HISTO_PROFILE unset → ValueError."""
    import os as _os
    from nlp_histo.pipeline.stages.knowledge_extraction.batch.voter_configs import get_profile

    saved = _os.environ.pop("NLP_HISTO_PROFILE", None)
    try:
        try:
            get_profile(None)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "get_profile(None) must raise when $NLP_HISTO_PROFILE is unset"
            )
    finally:
        if saved is not None:
            _os.environ["NLP_HISTO_PROFILE"] = saved


def test_unknown_profile_raises():
    """Typos in profile name must raise rather than silently pick default."""
    from nlp_histo.pipeline.stages.knowledge_extraction.batch.voter_configs import get_profile

    try:
        get_profile("smoke_haiku")
    except ValueError:
        pass
    else:
        raise AssertionError("get_profile('smoke_haiku') should raise after rename")


def test_version_constants():
    from nlp_histo.pipeline.stages.knowledge_extraction import models
    assert models.MAP_SCHEMA_VERSION == "map_v10_nli_and_voter_config_in_key"
    assert models.MAP_PROMPT_VERSION == "map_prompt_v5_expression_absent_vs_negative"
    assert models.MAP_STAGE_NAME == "map"


def test_cascade_signature_stable_and_sensitive():
    from nlp_histo.pipeline.stages.knowledge_extraction.models import compute_cascade_signature
    a = compute_cascade_signature([("openai", "gpt-4o-mini"), ("anthropic", "claude-haiku-4-5-20251001")])
    b = compute_cascade_signature([("openai", "gpt-4o-mini"), ("anthropic", "claude-haiku-4-5-20251001")])
    c = compute_cascade_signature([("anthropic", "claude-haiku-4-5-20251001"), ("openai", "gpt-4o-mini")])
    d = compute_cascade_signature([("openai", "gpt-4o-mini"), ("anthropic", "claude-sonnet-4-6-20251001")])
    assert a == b
    assert a != c, "signature must be order-sensitive"
    assert a != d, "signature must change when a model changes"


def test_map_cache_key_includes_versions_and_cascade():
    import tempfile
    from pathlib import Path as _P
    from nlp_histo.pipeline.stages.knowledge_extraction.cache import PipelineCache
    from nlp_histo.pipeline.stages.knowledge_extraction.models import (
        MAP_PROMPT_VERSION, MAP_SCHEMA_VERSION, MAP_STAGE_NAME, MapRunMetadata,
    )

    sentences = [{"text_element_id": 1}, {"text_element_id": 2}]
    md_a = MapRunMetadata(provider="openai", model="gpt-4o-mini",
                          cascade_signature="aaaaaaaaaaaaaaaa")
    md_b = MapRunMetadata(provider="openai", model="gpt-4o-mini",
                          cascade_signature="bbbbbbbbbbbbbbbb")
    key_a = PipelineCache._map_key(sentences, md_a)
    key_b = PipelineCache._map_key(sentences, md_b)
    assert key_a != key_b, "cascade_signature must affect MAP cache key"
    for piece in (MAP_SCHEMA_VERSION, MAP_PROMPT_VERSION, MAP_STAGE_NAME, "aaaaaaaaaaaaaaaa"):
        assert piece in key_a

    # Round-trip: set then get with matching metadata yields a hit.
    from nlp_histo.pipeline.stages.knowledge_extraction.models import (
        AuditMetadata, AuditableSummary, DirectionEnum, Finding, RelationTypeEnum,
    )
    f = Finding(
        category="morphology", claim="X has Y", evidence=["S1|PMC1|1"],
        confidence="high", verbatim_support="X has Y",
        subject_entity="X", outcome_entity="Y",
        relation_type=RelationTypeEnum.unclear,
        direction=DirectionEnum.no_direction,
        scope=None, grounding_score=None,
    )
    s = AuditableSummary(
        chunk_id="C1", findings=[f], summary_text="t",
        audit_metadata=AuditMetadata(sentences_analyzed=1, sentences_cited=["S1"],
                                     pmcids_referenced=["PMC1"], uncited_sentences=[]),
    )
    with tempfile.TemporaryDirectory() as td:
        cache = PipelineCache(_P(td) / "c.json")
        assert cache.get_map(sentences, md_a) is None  # miss (empty cache)
        cache.set_map(sentences, s, md_a)
        # Same metadata → hit
        hit = cache.get_map(sentences, md_a)
        assert hit is not None and hit.chunk_id == "C1"
        # Different cascade_signature → miss
        assert cache.get_map(sentences, md_b) is None


def test_map_run_metadata_summary_contains_required_fields():
    from langchain_core.runnables import RunnableLambda

    from nlp_histo.pipeline.stages.knowledge_extraction.stages.map_stage import MapStage

    class _DummyLLM(RunnableLambda):
        model_name = "dummy-model"
        def __init__(self):
            super().__init__(lambda x: None)
        def with_structured_output(self, *a, **kw):
            return self
        def with_retry(self, *a, **kw):
            return self

    stage = MapStage(
        voter_llms=[_DummyLLM()],
        level2_voter_llms=[_DummyLLM()],
        escalation_llm=_DummyLLM(),
        voter_specs=[("openai", "gpt-4o-mini")],
        level2_voter_specs=[("openai", "gpt-4.1-mini")],
        escalation_spec=("anthropic", "claude-sonnet-4-6-20251001"),
        cascade_profile="default",
    )
    md = stage.run_metadata_summary()
    for key in ("schema_version", "prompt_version", "stage_name",
                "cascade_profile", "cascade_signature",
                "l1_voters", "l2_voters", "l3_voter"):
        assert key in md, f"run_metadata_summary missing key {key!r}"
    assert md["cascade_profile"] == "default"
    assert md["l3_voter"] == {"provider": "anthropic", "model": "claude-sonnet-4-6-20251001"}


def test_batch_handle_persists_versions():
    import tempfile
    from pathlib import Path as _P
    from nlp_histo.pipeline.stages.knowledge_extraction.batch.models import BatchHandle, BatchPhase

    h = BatchHandle(
        pmcid="PMC1", phase=BatchPhase.L1_SUBMITTED,
        schema_version="map_v1_explicit_direction",
        prompt_version="map_prompt_v1_explicit_enums",
        stage_name="map",
        cascade_profile="smoke_haiku",
        cascade_signature="cafebabecafebabe",
    )
    with tempfile.TemporaryDirectory() as td:
        p = _P(td) / "h.json"
        h.save(p)
        h2 = BatchHandle.load(p)
        assert h2.schema_version == "map_v1_explicit_direction"
        assert h2.prompt_version == "map_prompt_v1_explicit_enums"
        assert h2.stage_name == "map"
        assert h2.cascade_profile == "smoke_haiku"
        assert h2.cascade_signature == "cafebabecafebabe"


# Runner

ALL_TESTS = [
    ("DirectionEnum has no_direction",           test_direction_enum_has_no_direction),
    ("RelationTypeEnum exact set",                test_relation_type_enum),
    ("Category Literal exact set",                test_category_literal_values),
    ("MAP prompt mentions every enum value",      test_prompt_lists_match_enums),
    ("OpenAI strict tool builds + validates",     test_openai_strict_tool_builds_and_validates),
    ("No default keys in MAP JSON schema",        test_no_default_fields_in_finding_or_scope_schema),
    ("Minimal AuditableSummary validates",        test_minimal_auditable_summary),
    ("Legacy null direction → no_direction",      test_legacy_null_direction_coerces_to_no_direction),
    ("Invalid relation_type → unclear",           test_invalid_relation_type_coerces_to_unclear),
    ("category 'demographics' → 'demographic'",   test_demographics_alias_repair),
    ("Bad finding is logged then dropped",        test_bad_finding_logged_when_dropped),
    ("Profiles cheap / real",                     test_profiles),
    ("No profile raises when unset",              test_no_profile_raises_when_unset),
    ("Unknown profile raises",                    test_unknown_profile_raises),
    ("Version constants present",                 test_version_constants),
    ("compute_cascade_signature stable+sensitive", test_cascade_signature_stable_and_sensitive),
    ("MAP cache key includes versions+cascade",   test_map_cache_key_includes_versions_and_cascade),
    ("run_metadata_summary required fields",      test_map_run_metadata_summary_contains_required_fields),
    ("BatchHandle persists versions",             test_batch_handle_persists_versions),
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} schema/enum checks...\n")
    passed = 0
    for name, fn in ALL_TESTS:
        if check(name, fn):
            passed += 1
    print(f"\n{passed}/{len(ALL_TESTS)} passed.")
    return 0 if passed == len(ALL_TESTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
