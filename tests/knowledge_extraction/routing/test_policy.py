"""
Tests for routing policy infrastructure.

Covers:
- RoutingPolicySpec construction and to_dict()
- PolicyEvaluationResult serialization round-trip
- PolicySelectionResult serialization round-trip
- PolicyEvaluationStore save/load/append/iter_results
- select_with_pulp: feasible selection, infeasible (relax recall), infeasible (all)
"""
from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from nlp_histo.pipeline.stages.knowledge_extraction.routing.policy import (
    PolicyEvaluationResult,
    PolicyEvaluationStore,
    PolicySelectionResult,
    RoutingPolicySpec,
)
from tests.paths import SCRIPTS

_SCRIPTS_DIR = SCRIPTS


def _import_select_with_pulp():
    """Load select_with_pulp from scripts/select_policy.py via importlib."""
    spec = importlib.util.spec_from_file_location(
        "select_policy", _SCRIPTS_DIR / "select_policy.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules.setdefault("select_policy", mod)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.select_with_pulp


# Fixtures

def _make_spec(**overrides) -> RoutingPolicySpec:
    defaults = dict(
        policy_id="emb_theta07",
        scorer_name="EmbeddingScorer",
        theta=0.70,
        voter_models=["gpt-4o-mini", "claude-haiku-4-5-20251001"],
        escalation_model="gpt-4o",
    )
    defaults.update(overrides)
    return RoutingPolicySpec(**defaults)


def _make_result(**overrides) -> PolicyEvaluationResult:
    defaults = dict(
        policy_id="emb_theta07",
        scorer_name="EmbeddingScorer",
        theta=0.70,
    )
    defaults.update(overrides)
    return PolicyEvaluationResult(**defaults)


# RoutingPolicySpec

class TestRoutingPolicySpec:
    def test_defaults(self):
        spec = _make_spec()
        assert spec.single_voter_policy == "escalate"
        assert spec.reject_on_low_agreement is False
        assert spec.fabricated_threshold == 0.25
        assert spec.weak_threshold == 0.60
        assert spec.notes == ""

    def test_to_dict_roundtrip(self):
        spec = _make_spec(notes="test run")
        d = spec.to_dict()
        assert d["policy_id"] == "emb_theta07"
        assert d["voter_models"] == ["gpt-4o-mini", "claude-haiku-4-5-20251001"]
        assert d["notes"] == "test run"
        # voter_models is a copy, not the same object
        assert d["voter_models"] is not spec.voter_models

    def test_override_fields(self):
        spec = _make_spec(
            single_voter_policy="keep",
            reject_on_low_agreement=True,
            fabricated_threshold=0.30,
        )
        assert spec.single_voter_policy == "keep"
        assert spec.reject_on_low_agreement is True
        assert spec.fabricated_threshold == 0.30


# PolicyEvaluationResult

class TestPolicyEvaluationResult:
    def test_defaults_are_none(self):
        r = _make_result()
        assert r.keep_precision is None
        assert r.false_accept_rate is None
        assert r.recall is None
        assert r.keep_rate is None
        assert r.escalation_rate is None
        assert r.selection_objective is None
        assert r.warnings == []

    def test_round_trip_json_minimal(self):
        r = _make_result()
        restored = PolicyEvaluationResult.model_validate_json(r.model_dump_json())
        assert restored.policy_id == "emb_theta07"
        assert restored.theta == pytest.approx(0.70)
        assert restored.keep_precision is None

    def test_round_trip_json_with_metrics(self):
        r = _make_result(
            keep_precision=0.92,
            false_accept_rate=0.08,
            recall=0.75,
            escalation_rate=0.30,
            selection_objective=0.21,
            selection_objective_alpha=0.6,
            warnings=["scorer mismatch"],
        )
        restored = PolicyEvaluationResult.model_validate_json(r.model_dump_json())
        assert restored.keep_precision == pytest.approx(0.92)
        assert restored.false_accept_rate == pytest.approx(0.08)
        assert restored.warnings == ["scorer mismatch"]

    def test_extra_fields_ignored(self):
        """Extra JSON fields (e.g. judge metadata) are silently dropped."""
        raw = json.loads(_make_result().model_dump_json())
        raw["unknown_future_field"] = "ignored"
        r = PolicyEvaluationResult.model_validate(raw)
        assert r.policy_id == "emb_theta07"


# PolicySelectionResult

class TestPolicySelectionResult:
    def test_round_trip_json(self):
        r = PolicySelectionResult(
            selected_policy_id="sem_theta074",
            feasible=True,
            candidates_evaluated=3,
            candidates_feasible=2,
            constraints={"max_false_accept": 0.10, "min_recall": 0.70},
            selection_objective_alpha=0.6,
            selected_scorer_name="SemanticAgreementScorer",
            selected_theta=0.74,
            selected_false_accept_rate=0.08,
            selected_recall=0.78,
            selected_escalation_rate=0.28,
            selected_selection_objective=0.20,
            runner_up_policy_id="emb_theta07",
        )
        restored = PolicySelectionResult.model_validate_json(r.model_dump_json())
        assert restored.selected_policy_id == "sem_theta074"
        assert restored.selected_theta == pytest.approx(0.74)
        assert restored.runner_up_policy_id == "emb_theta07"
        assert restored.feasible is True
        assert restored.candidates_feasible == 2

    def test_none_selected(self):
        r = PolicySelectionResult(
            selected_policy_id=None,
            feasible=False,
            candidates_evaluated=2,
            candidates_feasible=0,
            constraints={"max_false_accept": 0.05, "min_recall": 0.95},
            selection_objective_alpha=0.6,
            infeasibility_reason="No candidates satisfy constraints.",
        )
        restored = PolicySelectionResult.model_validate_json(r.model_dump_json())
        assert restored.selected_policy_id is None
        assert restored.feasible is False
        assert "No candidates" in restored.infeasibility_reason


# PolicyEvaluationStore

class TestPolicyEvaluationStore:
    def test_save_and_load(self, tmp_path):
        results = [_make_result(policy_id=f"pol_{i}") for i in range(3)]
        path = tmp_path / "results.jsonl"
        PolicyEvaluationStore.save(results, path)
        loaded = PolicyEvaluationStore.load(path)
        assert len(loaded) == 3
        assert [r.policy_id for r in loaded] == ["pol_0", "pol_1", "pol_2"]

    def test_append(self, tmp_path):
        path = tmp_path / "results.jsonl"
        PolicyEvaluationStore.append(_make_result(policy_id="first"), path)
        PolicyEvaluationStore.append(_make_result(policy_id="second"), path)
        loaded = PolicyEvaluationStore.load(path)
        assert len(loaded) == 2
        assert loaded[1].policy_id == "second"

    def test_iter_results(self, tmp_path):
        results = [_make_result(policy_id=f"pol_{i}") for i in range(4)]
        path = tmp_path / "results.jsonl"
        PolicyEvaluationStore.save(results, path)
        ids = [r.policy_id for r in PolicyEvaluationStore.iter_results(path)]
        assert ids == [f"pol_{i}" for i in range(4)]

    def test_iter_results_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.jsonl"
        assert list(PolicyEvaluationStore.iter_results(path)) == []

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "results.jsonl"
        PolicyEvaluationStore.save([_make_result()], path)
        assert path.exists()

    def test_one_json_per_line(self, tmp_path):
        path = tmp_path / "results.jsonl"
        PolicyEvaluationStore.save([_make_result(policy_id=f"p{i}") for i in range(3)], path)
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        assert len(lines) == 3
        for line in lines:
            obj = json.loads(line)
            assert "policy_id" in obj


# select_with_pulp

class TestSelectWithPulp:
    """Smoke tests for the PuLP ILP selector."""

    def _make_candidates(self) -> list[PolicyEvaluationResult]:
        return [
            _make_result(
                policy_id="pol_a",
                scorer_name="EmbeddingScorer",
                theta=0.70,
                false_accept_rate=0.12,
                recall=0.75,
                escalation_rate=0.30,
                selection_objective=0.23,
            ),
            _make_result(
                policy_id="pol_b",
                scorer_name="SemanticAgreementScorer",
                theta=0.74,
                false_accept_rate=0.08,
                recall=0.78,
                escalation_rate=0.28,
                selection_objective=0.20,
            ),
            _make_result(
                policy_id="pol_c",
                scorer_name="EmbeddingScorer",
                theta=0.80,
                false_accept_rate=0.05,
                recall=0.60,  # below min_recall
                escalation_rate=0.45,
                selection_objective=0.29,
            ),
        ]

    def test_feasible_selects_min_objective(self):
        pytest.importorskip("pulp")
        select_with_pulp = _import_select_with_pulp()

        candidates = self._make_candidates()
        selected, feasible, reason, runner_up = select_with_pulp(
            candidates, max_false_accept=0.10, min_recall=0.70, alpha=0.6
        )
        # pol_a has far=0.12 > 0.10 → infeasible; pol_b far=0.08 recall=0.78 → feasible
        # pol_c recall=0.60 < 0.70 → infeasible
        assert feasible is True
        assert selected is not None
        assert selected.policy_id == "pol_b"
        assert reason is None
        assert runner_up is None  # only one feasible

    def test_infeasible_relaxes_recall(self):
        """When recall constraint cannot be met, FAR constraint alone is used."""
        pytest.importorskip("pulp")
        select_with_pulp = _import_select_with_pulp()

        # pol_a: far=0.12 (fails FAR). pol_b: far=0.08 recall=0.20 (fails recall).
        # pol_c: far=0.05 recall=0.15 (fails recall).
        candidates = [
            _make_result(
                policy_id="pol_a", scorer_name="E", theta=0.7,
                false_accept_rate=0.12, recall=0.80, escalation_rate=0.3,
                selection_objective=0.22,
            ),
            _make_result(
                policy_id="pol_b", scorer_name="E", theta=0.74,
                false_accept_rate=0.08, recall=0.20, escalation_rate=0.28,
                selection_objective=0.20,
            ),
            _make_result(
                policy_id="pol_c", scorer_name="E", theta=0.80,
                false_accept_rate=0.05, recall=0.15, escalation_rate=0.45,
                selection_objective=0.29,
            ),
        ]
        selected, feasible, reason, runner_up = select_with_pulp(
            candidates, max_false_accept=0.10, min_recall=0.70, alpha=0.6
        )
        assert feasible is False
        assert selected is not None
        assert selected.policy_id in {"pol_b", "pol_c"}
        assert reason is not None

    def test_no_candidates_with_objective(self):
        pytest.importorskip("pulp")
        select_with_pulp = _import_select_with_pulp()

        candidates = [
            _make_result(policy_id="pol_x", scorer_name="E", theta=0.7),  # all None metrics
        ]
        selected, feasible, reason, runner_up = select_with_pulp(
            candidates, max_false_accept=0.10, min_recall=0.70, alpha=0.6
        )
        assert selected is None
        assert feasible is False
