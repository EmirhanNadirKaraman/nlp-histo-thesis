"""
Tests for RELATE sweep data path and offline re-classification.

Coverage:
  - _classify_pair_offline: all four label branches
  - run_relate_sweep: non-zero counts from fixture pairs
  - corpus_relate tuple-unpack regression: CorpusRelateStage must not crash
    when RelateStage.relate returns (relations, raw_pairs) tuple
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch


from eval.silver.pipeline_sweep import _classify_pair_offline, run_relate_sweep


# ── _classify_pair_offline ────────────────────────────────────────────────────

def test_classify_pair_offline_support():
    pair = {"ent_a_to_b": 0.9, "ent_b_to_a": 0.9, "con_a_to_b": 0.1, "con_b_to_a": 0.1}
    assert _classify_pair_offline(pair, ent_t=0.55, con_t=0.65) == "SUPPORT"


def test_classify_pair_offline_contradict():
    pair = {"ent_a_to_b": 0.1, "ent_b_to_a": 0.1, "con_a_to_b": 0.9, "con_b_to_a": 0.9}
    assert _classify_pair_offline(pair, ent_t=0.55, con_t=0.65) == "CONTRADICT"


def test_classify_pair_offline_scope_qualify_forward():
    pair = {"ent_a_to_b": 0.9, "ent_b_to_a": 0.2, "con_a_to_b": 0.1, "con_b_to_a": 0.1}
    assert _classify_pair_offline(pair, ent_t=0.55, con_t=0.65) == "SCOPE_QUALIFY"


def test_classify_pair_offline_scope_qualify_reverse():
    pair = {"ent_a_to_b": 0.2, "ent_b_to_a": 0.9, "con_a_to_b": 0.1, "con_b_to_a": 0.1}
    assert _classify_pair_offline(pair, ent_t=0.55, con_t=0.65) == "SCOPE_QUALIFY"


def test_classify_pair_offline_unrelated():
    pair = {"ent_a_to_b": 0.2, "ent_b_to_a": 0.2, "con_a_to_b": 0.2, "con_b_to_a": 0.2}
    assert _classify_pair_offline(pair, ent_t=0.55, con_t=0.65) == "UNRELATED"


def test_classify_pair_offline_contradict_priority_over_entailment():
    """Contradiction takes priority: if both con scores fire, label is CONTRADICT."""
    pair = {"ent_a_to_b": 0.9, "ent_b_to_a": 0.9, "con_a_to_b": 0.9, "con_b_to_a": 0.9}
    assert _classify_pair_offline(pair, ent_t=0.55, con_t=0.65) == "CONTRADICT"


# ── run_relate_sweep fixture test ─────────────────────────────────────────────

def test_run_relate_sweep_nonzero_from_fixture():
    """Sweep produces correct label counts from a small fixture pair list."""
    fake_pairs = [
        {"ent_a_to_b": 0.8, "ent_b_to_a": 0.8, "con_a_to_b": 0.1, "con_b_to_a": 0.1},  # SUPPORT
        {"ent_a_to_b": 0.1, "ent_b_to_a": 0.1, "con_a_to_b": 0.9, "con_b_to_a": 0.9},  # CONTRADICT
        {"ent_a_to_b": 0.8, "ent_b_to_a": 0.3, "con_a_to_b": 0.1, "con_b_to_a": 0.1},  # SCOPE_QUALIFY
        {"ent_a_to_b": 0.2, "ent_b_to_a": 0.2, "con_a_to_b": 0.2, "con_b_to_a": 0.2},  # UNRELATED
    ]
    rows = run_relate_sweep(
        all_pairs=fake_pairs,
        ent_thresholds=[0.5],
        con_thresholds=[0.6],
    )
    assert rows, "sweep returned no rows"
    assert len(rows) == 1
    row = rows[0]
    assert row["n_total_pairs"] == 4
    assert row["n_support"] == 1
    assert row["n_contradict"] == 1
    assert row["n_scope_qualify"] == 1
    assert row["n_unrelated"] == 1


def test_run_relate_sweep_empty_pairs_returns_zero_counts():
    """Sweep with no pairs returns rows with all-zero counts."""
    rows = run_relate_sweep(all_pairs=[], ent_thresholds=[0.5], con_thresholds=[0.6])
    assert rows
    row = rows[0]
    assert row["n_total_pairs"] == 0
    assert row["n_support"] == 0
    assert row["n_contradict"] == 0


def test_run_relate_sweep_multiple_thresholds():
    """Grid of (ent, con) thresholds produces one row per combo."""
    rows = run_relate_sweep(
        all_pairs=[{"ent_a_to_b": 0.5, "ent_b_to_a": 0.5,
                    "con_a_to_b": 0.3, "con_b_to_a": 0.3}],
        ent_thresholds=[0.3, 0.6],
        con_thresholds=[0.4, 0.7],
    )
    assert len(rows) == 4  # 2 ent × 2 con


# ── corpus_relate tuple-unpack regression ─────────────────────────────────────

_RULE_A = {
    "canonical_id": "CR_aaa",
    "group_id": "G_aaa",
    "subject_entity": "MGA",
    "outcome_entity": "CD34",
    "relation_type": "expression",
    "direction": "positive",
    "predicate_text": "MGA shows CD34 expression",
    "is_conflicted": False,
    "study_coverage": "single_study",
    "category": "IHC",
    "supporting_pmcids": ["PMC001"],
    "member_normal_ids": ["NF_001"],
    "mean_grounding_score": 0.9,
    "finding_count": 1,
}

_RULE_B = {
    "canonical_id": "CR_bbb",
    "group_id": "G_bbb",
    "subject_entity": "microglandular adenosis",
    "outcome_entity": "CD34",
    "relation_type": "expression",
    "direction": "negative",
    "predicate_text": "microglandular adenosis lacks CD34",
    "is_conflicted": False,
    "study_coverage": "single_study",
    "category": "IHC",
    "supporting_pmcids": ["PMC002"],
    "member_normal_ids": ["NF_002"],
    "mean_grounding_score": 0.8,
    "finding_count": 1,
}


def test_corpus_relate_tuple_unpack_does_not_crash():
    """
    Regression: CorpusRelateStage.relate_from_dir must correctly unpack the
    (relations, raw_pairs, skipped_pairs) tuple returned by RelateStage.relate.

    Before the fix, raw_relations was assigned the full tuple, causing
    len(raw_relations) == 3 and _enrich() to iterate over (list, list, list)
    instead of list[Relation], producing silent garbage or a TypeError.
    """
    from pipeline.stages.knowledge_extraction.stages.corpus_relate import CorpusRelateStage

    stage = CorpusRelateStage()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        # Single manifest file with two canonical rules from two different PMCIDs
        manifest_data = {
            "status": "success",
            "pmcid": "PMC001",
            "run_id": "test_run",
            "canonical_rules": [_RULE_A, _RULE_B],
        }
        manifest_file = tmp / "PMC001.json"
        manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

        output_path = tmp / "corpus_out.json"

        # Mock RelateStage.relate to return the correct 3-tuple
        # (relations, raw_pairs, skipped_pairs) without running NLI
        with patch.object(stage._relate, "relate", return_value=([], [], [])) as mock_relate:
            result = stage.relate_from_dir(
                source_dir=None,
                output_path=output_path,
                manifest=[manifest_file],
            )

        mock_relate.assert_called_once()
        assert isinstance(result, list), "relate_from_dir must return list[CorpusRelation]"
        assert output_path.exists(), "output JSON must be written"

        payload = json.loads(output_path.read_text())
        assert "raw_nli_pairs" in payload, "output must include raw_nli_pairs key"
