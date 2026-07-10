"""
Tests for pipeline/stages/summarization/corpus_relate.py

Coverage:
  - _enrich(): comparison_scope and same_paper labeling
  - relate_from_dir(): JSON loading, canonical rule pooling, output artifact
  - relate_from_dir(): empty directory and missing canonical_rules handling
  - CorpusRelation model field presence
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.stages.summarization.helpers.corpus_relate import CorpusRelateStage

from pipeline.stages.summarization.models import (
    CorpusRelation,
    RelationTypeLabel,
)
from pipeline.stages.summarization.models import Relation


# ── Minimal fixture builders ───────────────────────────────────────────────────

def _minimal_canonical_rule(**overrides) -> dict:
    base = {
        "canonical_id": "CR_aaa_positive",
        "group_id": "GRP_aaa",
        "subject_entity": "CEAN",
        "outcome_entity": "benign",
        "relation_type": "has_feature",
        "direction": "positive",
        "predicate_text": "CEAN is a benign vascular tumour",
        "is_conflicted": False,
        "study_coverage": "single_study",
        "category": "morphology",
        "supporting_pmcids": ["PMC111"],
        "member_normal_ids": ["NF1"],
        "mean_grounding_score": 0.88,
        "finding_count": 3,
    }
    base.update(overrides)
    return base


def _minimal_paper_json(pmcid: str, canonical_rules: list[dict]) -> dict:
    return {
        "pmcid": pmcid,
        "run_id": f"run-{pmcid}",
        "status": "success",
        "canonical_rules": canonical_rules,
        "relations": [],
        "final_rules": [],
    }


def _make_raw_relation(
    cid_a: str,
    cid_b: str,
    rel_type: RelationTypeLabel = RelationTypeLabel.CONTRADICT,
    score_ab: float = 0.82,
    score_ba: float = 0.78,
) -> Relation:
    return Relation(
        rule_id_a=cid_a,
        rule_id_b=cid_b,
        relation_type=rel_type,
        nli_score_a_to_b=score_ab,
        nli_score_b_to_a=score_ba,
    )


# ── 1. CorpusRelation model ────────────────────────────────────────────────────

class TestCorpusRelationModel:
    def test_required_fields_present(self):
        fields = set(CorpusRelation.model_fields.keys())
        for f in [
            "relation_id", "comparison_scope", "same_paper",
            "pmcid_a", "pmcid_b",
            "canonical_id_a", "canonical_id_b",   # inherited: rule_id_a/b
            "relation_type",                        # inherited
            "nli_score_a_to_b", "nli_score_b_to_a",  # inherited
            "subject_entity", "outcome_entity", "category",
            "direction_a", "direction_b",
            "predicate_a", "predicate_b",
            "mean_grounding_a", "mean_grounding_b",
            "finding_count_a", "finding_count_b",
            "supporting_pmcids_a", "supporting_pmcids_b",
            "is_conflicted_a", "study_coverage_a",
            "is_conflicted_b", "study_coverage_b",
            "scope_check_result", "scope_note",
        ]:
            # rule_id_a / rule_id_b are the Relation base names
            actual_f = f.replace("canonical_id_a", "rule_id_a").replace("canonical_id_b", "rule_id_b")
            assert actual_f in fields, f"Expected field {actual_f!r} missing from CorpusRelation"

    def test_scope_check_result_defaults_to_scope_unknown(self):
        cr = CorpusRelation(
            rule_id_a="CR_a", rule_id_b="CR_b",
            relation_type=RelationTypeLabel.CONTRADICT,
            nli_score_a_to_b=0.8, nli_score_b_to_a=0.7,
            relation_id="COR_test",
            comparison_scope="cross_paper", same_paper=False,
            pmcid_a="PMC1", pmcid_b="PMC2",
            subject_entity="X", outcome_entity="Y",
            category="morphology", relation_type_structural="has_feature",
            direction_a="positive", direction_b="negative",
            predicate_a="X is Y", predicate_b="X is not Y",
            mean_grounding_a=0.9, mean_grounding_b=0.8,
            finding_count_a=2, finding_count_b=1,
            supporting_pmcids_a=["PMC1"], supporting_pmcids_b=["PMC2"],
            is_conflicted_a=False, study_coverage_a="single_study",
            is_conflicted_b=False, study_coverage_b="single_study",
        )
        assert cr.scope_check_result == "scope_unknown"
        assert cr.scope_note == ""


# ── 2. _enrich(): comparison_scope and same_paper labeling ────────────────────

class TestEnrich:
    def _make_stage(self) -> CorpusRelateStage:
        stage = CorpusRelateStage.__new__(CorpusRelateStage)
        # No RelateStage needed — only testing _enrich()
        return stage

    def test_cross_paper_relation_labeled_correctly(self):
        stage = self._make_stage()
        raw = [_make_raw_relation("CR_a_pos", "CR_b_neg")]
        id_to_pmcid = {"CR_a_pos": "PMC111", "CR_b_neg": "PMC222"}
        meta = {
            "CR_a_pos": _minimal_canonical_rule(canonical_id="CR_a_pos"),
            "CR_b_neg": _minimal_canonical_rule(
                canonical_id="CR_b_neg",
                direction="negative",
                supporting_pmcids=["PMC222"],
            ),
        }
        result = stage._enrich(raw, id_to_pmcid, meta)

        assert len(result) == 1
        rel = result[0]
        assert rel.same_paper is False
        assert rel.comparison_scope == "cross_paper"
        assert rel.pmcid_a == "PMC111"
        assert rel.pmcid_b == "PMC222"

    def test_intra_paper_relation_labeled_correctly(self):
        stage = self._make_stage()
        raw = [_make_raw_relation("CR_a_pos", "CR_a_neg", rel_type=RelationTypeLabel.CONTRADICT)]
        id_to_pmcid = {"CR_a_pos": "PMC111", "CR_a_neg": "PMC111"}
        meta = {
            "CR_a_pos": _minimal_canonical_rule(canonical_id="CR_a_pos"),
            "CR_a_neg": _minimal_canonical_rule(
                canonical_id="CR_a_neg", direction="negative"
            ),
        }
        result = stage._enrich(raw, id_to_pmcid, meta)

        rel = result[0]
        assert rel.same_paper is True
        assert rel.comparison_scope == "intra_paper"
        assert rel.pmcid_a == "PMC111"
        assert rel.pmcid_b == "PMC111"

    def test_relation_id_is_deterministic(self):
        stage = self._make_stage()
        raw = [_make_raw_relation("CR_x", "CR_y")]
        id_to_pmcid = {"CR_x": "PMC1", "CR_y": "PMC2"}
        meta = {
            "CR_x": _minimal_canonical_rule(canonical_id="CR_x"),
            "CR_y": _minimal_canonical_rule(canonical_id="CR_y"),
        }
        r1 = stage._enrich(raw, id_to_pmcid, meta)
        r2 = stage._enrich(raw, id_to_pmcid, meta)
        assert r1[0].relation_id == r2[0].relation_id

    def test_relation_id_uses_sorted_pair(self):
        """relation_id for (A, B) and (B, A) should be identical."""
        stage = self._make_stage()
        id_to_pmcid = {"CR_x": "PMC1", "CR_y": "PMC2"}
        meta = {
            "CR_x": _minimal_canonical_rule(canonical_id="CR_x"),
            "CR_y": _minimal_canonical_rule(canonical_id="CR_y"),
        }
        raw_ab = [_make_raw_relation("CR_x", "CR_y")]
        raw_ba = [_make_raw_relation("CR_y", "CR_x")]
        r_ab = stage._enrich(raw_ab, id_to_pmcid, meta)
        r_ba = stage._enrich(raw_ba, id_to_pmcid, meta)
        assert r_ab[0].relation_id == r_ba[0].relation_id

    def test_predicate_and_direction_copied(self):
        stage = self._make_stage()
        raw = [_make_raw_relation("CR_a", "CR_b", rel_type=RelationTypeLabel.SUPPORT)]
        id_to_pmcid = {"CR_a": "PMC1", "CR_b": "PMC1"}
        meta = {
            "CR_a": _minimal_canonical_rule(
                canonical_id="CR_a",
                predicate_text="CEAN is benign",
                direction="positive",
            ),
            "CR_b": _minimal_canonical_rule(
                canonical_id="CR_b",
                predicate_text="CEAN has benign behaviour",
                direction="positive",
            ),
        }
        rel = stage._enrich(raw, id_to_pmcid, meta)[0]
        assert rel.predicate_a == "CEAN is benign"
        assert rel.predicate_b == "CEAN has benign behaviour"
        assert rel.direction_a == "positive"
        assert rel.direction_b == "positive"
        assert rel.scope_check_result == "scope_unknown"

    def test_nli_scores_preserved(self):
        stage = self._make_stage()
        raw = [_make_raw_relation("CR_a", "CR_b", score_ab=0.91, score_ba=0.85)]
        id_to_pmcid = {"CR_a": "PMC1", "CR_b": "PMC2"}
        meta = {
            "CR_a": _minimal_canonical_rule(canonical_id="CR_a"),
            "CR_b": _minimal_canonical_rule(canonical_id="CR_b"),
        }
        rel = stage._enrich(raw, id_to_pmcid, meta)[0]
        assert rel.nli_score_a_to_b == pytest.approx(0.91)
        assert rel.nli_score_b_to_a == pytest.approx(0.85)


# ── 3. relate_from_dir() integration (NLI mocked) ────────────────────────────

class TestRelateFromDir:
    """
    relate_from_dir() is tested with RelateStage.relate patched to return
    a controlled list of Relation objects.  This avoids loading the NLI model.
    """

    def _write_paper_json(self, tmp_path: Path, pmcid: str, rules: list[dict]) -> Path:
        p = tmp_path / f"{pmcid}.json"
        p.write_text(
            json.dumps(_minimal_paper_json(pmcid, rules)), encoding="utf-8"
        )
        return p

    def test_output_json_written_with_correct_counts(self, tmp_path):
        src = tmp_path / "summaries"
        src.mkdir()
        out = tmp_path / "corpus_relations.json"

        rule_a = _minimal_canonical_rule(
            canonical_id="CR_a_pos", supporting_pmcids=["PMC111"]
        )
        rule_b = _minimal_canonical_rule(
            canonical_id="CR_b_neg",
            direction="negative",
            supporting_pmcids=["PMC222"],
        )
        self._write_paper_json(src, "PMC111", [rule_a])
        self._write_paper_json(src, "PMC222", [rule_b])

        fake_relation = _make_raw_relation("CR_a_pos", "CR_b_neg")

        stage = CorpusRelateStage()
        with patch.object(stage._relate, "relate", return_value=[fake_relation]):
            relations = stage.relate_from_dir(src, out)

        assert out.exists()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["total_relations"] == 1
        assert payload["cross_paper_count"] == 1
        assert payload["intra_paper_count"] == 0
        assert set(payload["pmcids_loaded"]) == {"PMC111", "PMC222"}
        assert payload["canonical_rules_loaded"] == 2

    def test_intra_paper_pair_labeled_intra(self, tmp_path):
        src = tmp_path / "summaries"
        src.mkdir()
        out = tmp_path / "corpus_relations.json"

        rule_a = _minimal_canonical_rule(
            canonical_id="CR_a_pos", direction="positive"
        )
        rule_b = _minimal_canonical_rule(
            canonical_id="CR_a_neg",
            direction="negative",
        )
        self._write_paper_json(src, "PMC111", [rule_a, rule_b])

        fake_relation = _make_raw_relation("CR_a_pos", "CR_a_neg")

        stage = CorpusRelateStage()
        with patch.object(stage._relate, "relate", return_value=[fake_relation]):
            stage.relate_from_dir(src, out)

        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["intra_paper_count"] == 1
        assert payload["cross_paper_count"] == 0
        rel = payload["relations"][0]
        assert rel["same_paper"] is True
        assert rel["comparison_scope"] == "intra_paper"
        assert rel["pmcid_a"] == "PMC111"
        assert rel["pmcid_b"] == "PMC111"

    def test_cross_paper_pair_labeled_cross(self, tmp_path):
        src = tmp_path / "summaries"
        src.mkdir()
        out = tmp_path / "corpus_relations.json"

        self._write_paper_json(
            src, "PMC111",
            [_minimal_canonical_rule(canonical_id="CR_111", supporting_pmcids=["PMC111"])],
        )
        self._write_paper_json(
            src, "PMC222",
            [_minimal_canonical_rule(
                canonical_id="CR_222", direction="negative",
                supporting_pmcids=["PMC222"],
            )],
        )

        fake_relation = _make_raw_relation("CR_111", "CR_222")
        stage = CorpusRelateStage()
        with patch.object(stage._relate, "relate", return_value=[fake_relation]):
            stage.relate_from_dir(src, out)

        payload = json.loads(out.read_text(encoding="utf-8"))
        rel = payload["relations"][0]
        assert rel["same_paper"] is False
        assert rel["comparison_scope"] == "cross_paper"
        assert rel["pmcid_a"] == "PMC111"
        assert rel["pmcid_b"] == "PMC222"

    def test_empty_source_dir_returns_empty_and_no_crash(self, tmp_path):
        src = tmp_path / "empty"
        src.mkdir()
        out = tmp_path / "corpus_relations.json"

        stage = CorpusRelateStage()
        result = stage.relate_from_dir(src, out)
        assert result == []
        assert not out.exists()

    def test_json_without_canonical_rules_is_skipped(self, tmp_path):
        src = tmp_path / "summaries"
        src.mkdir()
        out = tmp_path / "corpus_relations.json"

        # Paper without canonical_rules key (e.g., error result or old format)
        bad = tmp_path / "summaries" / "PMC999.json"
        bad.write_text(json.dumps({"pmcid": "PMC999", "status": "error"}), encoding="utf-8")

        stage = CorpusRelateStage()
        with patch.object(stage._relate, "relate", return_value=[]):
            stage.relate_from_dir(src, out)
        # Should not crash; output may or may not be written (no rules = nothing to compare)

    def test_invalid_json_skipped_without_crash(self, tmp_path):
        src = tmp_path / "summaries"
        src.mkdir()
        out = tmp_path / "corpus_relations.json"

        (src / "broken.json").write_text("{NOT VALID", encoding="utf-8")
        valid = _minimal_canonical_rule(canonical_id="CR_good")
        self._write_paper_json(src, "PMC111", [valid])

        stage = CorpusRelateStage()
        with patch.object(stage._relate, "relate", return_value=[]):
            stage.relate_from_dir(src, out)  # must not raise


# ── 4. Run-selection ───────────────────────────────────────────────────────────

class TestRunSelection:
    """
    Tests for _latest_per_pmcid, _filter_valid, and _load_manifest.
    All tests exercise the selection logic without loading the NLI model.
    """

    def _write_json(self, path: Path, pmcid: str, rules: list[dict],
                    run_id: str = "", status: str = "success") -> None:
        payload = {
            "pmcid": pmcid,
            "run_id": run_id,
            "status": status,
            "canonical_rules": rules,
            "relations": [],
            "final_rules": [],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_latest_per_pmcid_selects_newer_run(self, tmp_path):
        """When two runs exist for the same PMCID, the one with the later
        run_id timestamp is selected."""
        src = tmp_path / "summaries"
        src.mkdir()
        out = tmp_path / "corpus_relations.json"

        rule_a = _minimal_canonical_rule(canonical_id="CR_old")
        rule_b = _minimal_canonical_rule(canonical_id="CR_new")

        old_file = src / "PMC111_old.json"
        new_file = src / "PMC111_new.json"
        self._write_json(old_file, "PMC111", [rule_a], run_id="PMC111_20260101T120000")
        self._write_json(new_file, "PMC111", [rule_b], run_id="PMC111_20260201T120000")

        stage = CorpusRelateStage()
        with patch.object(stage._relate, "relate", return_value=[]):
            stage.relate_from_dir(src, out, run_selection="latest_per_pmcid")

        # Only the newer run's rule should be loaded
        payload = json.loads(out.read_text(encoding="utf-8"))
        # canonical_rules_loaded == 1 (one PMCID, one run)
        assert payload["canonical_rules_loaded"] == 1
        assert payload["pmcids_loaded"] == ["PMC111"]

    def test_latest_per_pmcid_skips_failed_runs(self, tmp_path):
        """A failed run is skipped in favour of a successful one."""
        src = tmp_path / "summaries"
        src.mkdir()
        out = tmp_path / "corpus_relations.json"

        rule = _minimal_canonical_rule(canonical_id="CR_ok")
        failed_file = src / "PMC222_failed.json"
        ok_file = src / "PMC222_ok.json"
        self._write_json(failed_file, "PMC222", [rule],
                         run_id="PMC222_20260201T120000", status="error")
        self._write_json(ok_file, "PMC222", [rule],
                         run_id="PMC222_20260101T120000", status="success")

        stage = CorpusRelateStage()
        with patch.object(stage._relate, "relate", return_value=[]):
            stage.relate_from_dir(src, out, run_selection="latest_per_pmcid")

        # The older-but-successful run must be loaded
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["canonical_rules_loaded"] == 1

    def test_latest_per_pmcid_skips_empty_canonical_rules(self, tmp_path):
        """A run with empty canonical_rules is skipped even if it is newer."""
        src = tmp_path / "summaries"
        src.mkdir()
        out = tmp_path / "corpus_relations.json"

        rule = _minimal_canonical_rule(canonical_id="CR_has_rules")
        empty_file = src / "PMC333_empty.json"
        ok_file = src / "PMC333_ok.json"
        # newer timestamp, but empty rules
        self._write_json(empty_file, "PMC333", [],
                         run_id="PMC333_20260301T000000", status="success")
        self._write_json(ok_file, "PMC333", [rule],
                         run_id="PMC333_20260101T000000", status="success")

        stage = CorpusRelateStage()
        with patch.object(stage._relate, "relate", return_value=[]):
            stage.relate_from_dir(src, out, run_selection="latest_per_pmcid")

        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["canonical_rules_loaded"] == 1

    def test_run_selection_all_includes_both_runs(self, tmp_path):
        """run_selection='all' pools both runs even when they share a PMCID."""
        src = tmp_path / "summaries"
        src.mkdir()
        out = tmp_path / "corpus_relations.json"

        rule_a = _minimal_canonical_rule(canonical_id="CR_run1")
        rule_b = _minimal_canonical_rule(canonical_id="CR_run2")
        self._write_json(src / "PMC444_run1.json", "PMC444", [rule_a],
                         run_id="PMC444_20260101T000000")
        self._write_json(src / "PMC444_run2.json", "PMC444", [rule_b],
                         run_id="PMC444_20260201T000000")

        stage = CorpusRelateStage()
        with patch.object(stage._relate, "relate", return_value=[]):
            stage.relate_from_dir(src, out, run_selection="all")

        payload = json.loads(out.read_text(encoding="utf-8"))
        # Both runs' rules must be present
        assert payload["canonical_rules_loaded"] == 2

    def test_manifest_overrides_dir_selection(self, tmp_path):
        """manifest= causes exactly those files to be loaded, ignoring source_dir."""
        src = tmp_path / "summaries"
        src.mkdir()
        out = tmp_path / "corpus_relations.json"

        rule_a = _minimal_canonical_rule(canonical_id="CR_selected")
        rule_b = _minimal_canonical_rule(canonical_id="CR_ignored")
        selected_file = src / "PMC555.json"
        ignored_file = src / "PMC666.json"
        self._write_json(selected_file, "PMC555", [rule_a])
        self._write_json(ignored_file, "PMC666", [rule_b])

        stage = CorpusRelateStage()
        with patch.object(stage._relate, "relate", return_value=[]):
            stage.relate_from_dir(src, out, manifest=[selected_file])

        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["pmcids_loaded"] == ["PMC555"]
        assert payload["canonical_rules_loaded"] == 1
        assert payload["run_selection"] == "manifest"

    def test_missing_run_id_falls_back_to_mtime(self, tmp_path):
        """When run_id has no timestamp suffix, file mtime is used as tiebreaker."""
        src = tmp_path / "summaries"
        src.mkdir()
        out = tmp_path / "corpus_relations.json"

        rule_old = _minimal_canonical_rule(canonical_id="CR_older_mtime")
        rule_new = _minimal_canonical_rule(canonical_id="CR_newer_mtime")

        old_file = src / "PMC777_old.json"
        new_file = src / "PMC777_new.json"
        self._write_json(old_file, "PMC777", [rule_old], run_id="PMC777_noTS")
        self._write_json(new_file, "PMC777", [rule_new], run_id="PMC777_noTS")

        # Force old_file to have an earlier mtime
        import os
        os.utime(old_file, (1_000_000, 1_000_000))  # epoch +~11 days
        os.utime(new_file, (2_000_000, 2_000_000))  # epoch +~23 days

        stage = CorpusRelateStage()
        with patch.object(stage._relate, "relate", return_value=[]):
            stage.relate_from_dir(src, out, run_selection="latest_per_pmcid")

        # Exactly one PMCID, one rule
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["canonical_rules_loaded"] == 1
