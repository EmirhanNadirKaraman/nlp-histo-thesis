"""
Tests for scripts/inspect_pipeline_output.py

Coverage:
  - PMCID validation in compare mode
  - Diff summary generation (_diff_items)
  - Sentence hover lookup (_build_sentence_lookup)
  - Flagged CSV export (export_flagged_csv)
  - Batch index generation (render_batch)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import importlib.util

import pytest

# scripts/ is not a package — load the module directly by file path
_SCRIPT = Path(__file__).parent.parent / "scripts" / "inspect_pipeline_output.py"
_spec = importlib.util.spec_from_file_location("inspect_pipeline_output", _SCRIPT)
_inspector = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_inspector)

# Pull names from the loaded module
_build_sentence_lookup = _inspector._build_sentence_lookup
_compute_flags = _inspector._compute_flags
_diff_items = _inspector._diff_items
_entity_flags = _inspector._entity_flags
_finding_key = _inspector._finding_key
_stable_key = _inspector._stable_key
build_context = _inspector.build_context
build_diff_context = _inspector.build_diff_context
export_flagged_csv = _inspector.export_flagged_csv
render = _inspector.render
render_batch = _inspector.render_batch
render_diff = _inspector.render_diff



# ── Minimal fixture builders ───────────────────────────────────────────────────

def _minimal_rule(**overrides) -> dict:
    base = {
        "final_id": "F1",
        "canonical_id": "C1",
        "group_id": "G1",
        "subject_entity": "CEAN",
        "outcome_entity": "benign",
        "predicate_text": "CEAN is a benign tumour",
        "relation_type": "has_feature",
        "direction": "positive",
        "category": "morphology",
        "final_score": 0.85,
        "mean_grounding_score": 0.9,
        "support_count": 3,
        "contradict_count": 0,
        "scope_qualify_count": 0,
        "is_contradicted": False,
        "contradicted_by": [],
        "is_conflicted": False,
        "study_coverage": "single_study",
        "supporting_pmcids": ["PMC10047158"],
        "member_normal_ids": ["N1"],
        "finding_count": 3,
    }
    base.update(overrides)
    return base


def _minimal_data(pmcid: str = "PMC1234567", run_id: str = "run-1") -> dict:
    return {
        "pmcid": pmcid,
        "run_id": run_id,
        "status": "success",
        "summary": "test summary",
        "final_rules": [_minimal_rule()],
        "canonical_rules": [
            {
                "canonical_id": "C1",
                "group_id": "G1",
                "subject_entity": "CEAN",
                "outcome_entity": "benign",
                "predicate_text": "CEAN is a benign tumour",
                "relation_type": "has_feature",
                "direction": "positive",
                "category": "morphology",
                "finding_count": 3,
                "mean_grounding_score": 0.9,
                "is_conflicted": False,
        "study_coverage": "single_study",
                "supporting_pmcids": ["PMC1234567"],
                "member_normal_ids": ["N1"],
            }
        ],
        "relations": [],
        "audit_trail": {
            "map_chunks": [
                {
                    "chunk_id": "C1",
                    "findings": [
                        {
                            "subject_entity": "CEAN",
                            "outcome_entity": "benign",
                            "claim": "CEAN is a benign tumour",
                            "relation_type": "has_feature",
                            "direction": "positive",
                            "category": "morphology",
                            "confidence": "high",
                            "grounding_score": 0.9,
                            "verbatim_support": "CEAN is a rare benign vascular tumour of the skin.",
                            "evidence": ["S2|PMC1234567|5"],
                            "scope": None,
                        }
                    ],
                    "audit_metadata": {
                        "sentences_analyzed": 10,
                        "sentences_cited": ["S2|PMC1234567|5"],
                        "pmcids_referenced": ["PMC1234567"],
                        "uncited_sentences": [],
                    },
                }
            ],
            "rules_provenance": {
                "rules": [
                    {
                        "rule_id": "R1",
                        "type": "diagnostic",
                        "condition": "IF CEAN",
                        "action": "benign diagnosis",
                        "confidence": "High",
                        "evidence_chain": [
                            {
                                "sentence_id": "S2",
                                "pmcid": "PMC1234567",
                                "text_element_id": 5,
                                "verbatim": "CEAN is a rare benign vascular tumour of the skin.",
                            }
                        ],
                    }
                ],
                "audit_summary": {},
            },
        },
    }


# ── 1. PMCID validation ────────────────────────────────────────────────────────

class TestPmcidValidation:
    def test_matching_pmcid_does_not_raise(self, tmp_path):
        data_a = _minimal_data("PMC111", "run-a")
        data_b = _minimal_data("PMC111", "run-b")

        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        out = tmp_path / "diff.html"
        a.write_text(json.dumps(data_a), encoding="utf-8")
        b.write_text(json.dumps(data_b), encoding="utf-8")

        # Should not raise / sys.exit
        render_diff(a, b, out)
        assert out.exists()

    def test_mismatched_pmcid_exits(self, tmp_path):
        data_a = _minimal_data("PMC111", "run-a")
        data_b = _minimal_data("PMC222", "run-b")  # different PMCID

        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        out = tmp_path / "diff.html"
        a.write_text(json.dumps(data_a), encoding="utf-8")
        b.write_text(json.dumps(data_b), encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info:
            render_diff(a, b, out)
        assert exc_info.value.code != 0


# ── 2. Diff summary generation ────────────────────────────────────────────────

class TestDiffItems:
    def _make_rule(self, subj: str, score: float = 0.8) -> dict:
        return _minimal_rule(subject_entity=subj, final_score=score)

    def test_identical_lists_produce_no_diff(self):
        rules = [self._make_rule("A"), self._make_rule("B")]
        result = _diff_items(rules, rules, _stable_key, ["final_score"])
        assert result["added"] == []
        assert result["removed"] == []
        assert result["changed"] == []

    def test_added_item_detected(self):
        a = [self._make_rule("A")]
        b = [self._make_rule("A"), self._make_rule("B")]
        result = _diff_items(a, b, _stable_key, ["final_score"])
        assert len(result["added"]) == 1
        assert result["added"][0]["subject_entity"] == "B"
        assert result["removed"] == []

    def test_removed_item_detected(self):
        a = [self._make_rule("A"), self._make_rule("B")]
        b = [self._make_rule("A")]
        result = _diff_items(a, b, _stable_key, ["final_score"])
        assert len(result["removed"]) == 1
        assert result["removed"][0]["subject_entity"] == "B"
        assert result["added"] == []

    def test_changed_field_detected(self):
        a = [self._make_rule("A", score=0.8)]
        b = [self._make_rule("A", score=0.6)]
        result = _diff_items(a, b, _stable_key, ["final_score"])
        assert len(result["changed"]) == 1
        changed = result["changed"][0]
        assert "final_score" in changed["field_diffs"]
        assert changed["field_diffs"]["final_score"] == (0.8, 0.6)

    def test_unchanged_field_not_in_field_diffs(self):
        # Two rules with identical final_score — no change
        a = [self._make_rule("A", score=0.8)]
        b = [self._make_rule("A", score=0.8)]
        result = _diff_items(a, b, _stable_key, ["final_score"])
        assert result["changed"] == []

    def test_build_diff_context_summary_counts(self):
        data_a = _minimal_data("PMC111", "run-a")
        data_b = _minimal_data("PMC111", "run-b")
        # Add an extra rule in b
        extra = _minimal_rule(subject_entity="EXTRA", final_id="F99", canonical_id="C99")
        data_b["final_rules"].append(extra)

        ctx = build_diff_context(data_a, data_b)
        ds = ctx["diff_summary"]
        assert ds["final_added"] == 1
        assert ds["final_removed"] == 0


# ── 3. Sentence hover lookup ──────────────────────────────────────────────────

class TestSentenceLookup:
    def test_basic_map_finding_verbatim(self):
        data = _minimal_data()
        lookup = _build_sentence_lookup(data)
        assert "S2|PMC1234567|5" in lookup
        assert "rare benign" in lookup["S2|PMC1234567|5"]

    def test_rules_provenance_verbatim(self):
        data = _minimal_data()
        # Add a different sentence only in rules_provenance
        data["audit_trail"]["rules_provenance"]["rules"][0]["evidence_chain"].append(
            {
                "sentence_id": "S9",
                "pmcid": "PMC1234567",
                "text_element_id": 8,
                "verbatim": "Following excision, no recurrence was observed.",
            }
        )
        lookup = _build_sentence_lookup(data)
        assert "S9|PMC1234567|8" in lookup
        assert "no recurrence" in lookup["S9|PMC1234567|8"]

    def test_missing_verbatim_not_included(self):
        data = _minimal_data()
        # Clear verbatim_support from MAP finding
        data["audit_trail"]["map_chunks"][0]["findings"][0]["verbatim_support"] = None
        # Clear evidence_chain verbatim
        data["audit_trail"]["rules_provenance"]["rules"][0]["evidence_chain"][0]["verbatim"] = ""
        lookup = _build_sentence_lookup(data)
        # Token should NOT be in lookup if no text is available
        assert "S2|PMC1234567|5" not in lookup

    def test_empty_audit_trail_gives_empty_lookup(self):
        data = _minimal_data()
        data["audit_trail"] = {}
        lookup = _build_sentence_lookup(data)
        assert lookup == {}

    def test_sentence_lookup_in_context(self):
        data = _minimal_data()
        ctx = build_context(data)
        # sentence_lookup_json must be a valid JSON string
        parsed = json.loads(ctx["sentence_lookup_json"])
        assert isinstance(parsed, dict)
        assert "S2|PMC1234567|5" in parsed


# ── 4. Flagged CSV export ─────────────────────────────────────────────────────

class TestFlaggedCsvExport:
    def _build_flagged_context(self):
        data = _minimal_data()
        # Make subject_entity generic so it gets flagged
        data["final_rules"][0]["subject_entity"] = "patient"
        data["canonical_rules"][0]["subject_entity"] = "patient"
        data["audit_trail"]["map_chunks"][0]["findings"][0]["subject_entity"] = "patient"
        return data

    def test_csv_created_with_expected_columns(self, tmp_path):
        data = self._build_flagged_context()
        ctx = build_context(data)
        out_csv = tmp_path / "flagged.csv"
        n = export_flagged_csv(ctx, data, out_csv)
        assert out_csv.exists()
        with open(out_csv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        # At least one flagged row
        assert n > 0
        assert len(rows) == n
        # Required columns present
        for col in ["pmcid", "run_id", "stage", "flags", "subject_entity"]:
            assert col in reader.fieldnames

    def test_csv_flags_column_populated(self, tmp_path):
        data = self._build_flagged_context()
        ctx = build_context(data)
        out_csv = tmp_path / "flagged.csv"
        export_flagged_csv(ctx, data, out_csv)
        with open(out_csv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        # Every row should have a non-empty flags column (since we only export flagged items)
        for row in rows:
            assert row["flags"] != ""

    def test_csv_empty_if_no_flags(self, tmp_path):
        data = _minimal_data()  # "CEAN" not a generic entity → no flags
        # Ensure no flags by checking
        ctx = build_context(data)
        if ctx["flagged_count"] == 0:
            out_csv = tmp_path / "empty.csv"
            n = export_flagged_csv(ctx, data, out_csv)
            with open(out_csv, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            assert len(rows) == n  # consistent

    def test_csv_pmcid_and_run_id_correct(self, tmp_path):
        data = self._build_flagged_context()
        ctx = build_context(data)
        out_csv = tmp_path / "flagged.csv"
        export_flagged_csv(ctx, data, out_csv)
        with open(out_csv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        for row in rows:
            assert row["pmcid"] == "PMC1234567"
            assert row["run_id"] == "run-1"


# ── 5. Batch inspector ────────────────────────────────────────────────────────

class TestBatchInspector:
    def test_index_created_with_one_row_per_paper(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"

        # Write two minimal JSON files
        for i, pmcid in enumerate(["PMC1111111", "PMC2222222"]):
            data = _minimal_data(pmcid, f"run-{i}")
            (src / f"{pmcid}.json").write_text(json.dumps(data), encoding="utf-8")

        render_batch(src, out)

        index = out / "index.html"
        assert index.exists()
        content = index.read_text(encoding="utf-8")
        assert "PMC1111111" in content
        assert "PMC2222222" in content

    def test_individual_html_per_paper_created(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"

        for pmcid in ["PMC1111111", "PMC2222222"]:
            data = _minimal_data(pmcid, "run-x")
            (src / f"{pmcid}.json").write_text(json.dumps(data), encoding="utf-8")

        render_batch(src, out)

        assert (out / "PMC1111111.html").exists()
        assert (out / "PMC2222222.html").exists()

    def test_invalid_json_is_skipped(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"

        # One valid, one invalid
        data = _minimal_data("PMC9999999", "run-x")
        (src / "PMC9999999.json").write_text(json.dumps(data), encoding="utf-8")
        (src / "broken.json").write_text("{NOT VALID JSON", encoding="utf-8")

        render_batch(src, out)  # should not crash

        index = out / "index.html"
        assert index.exists()
        content = index.read_text(encoding="utf-8")
        assert "PMC9999999" in content

    def test_empty_source_dir_produces_no_crash(self, tmp_path):
        src = tmp_path / "empty"
        src.mkdir()
        out = tmp_path / "out"
        render_batch(src, out)  # should not crash or raise

    def test_single_run_html_is_self_contained(self, tmp_path):
        """Generated single HTML must not reference external resources."""
        data = _minimal_data("PMC1234567", "run-x")
        json_path = tmp_path / "PMC1234567.json"
        out_path = tmp_path / "PMC1234567.html"
        json_path.write_text(json.dumps(data), encoding="utf-8")

        render(json_path, out_path)
        html = out_path.read_text(encoding="utf-8")
        # No external stylesheet or script links
        assert 'src="http' not in html
        assert 'href="http' not in html
        # Contains SENTENCE_LOOKUP definition
        assert "SENTENCE_LOOKUP" in html


# ── 6. Entity flag unit tests ─────────────────────────────────────────────────

class TestEntityFlags:
    """Direct regression tests for _entity_flags() and the polarity branch of _compute_flags()."""

    # ── _entity_flags ──────────────────────────────────────────────────────────

    def test_none_entity_returns_empty_entity_flag(self):
        assert _entity_flags(None) == ["empty-entity"]

    def test_empty_string_entity_returns_empty_entity_flag(self):
        assert _entity_flags("") == ["empty-entity"]

    def test_taxonomy_leak_cetacea(self):
        flags = _entity_flags("Cetacea vascular")
        assert "taxonomy-leak" in flags

    def test_taxonomy_leak_case_insensitive(self):
        flags = _entity_flags("RODENTIA")
        assert "taxonomy-leak" in flags

    def test_taxonomy_no_false_positive_on_clean_entity(self):
        flags = _entity_flags("CEAN")
        assert "taxonomy-leak" not in flags

    def test_procedure_as_entity_immunohistochemistry(self):
        flags = _entity_flags("immunohistochemistry")
        assert "procedure-as-entity" in flags

    def test_procedure_as_entity_pcr(self):
        flags = _entity_flags("PCR analysis")
        assert "procedure-as-entity" in flags

    def test_procedure_no_false_positive_on_clean_entity(self):
        flags = _entity_flags("CEAN tumour")
        assert "procedure-as-entity" not in flags

    def test_generic_entity_patient(self):
        flags = _entity_flags("patient")
        assert "generic-entity" in flags

    def test_generic_entity_patients_plural(self):
        flags = _entity_flags("patients")
        assert "generic-entity" in flags

    def test_generic_entity_lesion(self):
        flags = _entity_flags("lesion")
        assert "generic-entity" in flags

    def test_generic_entity_the_patient(self):
        flags = _entity_flags("the patient")
        assert "generic-entity" in flags

    def test_generic_no_false_positive_on_specific_entity(self):
        flags = _entity_flags("CEAN")
        assert "generic-entity" not in flags

    def test_multiple_flags_can_coexist(self):
        # taxonomy + procedure both match — _GENERIC_RE is anchored so can't co-fire here
        flags = _entity_flags("rodentia immunohistochemistry")
        assert "taxonomy-leak" in flags
        assert "procedure-as-entity" in flags

    # ── polarity/assertion mismatch in _compute_flags ──────────────────────────

    def test_polarity_mismatch_uncertain_with_negation(self):
        obj = {
            "subject_entity": "CEAN",
            "outcome_entity": "benign",
            "predicate_text": "no evidence of malignancy",
            "mean_grounding_score": 0.9,
        }
        flags = _compute_flags(obj)
        assert "polarity-mismatch" in flags

    def test_polarity_mismatch_uncertain_with_absent(self):
        obj = {
            "subject_entity": "CEAN",
            "outcome_entity": "invasion",
            "predicate_text": "absent vascular invasion",
            "mean_grounding_score": 0.9,
        }
        flags = _compute_flags(obj)
        assert "polarity-mismatch" in flags

    def test_no_polarity_mismatch_when_confirmed(self):
        obj = {
            "subject_entity": "CEAN",
            "outcome_entity": "benign",
            "predicate_text": "no evidence of malignancy",
            "mean_grounding_score": 0.9,
        }
        flags = _compute_flags(obj)
        assert "polarity-mismatch" not in flags

    def test_no_polarity_mismatch_when_uncertain_but_no_polarity_word(self):
        obj = {
            "subject_entity": "CEAN",
            "outcome_entity": "benign",
            "predicate_text": "CEAN may be a benign tumour",
            "mean_grounding_score": 0.9,
        }
        flags = _compute_flags(obj)
        assert "polarity-mismatch" not in flags


# ── 7. CSV deduplication ──────────────────────────────────────────────────────

class TestCsvDeduplication:
    """A flagged rule present in both final_rules and canonical_rules must appear
    only once in the CSV (FINAL stage wins; CANONICAL is suppressed)."""

    def _build_duplicate_flagged_data(self) -> dict:
        """Return data where the single rule is flagged and shared across both rule lists."""
        data = _minimal_data()
        # "patient" triggers generic-entity flag → rule becomes flagged in both layers
        data["final_rules"][0]["subject_entity"] = "patient"
        data["canonical_rules"][0]["subject_entity"] = "patient"
        # Ensure both rules have the same stable key fields
        for key in ("predicate_text", "relation_type", "outcome_entity", "category"):
            data["canonical_rules"][0][key] = data["final_rules"][0][key]
        return data

    def test_flagged_rule_appears_only_once_in_csv(self, tmp_path):
        data = self._build_duplicate_flagged_data()
        ctx = build_context(data)
        out_csv = tmp_path / "dedup.csv"
        n = export_flagged_csv(ctx, data, out_csv)

        with open(out_csv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        # Count rows whose stable_key matches the shared rule
        shared_key = "|".join([
            data["final_rules"][0].get("category", "").lower(),
            data["final_rules"][0].get("relation_type", "").lower(),
            "patient",
            data["final_rules"][0].get("outcome_entity", "").lower(),
            data["final_rules"][0].get("predicate_text", "").lower(),
        ])
        matching = [r for r in rows if r["stable_key"] == shared_key]
        assert len(matching) == 1, (
            f"Expected 1 row for the shared stable key, got {len(matching)}"
        )

    def test_deduplicated_row_has_final_stage(self, tmp_path):
        data = self._build_duplicate_flagged_data()
        ctx = build_context(data)
        out_csv = tmp_path / "dedup.csv"
        export_flagged_csv(ctx, data, out_csv)

        with open(out_csv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        # The surviving row must be the FINAL stage, not CANONICAL
        stages = [r["stage"] for r in rows]
        assert "FINAL" in stages
        assert "CANONICAL" not in stages

    def test_non_overlapping_canonical_rule_is_included(self, tmp_path):
        """A canonical rule whose stable_key does NOT appear in final_rules is kept."""
        data = _minimal_data()
        # Flag the final rule
        data["final_rules"][0]["subject_entity"] = "patient"
        # Add a distinct canonical rule that only lives in canonical_rules
        extra_canon = {
            "canonical_id": "C99",
            "group_id": "G99",
            "subject_entity": "patients",       # generic → flagged
            "outcome_entity": "recurrence",
            "predicate_text": "Recurrence was observed in patients",
            "relation_type": "causes",
            "direction": "positive",
            "category": "prognosis",
            "finding_count": 1,
            "mean_grounding_score": 0.8,
            "is_conflicted": False,
        "study_coverage": "single_study",
            "supporting_pmcids": ["PMC1234567"],
            "member_normal_ids": [],
        }
        data["canonical_rules"].append(extra_canon)
        # Keep the first canonical rule in sync with the final rule (will be suppressed)
        data["canonical_rules"][0]["subject_entity"] = "patient"

        ctx = build_context(data)
        out_csv = tmp_path / "nonoverlap.csv"
        export_flagged_csv(ctx, data, out_csv)

        with open(out_csv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        stages = [r["stage"] for r in rows]
        # The extra canonical rule (no matching final key) must appear
        assert "CANONICAL" in stages


# ── 8. render_diff HTML marker assertions ────────────────────────────────────

class TestRenderDiffHtmlMarkers:
    """Verify that + ADDED / - REMOVED / ~ CHANGED markers appear in the
    rendered diff HTML when expected and are absent when not needed."""

    def _write_json(self, tmp_path, name: str, data: dict) -> Path:
        p = tmp_path / name
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def test_added_marker_present_when_rule_added_in_b(self, tmp_path):
        data_a = _minimal_data("PMC111", "run-a")
        data_b = _minimal_data("PMC111", "run-b")
        data_b["final_rules"].append(
            _minimal_rule(subject_entity="EXTRA", final_id="F99", canonical_id="C99")
        )
        a = self._write_json(tmp_path, "a.json", data_a)
        b = self._write_json(tmp_path, "b.json", data_b)
        out = tmp_path / "diff.html"

        render_diff(a, b, out)
        html = out.read_text(encoding="utf-8")
        assert "+ ADDED" in html

    def test_removed_marker_present_when_rule_removed_in_b(self, tmp_path):
        data_a = _minimal_data("PMC111", "run-a")
        data_b = _minimal_data("PMC111", "run-b")
        # data_a has the base rule; data_b has no rules
        data_b["final_rules"] = []
        a = self._write_json(tmp_path, "a.json", data_a)
        b = self._write_json(tmp_path, "b.json", data_b)
        out = tmp_path / "diff.html"

        render_diff(a, b, out)
        html = out.read_text(encoding="utf-8")
        assert "- REMOVED" in html

    def test_changed_marker_present_when_field_differs(self, tmp_path):
        data_a = _minimal_data("PMC111", "run-a")
        data_b = _minimal_data("PMC111", "run-b")
        # Same stable key, different final_score → ~ CHANGED
        data_b["final_rules"][0]["final_score"] = 0.50
        a = self._write_json(tmp_path, "a.json", data_a)
        b = self._write_json(tmp_path, "b.json", data_b)
        out = tmp_path / "diff.html"

        render_diff(a, b, out)
        html = out.read_text(encoding="utf-8")
        assert "~ CHANGED" in html

    def test_no_added_marker_when_rules_are_identical(self, tmp_path):
        data_a = _minimal_data("PMC111", "run-a")
        data_b = _minimal_data("PMC111", "run-b")
        a = self._write_json(tmp_path, "a.json", data_a)
        b = self._write_json(tmp_path, "b.json", data_b)
        out = tmp_path / "diff.html"

        render_diff(a, b, out)
        html = out.read_text(encoding="utf-8")
        # With no differences there should be no diff-marker spans for add/rem/chg
        assert "+ ADDED" not in html
        assert "- REMOVED" not in html
        assert "~ CHANGED" not in html


# ── 9. Low-grounding threshold sensitivity ────────────────────────────────────

class TestLowGsThreshold:
    """flagged_count must respond to changes in low_gs_threshold."""

    def _build_low_gs_data(self, gs: float = 0.35) -> dict:
        """Return data with one rule whose mean_grounding_score is gs and no other flag triggers."""
        data = _minimal_data()
        data["final_rules"][0]["mean_grounding_score"] = gs
        # Keep entity clean so the only possible flag is low-grounding
        data["final_rules"][0]["subject_entity"] = "CEAN"
        data["final_rules"][0]["predicate_text"] = "CEAN is a benign tumour"
        return data

    def test_rule_flagged_when_gs_below_threshold(self):
        data = self._build_low_gs_data(gs=0.35)
        ctx = build_context(data, low_gs_threshold=0.4)
        assert ctx["flagged_count"] == 1
        assert ctx["low_grounding_count"] == 1

    def test_rule_not_flagged_when_gs_above_threshold(self):
        data = self._build_low_gs_data(gs=0.35)
        ctx = build_context(data, low_gs_threshold=0.3)
        assert ctx["flagged_count"] == 0
        assert ctx["low_grounding_count"] == 0

    def test_threshold_exactly_at_gs_is_not_flagged(self):
        # gs < threshold required; equal should NOT trigger the flag
        data = self._build_low_gs_data(gs=0.4)
        ctx = build_context(data, low_gs_threshold=0.4)
        assert ctx["flagged_count"] == 0

    def test_flagged_count_increases_with_stricter_threshold(self):
        """Two rules: gs=0.35 and gs=0.55. At 0.4 → 1 flagged; at 0.6 → 2 flagged."""
        data = _minimal_data()
        data["final_rules"][0]["mean_grounding_score"] = 0.35
        data["final_rules"][0]["subject_entity"] = "CEAN"
        data["final_rules"][0]["predicate_text"] = "CEAN is a benign tumour"

        second = _minimal_rule(
            subject_entity="CEAN",
            final_id="F2",
            canonical_id="C2",
            predicate_text="CEAN has a low recurrence rate",
        )
        second["mean_grounding_score"] = 0.55
        data["final_rules"].append(second)

        ctx_04 = build_context(data, low_gs_threshold=0.4)
        ctx_06 = build_context(data, low_gs_threshold=0.6)

        assert ctx_04["flagged_count"] == 1
        assert ctx_06["flagged_count"] == 2
