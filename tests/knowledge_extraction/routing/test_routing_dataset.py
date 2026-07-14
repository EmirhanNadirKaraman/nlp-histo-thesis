"""
Tests for RoutingRecord / RoutingDataset serialization and MapStage collector hook.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from nlp_histo.pipeline.stages.knowledge_extraction.agreement.checker import AgreementChecker
from nlp_histo.pipeline.stages.knowledge_extraction.interfaces.scoring import ChunkDecision, ScoreBundle
from nlp_histo.pipeline.stages.knowledge_extraction.models import AuditMetadata, AuditableSummary, Finding
from nlp_histo.pipeline.stages.knowledge_extraction.routing.routing_dataset import (
    RoutingDataset,
    RoutingRecord,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

PMCID = "PMC9999999"


def _make_finding(claim: str = "CD31 -> Positive") -> Finding:
    return Finding(
        category="IHC",
        claim=claim,
        evidence=[f"S1|{PMCID}|1"],
        confidence="high",
        verbatim_support="CD31 was positive.",
    )


def _make_summary(chunk_id: str = "C1") -> AuditableSummary:
    return AuditableSummary(
        chunk_id=chunk_id,
        findings=[_make_finding()],
        summary_text="Test summary.",
        audit_metadata=AuditMetadata(
            sentences_analyzed=1,
            sentences_cited=["1"],
            pmcids_referenced=[PMCID],
            uncited_sentences=[],
        ),
    )


def _make_record(**overrides) -> RoutingRecord:
    defaults = dict(
        pmcid=PMCID,
        chunk_id="C1",
        chunk_text="[S1|PMC9999999|1] CD31 was positive.",
        n_voters=2,
        n_eligible=2,
        gate_origin="agreement_gate",
        decision="keep",
        reason_codes=[],
        escalated=False,
        used_agreement_gate=True,
        deferral_score=0.85,
        agreement_scorer="SemanticAgreementScorer",
        output=_make_summary().model_dump(),
        best_eligible_output=_make_summary().model_dump(),
        best_eligible_exists=True,
        best_eligible_voter_index=0,
    )
    defaults.update(overrides)
    return RoutingRecord(**defaults)


# ── RoutingRecord serialization ────────────────────────────────────────────────

class TestRoutingRecord:
    def test_round_trip_json(self):
        record = _make_record()
        json_str = record.model_dump_json()
        restored = RoutingRecord.model_validate_json(json_str)
        assert restored.pmcid == record.pmcid
        assert restored.deferral_score == record.deferral_score
        assert restored.decision == record.decision

    def test_null_optional_fields(self):
        record = _make_record(
            deferral_score=None,
            best_eligible_output=None,
            best_eligible_exists=False,
            best_eligible_voter_index=None,
        )
        restored = RoutingRecord.model_validate_json(record.model_dump_json())
        assert restored.deferral_score is None
        assert restored.best_eligible_output is None
        assert restored.best_eligible_voter_index is None

    def test_label_fields_default_none(self):
        record = _make_record()
        assert record.keep_ok is None
        assert record.strong_judge_score is None
        assert record.strong_judge_preferred is None

    def test_label_fields_roundtrip(self):
        record = _make_record(keep_ok=True, strong_judge_score=0.9, strong_judge_preferred="output")
        restored = RoutingRecord.model_validate_json(record.model_dump_json())
        assert restored.keep_ok is True
        assert restored.strong_judge_score == pytest.approx(0.9)
        assert restored.strong_judge_preferred == "output"


# ── RoutingDataset I/O ─────────────────────────────────────────────────────────

class TestRoutingDataset:
    def test_save_and_load(self, tmp_path):
        records = [_make_record(chunk_id=f"C{i}") for i in range(3)]
        path = tmp_path / "records.jsonl"
        RoutingDataset.save(records, path)

        loaded = RoutingDataset.load(path)
        assert len(loaded) == 3
        assert [r.chunk_id for r in loaded] == ["C0", "C1", "C2"]

    def test_iter_records(self, tmp_path):
        records = [_make_record(chunk_id=f"C{i}") for i in range(5)]
        path = tmp_path / "records.jsonl"
        RoutingDataset.save(records, path)

        chunk_ids = [r.chunk_id for r in RoutingDataset.iter_records(path)]
        assert chunk_ids == [f"C{i}" for i in range(5)]

    def test_append_instance(self, tmp_path):
        path = tmp_path / "records.jsonl"
        ds = RoutingDataset(path)
        ds.append(_make_record(chunk_id="C1"))
        ds.append(_make_record(chunk_id="C2"))

        loaded = RoutingDataset.load(path)
        assert len(loaded) == 2
        assert loaded[1].chunk_id == "C2"

    def test_append_crash_safe_one_json_per_line(self, tmp_path):
        """Each record is exactly one line — partial writes don't corrupt others."""
        path = tmp_path / "records.jsonl"
        ds = RoutingDataset(path)
        ds.append(_make_record(chunk_id="C1"))
        ds.append(_make_record(chunk_id="C2"))

        lines = [line for line in path.read_text().splitlines() if line.strip()]
        assert len(lines) == 2
        # Each line is valid JSON
        for line in lines:
            obj = json.loads(line)
            assert "pmcid" in obj

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "records.jsonl"
        RoutingDataset.save([_make_record()], path)
        assert path.exists()


# ── MapStage routing collector integration ─────────────────────────────────────

class TestMapStageRoutingCollector:
    """Verify that _cascade() writes RoutingRecord to collector on the router path."""

    def _make_map_stage(self, routing_collector=None, agreement_decision=ChunkDecision.KEEP):
        from nlp_histo.pipeline.stages.knowledge_extraction.stages.map_stage import MapStage
        from nlp_histo.pipeline.stages.knowledge_extraction.routing.router import MapOutputRouter

        mock_voter_llm = MagicMock()
        mock_escalation_llm = MagicMock()

        # The MapStage's chunk_id round-trip check (added 2026-05-15) raises
        # ValueError when a voter returns a summary whose chunk_id doesn't
        # match the expected one. Return a fresh summary per invocation that
        # echoes whatever chunk_id the caller passed in `inp`.
        def _voter_side_effect(inp, **_kwargs):
            return _make_summary(inp.get("chunk_id", "C1"))

        mock_voter_chain = MagicMock()
        mock_voter_chain.invoke.side_effect = _voter_side_effect
        mock_escalation_chain = MagicMock()
        mock_escalation_chain.invoke.side_effect = _voter_side_effect

        mock_checker = MagicMock(spec=AgreementChecker)
        mock_checker.compute.return_value = ScoreBundle(
            confidence=0.9,
            embedding_agreement=0.9,
            decision=agreement_decision,
        )
        mock_checker.best.side_effect = lambda outputs, bundle=None: outputs[0]
        mock_checker._scorer = MagicMock()
        mock_checker._scorer.__class__.__name__ = "MockScorer"

        router = MapOutputRouter(agreement_checker=mock_checker)

        with patch("nlp_histo.pipeline.stages.knowledge_extraction.stages.map_stage.build_map_chain") as mock_build:
            # 2 L1 voters + 2 L2 voters + 1 L3 escalation = 5 build_map_chain calls
            mock_build.side_effect = [
                mock_voter_chain,
                mock_voter_chain,
                mock_voter_chain,
                mock_voter_chain,
                mock_escalation_chain,
            ]
            stage = MapStage(
                voter_llms=[mock_voter_llm, mock_voter_llm],         # L1 (2 voters)
                level2_voter_llms=[mock_voter_llm, mock_voter_llm],  # L2 (2 voters)
                escalation_llm=mock_escalation_llm,                  # L3
                router=router,
                routing_collector=routing_collector,
            )

        # Inject the mocked checker and chains
        stage._agreement = mock_checker
        stage._voter_chains = [mock_voter_chain, mock_voter_chain]
        stage._level2_voter_chains = [mock_voter_chain, mock_voter_chain]
        stage._escalation_chain = mock_escalation_chain

        # _cascade() reads from _last_escalation_counts, which is normally
        # initialized in process().  Seed it here so we can call _cascade()
        # directly without going through the full pipeline.
        stage._last_escalation_counts = {
            "total": 0, "l1_kept": 0, "l2_escalated": 0, "l2_kept": 0,
            "l3_escalated": 0, "l3_kept": 0, "finalized": 0, "dropped": 0,
        }

        return stage

    def test_collector_none_no_side_effects(self, tmp_path):
        """No collector → _cascade() behaves identically to before."""
        stage = self._make_map_stage(routing_collector=None)
        chunk = [{"pmcid": PMCID, "text_element_id": 1, "sentence": "CD31 was positive."}]
        result = stage._cascade(chunk, PMCID, "C1")
        assert result is not None

    def test_keep_record_written(self, tmp_path):
        path = tmp_path / "out.jsonl"
        ds = RoutingDataset(path)
        stage = self._make_map_stage(routing_collector=ds, agreement_decision=ChunkDecision.KEEP)

        chunk = [{"pmcid": PMCID, "text_element_id": 1, "sentence": "CD31 was positive."}]
        stage._cascade(chunk, PMCID, "C1")

        records = RoutingDataset.load(path)
        assert len(records) == 1
        rec = records[0]
        assert rec.pmcid == PMCID
        assert rec.chunk_id == "C1"
        assert rec.decision == "keep"
        assert rec.escalated is False
        assert rec.best_eligible_exists is True
        # KEEP: output == best_eligible_output
        assert rec.output == rec.best_eligible_output

    def test_escalate_record_written(self, tmp_path):
        path = tmp_path / "out.jsonl"
        ds = RoutingDataset(path)
        stage = self._make_map_stage(
            routing_collector=ds, agreement_decision=ChunkDecision.ESCALATE
        )

        chunk = [{"pmcid": PMCID, "text_element_id": 1, "sentence": "CD31 was positive."}]
        stage._cascade(chunk, PMCID, "C1")

        records = RoutingDataset.load(path)
        assert len(records) == 1
        rec = records[0]
        assert rec.decision == "escalate"
        assert rec.escalated is True

    def test_multiple_chunks_multiple_records(self, tmp_path):
        path = tmp_path / "out.jsonl"
        ds = RoutingDataset(path)
        stage = self._make_map_stage(routing_collector=ds)

        chunk = [{"pmcid": PMCID, "text_element_id": 1, "sentence": "CD31 was positive."}]
        stage._cascade(chunk, PMCID, "C1")
        stage._cascade(chunk, PMCID, "C2")

        records = RoutingDataset.load(path)
        assert len(records) == 2
        assert records[0].chunk_id == "C1"
        assert records[1].chunk_id == "C2"
