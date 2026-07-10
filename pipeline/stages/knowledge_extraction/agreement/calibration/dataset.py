"""Agreement dataset: collection, storage, and loading of labeled examples."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel

from pipeline.stages.knowledge_extraction.interfaces.scoring import ChunkDecision
from pipeline.stages.knowledge_extraction.models import AuditableSummary

logger = logging.getLogger(__name__)

# (chunk_id, pmcid, voter_outputs, source_text)
ChunkTuple = tuple[str, str, list[AuditableSummary], str]


class DatasetRecord(BaseModel):
    """A single labeled example for threshold optimization."""

    chunk_id: str
    pmcid: str
    embedding_score: float
    ner_score: float
    gold_label: ChunkDecision
    rationale: str
    voter_outputs: list[dict]   # serialized AuditableSummary.model_dump()
    source_text: str


class DatasetCollector:
    """
    Assembles a labeled dataset from voter outputs and gold labels.

    For each chunk, runs the hybrid scorer to extract (emb, ner) features,
    then runs the gold labeler to assign a three-class label.

    Parameters
    ----------
    hybrid_scorer:
        HybridScorer instance for feature computation.
    gold_labeler:
        GoldLabeler instance for gold label generation.
    """

    def __init__(self, hybrid_scorer: object, gold_labeler: object) -> None:
        self._scorer = hybrid_scorer
        self._labeler = gold_labeler

    def collect(self, chunks: list[ChunkTuple]) -> list[DatasetRecord]:
        """
        Build a list of DatasetRecord from a batch of chunks.

        Parameters
        ----------
        chunks:
            List of (chunk_id, pmcid, voter_outputs, source_text) tuples.
        """
        records: list[DatasetRecord] = []
        for chunk_id, pmcid, voter_outputs, source_text in chunks:
            try:
                bundle = self._scorer.compute(voter_outputs, source_text)
                gold = self._labeler.label(voter_outputs, source_text)
                records.append(DatasetRecord(
                    chunk_id=chunk_id,
                    pmcid=pmcid,
                    embedding_score=bundle.embedding_agreement or 0.0,
                    ner_score=bundle.entity_overlap or 0.0,
                    gold_label=gold.decision,
                    rationale=gold.rationale,
                    voter_outputs=[o.model_dump() for o in voter_outputs],
                    source_text=source_text,
                ))
                logger.debug(
                    "Collected %s: emb=%.3f ner=%.3f label=%s",
                    chunk_id,
                    bundle.embedding_agreement or 0.0,
                    bundle.entity_overlap or 0.0,
                    gold.decision.value,
                )
            except Exception:
                logger.exception("Failed to collect record for chunk %s", chunk_id)
        return records


class AgreementDataset:
    """Serializes and deserializes DatasetRecord collections as JSONL."""

    @staticmethod
    def save(records: list[DatasetRecord], path: str | Path) -> None:
        """Write records to a JSONL file (one JSON object per line)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(record.model_dump_json() + "\n")
        logger.info("Saved %d records to %s", len(records), path)

    @staticmethod
    def load(path: str | Path) -> list[DatasetRecord]:
        """Load all records from a JSONL file into memory."""
        path = Path(path)
        records: list[DatasetRecord] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(DatasetRecord.model_validate_json(line))
        logger.info("Loaded %d records from %s", len(records), path)
        return records

    @staticmethod
    def iter_records(path: str | Path) -> Iterator[DatasetRecord]:
        """Iterate over records without loading all into memory."""
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield DatasetRecord.model_validate_json(line)
