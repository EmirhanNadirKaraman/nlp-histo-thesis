"""Routing dataset: per-chunk routing decision export for MAP-stage evaluation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RoutingRecord(BaseModel):
    # Identity
    pmcid: str
    chunk_id: str
    chunk_text: str            # formatted [S1|..] source — for inspection + judge prompt

    # Routing summary
    n_voters: int              # total voters before routing
    n_eligible: int            # voters that passed all validation gates
    gate_origin: str           # "schema_gate" | "grounding_gate" | "agreement_gate"
    decision: str              # "keep" | "escalate" | "reject"
    reason_codes: list[str]
    escalated: bool            # True when escalation model was called
    used_agreement_gate: bool  # True when agreement gate produced the final decision
    deferral_score: float | None  # max(avg_sim); None when agreement gate not reached
    agreement_scorer: str      # class name of the scorer used

    # Outputs
    # output: what _cascade() actually returned to the caller
    #   KEEP          → best eligible voter output (cheap path)
    #   ESCALATE/REJECT → escalation model output (expensive path)
    output: dict

    # best_eligible_output: best eligible voter output, always when one existed
    #   KEEP          → identical to output (both are the selected eligible voter)
    #   ESCALATE/REJECT → best eligible voter if any existed; None if none usable
    best_eligible_output: dict | None
    best_eligible_exists: bool
    best_eligible_voter_index: int | None   # original voter index in the voters list

    # Post-hoc label fields (filled by human reviewer or strong judge)
    keep_ok: bool | None = None
    strong_judge_score: float | None = None
    strong_judge_preferred: str | None = None  # "output" | "voter" | "neither"


class RoutingDataset:
    """
    Per-chunk routing decision collector with JSONL persistence.

    Instantiate with a path and pass to ``MapStage`` as ``routing_collector``.
    The file is opened in append mode for each record — crash-safe for long runs.

    Example
    -------
    >>> ds = RoutingDataset("out/routing_records.jsonl")
    >>> runner = MapStage(..., routing_collector=ds)
    >>> records = list(RoutingDataset.iter_records("out/routing_records.jsonl"))
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: RoutingRecord) -> None:
        """Append a single record (one JSON line).  Safe to call mid-run."""
        with self._path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

    # Static I/O (mirrors AgreementDataset API)

    @staticmethod
    def save(records: list[RoutingRecord], path: str | Path) -> None:
        """Write records to a JSONL file (overwrites any existing file)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(record.model_dump_json() + "\n")
        logger.info("Saved %d records to %s", len(records), path)

    @staticmethod
    def load(path: str | Path) -> list[RoutingRecord]:
        """Load all records from a JSONL file into memory."""
        path = Path(path)
        records: list[RoutingRecord] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(RoutingRecord.model_validate_json(line))
        logger.info("Loaded %d records from %s", len(records), path)
        return records

    @staticmethod
    def iter_records(path: str | Path) -> Iterator[RoutingRecord]:
        """Iterate over records without loading all into memory."""
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield RoutingRecord.model_validate_json(line)
