"""
TraceCollector — mutable accumulator for one pipeline run.

Created at the start of KnowledgeExtractionRunner.process() and flushed at the end.
Each stage calls the appropriate record_*() method; all calls are no-ops when
the collector is None (the default when tracing is disabled).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .models import (
    ChunkTrace,
    ChunkingTrace,
    ExportTrace,
    GroundingFilterTrace,
    IngestionTrace,
    MapStageTrace,
    RunTrace,
)


class TraceCollector:
    """
    Accumulates structured trace data for one paper run.

    Usage
    -----
    collector = TraceCollector(run_id, pmcid, config_snapshot)
    # ... pipeline runs, stages call collector.record_*() ...
    run_trace = collector.finalize("success")
    # caller writes run_trace + chunk traces to JSONL
    """

    def __init__(
        self,
        run_id: str,
        pmcid: str,
        config_snapshot: dict[str, Any],
    ) -> None:
        self._run_id = run_id
        self._pmcid = pmcid
        self._config = config_snapshot
        self._started_at = datetime.now(tz=timezone.utc)
        self._warnings: list[str] = []
        self._chunk_traces: list[ChunkTrace] = []
        # Stage slots — filled in as the pipeline progresses
        self._ingestion: IngestionTrace | None = None
        self._chunking: ChunkingTrace | None = None
        self._map_stage: MapStageTrace | None = None
        self._grounding_map: GroundingFilterTrace | None = None
        self._export: ExportTrace = ExportTrace()
        self._ended_at: datetime | None = None
        self._status: str = "running"
        self._error: str | None = None

    # ── Read-only properties ───────────────────────────────────────────────────

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def pmcid(self) -> str:
        return self._pmcid

    @property
    def chunk_traces(self) -> list[ChunkTrace]:
        return list(self._chunk_traces)

    # ── Stage recording ────────────────────────────────────────────────────────

    def warn(self, msg: str) -> None:
        self._warnings.append(msg)

    def record_ingestion(self, sentence_count: int, te_count: int) -> None:
        self._ingestion = IngestionTrace(
            sentence_count=sentence_count,
            te_count=te_count,
        )

    def record_chunking(self, total_chunks: int, chunk_size: int, chunk_overlap: int = 0) -> None:
        self._chunking = ChunkingTrace(
            total_chunks=total_chunks,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def add_chunk_trace(self, trace: ChunkTrace) -> None:
        self._chunk_traces.append(trace)

    def record_map_stage(
        self,
        total_chunks: int,
        cache_hits: int,
        cache_misses: int,
        escalations: int,
        l2_escalations: int,
        keeps: int,
        rejects: int,
        total_findings_out: int,
    ) -> None:
        self._map_stage = MapStageTrace(
            total_chunks=total_chunks,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            escalations=escalations,
            l2_escalations=l2_escalations,
            keeps=keeps,
            rejects=rejects,
            total_findings_out=total_findings_out,
        )

    def record_grounding(self, stage: str, items_before: int, items_after: int) -> None:
        """
        Parameters
        ----------
        stage:
            "map_findings" (the only grounding pass on the live path).
        """
        self._grounding_map = GroundingFilterTrace(
            stage=stage,
            items_before=items_before,
            items_after=items_after,
            dropped=items_before - items_after,
        )

    def add_artifact(
        self, path: str, artifact_type: str, count: int | None = None
    ) -> None:
        self._export.artifacts.append(
            {"path": path, "type": artifact_type, "count": count}
        )

    # ── Finalization ───────────────────────────────────────────────────────────

    def finalize(self, status: str, error: str | None = None) -> RunTrace:
        """
        Seal the run trace.  May be called multiple times (idempotent after first).
        Returns a fully-populated RunTrace ready for serialization.
        """
        if self._ended_at is None:
            self._ended_at = datetime.now(tz=timezone.utc)
        self._status = status
        self._error = error
        duration = (self._ended_at - self._started_at).total_seconds()
        return RunTrace(
            run_id=self._run_id,
            pmcid=self._pmcid,
            started_at=self._started_at.isoformat(),
            ended_at=self._ended_at.isoformat(),
            duration_s=round(duration, 3),
            status=self._status,
            error=self._error,
            warnings=list(self._warnings),
            config_snapshot=self._config,
            ingestion=self._ingestion,
            chunking=self._chunking,
            map_stage=self._map_stage,
            grounding_map=self._grounding_map,
            export=self._export if self._export.artifacts else None,
        )
