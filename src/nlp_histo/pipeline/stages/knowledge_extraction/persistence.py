"""Filesystem persistence for knowledge_extraction-stage artifacts.

This module is purely observational: it serialises stage outputs to JSONL +
JSON without changing any extraction logic. Persistence is off by default —
it is activated by passing ``artifact_root`` to :class:`KnowledgeExtractionRunner`.

Layout::

    runs/{run_id}/
        manifest.json
        map/{pmcid}/findings.jsonl
        map/{pmcid}/chunks.jsonl
        map/{pmcid}/rejected_findings.jsonl
        map/{pmcid}/bad_findings.jsonl              (if collected)
        map/{pmcid}/enum_observations.jsonl         (if collected)
        normalize/{pmcid}/normal_findings.jsonl
        normalize/{pmcid}/entity_links.jsonl
        normalize/{pmcid}/dedup_trace.jsonl
        group/{pmcid}/groups.jsonl
        group/{pmcid}/non_groupable.jsonl
        canonicalize/{pmcid}/canonical_rules.jsonl
        relate/{pmcid}/relations.jsonl
        relate/{pmcid}/raw_pairs.jsonl
        relate/corpus/relations.jsonl               (when corpus relate emits)
        resolve/{pmcid}/final_rules.jsonl
        resolve/{pmcid}/score_trace.jsonl
        logs/

"""
from __future__ import annotations

import csv
import dataclasses
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# JSON coercion

def _to_jsonable(obj: Any) -> Any:
    """Coerce an arbitrary object into a JSON-safe structure.

    Handles Pydantic v2 models, dataclasses, enums, sets/frozensets, datetimes,
    Paths, and plain containers. Falls back to ``repr()`` for unknown types so
    serialisation never crashes — telemetry must never break the pipeline.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    # Pydantic v2
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_jsonable(model_dump())
        except Exception:
            pass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [_to_jsonable(v) for v in sorted(obj, key=lambda x: str(x))]
    # Best-effort fallback: object's __dict__, then repr
    if hasattr(obj, "__dict__"):
        try:
            return _to_jsonable({k: v for k, v in vars(obj).items()
                                 if not k.startswith("_")})
        except Exception:
            pass
    return repr(obj)


def _dumps(obj: Any) -> str:
    return json.dumps(_to_jsonable(obj), ensure_ascii=False, default=str)


# Low-level writers

def write_jsonl(path: Path, records: Iterable[Any]) -> int:
    """Bulk-write records as JSONL via temp-file + rename. Returns row count.

    Required output — raises on failure (caller decides whether to swallow).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent),
    )
    n = 0
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(_dumps(rec))
                fh.write("\n")
                n += 1
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return n


def append_jsonl(path: Path, record: Any) -> None:
    """Append one record; create parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(_dumps(record))
        fh.write("\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(_to_jsonable(obj), fh, ensure_ascii=False, indent=2, default=str)
            fh.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> int:
    """Best-effort CSV writer. Returns rows written; logs and returns 0 on failure."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: _csv_value(row.get(k)) for k in columns})
        return len(rows)
    except Exception as exc:
        logger.warning("CSV write failed at %s: %s", path, exc)
        return 0


def _csv_value(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return _dumps(v)


# Manifest

@dataclass
class RunManifest:
    """In-memory manifest with stable on-disk shape.

    Fields are intentionally permissive — anything optional may be left None
    or empty. The writer never blocks on missing data.
    """
    run_id: str
    artifact_root: Path
    status: str = "running"               # running | completed | failed
    timestamp_start: str = ""
    timestamp_end: str | None = None
    papers: list[str] = field(default_factory=list)
    stages_attempted: list[str] = field(default_factory=list)
    stages_completed: list[str] = field(default_factory=list)
    schema_version: str | None = None
    prompt_version: str | None = None
    cascade_signature: str | None = None
    git_commit: str | None = None
    config: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    chunk_size: int | None = None
    error: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "papers": list(self.papers),
            "stages_attempted": list(self.stages_attempted),
            "stages_completed": list(self.stages_completed),
            "schema_version": self.schema_version,
            "prompt_version": self.prompt_version,
            "cascade_signature": self.cascade_signature,
            "git_commit": self.git_commit,
            "config": self.config,
            "models": self.models,
            "thresholds": self.thresholds,
            "chunk_size": self.chunk_size,
            "error": self.error,
            "extra": self.extra,
        }


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def compute_pipeline_config_hash(
    *,
    config: Any = None,
    thresholds: Any = None,
    models: Any = None,
    schema_version: str | None = None,
    prompt_version: str | None = None,
    cascade_signature: str | None = None,
) -> str:
    """Stable 16-char hash over the run-defining configuration.

    Used in `manifest.extra["pipeline_config_hash"]` so two runs that differ
    only in `run_id` / timestamps but share the same effective config produce
    the same hash (and the converse: any threshold / model / schema / prompt
    change flips the hash).

    All inputs are coerced via `_to_jsonable` so Pydantic models, dataclasses
    and enums hash deterministically. Keys are emitted in sorted order.
    """
    import hashlib
    payload = {
        "config":            _to_jsonable(config),
        "thresholds":        _to_jsonable(thresholds),
        "models":            _to_jsonable(models),
        "schema_version":    schema_version,
        "prompt_version":    prompt_version,
        "cascade_signature": cascade_signature,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _try_git_commit() -> str | None:
    """Best-effort git HEAD lookup. Never raises."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        return None
    return None


# Top-level writer

class RunArtifactWriter:
    """Per-run filesystem writer for knowledge_extraction-stage outputs.

    All required outputs (``write_stage_jsonl`` with required=True) raise on
    failure. Optional/trace data is logged and swallowed so missing optional
    fields never crash the pipeline.

    When a run directory already contains a ``manifest.json`` (e.g. when a
    batch of papers shares a ``run_id``), the existing manifest is loaded and
    merged with the supplied one so cumulative state (papers, stages_completed)
    isn't clobbered.
    """

    def __init__(
        self,
        run_id: str,
        root_dir: Path | str = "runs",
        manifest: RunManifest | None = None,
    ) -> None:
        self.run_id = run_id
        self.run_dir: Path = Path(root_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)

        new_manifest = manifest or RunManifest(
            run_id=run_id,
            artifact_root=Path(root_dir),
            timestamp_start=_now_iso(),
            git_commit=_try_git_commit(),
        )
        existing = self._load_existing_manifest()
        if existing is not None:
            new_manifest = self._merge_manifests(existing, new_manifest)
        self._manifest: RunManifest = new_manifest
        self.write_manifest()

    def _load_existing_manifest(self) -> dict | None:
        path = self.run_dir / "manifest.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not parse existing manifest %s: %s", path, exc)
            return None

    @staticmethod
    def _merge_manifests(prior: dict, current: RunManifest) -> RunManifest:
        """Carry forward cumulative state from a prior manifest on disk.

        Papers, stages_attempted, stages_completed, timestamp_start are taken
        from the prior manifest (so a multi-paper batch run keeps growing the
        roster). Per-paper config/threshold fields fall through to the new
        manifest because they may have been refined since the previous call.
        """
        merged = RunManifest(
            run_id=current.run_id,
            artifact_root=current.artifact_root,
            status="running",
            timestamp_start=prior.get("timestamp_start") or current.timestamp_start,
            timestamp_end=None,
            papers=list(dict.fromkeys([*prior.get("papers", []), *current.papers])),
            stages_attempted=list(dict.fromkeys([
                *prior.get("stages_attempted", []), *current.stages_attempted,
            ])),
            stages_completed=list(dict.fromkeys([
                *prior.get("stages_completed", []), *current.stages_completed,
            ])),
            schema_version=current.schema_version or prior.get("schema_version"),
            prompt_version=current.prompt_version or prior.get("prompt_version"),
            cascade_signature=current.cascade_signature or prior.get("cascade_signature"),
            git_commit=current.git_commit or prior.get("git_commit"),
            config=current.config or prior.get("config", {}),
            models=current.models or prior.get("models", {}),
            thresholds=current.thresholds or prior.get("thresholds", {}),
            chunk_size=current.chunk_size if current.chunk_size is not None
                       else prior.get("chunk_size"),
            error=None,
            extra={**prior.get("extra", {}), **current.extra},
        )
        return merged

    # manifest

    @property
    def manifest(self) -> RunManifest:
        return self._manifest

    def update_manifest(self, **fields: Any) -> None:
        for key, value in fields.items():
            if hasattr(self._manifest, key):
                setattr(self._manifest, key, value)
            else:
                self._manifest.extra[key] = value
        self.write_manifest()

    def mark_stage_attempted(self, stage: str) -> None:
        if stage not in self._manifest.stages_attempted:
            self._manifest.stages_attempted.append(stage)
        self.write_manifest()

    def mark_stage_completed(self, stage: str) -> None:
        self.mark_stage_attempted(stage)
        if stage not in self._manifest.stages_completed:
            self._manifest.stages_completed.append(stage)
        self.write_manifest()

    def add_paper(self, pmcid: str) -> None:
        if pmcid and pmcid not in self._manifest.papers:
            self._manifest.papers.append(pmcid)
            self.write_manifest()

    def finalize(self, status: str, error: str | None = None) -> None:
        self._manifest.status = status
        self._manifest.timestamp_end = _now_iso()
        if error is not None:
            self._manifest.error = error
        self.write_manifest()

    def write_manifest(self) -> None:
        try:
            write_json(self.run_dir / "manifest.json", self._manifest.to_dict())
        except Exception as exc:
            logger.error("Failed to write manifest: %s", exc, exc_info=True)
            raise

    # stage path helpers

    def stage_dir(self, stage: str, pmcid: str | None = None) -> Path:
        d = self.run_dir / stage
        if pmcid:
            d = d / pmcid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def stage_path(self, stage: str, pmcid: str | None, name: str) -> Path:
        return self.stage_dir(stage, pmcid) / name

    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    # stage writers

    def write_stage_jsonl(
        self,
        stage: str,
        name: str,
        records: Iterable[Any],
        *,
        pmcid: str | None = None,
        required: bool = False,
    ) -> int:
        """Write a JSONL artifact. ``required=True`` raises on failure."""
        path = self.stage_path(stage, pmcid, name)
        try:
            n = write_jsonl(path, records)
            logger.debug("[%s/%s] wrote %d records to %s", stage, pmcid or "-", n, path)
            return n
        except Exception as exc:
            if required:
                logger.error("Required artifact failed: %s — %s", path, exc)
                raise
            logger.warning("Optional artifact failed: %s — %s", path, exc)
            return 0

    def write_stage_json(
        self,
        stage: str,
        name: str,
        obj: Any,
        *,
        pmcid: str | None = None,
        required: bool = False,
    ) -> None:
        path = self.stage_path(stage, pmcid, name)
        try:
            write_json(path, obj)
        except Exception as exc:
            if required:
                logger.error("Required artifact failed: %s — %s", path, exc)
                raise
            logger.warning("Optional artifact failed: %s — %s", path, exc)

    def write_error(self, stage: str, error: str, *, pmcid: str | None = None) -> None:
        """Append a stage-level error to the run's logs/ directory."""
        path = self.logs_dir() / "stage_errors.jsonl"
        try:
            append_jsonl(path, {
                "ts": _now_iso(),
                "stage": stage,
                "pmcid": pmcid,
                "error": error,
            })
        except Exception as exc:
            logger.warning("Failed to write stage error log: %s", exc)

    def write_stage_csv(
        self,
        stage: str,
        name: str,
        rows: list[dict],
        columns: list[str],
        *,
        pmcid: str | None = None,
    ) -> int:
        """Best-effort CSV summary; never raises."""
        path = self.stage_path(stage, pmcid, name)
        return write_csv(path, rows, columns)


# Stage persisters (shared by sync + batch runners)
#
# These take a fully-built RunArtifactWriter and the in-memory stage outputs.
# Each function is a no-op when the writer is None, so callers can pass
# ``writer = self._make_artifact_writer(...)`` unconditionally instead of
# branching on artifact_root.


def non_groupable_reason(nf: Any) -> str:
    """Human-readable reason a NormalFinding failed is_groupable()."""
    from .models import RelationTypeEnum  # noqa: PLC0415 — defer to avoid circular import
    parts = []
    if getattr(nf, "subject_entity", None) is None:
        parts.append("subject_entity=None")
    if getattr(nf, "outcome_entity", None) is None:
        parts.append("outcome_entity=None")
    if getattr(nf, "relation_type", None) is RelationTypeEnum.unclear:
        parts.append("relation_type=unclear")
    return ", ".join(parts) if parts else "unknown"


def build_rejection_summary(
    pmcid: str,
    grounding_threshold: float | None,
    map_findings_total: int,
    grounding_rejected: list,
    normal_findings: list,
    non_groupable_nfs: list,
):
    """Build a :class:`RejectionSummary` from collected rejection data for one paper.

    Shared by sync and batch runners. ``grounding_rejected`` is a list of
    ``(chunk_id, Finding)`` tuples. Returns a populated ``RejectionSummary``.
    """
    from collections import Counter  # noqa: PLC0415
    from .models import (  # noqa: PLC0415 — defer to avoid circular import
        RejectedFinding,
        RejectionSummary,
        RelationTypeEnum,
    )

    rejected_items: list = []

    for chunk_id, f in grounding_rejected:
        score = f.grounding_score
        reason = (
            f"grounding_score={score:.3f} < threshold={grounding_threshold}"
            if score is not None and grounding_threshold is not None
            else "grounding_score below threshold"
        )
        rejected_items.append(RejectedFinding(
            stage="grounding_map",
            reason=reason,
            claim=f.claim,
            category=f.category,
            chunk_id=chunk_id,
            grounding_score=score,
            subject_entity=f.subject_entity,
            outcome_entity=f.outcome_entity,
            relation_type=f.relation_type.value if f.relation_type else None,
        ))

    for nf in non_groupable_nfs:
        rejected_items.append(RejectedFinding(
            stage="group_non_groupable",
            reason=non_groupable_reason(nf),
            claim=nf.predicate_text,
            category=nf.category,
            grounding_score=nf.mean_grounding_score,
            subject_entity=nf.subject_entity,
            outcome_entity=nf.outcome_entity,
            relation_type=nf.relation_type.value if nf.relation_type else None,
        ))

    n_total = map_findings_total
    n_grounding = len(grounding_rejected)
    n_normal = len(normal_findings)
    n_non_groupable = len(non_groupable_nfs)

    cat_counts: Counter = Counter(f.category for _, f in grounding_rejected)

    return RejectionSummary(
        pmcid=pmcid,
        grounding_threshold=grounding_threshold,
        map_findings_total=n_total,
        map_grounding_rejected=n_grounding,
        normal_findings_total=n_normal,
        non_groupable_total=n_non_groupable,
        non_groupable_no_subject=sum(
            1 for nf in non_groupable_nfs if nf.subject_entity is None
        ),
        non_groupable_no_outcome=sum(
            1 for nf in non_groupable_nfs if nf.outcome_entity is None
        ),
        non_groupable_unclear_relation=sum(
            1 for nf in non_groupable_nfs
            if nf.relation_type is RelationTypeEnum.unclear
        ),
        grounding_rejection_rate=round(n_grounding / n_total, 4) if n_total else 0.0,
        non_groupable_rate=round(n_non_groupable / n_normal, 4) if n_normal else 0.0,
        grounding_rejected_by_category=dict(cat_counts),
        rejected=rejected_items,
    )


# DB persistence (shared between sync + batch runners)
# These functions encapsulate every write into the ``sum_*`` tables. Both
# ``KnowledgeExtractionRunner`` and ``BatchKnowledgeExtractionRunner`` forward to them so
# that schema/order changes happen in exactly one place. All take ``db`` as
# the first parameter (a ``DatabaseConnection`` instance, or ``None`` to no-op).


def create_pipeline_run(db, pmcid: str, run_id: str, config_snapshot: dict) -> int | None:
    """Insert a ``pipeline_runs`` row with status='running'. Returns the row id."""
    if db is None:
        return None
    try:
        from nlp_histo.database import Document, PipelineRun  # noqa: PLC0415
        with db.session_scope() as session:
            doc = session.query(Document).filter_by(pmcid=pmcid).first()
            run = PipelineRun(
                run_id=run_id,
                pmcid=pmcid,
                document_id=doc.id if doc else None,
                status="running",
                config_snapshot=config_snapshot,
                started_at=datetime.now(tz=timezone.utc),
            )
            session.add(run)
            session.flush()
            return run.id
    except Exception as exc:
        logger.warning("[%s] DB: failed to create pipeline_run: %s", pmcid, exc)
        return None


def finish_pipeline_run(
    db,
    db_id: int | None,
    status: str,
    error: str | None = None,
    narrative_summary: str | None = None,
) -> None:
    """Update a ``pipeline_runs`` row to its final status. No-op when ``db_id`` is None."""
    if db_id is None or db is None:
        return
    try:
        from nlp_histo.database import PipelineRun  # noqa: PLC0415
        with db.session_scope() as session:
            run = session.query(PipelineRun).filter_by(id=db_id).first()
            if run is not None:
                run.status = status
                run.finished_at = datetime.now(tz=timezone.utc)
                run.error = error
                if narrative_summary is not None:
                    run.narrative_summary = narrative_summary
    except Exception as exc:
        logger.warning("DB: failed to update pipeline_run id=%s: %s", db_id, exc)


def clear_normalized_run_data(db, db_id: int, pmcid: str) -> None:
    """Delete all post-MAP rows for (pipeline_run_id, pmcid) in FK-safe order."""
    if db is None:
        return
    try:
        from nlp_histo.database.models import (  # noqa: PLC0415
            SumCanonicalRule, SumFinalRule, SumFindingGroup, SumGroupMember,
            SumNormalFinding, SumNormalFindingSpan, SumRelation,
        )
        with db.engine.begin() as conn:
            t = SumFinalRule.__table__
            conn.execute(t.delete().where(
                (t.c.pipeline_run_id == db_id) & (t.c.pmcid == pmcid)))
            t = SumRelation.__table__
            conn.execute(t.delete().where(
                (t.c.pipeline_run_id == db_id) & (t.c.pmcid == pmcid)))
            t = SumCanonicalRule.__table__
            conn.execute(t.delete().where(
                (t.c.pipeline_run_id == db_id) & (t.c.pmcid == pmcid)))
            fg_table = SumFindingGroup.__table__
            subq = fg_table.select().where(
                (fg_table.c.pipeline_run_id == db_id) & (fg_table.c.pmcid == pmcid)
            ).with_only_columns(fg_table.c.id)
            t = SumGroupMember.__table__
            conn.execute(t.delete().where(t.c.finding_group_id.in_(subq)))
            t = SumFindingGroup.__table__
            conn.execute(t.delete().where(
                (t.c.pipeline_run_id == db_id) & (t.c.pmcid == pmcid)))
            nf_table = SumNormalFinding.__table__
            subq2 = nf_table.select().where(
                (nf_table.c.pipeline_run_id == db_id) & (nf_table.c.pmcid == pmcid)
            ).with_only_columns(nf_table.c.id)
            t = SumNormalFindingSpan.__table__
            conn.execute(t.delete().where(t.c.normal_finding_id.in_(subq2)))
            t = SumNormalFinding.__table__
            conn.execute(t.delete().where(
                (t.c.pipeline_run_id == db_id) & (t.c.pmcid == pmcid)))
    except Exception as exc:
        logger.warning(
            "[%s] DB: failed to clear normalized run data (run_id=%d): %s",
            pmcid, db_id, exc, exc_info=True,
        )


def replace_verbatim_from_db(db, chunk_summaries: list) -> None:
    """Replace each Finding's ``verbatim_support`` with the cited TextElement text.

    Evidence strings have format ``S{i}|{pmcid}|{te_id}``. Collects all cited
    te_ids, batch-queries ``TextElement.text_content`` in one round-trip, and
    overwrites ``verbatim_support`` in place. Falls back to the LLM-supplied
    value when the DB is unavailable, the te_id has no match, or any error
    occurs. No-ops when ``db`` is None.
    """
    if db is None:
        return
    te_ids: set[int] = set()
    for cs in chunk_summaries:
        for f in cs.findings:
            for ev in f.evidence:
                parts = ev.split("|")
                if len(parts) == 3:
                    try:
                        te_ids.add(int(parts[2]))
                    except ValueError:
                        pass
    if not te_ids:
        return
    try:
        from nlp_histo.database import TextElement  # noqa: PLC0415
        te_map: dict[int, str] = {}
        with db.session_scope() as session:
            rows = (
                session.query(TextElement.id, TextElement.text_content)
                .filter(TextElement.id.in_(te_ids))
                .all()
            )
            te_map = {row.id: row.text_content for row in rows}
    except Exception as exc:
        logger.warning("[verbatim] DB lookup failed — keeping LLM verbatim: %s", exc)
        return
    replaced = 0
    for cs in chunk_summaries:
        for f in cs.findings:
            if not f.evidence:
                continue
            parts = f.evidence[0].split("|")
            if len(parts) == 3:
                try:
                    te_id = int(parts[2])
                    text = te_map.get(te_id)
                    if text:
                        f.verbatim_support = text
                        replaced += 1
                except ValueError:
                    pass
    logger.info("[verbatim] replaced %d/%d finding verbatims from DB",
                replaced, sum(len(cs.findings) for cs in chunk_summaries))


def persist_map_findings(db, db_id: int | None, pmcid: str, chunk_summaries: list) -> None:
    """Persist scored MAP findings (post-grounding) to ``sum_map_findings``.

    Unlike the sibling ``persist_*`` helpers this re-raises on failure rather
    than swallowing — a silent 0-row insert here corrupts every downstream
    experiment that reads ``sum_map_findings`` while the run still reports
    ``success`` (B-055). Both runners wrap the call in a finalize-level handler
    that marks the ``pipeline_run`` failed, so re-raising surfaces the real DB
    error instead of hiding it behind a WARNING that goes to unrecorded stdout.
    """
    if db_id is None or db is None:
        return
    total_findings = sum(len(cs.findings) for cs in chunk_summaries)
    logger.info(
        "[%s] persist_map_findings: entering with %d chunks (%d findings) run_id=%s",
        pmcid, len(chunk_summaries), total_findings, db_id,
    )
    try:
        from nlp_histo.database.models import SumMapFinding  # noqa: PLC0415
        from .models import FindingScope  # noqa: PLC0415
        rows = []
        for cs in chunk_summaries:
            for pos, f in enumerate(cs.findings):
                scope = f.scope or FindingScope()
                rows.append({
                    "pipeline_run_id":       db_id,
                    "pmcid":                 pmcid,
                    "chunk_id":              cs.chunk_id,
                    "position_in_chunk":     pos,
                    "category":              f.category,
                    "claim":                 f.claim,
                    "confidence":            f.confidence,
                    "verbatim_support":      f.verbatim_support,
                    "subject_entity":        f.subject_entity,
                    "outcome_entity":        f.outcome_entity,
                    "relation_type":         f.relation_type.value,
                    "direction":             f.direction.value if f.direction else None,
                    "raw_relation_type":     f.raw_relation_type,
                    "raw_direction":         f.raw_direction,
                    "raw_category":          f.raw_category,
                    "grounding_score":       f.grounding_score,
                    "evidence_refs":         list(f.evidence) if f.evidence else [],
                    "scope_disease_subtype":   scope.disease_subtype,
                    "scope_cohort_n":          scope.cohort_n,
                    "scope_assay_method":      scope.assay_method,
                    "scope_biomarker_cutoff":  scope.biomarker_cutoff,
                    "scope_tissue_site":       scope.tissue_site,
                    "scope_treatment_context": scope.treatment_context,
                    "scope_endpoint":          scope.endpoint,
                    "scope_study_design":      scope.study_design,
                })
        if not rows:
            return
        with db.engine.begin() as conn:
            conn.execute(
                SumMapFinding.__table__.delete().where(
                    SumMapFinding.__table__.c.pipeline_run_id == db_id
                )
            )
            conn.execute(SumMapFinding.__table__.insert(), rows)
        logger.info(
            "[%s] DB: persisted %d map findings (run_id=%d)",
            pmcid, len(rows), db_id,
        )
    except Exception as exc:
        # B-055: do not swallow. A failed bulk insert here means zero map
        # findings persist while the run otherwise completes — silent
        # corruption. Re-raise so the finalize-level handler fails the run.
        logger.error(
            "[%s] DB: failed to persist map findings (run_id=%s) — re-raising: %s",
            pmcid, db_id, exc, exc_info=True,
        )
        raise


def persist_normal_findings(
    db, db_id: int | None, pmcid: str, normal_findings: list,
) -> dict[str, int]:
    """Persist NormalFindings + evidence spans. Returns ``{normal_id: db_row_id}``.

    Clears prior post-MAP rows for ``(db_id, pmcid)`` before insert so reruns
    don't trip FK / unique constraints.
    """
    if db_id is None or db is None:
        return {}
    clear_normalized_run_data(db, db_id, pmcid)
    nf_id_map: dict[str, int] = {}
    try:
        from nlp_histo.database.models import SumNormalFinding, SumNormalFindingSpan  # noqa: PLC0415
        from .models import FindingScope  # noqa: PLC0415
        # Dedupe by normal_id (last-wins, matching the in-memory ``nf_by_id`` dict):
        # the pipeline treats normal_id as unique and the DB enforces
        # UNIQUE(pipeline_run_id, normal_id). Without this, a duplicate normal_id
        # from NORMALIZE aborts the whole insert → 0 rows persisted (B-076).
        deduped_nfs = list({nf.normal_id: nf for nf in normal_findings}.values())
        with db.session_scope() as session:
            for nf in deduped_nfs:
                scope = nf.scope or FindingScope()
                row = SumNormalFinding(
                    pipeline_run_id     = db_id,
                    pmcid               = pmcid,
                    normal_id           = nf.normal_id,
                    subject_entity      = nf.subject_entity,
                    outcome_entity      = nf.outcome_entity,
                    relation_type       = nf.relation_type.value,
                    direction           = nf.direction.value if nf.direction else None,
                    category            = nf.category,
                    predicate_text      = nf.predicate_text,
                    mean_grounding_score= nf.mean_grounding_score,
                    pmcids              = list(nf.pmcids) if nf.pmcids else [],
                    scope_disease_subtype   = scope.disease_subtype,
                    scope_cohort_n          = scope.cohort_n,
                    scope_assay_method      = scope.assay_method,
                    scope_biomarker_cutoff  = scope.biomarker_cutoff,
                    scope_tissue_site       = scope.tissue_site,
                    scope_treatment_context = scope.treatment_context,
                    scope_endpoint          = scope.endpoint,
                    scope_study_design      = scope.study_design,
                )
                session.add(row)
                session.flush()
                nf_id_map[nf.normal_id] = row.id
                for span in (nf.evidence or []):
                    session.add(SumNormalFindingSpan(
                        normal_finding_id = row.id,
                        sentence_id       = span.sentence_id,
                        pmcid             = span.pmcid,
                        text_element_id   = span.text_element_id or None,
                        verbatim          = span.verbatim,
                    ))
        logger.info(
            "[%s] DB: persisted %d normal findings + spans (run_id=%d)",
            pmcid, len(nf_id_map), db_id,
        )
    except Exception as exc:
        logger.warning("[%s] DB: failed to persist normal findings: %s", pmcid, exc)
        return {}
    return nf_id_map


def persist_finding_groups(
    db, db_id: int | None, pmcid: str, finding_groups: list,
    nf_db_id_map: dict[str, int],
) -> dict[str, int]:
    """Persist FindingGroups + member links. Returns ``{group_id: db_row_id}``."""
    if db_id is None or db is None:
        return {}
    group_id_map: dict[str, int] = {}
    try:
        from nlp_histo.database.models import SumFindingGroup, SumGroupMember  # noqa: PLC0415
        with db.session_scope() as session:
            for fg in finding_groups:
                row = SumFindingGroup(
                    pipeline_run_id     = db_id,
                    pmcid               = pmcid,
                    group_id            = fg.group_id,
                    subject_entity      = fg.subject_entity,
                    outcome_entity      = fg.outcome_entity,
                    relation_type       = fg.relation_type.value,
                    category            = fg.category,
                    scope_heterogeneity = fg.scope_heterogeneity,
                    direction_counts    = dict(fg.direction_counts),
                )
                session.add(row)
                session.flush()
                group_id_map[fg.group_id] = row.id
                missing = 0
                seen_members: set[str] = set()
                for normal_id in fg.member_ids:
                    # uq_sum_group_member is (finding_group_id, normal_id); a group
                    # listing the same normal_id twice would abort the insert (B-076).
                    if normal_id in seen_members:
                        continue
                    seen_members.add(normal_id)
                    nf_db_id = nf_db_id_map.get(normal_id)
                    if nf_db_id is None:
                        missing += 1
                    session.add(SumGroupMember(
                        finding_group_id  = row.id,
                        normal_finding_id = nf_db_id,
                        normal_id         = normal_id,
                    ))
                if missing:
                    logger.debug(
                        "[%s] group %s: %d member(s) have no NF DB id (Phase 2 incomplete)",
                        pmcid, fg.group_id, missing,
                    )
        logger.info(
            "[%s] DB: persisted %d finding groups (run_id=%d)",
            pmcid, len(group_id_map), db_id,
        )
    except Exception as exc:
        logger.warning("[%s] DB: failed to persist finding groups: %s", pmcid, exc)
        return {}
    return group_id_map


def persist_canonical_rules(
    db, db_id: int | None, pmcid: str, canonical_rules: list,
    fg_db_id_map: dict[str, int],
) -> dict[str, int]:
    """Persist CanonicalRules. Returns ``{canonical_id: db_row_id}``."""
    if db_id is None or db is None:
        return {}
    cr_id_map: dict[str, int] = {}
    try:
        from nlp_histo.database.models import SumCanonicalRule  # noqa: PLC0415
        rows = []
        for cr in canonical_rules:
            fg_db_id = fg_db_id_map.get(cr.group_id)
            row = SumCanonicalRule(
                pipeline_run_id      = db_id,
                pmcid                = pmcid,
                canonical_id         = cr.canonical_id,
                finding_group_id     = fg_db_id,
                group_id             = cr.group_id,
                subject_entity       = cr.subject_entity,
                outcome_entity       = cr.outcome_entity,
                relation_type        = cr.relation_type.value,
                direction            = cr.direction.value if cr.direction else None,
                predicate_text       = cr.predicate_text,
                is_conflicted        = cr.is_conflicted,
                study_coverage       = cr.study_coverage,
                category             = cr.category,
                pmcids               = list(cr.supporting_pmcids) if cr.supporting_pmcids else [],
                member_normal_ids    = list(cr.member_normal_ids) if cr.member_normal_ids else [],
                mean_grounding_score = cr.mean_grounding_score,
                finding_count        = cr.finding_count,
                subject_cui          = cr.subject_cui,
                outcome_cui          = cr.outcome_cui,
            )
            rows.append(row)
        with db.session_scope() as session:
            for row in rows:
                session.add(row)
            session.flush()
            for row in rows:
                cr_id_map[row.canonical_id] = row.id
        logger.info(
            "[%s] DB: persisted %d canonical rules (run_id=%d)",
            pmcid, len(cr_id_map), db_id,
        )
    except Exception as exc:
        logger.warning("[%s] DB: failed to persist canonical rules: %s", pmcid, exc)
        return {}
    return cr_id_map


def persist_relations(
    db, db_id: int | None, pmcid: str, relations: list,
    cr_db_id_map: dict[str, int],
) -> None:
    """Persist Relations to ``sum_relations``."""
    if db_id is None or db is None:
        return
    try:
        from nlp_histo.database.models import SumRelation  # noqa: PLC0415
        rows = []
        for rel in relations:
            rows.append(SumRelation(
                pipeline_run_id     = db_id,
                pmcid               = pmcid,
                rule_id_a           = rel.rule_id_a,
                rule_id_b           = rel.rule_id_b,
                canonical_rule_a_id = cr_db_id_map.get(rel.rule_id_a),
                canonical_rule_b_id = cr_db_id_map.get(rel.rule_id_b),
                relation_type       = rel.relation_type.value,
                nli_score_a_to_b    = rel.nli_score_a_to_b,
                nli_score_b_to_a    = rel.nli_score_b_to_a,
            ))
        if rows:
            with db.session_scope() as session:
                session.bulk_save_objects(rows)
            logger.info(
                "[%s] DB: persisted %d relations (run_id=%d)",
                pmcid, len(rows), db_id,
            )
    except Exception as exc:
        logger.warning("[%s] DB: failed to persist relations: %s", pmcid, exc)


def persist_final_rules(
    db, db_id: int | None, pmcid: str, final_rules: list,
    cr_db_id_map: dict[str, int],
) -> None:
    """Persist FinalRules (scored canonical rules) to ``sum_final_rules``."""
    if db_id is None or db is None:
        return
    try:
        from nlp_histo.database.models import SumFinalRule  # noqa: PLC0415
        rows = []
        for fr in final_rules:
            rows.append(SumFinalRule(
                pipeline_run_id      = db_id,
                pmcid                = pmcid,
                final_id             = fr.final_id,
                canonical_rule_id    = cr_db_id_map.get(fr.canonical_id),
                canonical_id         = fr.canonical_id,
                subject_entity       = fr.subject_entity,
                outcome_entity       = fr.outcome_entity,
                relation_type        = fr.relation_type.value,
                direction            = fr.direction.value if fr.direction else None,
                predicate_text       = fr.predicate_text,
                category             = fr.category,
                final_score          = fr.final_score,
                support_count        = fr.support_count,
                contradict_count     = fr.contradict_count,
                scope_qualify_count  = fr.scope_qualify_count,
                is_contradicted      = fr.is_contradicted,
                contradicted_by      = list(fr.contradicted_by) if fr.contradicted_by else [],
                score_mode           = fr.score_mode,
            ))
        if rows:
            with db.session_scope() as session:
                session.bulk_save_objects(rows)
            logger.info(
                "[%s] DB: persisted %d final rules (run_id=%d)",
                pmcid, len(rows), db_id,
            )
    except Exception as exc:
        logger.warning("[%s] DB: failed to persist final rules: %s", pmcid, exc)


def persist_rejection_summary(db, db_id: int | None, rejection_summary) -> None:
    """Persist a ``RejectionSummary`` + detail rows."""
    if db_id is None or db is None:
        return
    try:
        from nlp_histo.database.models import SumRejectionSummary, SumRejectedFinding  # noqa: PLC0415
        with db.session_scope() as session:
            summary_row = SumRejectionSummary(
                pipeline_run_id                = db_id,
                pmcid                          = rejection_summary.pmcid,
                grounding_threshold            = rejection_summary.grounding_threshold,
                map_findings_total             = rejection_summary.map_findings_total,
                map_grounding_rejected         = rejection_summary.map_grounding_rejected,
                normal_findings_total          = rejection_summary.normal_findings_total,
                non_groupable_total            = rejection_summary.non_groupable_total,
                non_groupable_no_subject       = rejection_summary.non_groupable_no_subject,
                non_groupable_no_outcome       = rejection_summary.non_groupable_no_outcome,
                non_groupable_unclear_relation = rejection_summary.non_groupable_unclear_relation,
                grounding_rejection_rate       = rejection_summary.grounding_rejection_rate,
                non_groupable_rate             = rejection_summary.non_groupable_rate,
                grounding_rejected_by_category = rejection_summary.grounding_rejected_by_category,
            )
            session.add(summary_row)
            session.flush()
            for item in rejection_summary.rejected:
                session.add(SumRejectedFinding(
                    pipeline_run_id      = db_id,
                    rejection_summary_id = summary_row.id,
                    pmcid                = rejection_summary.pmcid,
                    stage                = item.stage,
                    reason               = item.reason,
                    claim                = item.claim,
                    category             = item.category,
                    chunk_id             = item.chunk_id,
                    grounding_score      = item.grounding_score,
                    subject_entity       = item.subject_entity,
                    outcome_entity       = item.outcome_entity,
                    relation_type        = item.relation_type,
                ))
        logger.info(
            "[%s] DB: persisted rejection summary (%d grounding, %d non-groupable, run_id=%d)",
            rejection_summary.pmcid,
            rejection_summary.map_grounding_rejected,
            rejection_summary.non_groupable_total,
            db_id,
        )
    except Exception as exc:
        logger.warning(
            "[%s] DB: failed to persist rejection summary: %s",
            rejection_summary.pmcid, exc, exc_info=True,
        )


def _copy_enum_logs(writer: "RunArtifactWriter", pmcid: str) -> None:
    """Mirror logs/{enum_observations,bad_findings}.jsonl into the per-PMCID dir.

    Missing files are skipped without error.
    """
    for name in ("enum_observations.jsonl", "bad_findings.jsonl"):
        src = writer.logs_dir() / name
        if not src.exists():
            continue
        try:
            dst = writer.stage_path("map", pmcid, name)
            dst.write_bytes(src.read_bytes())
        except Exception as exc:
            logger.debug("[%s] enum log copy failed for %s: %s", pmcid, name, exc)


def persist_map_artifacts(
    writer: "RunArtifactWriter | None",
    pmcid: str,
    chunk_summaries: list,
    grounding_rejected: list,
) -> None:
    """Write findings.jsonl + chunks.jsonl + rejected_findings.jsonl + enum logs."""
    if writer is None:
        return
    writer.add_paper(pmcid)
    writer.mark_stage_attempted("map")
    try:
        # finding_id is a PrivateAttr on Finding so it does not appear in
        # model_dump(); persist it explicitly. (chunk_id, position_in_chunk)
        # remain for human-readable coordinates.
        findings_rows: list[dict] = []
        for cs in chunk_summaries:
            for pos, f in enumerate(cs.findings):
                fd = f.model_dump()
                fd["finding_id"] = f.finding_id
                fd["pmcid"] = pmcid
                fd["chunk_id"] = cs.chunk_id
                fd["position_in_chunk"] = pos
                findings_rows.append(fd)
        writer.write_stage_jsonl(
            "map", "findings.jsonl", findings_rows, pmcid=pmcid, required=True,
        )

        chunk_rows = [
            {
                "pmcid": pmcid,
                "chunk_id": cs.chunk_id,
                "n_findings": len(cs.findings),
                "summary_text": cs.summary_text,
                "audit_metadata": cs.audit_metadata.model_dump(),
            }
            for cs in chunk_summaries
        ]
        writer.write_stage_jsonl(
            "map", "chunks.jsonl", chunk_rows, pmcid=pmcid, required=True,
        )

        rejected_rows = [
            {
                "pmcid": pmcid,
                "chunk_id": chunk_id,
                "finding_id": f.finding_id,
                "claim": f.claim,
                "category": f.category,
                "subject_entity": f.subject_entity,
                "outcome_entity": f.outcome_entity,
                "relation_type": f.relation_type.value if f.relation_type else None,
                "direction": f.direction.value if f.direction else None,
                "grounding_score": f.grounding_score,
                "evidence": list(f.evidence) if f.evidence else [],
                "verbatim_support": f.verbatim_support,
                "reason": "grounding_score_below_threshold",
            }
            for chunk_id, f in grounding_rejected
        ]
        writer.write_stage_jsonl(
            "map", "rejected_findings.jsonl", rejected_rows, pmcid=pmcid,
        )

        _copy_enum_logs(writer, pmcid)
        writer.mark_stage_completed("map")
    except Exception as exc:
        writer.write_error(stage="map", error=str(exc), pmcid=pmcid)
        raise


def persist_normalize_artifacts(
    writer: "RunArtifactWriter | None",
    pmcid: str,
    normal_findings: list,
) -> None:
    if writer is None:
        return
    writer.mark_stage_attempted("normalize")
    try:
        writer.write_stage_jsonl(
            "normalize", "normal_findings.jsonl",
            [nf.model_dump() for nf in normal_findings],
            pmcid=pmcid, required=True,
        )
        entity_links = [
            {
                "normal_id": nf.normal_id,
                "subject_entity": nf.subject_entity,
                "subject_cui":    nf.subject_cui,
                "outcome_entity": nf.outcome_entity,
                "outcome_cui":    nf.outcome_cui,
            }
            for nf in normal_findings
        ]
        writer.write_stage_jsonl(
            "normalize", "entity_links.jsonl", entity_links, pmcid=pmcid,
        )
        dedup_rows = [
            {
                "normal_id": nf.normal_id,
                "n_evidence_spans": len(nf.evidence) if nf.evidence else 0,
                "evidence_coords": [
                    {
                        "sentence_id": s.sentence_id,
                        "pmcid": s.pmcid,
                        "text_element_id": s.text_element_id,
                    }
                    for s in (nf.evidence or [])
                ],
                "source_finding_ids": list(nf.source_finding_ids or []),
            }
            for nf in normal_findings
        ]
        writer.write_stage_jsonl(
            "normalize", "dedup_trace.jsonl", dedup_rows, pmcid=pmcid,
        )
        writer.mark_stage_completed("normalize")
    except Exception as exc:
        writer.write_error(stage="normalize", error=str(exc), pmcid=pmcid)
        raise


def persist_group_artifacts(
    writer: "RunArtifactWriter | None",
    pmcid: str,
    groups: list,
    non_groupable_nfs: list,
) -> None:
    if writer is None:
        return
    writer.mark_stage_attempted("group")
    try:
        writer.write_stage_jsonl(
            "group", "groups.jsonl",
            [g.model_dump() for g in groups],
            pmcid=pmcid, required=True,
        )
        non_groupable_rows = [
            {
                "normal_id": nf.normal_id,
                "subject_entity": nf.subject_entity,
                "outcome_entity": nf.outcome_entity,
                "relation_type": nf.relation_type.value if nf.relation_type else None,
                "category": nf.category,
                "predicate_text": nf.predicate_text,
                "reason": non_groupable_reason(nf),
            }
            for nf in non_groupable_nfs
        ]
        writer.write_stage_jsonl(
            "group", "non_groupable.jsonl", non_groupable_rows, pmcid=pmcid,
        )
        writer.mark_stage_completed("group")
    except Exception as exc:
        writer.write_error(stage="group", error=str(exc), pmcid=pmcid)
        raise


def persist_canonicalize_artifacts(
    writer: "RunArtifactWriter | None",
    pmcid: str,
    canonical_rules: list,
) -> None:
    if writer is None:
        return
    writer.mark_stage_attempted("canonicalize")
    try:
        writer.write_stage_jsonl(
            "canonicalize", "canonical_rules.jsonl",
            [cr.model_dump() for cr in canonical_rules],
            pmcid=pmcid, required=True,
        )
        writer.mark_stage_completed("canonicalize")
    except Exception as exc:
        writer.write_error(stage="canonicalize", error=str(exc), pmcid=pmcid)
        raise


def persist_relate_artifacts(
    writer: "RunArtifactWriter | None",
    pmcid: str,
    relations: list,
    raw_pairs: list,
    skipped_pairs: list | None = None,
) -> None:
    if writer is None:
        return
    writer.mark_stage_attempted("relate")
    try:
        writer.write_stage_jsonl(
            "relate", "relations.jsonl",
            [r.model_dump() for r in relations],
            pmcid=pmcid, required=True,
        )
        writer.write_stage_jsonl(
            "relate", "raw_pairs.jsonl",
            [p.model_dump() for p in raw_pairs],
            pmcid=pmcid, required=True,
        )
        # Pre-NLI gate rejections. Default to [] when caller did not supply
        # them so legacy callers preserve their existing on-disk shape.
        skipped_rows = (
            [s.model_dump() for s in (skipped_pairs or [])]
        )
        writer.write_stage_jsonl(
            "relate", "skipped_pairs.jsonl", skipped_rows, pmcid=pmcid,
        )
        writer.mark_stage_completed("relate")
    except Exception as exc:
        writer.write_error(stage="relate", error=str(exc), pmcid=pmcid)
        raise


def persist_resolve_artifacts(
    writer: "RunArtifactWriter | None",
    pmcid: str,
    final_rules: list,
    relations: list,
) -> None:
    if writer is None:
        return
    writer.mark_stage_attempted("resolve")
    try:
        writer.write_stage_jsonl(
            "resolve", "final_rules.jsonl",
            [fr.model_dump() for fr in final_rules],
            pmcid=pmcid, required=True,
        )
        score_rows = [
            {
                "final_id":          fr.final_id,
                "canonical_id":      fr.canonical_id,
                "group_id":          fr.group_id,
                "member_normal_ids": list(fr.member_normal_ids or []),
                "final_score":       fr.final_score,
                "support_count":     fr.support_count,
                "contradict_count":  fr.contradict_count,
                "scope_qualify_count": fr.scope_qualify_count,
                "is_contradicted":   fr.is_contradicted,
                "contradicted_by":   list(fr.contradicted_by or []),
                "mean_grounding_score": fr.mean_grounding_score,
                "finding_count":     fr.finding_count,
                "study_coverage":    fr.study_coverage,
            }
            for fr in final_rules
        ]
        writer.write_stage_jsonl(
            "resolve", "score_trace.jsonl", score_rows, pmcid=pmcid,
        )
        writer.mark_stage_completed("resolve")
    except Exception as exc:
        writer.write_error(stage="resolve", error=str(exc), pmcid=pmcid)
        raise
