"""
eval.llm_judge.cache — Postgres-backed result cache.

Cache key includes: model, prompt_version, schema_version, task,
and a normalized hash of the request payload.  Failed / malformed /
skipped responses are never cached.

The cache uses the project's existing DatabaseConnection infrastructure —
no SQLite, no separate connection pool.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from . import LABEL_SOURCE, MODEL, PROMPT_VERSION, SCHEMA_VERSION

logger = logging.getLogger(__name__)


class JudgeCache:
    """Postgres-backed evaluation cache using the project DB connection."""

    def __init__(self, db: Any) -> None:
        """
        Parameters
        ----------
        db:
            A DatabaseConnection instance (from database.get_db_connection()).
            Each cache operation opens its own session via session_scope().
        """
        self._db = db

    # Key

    @staticmethod
    def make_key(
        task: str,
        model: str = MODEL,
        prompt_version: str = PROMPT_VERSION,
        schema_version: str = SCHEMA_VERSION,
        **inputs: Any,
    ) -> str:
        """Deterministic cache key from model + versions + request payload."""
        blob = json.dumps(
            {
                "model": model,
                "pv": prompt_version,
                "sv": schema_version,
                "task": task,
                **inputs,
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    # CRUD

    def get(self, key: str) -> dict | None:
        """Return the raw result_payload (Opus judgment) for this key, or None on miss.

        Callers must pass the result through parse_*_result() to derive fields
        such as fields_changed, is_correct, and F1 metrics.
        """
        from nlp_histo.database import LlmJudgeCache  # noqa: PLC0415

        with self._db.session_scope() as session:
            row = session.query(LlmJudgeCache).filter_by(cache_key=key).first()
            if row is None:
                return None
            return dict(row.result_payload)

    def set(
        self,
        key: str,
        request_payload: dict,
        result_payload: dict,
        task: str,
    ) -> None:
        """
        Store a judge result.  Only call this for successful, parseable responses.

        Parameters
        ----------
        key:
            Cache key from make_key().
        request_payload:
            The request context (JudgeRequest.metadata) — what was sent to the judge.
        result_payload:
            The parsed Opus judgment dict — what the judge returned.
        task:
            Task name, e.g. "q1_precision".
        """
        from nlp_histo.database import LlmJudgeCache  # noqa: PLC0415
        from sqlalchemy.dialects.postgresql import insert  # noqa: PLC0415

        pmcid: str | None = request_payload.get("pmcid")
        pipeline_run_id: int | None = request_payload.get("pipeline_run_id")
        source_id: int | None = (
            request_payload.get("finding_id")
            or request_payload.get("relation_id")
            or request_payload.get("text_element_id")
            or request_payload.get("rejected_finding_id")
        )

        with self._db.session_scope() as session:
            stmt = (
                insert(LlmJudgeCache)
                .values(
                    cache_key=key,
                    task=task,
                    judge_model=MODEL,
                    prompt_version=PROMPT_VERSION,
                    schema_version=SCHEMA_VERSION,
                    label_source=LABEL_SOURCE,
                    request_payload=request_payload,
                    result_payload=result_payload,
                    pmcid=pmcid,
                    pipeline_run_id=pipeline_run_id,
                    source_id=source_id,
                )
                .on_conflict_do_update(
                    index_elements=["cache_key"],
                    set_={
                        "result_payload": result_payload,
                        "updated_at": sa_now(),
                    },
                )
            )
            session.execute(stmt)

    def clear_version(self, tasks: set[str] | None = None) -> int:
        """
        Delete cache rows for the current model/prompt/schema version.

        Parameters
        ----------
        tasks:
            If provided, only delete rows whose task is in this set.
            If None, delete all rows for the current version.

        Returns the number of rows deleted.
        """
        from nlp_histo.database import LlmJudgeCache  # noqa: PLC0415

        with self._db.session_scope() as session:
            q = session.query(LlmJudgeCache).filter_by(
                judge_model=MODEL,
                prompt_version=PROMPT_VERSION,
                schema_version=SCHEMA_VERSION,
            )
            if tasks is not None:
                q = q.filter(LlmJudgeCache.task.in_(tasks))
            n = q.delete(synchronize_session=False)
            logger.info(
                "Cache: cleared %d rows (model=%s pv=%s sv=%s tasks=%s)",
                n, MODEL, PROMPT_VERSION, SCHEMA_VERSION,
                sorted(tasks) if tasks else "all",
            )
            return n

    def count(self) -> int:
        """Return total number of rows in the cache table."""
        from nlp_histo.database import LlmJudgeCache  # noqa: PLC0415

        with self._db.session_scope() as session:
            return session.query(LlmJudgeCache).count()

    def close(self) -> None:
        """No-op — connection lifecycle is managed by DatabaseConnection."""


# Helpers

def sa_now():
    """Return a SQLAlchemy server-side now() expression."""
    from sqlalchemy import func  # noqa: PLC0415
    return func.now()
