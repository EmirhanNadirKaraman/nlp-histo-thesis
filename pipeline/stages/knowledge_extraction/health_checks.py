"""Startup health checks for the summarisation pipeline.

Probes each non-LLM component the pipeline depends on and reports whether
it's in a working state — guards against the "looks fine, isn't fine"
class of bug (B-019, B-020 were LLM-side silent failures; the components
checked here can fail in equally invisible ways).

Probed:
  * NLI grounding model — actually load and run a one-pair inference.
  * UMLS singleton — confirm enabled state and that scispaCy + linker load.
  * Embedding provider — embed one string, assert non-degenerate vector.
  * Database — query a trivial row count if a connection is supplied.

Each probe returns a `HealthCheckResult`. The aggregate is logged as a
single block + returned to the caller so it can be persisted to the run
artifact for thesis-trail traceability.

Usage:
    from pipeline.stages.knowledge_extraction.health_checks import run_health_checks
    results = run_health_checks(db=db_conn)
    if any(not r.ok for r in results):
        # operator decides whether to abort or proceed; we don't raise
        # by default because some checks (UMLS) are legitimately opt-out.
        ...

This module never raises — every probe catches its own exceptions so
health-check failures cannot kill a pipeline run that the operator
launched intentionally.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """One probe's outcome — name, ok flag, elapsed ms, freeform detail."""
    name: str
    ok: bool
    elapsed_ms: float
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _check_nli_model() -> HealthCheckResult:
    """Load the active NLI grounding model and run a one-pair inference.

    Catches the silent-fallback scenario where the model fails to load and
    GroundingFilter silently accepts everything (or rejects everything).
    """
    t0 = time.perf_counter()
    try:
        from .helpers.grounding_filter import GroundingFilter, _score_pairs  # noqa: PLC0415

        gf = GroundingFilter(batch_size=1)
        nli_pipe = gf._pipe  # noqa: SLF001 — intentional white-box probe of cached pipeline
        scores = _score_pairs(
            [("The patient was diagnosed with lymphoma.", "The patient has cancer.")],
            nli_pipe,
        )
        elapsed = (time.perf_counter() - t0) * 1000.0
        if not scores or not isinstance(scores[0], (float, int)):
            return HealthCheckResult(
                name="nli_model", ok=False, elapsed_ms=elapsed,
                detail=f"NLI returned malformed score: {scores!r}",
                metadata={"model": gf._model_name},
            )
        return HealthCheckResult(
            name="nli_model", ok=True, elapsed_ms=elapsed,
            detail=f"loaded {gf._model_name}, smoke entailment score={scores[0]:.3f}",
            metadata={"model": gf._model_name, "score": scores[0]},
        )
    except Exception as exc:  # noqa: BLE001 — health check must never raise
        elapsed = (time.perf_counter() - t0) * 1000.0
        return HealthCheckResult(
            name="nli_model", ok=False, elapsed_ms=elapsed,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _check_umls() -> HealthCheckResult:
    """Confirm UMLS enablement and that the scispaCy + linker pair loads.

    UMLS is intentionally opt-out (env var `NLP_HISTO_DISABLE_UMLS=1`), so
    `ok=False` when disabled is a STATUS, not a failure — the metadata
    distinguishes "disabled by operator" from "disabled by load failure".
    """
    t0 = time.perf_counter()
    try:
        from .umls_resources import get_linker, get_nlp, umls_disabled  # noqa: PLC0415

        if umls_disabled():
            elapsed = (time.perf_counter() - t0) * 1000.0
            return HealthCheckResult(
                name="umls", ok=True, elapsed_ms=elapsed,
                detail="disabled via $NLP_HISTO_DISABLE_UMLS (intentional)",
                metadata={"disabled_by_env": True},
            )
        nlp = get_nlp()
        linker = get_linker()
        elapsed = (time.perf_counter() - t0) * 1000.0
        if nlp is None or linker is None:
            return HealthCheckResult(
                name="umls", ok=False, elapsed_ms=elapsed,
                detail=f"UMLS not disabled but nlp={nlp!r} linker={linker!r}",
            )
        # Cheap probe: how many concepts does the linker hold? Should be
        # a large positive number (>100k for the default UMLS subset).
        kb_size: int | None = None
        try:
            kb = getattr(linker, "kb", None)
            cui_to_entity = getattr(kb, "cui_to_entity", None)
            if cui_to_entity is not None:
                kb_size = len(cui_to_entity)
        except Exception:  # noqa: BLE001
            kb_size = None
        return HealthCheckResult(
            name="umls", ok=True, elapsed_ms=elapsed,
            detail=f"scispaCy + linker loaded (kb_size={kb_size})",
            metadata={"kb_size": kb_size},
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - t0) * 1000.0
        return HealthCheckResult(
            name="umls", ok=False, elapsed_ms=elapsed,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _check_embedding_provider() -> HealthCheckResult:
    """Embed one string and assert the returned vector is non-degenerate.

    Guards against the "API errors silently fall back to all-zero vectors"
    failure mode that would make every pair-wise agreement score 1.0.
    """
    t0 = time.perf_counter()
    try:
        from .agreement.providers import GeminiEmbedder, OpenAIEmbedder  # noqa: PLC0415

        # Prefer Gemini if its key is set (matches production cheap-profile
        # default); fall back to OpenAI.
        if os.environ.get("GOOGLE_API_KEY"):
            embedder = GeminiEmbedder()
            label = "gemini"
        elif os.environ.get("OPENAI_API_KEY"):
            embedder = OpenAIEmbedder()
            label = "openai"
        else:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return HealthCheckResult(
                name="embedding", ok=False, elapsed_ms=elapsed,
                detail="neither GOOGLE_API_KEY nor OPENAI_API_KEY set",
            )

        vectors = embedder(["the patient was diagnosed with lymphoma"])
        elapsed = (time.perf_counter() - t0) * 1000.0
        if not vectors or not vectors[0]:
            return HealthCheckResult(
                name="embedding", ok=False, elapsed_ms=elapsed,
                detail=f"embedder returned empty: {vectors!r}",
                metadata={"provider": label},
            )
        v = vectors[0]
        # Non-degenerate check: at least one non-zero component AND the
        # vector is not all-equal (all-equal also produces agreement=1.0
        # for cosine similarity).
        all_zero = not any(abs(x) > 1e-9 for x in v)
        all_equal = len(set(v[:10])) == 1 if len(v) >= 10 else False
        if all_zero or all_equal:
            return HealthCheckResult(
                name="embedding", ok=False, elapsed_ms=elapsed,
                detail=(
                    f"degenerate vector (all_zero={all_zero}, "
                    f"all_equal={all_equal}, len={len(v)})"
                ),
                metadata={"provider": label, "dim": len(v)},
            )
        return HealthCheckResult(
            name="embedding", ok=True, elapsed_ms=elapsed,
            detail=f"{label}: dim={len(v)}, sample[0:3]={v[:3]}",
            metadata={"provider": label, "dim": len(v)},
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - t0) * 1000.0
        return HealthCheckResult(
            name="embedding", ok=False, elapsed_ms=elapsed,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _check_database(db) -> HealthCheckResult:
    """Run a trivial query through the supplied DB connection.

    `db` is a DatabaseConnection instance (from `database/db_connection.py`)
    or None. None is treated as ok (DB is opt-in).
    """
    t0 = time.perf_counter()
    try:
        if db is None:
            return HealthCheckResult(
                name="database", ok=True, elapsed_ms=0.0,
                detail="no db connection supplied — DB persistence disabled",
                metadata={"db_supplied": False},
            )
        with db.session_scope() as session:
            row = session.execute(__import__("sqlalchemy").text("SELECT 1")).scalar()
        elapsed = (time.perf_counter() - t0) * 1000.0
        if row != 1:
            return HealthCheckResult(
                name="database", ok=False, elapsed_ms=elapsed,
                detail=f"SELECT 1 returned {row!r}",
            )
        return HealthCheckResult(
            name="database", ok=True, elapsed_ms=elapsed,
            detail="SELECT 1 = 1",
            metadata={"db_supplied": True},
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - t0) * 1000.0
        return HealthCheckResult(
            name="database", ok=False, elapsed_ms=elapsed,
            detail=f"{type(exc).__name__}: {exc}",
        )


def run_health_checks(*, db=None, skip_nli: bool = False) -> list[HealthCheckResult]:
    """Run every probe and return the list of results.

    Logs a single aggregated block at INFO level so operators see the
    summary in stdout. `skip_nli=True` lets fast unit tests skip the
    HuggingFace download — production callers should leave it False.
    """
    results: list[HealthCheckResult] = []
    if not skip_nli:
        results.append(_check_nli_model())
    results.append(_check_umls())
    results.append(_check_embedding_provider())
    results.append(_check_database(db))

    # Single aggregated log line so operators don't have to grep.
    lines = ["Pipeline health checks:"]
    for r in results:
        marker = "OK " if r.ok else "FAIL"
        lines.append(f"  [{marker}] {r.name:12s} ({r.elapsed_ms:6.0f}ms)  {r.detail}")
    aggregated = "\n".join(lines)
    if all(r.ok for r in results):
        logger.info(aggregated)
    else:
        logger.warning(aggregated)
    return results


def assert_persistence_row_counts(
    db,
    pmcid: str,
    *,
    expected_map_findings_min: int = 1,
) -> list[HealthCheckResult]:
    """Post-run sanity: count rows in each `sum_*` table for one paper.

    Catches the partial-persistence failure mode where one of the per-stage
    `_persist_*` methods threw an exception that was swallowed at the call
    site (B-017 class). If `sum_map_findings` has rows but `sum_final_rules`
    has zero, the run looks complete in the JSON artifact but the DB-backed
    inspector / corpus_relate gate sees nothing.

    Never raises. Returns one HealthCheckResult per table probed.
    """
    results: list[HealthCheckResult] = []
    if db is None:
        return results
    table_specs = [
        ("sum_map_findings",        expected_map_findings_min),
        ("sum_normal_findings",     1),
        ("sum_finding_groups",      1),
        ("sum_canonical_rules",     1),
        ("sum_final_rules",         1),
    ]
    try:
        from sqlalchemy import text  # noqa: PLC0415
        with db.session_scope() as session:
            for table_name, expected_min in table_specs:
                t0 = time.perf_counter()
                try:
                    count = session.execute(
                        text(f"SELECT COUNT(*) FROM {table_name} WHERE pmcid = :pmcid"),
                        {"pmcid": pmcid},
                    ).scalar() or 0
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    ok = count >= expected_min
                    results.append(HealthCheckResult(
                        name=f"rows:{table_name}", ok=ok, elapsed_ms=elapsed,
                        detail=f"{count} rows for pmcid={pmcid} (expected ≥{expected_min})",
                        metadata={"count": count, "pmcid": pmcid, "table": table_name},
                    ))
                except Exception as exc:  # noqa: BLE001
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    results.append(HealthCheckResult(
                        name=f"rows:{table_name}", ok=False, elapsed_ms=elapsed,
                        detail=f"query failed: {type(exc).__name__}: {exc}",
                    ))
    except Exception as exc:  # noqa: BLE001
        results.append(HealthCheckResult(
            name="rows:session", ok=False, elapsed_ms=0.0,
            detail=f"session_scope failed: {type(exc).__name__}: {exc}",
        ))

    # Log only the failures + a single OK line if everything passed.
    failed = [r for r in results if not r.ok]
    if failed:
        msg_lines = [f"[{pmcid}] Post-run row-count audit — {len(failed)} table(s) below expected:"]
        for r in failed:
            msg_lines.append(f"  [FAIL] {r.name:32s}  {r.detail}")
        logger.warning("\n".join(msg_lines))
    else:
        logger.info(
            "[%s] Post-run row-count audit OK (%d table(s) checked)",
            pmcid, len(results),
        )
    return results
