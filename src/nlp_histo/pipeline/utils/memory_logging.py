"""
Lightweight stage-level memory checkpoint logging.

Each ``checkpoint`` call emits a single grep-friendly ``MEMORY`` line with the
stage name, the event, elapsed time since the previous checkpoint, and — when
``psutil`` is installed — process RSS / VMS in MB plus the RSS delta since the
previous checkpoint.

Designed to be cheap (one ``psutil.Process().memory_info()`` call) so it can be
sprinkled liberally without affecting pipeline behaviour.

If ``psutil`` is not installed, the logger warns once and continues emitting
MEMORY lines with the stage/event/elapsed fields only — the rss_mb / vms_mb /
delta_rss_mb fields are omitted.  Pipelines never crash for lack of psutil.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any

try:  # psutil is an optional dependency.
    import psutil  # type: ignore
    _PSUTIL_AVAILABLE = True
except Exception:  # pragma: no cover — exercised only when psutil missing
    psutil = None  # type: ignore
    _PSUTIL_AVAILABLE = False

_DEFAULT_LOGGER = logging.getLogger("pipeline.memory")
_PSUTIL_WARNED = False


def _warn_psutil_missing_once(logger: logging.Logger) -> None:
    global _PSUTIL_WARNED
    if _PSUTIL_WARNED:
        return
    _PSUTIL_WARNED = True
    logger.warning(
        "MEMORY logging degraded (no RSS/VMS): psutil not installed — "
        "`pip install psutil` to enable memory metrics; "
        "MEMORY lines will still report stage/event/elapsed",
    )


class MemoryLogger:
    """Emit ``MEMORY ...`` log lines at pipeline checkpoints.

    Usage::

        mem = MemoryLogger()
        mem.checkpoint("NORMALIZE", "before")
        ...
        mem.checkpoint("NORMALIZE", "after")

    Or with the context manager (emits ``before`` / ``after`` automatically and
    ``failed`` if the block raises)::

        with mem.stage("NORMALIZE"):
            normalize_stage.run(...)
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        pmcid: str | None = None,
    ) -> None:
        self._logger = logger or _DEFAULT_LOGGER
        self._pmcid = pmcid
        self._prev_t: float | None = None
        self._prev_rss_mb: float | None = None
        self._last_stage: str | None = None
        if not _PSUTIL_AVAILABLE:
            _warn_psutil_missing_once(self._logger)

    @property
    def last_stage(self) -> str | None:
        """Most recent stage name passed to ``checkpoint``.  Useful for the
        outermost ``except`` block to label a ``failed`` event."""
        return self._last_stage

    def _sample(self) -> tuple[float | None, float | None]:
        if not _PSUTIL_AVAILABLE:
            return None, None
        try:
            info = psutil.Process().memory_info()  # type: ignore[union-attr]
            return info.rss / (1024 * 1024), info.vms / (1024 * 1024)
        except Exception:  # pragma: no cover
            return None, None

    def checkpoint(
        self,
        stage: str,
        event: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Emit one ``MEMORY`` line for the given stage/event.

        Always safe to call: when psutil is unavailable the rss_mb / vms_mb /
        delta_rss_mb fields are omitted but stage/event/elapsed are still
        logged, so the timeline of stage transitions stays intact.
        """
        self._last_stage = stage
        rss_mb, vms_mb = self._sample()
        now = time.perf_counter()
        elapsed = 0.0 if self._prev_t is None else now - self._prev_t
        delta_rss = (
            None
            if rss_mb is None or self._prev_rss_mb is None
            else rss_mb - self._prev_rss_mb
        )

        parts = []
        if self._pmcid is not None:
            parts.append(f"pmcid={self._pmcid}")
        parts.append(f"stage={stage}")
        parts.append(f"event={event}")
        if rss_mb is not None:
            parts.append(f"rss_mb={rss_mb:.1f}")
        if vms_mb is not None:
            parts.append(f"vms_mb={vms_mb:.1f}")
        parts.append(f"elapsed_s={elapsed:.1f}")
        if delta_rss is not None:
            parts.append(f"delta_rss_mb={delta_rss:+.1f}")
        if extra:
            for k, v in extra.items():
                parts.append(f"{k}={v}")

        self._logger.info("MEMORY " + " ".join(parts))

        self._prev_t = now
        if rss_mb is not None:
            self._prev_rss_mb = rss_mb

    @contextmanager
    def stage(self, name: str, extra: dict[str, Any] | None = None):
        """Context manager that emits ``before`` / ``after`` (or ``failed``)."""
        self.checkpoint(name, "before", extra)
        try:
            yield
        except BaseException:
            self.checkpoint(name, "failed", extra)
            raise
        else:
            self.checkpoint(name, "after", extra)


_GLOBAL_LOGGER: MemoryLogger | None = None


def get_default_memory_logger() -> MemoryLogger:
    """Return a process-wide ``MemoryLogger`` for use outside a per-paper run
    (e.g. one-time model loading inside ``umls_resources``)."""
    global _GLOBAL_LOGGER
    if _GLOBAL_LOGGER is None:
        _GLOBAL_LOGGER = MemoryLogger()
    return _GLOBAL_LOGGER
