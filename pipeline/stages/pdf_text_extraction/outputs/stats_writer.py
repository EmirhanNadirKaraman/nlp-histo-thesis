"""
DocStatsCollector — per-document observability artifact for the PDF text
extraction pipeline.

Writes one JSON file per processed document to
``{run_metadata_dir}/{pmcid}_stats.json`` containing:

* status (``ok`` / ``failed`` / ``skipped``) and optional error message
* per-stage wall-clock timings
* counts of layout elements, kept/rejected nodes, output text rows, figures,
  tables, masked regions
* histogram of NodeScorer rejection reasons (R0 / R1 / R2 / R3 / R-color)
  plus an "in header zone" sub-tally (R4 location hint)
* a deterministic ``config_digest`` and a compact ``config_used`` block so
  two runs can be diffed knob-by-knob

Design contract (do NOT break)
------------------------------
This module is observability-only.  It must never influence extraction
behavior.  The collector exposes ``record_*`` methods that accumulate state
and a single ``write()`` that emits JSON.  All recording operations are
best-effort: a failure to record a stat must not raise into the runner.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


# ── Reason-code mapping ───────────────────────────────────────────────────────
#
# NodeScorer emits human-readable rejection strings.  We map each canonical
# prefix to a stable short code so histograms are comparable across runs even
# if the reason wording changes.  Unknown reasons fall into ``OTHER`` with the
# raw string preserved under ``reason_unknown_examples``.

#
# Source of truth: prefixes match the strings emitted by
# pipeline/stages/pdf_text_extraction/components/node_scorer.py::_score_text_node.
# Ordering matters when prefixes are not mutually exclusive — first match wins.
_REASON_PREFIXES = (
    ("empty text",              "R0_empty"),
    ("white-text ghost layer",  "R_color_white"),
    ("hidden text layer",       "R3_dense_text"),
    ("visually blank region",   "R1_blank_pixels"),
    ("no fitz words",           "R2_no_fitz_words"),
)
_HEADER_ZONE_MARKER = "in header zone"


def classify_reason(reason: str) -> str:
    """Map a NodeScorer rejection string to a stable short code."""
    if not reason:
        return "OTHER"
    for prefix, code in _REASON_PREFIXES:
        if reason.startswith(prefix):
            return code
    return "OTHER"


# ── Stats dataclass ───────────────────────────────────────────────────────────

@dataclass
class _StageTiming:
    name: str
    seconds: float
    ok: bool = True
    error: Optional[str] = None


@dataclass
class DocStats:
    pmcid: str
    pdf_path: str
    run_id: Optional[str] = None
    started_at: str = ""
    finished_at: str = ""
    wall_seconds: float = 0.0
    status: str = "running"          # running | ok | failed | skipped
    error: Optional[str] = None
    failed_stage: Optional[str] = None

    stage_timings: List[_StageTiming] = field(default_factory=list)

    counts: Dict[str, int] = field(default_factory=dict)
    """
    Slots populated by the runner as available:
      layout_elements_full, layout_elements_masked,
      after_filter, text_rows,
      figures_cropped, tables_cropped,
      mask_regions, table_regions_detected,
      two_pass_scored, two_pass_kept, two_pass_rejected
    """

    reason_histogram: Dict[str, int] = field(default_factory=dict)
    """Counts keyed by canonical reason code (R0_empty, R1_blank_pixels, …)."""

    reason_in_header_zone: int = 0
    """How many rejections had the 'in header zone' marker (R4 amendment)."""

    reason_unknown_examples: List[str] = field(default_factory=list)
    """Up to 5 raw rejection strings classified as OTHER, for forensic review."""

    table_detection: Dict[str, Any] = field(default_factory=dict)
    """Reserved for {detector, n_regions, per_source_counts}."""

    config_digest: Optional[str] = None
    config_used: Dict[str, Any] = field(default_factory=dict)


# ── Collector ─────────────────────────────────────────────────────────────────

class DocStatsCollector:
    """
    Mutable accumulator. Construct ONE per document, BEFORE the first
    fail-prone extraction step.  Call ``write()`` from a ``finally`` block so
    failed documents still produce stats.
    """

    _MAX_UNKNOWN_EXAMPLES = 5

    def __init__(
        self,
        pmcid: str,
        pdf_path: Path,
        run_metadata_dir: Path,
        run_id: Optional[str] = None,
        config_digest: Optional[str] = None,
        config_used: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._dir = Path(run_metadata_dir)
        self._stats = DocStats(
            pmcid=pmcid,
            pdf_path=str(pdf_path),
            run_id=run_id,
            started_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            config_digest=config_digest,
            config_used=config_used or {},
        )
        self._t0 = time.monotonic()
        self._written = False

    # ── Recording ─────────────────────────────────────────────────────────────

    @contextlib.contextmanager
    def stage(self, name: str):
        """Time a stage.  Errors are recorded and re-raised."""
        t = time.monotonic()
        try:
            yield
        except Exception as exc:
            self._safe(lambda: self._stats.stage_timings.append(
                _StageTiming(name=name, seconds=time.monotonic() - t,
                             ok=False, error=type(exc).__name__)
            ))
            self._stats.failed_stage = name
            raise
        else:
            self._safe(lambda: self._stats.stage_timings.append(
                _StageTiming(name=name, seconds=time.monotonic() - t)
            ))

    def set_count(self, key: str, value: int) -> None:
        self._safe(lambda: self._stats.counts.__setitem__(key, int(value)))

    def add_count(self, key: str, delta: int = 1) -> None:
        self._safe(lambda: self._stats.counts.__setitem__(
            key, int(self._stats.counts.get(key, 0)) + int(delta)
        ))

    def record_scored_nodes(self, scored_nodes: Iterable) -> None:
        """Tally rejection reasons from a sequence of ScoredNode objects."""
        def _do():
            kept = rejected = 0
            for n in scored_nodes:
                if getattr(n, "keep", True):
                    kept += 1
                    continue
                rejected += 1
                raw = getattr(n, "rejection_reason", "") or ""
                code = classify_reason(raw)
                self._stats.reason_histogram[code] = \
                    self._stats.reason_histogram.get(code, 0) + 1
                if _HEADER_ZONE_MARKER in raw:
                    self._stats.reason_in_header_zone += 1
                if code == "OTHER" and raw and \
                        len(self._stats.reason_unknown_examples) < self._MAX_UNKNOWN_EXAMPLES:
                    self._stats.reason_unknown_examples.append(raw)
            self._stats.counts["two_pass_kept"] = kept
            self._stats.counts["two_pass_rejected"] = rejected
            self._stats.counts["two_pass_scored"] = kept + rejected
        self._safe(_do)

    def record_table_detection(self, detector: str, n_regions: int,
                               per_source_counts: Optional[Dict[str, int]] = None) -> None:
        self._safe(lambda: self._stats.table_detection.update({
            "detector": detector,
            "n_regions": int(n_regions),
            "per_source_counts": per_source_counts or {},
        }))

    def mark_skipped(self, reason: str) -> None:
        self._stats.status = "skipped"
        self._stats.error = reason

    def mark_failed(self, exc: BaseException, stage: Optional[str] = None) -> None:
        self._stats.status = "failed"
        self._stats.error = f"{type(exc).__name__}: {exc}"
        if stage and not self._stats.failed_stage:
            self._stats.failed_stage = stage

    def mark_ok(self) -> None:
        if self._stats.status == "running":
            self._stats.status = "ok"

    # ── Persistence ───────────────────────────────────────────────────────────

    def write(self) -> Optional[Path]:
        """Idempotent: writes at most once."""
        if self._written:
            return None
        self._written = True
        self._stats.finished_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        self._stats.wall_seconds = round(time.monotonic() - self._t0, 4)
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            out = self._dir / f"{self._stats.pmcid}_stats.json"
            payload = _to_jsonable(self._stats)
            tmp = out.with_suffix(out.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            tmp.replace(out)
            logger.info("DocStatsCollector: wrote %s", out.name)
            return out
        except Exception:
            logger.exception("DocStatsCollector: failed to write stats for %s",
                             self._stats.pmcid)
            return None

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _safe(fn) -> None:
        try:
            fn()
        except Exception:
            logger.exception("DocStatsCollector: record op failed (suppressed)")


# ── Config digest helpers ─────────────────────────────────────────────────────

_PATHCONFIG_FIELDS_TO_EXCLUDE = {
    "project_root", "output_root", "files_root",
    "masked_pdf_dir", "docling_full_dir", "docling_masked_dir",
    "text_dir", "text_raw_dir", "json_dir", "vis_dir",
    "figures_dir", "tables_dir", "blacklist_file",
    "completed_file", "run_metadata_dir",
}


def config_digest(cfg: Any, length: int = 12) -> str:
    """
    Stable short hash of a PipelineConfig (or any dataclass).  Paths are
    excluded so the digest reflects only knobs that change extraction
    behavior — not where outputs are written.
    """
    try:
        payload = _config_payload(cfg)
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:length]
    except Exception:
        logger.exception("config_digest: failed; returning 'unknown'")
        return "unknown"


def config_snapshot(cfg: Any) -> Dict[str, Any]:
    """Compact JSON-friendly snapshot of cfg (paths excluded from digest input
    but kept here for forensic value)."""
    try:
        return _to_jsonable(cfg)
    except Exception:
        logger.exception("config_snapshot: failed")
        return {}


def _config_payload(cfg: Any) -> Dict[str, Any]:
    """asdict(cfg) but strip the PathConfig fields that don't change behavior."""
    data = _to_jsonable(cfg)
    if isinstance(data, dict) and "paths" in data and isinstance(data["paths"], dict):
        data = dict(data)
        data["paths"] = {
            k: v for k, v in data["paths"].items()
            if k not in _PATHCONFIG_FIELDS_TO_EXCLUDE
        }
    return data


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "value") and obj.__class__.__name__ in {"TableDetectorType", "LogLevel", "OcrEngine", "BaselineMode"}:
        return obj.value
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
