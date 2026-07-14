"""
RunManifestWriter — captures one JSON manifest per batch invocation.

A manifest is the single artifact a thesis-mode comparison reads when diffing
two configurations.  It contains:

* run_id, started_at, finished_at
* git SHA + dirty flag (best-effort)
* host, python version
* the full PipelineConfig snapshot + a deterministic ``config_digest``
* input description (pdf_dir, glob, max_docs, attempted PMCID list)
* aggregated per-document stats summary (counts, reason histogram, wall time)
* per-document status / wall_seconds breakdown

Observability-only — must NOT influence extraction behavior.  All recording
operations and the final write are best-effort: failure to write a manifest
must not raise into the runner.
"""
from __future__ import annotations

import json
import logging
import socket
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from nlp_histo.pipeline.stages.pdf_text_extraction.outputs.stats_writer import (
    config_digest,
    config_snapshot,
)

logger = logging.getLogger(__name__)


def make_run_id() -> str:
    """ISO-ish timestamp + 8-char uuid suffix, filesystem-safe."""
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def _git_info() -> Dict[str, Any]:
    """Best-effort git SHA + dirty flag. Returns ``{}`` when git metadata is unavailable.

    Provenance, not configuration: an installed wheel is not inside a checkout, so
    there is nothing to walk up to. Ask git about the *working directory* instead of
    counting parents from ``__file__`` — outside a repository git simply exits
    non-zero and the manifest records no git fields (schema unchanged; consumers
    already treat them as optional). Never runs at import.
    """
    try:
        repo_root = Path.cwd()
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return {"sha": sha, "branch": branch, "dirty": bool(dirty_out.strip())}
    except Exception:
        return {"sha": None, "branch": None, "dirty": None}


class RunManifestWriter:
    """
    Build one manifest per batch invocation.  Construct early, pass ``run_id``
    to each ``runner.run_document(..., run_id=...)`` call so per-doc stats are
    tagged, then call ``write(batch_stats)`` at the end.
    """

    def __init__(
        self,
        cfg: Any,
        run_metadata_dir: Path,
        run_id: Optional[str] = None,
    ) -> None:
        self._cfg = cfg
        self._dir = Path(run_metadata_dir)
        self.run_id = run_id or make_run_id()
        self._started_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        self._input_meta: Dict[str, Any] = {}
        self._attempted: List[str] = []

    def set_input(self, **kwargs: Any) -> None:
        """Record what was asked of this batch (pdf_dir / glob / max_docs / etc.)."""
        self._input_meta.update({k: _jsonable(v) for k, v in kwargs.items()})

    def set_attempted(self, pmcids: List[str]) -> None:
        self._attempted = list(pmcids)

    def write(self, batch_stats: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """Aggregate per-doc stats files and write the manifest.  Returns the path on success."""
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            per_doc, agg = self._aggregate()
            payload: Dict[str, Any] = {
                "run_id":        self.run_id,
                "started_at":    self._started_at,
                "finished_at":   datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "git":           _git_info(),
                "host":          socket.gethostname(),
                "python":        sys.version.split()[0],
                "input":         self._input_meta,
                "config_digest": config_digest(self._cfg),
                "config":        config_snapshot(self._cfg),
                "batch_stats":   batch_stats or {},
                "summary":       agg,
                "documents":     per_doc,
            }
            out = self._dir / f"run_{self.run_id}.json"
            tmp = out.with_suffix(out.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            tmp.replace(out)
            logger.info("RunManifestWriter: wrote %s", out.name)
            return out
        except Exception:
            logger.exception("RunManifestWriter: failed to write manifest")
            return None

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _aggregate(self) -> tuple:
        per_doc: List[Dict[str, Any]] = []
        agg_counts: Dict[str, int] = {}
        agg_reasons: Dict[str, int] = {}
        agg_in_header = 0
        total_wall = 0.0
        n_ok = n_failed = 0
        n_with_stats = 0

        for pmcid in self._attempted:
            path = self._dir / f"{pmcid}_stats.json"
            if not path.exists():
                per_doc.append({"pmcid": pmcid, "status": "no_stats", "wall_seconds": None})
                continue
            try:
                data = json.loads(path.read_text())
            except Exception:
                logger.exception("RunManifestWriter: bad stats file %s", path.name)
                continue
            n_with_stats += 1
            per_doc.append({
                "pmcid":         data.get("pmcid"),
                "status":        data.get("status"),
                "wall_seconds":  data.get("wall_seconds"),
                "failed_stage":  data.get("failed_stage"),
                "n_text_rows":   data.get("counts", {}).get("text_rows"),
                "n_figures":     data.get("counts", {}).get("figures_cropped"),
                "n_tables":      data.get("counts", {}).get("tables_cropped"),
            })
            if data.get("status") == "ok":
                n_ok += 1
            elif data.get("status") == "failed":
                n_failed += 1
            total_wall += float(data.get("wall_seconds") or 0)
            for k, v in (data.get("counts") or {}).items():
                try:
                    agg_counts[k] = agg_counts.get(k, 0) + int(v)
                except (TypeError, ValueError):
                    pass
            for k, v in (data.get("reason_histogram") or {}).items():
                try:
                    agg_reasons[k] = agg_reasons.get(k, 0) + int(v)
                except (TypeError, ValueError):
                    pass
            try:
                agg_in_header += int(data.get("reason_in_header_zone") or 0)
            except (TypeError, ValueError):
                pass

        summary = {
            "n_attempted":      len(self._attempted),
            "n_with_stats":     n_with_stats,
            "n_ok":             n_ok,
            "n_failed":         n_failed,
            "total_wall_seconds": round(total_wall, 4),
            "mean_wall_seconds": round(total_wall / n_with_stats, 4) if n_with_stats else 0.0,
            "counts_sum":       agg_counts,
            "reason_histogram_sum": agg_reasons,
            "reason_in_header_zone_sum": agg_in_header,
        }
        return per_doc, summary


# ── Helpers ───────────────────────────────────────────────────────────────────

def _jsonable(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)
