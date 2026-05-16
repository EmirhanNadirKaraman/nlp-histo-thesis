"""Pure file I/O helpers for the Phase 1 evaluation harness.

Strict invariant: this module never imports NLI/LLM/embedding libraries
(transformers, torch, openai, anthropic, google.genai, vertexai,
sentence_transformers) nor any code from
``pipeline.stages.summarization.llm_providers``. Phase 1 scripts that import
``_lib`` must keep that invariant.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Summary JSONs
# ---------------------------------------------------------------------------

def iter_summary_paths(summaries_dir: Path) -> Iterator[Path]:
    if not summaries_dir.exists():
        return
    for child in sorted(summaries_dir.iterdir()):
        if not child.is_file() or child.suffix != ".json":
            continue
        if child.name.startswith("_"):
            continue
        yield child


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def coerce_pmcid_from_path(path: Path) -> str:
    return path.stem


# ---------------------------------------------------------------------------
# Trace artifacts
# ---------------------------------------------------------------------------

def iter_runs_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_runs_by_pmcid(path: Path) -> dict[str, dict[str, Any]]:
    """Most recent run per pmcid wins (by ``started_at`` then file order)."""
    latest: dict[str, dict[str, Any]] = {}
    for rec in iter_runs_jsonl(path):
        pmcid = rec.get("pmcid")
        if not pmcid:
            continue
        prev = latest.get(pmcid)
        if prev is None:
            latest[pmcid] = rec
            continue
        prev_started = prev.get("started_at") or ""
        this_started = rec.get("started_at") or ""
        if this_started >= prev_started:
            latest[pmcid] = rec
    return latest


def load_chunk_summary_csv(path: Path) -> dict[str, list[dict[str, str]]]:
    """Group chunk_summary.csv rows by pmcid."""
    grouped: dict[str, list[dict[str, str]]] = {}
    if not path.exists():
        return grouped
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped.setdefault(row.get("pmcid", ""), []).append(row)
    return grouped


def iter_cascade_decisions(decisions_dir: Path, pmcid: str) -> Iterator[dict[str, Any]]:
    path = decisions_dir / f"{pmcid}.jsonl"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------------------
# Cost artifacts
# ---------------------------------------------------------------------------

def iter_cost_reports(cost_root: Path) -> Iterator[dict[str, Any]]:
    if not cost_root.exists():
        return
    for run_dir in sorted(cost_root.iterdir()):
        if not run_dir.is_dir():
            continue
        report = run_dir / "cost_report.json"
        if not report.exists():
            continue
        try:
            with report.open("r", encoding="utf-8") as f:
                yield json.load(f)
        except (OSError, json.JSONDecodeError):
            continue


def iter_usage_records(cost_root: Path) -> Iterator[dict[str, Any]]:
    if not cost_root.exists():
        return
    for run_dir in sorted(cost_root.iterdir()):
        if not run_dir.is_dir():
            continue
        records = run_dir / "llm_usage_records.jsonl"
        if not records.exists():
            continue
        with records.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def aggregate_cost_by_paper(cost_root: Path) -> dict[str, dict[str, float]]:
    """Sum cost_report ``actual_usage.by_paper`` entries across all run dirs.

    Returns ``{paper_id: {calls, input_tokens, output_tokens, total_tokens, cost_usd}}``.
    """
    aggregated: dict[str, dict[str, float]] = {}
    for report in iter_cost_reports(cost_root):
        by_paper = (report.get("actual_usage") or {}).get("by_paper") or {}
        for paper_id, stats in by_paper.items():
            bucket = aggregated.setdefault(
                paper_id,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0},
            )
            bucket["calls"] += _safe_number(stats.get("calls"))
            bucket["input_tokens"] += _safe_number(stats.get("input_tokens"))
            bucket["output_tokens"] += _safe_number(stats.get("output_tokens"))
            bucket["total_tokens"] += _safe_number(stats.get("total_tokens"))
            bucket["cost_usd"] += _safe_number(stats.get("cost_usd"))
    return aggregated


def _safe_number(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Hash + git + time helpers
# ---------------------------------------------------------------------------

def get_git_commit(repo_dir: Path | None = None) -> str | None:
    """Return ``<sha>`` or ``<sha>-dirty``; ``None`` if not a git repo."""
    cwd = str(repo_dir) if repo_dir else None
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, capture_output=True, text=True, check=False, timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if rev.returncode != 0 or not rev.stdout.strip():
        return None
    sha = rev.stdout.strip()
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd, capture_output=True, text=True, check=False, timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return sha
    if status.returncode == 0 and status.stdout.strip():
        return f"{sha}-dirty"
    return sha


def sha256_short(payload: Any, length: int = 16) -> str:
    if isinstance(payload, (dict, list)):
        material = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    elif isinstance(payload, bytes):
        material = payload
    elif isinstance(payload, str):
        material = payload.encode("utf-8")
    else:
        material = str(payload).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:length]


def now_iso_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
