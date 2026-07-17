"""Reusable helpers for Layer A (frozen-artifact) calibration sweeps.

Strict invariant: this module never imports NLI / LLM / embedding /
transformers / torch libraries, nor any code from
``pipeline.stages.knowledge_extraction.llm.llm_providers``. Layer A sweeps are
pure-filesystem analyses of frozen summarisation artifacts.

The module exposes:

- :class:`SweepMetadata`, :class:`FindingRecord` — shared dataclasses.
- :func:`iter_summary_paths`, :func:`load_summary` — artifact loaders.
- :func:`gather_findings_with_grounding` — pulls every original MAP finding
  (surviving + grounding-rejected) and reports any invariant breakage as a
  list of human-readable warnings.
- :func:`simulate_threshold_sweep` — pure function: ``(records, thresholds)
  → per-threshold rows``. Denominator semantics are explicit:
  ``kept_pct_of_scored`` excludes missing-score records.
- :func:`write_sweep_csv`, :func:`write_sweep_markdown` — output writers
  that stamp the metadata block.
- :func:`get_git_commit`, :func:`now_iso_utc`, :func:`sanitize_for_json`.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = "v1"

GROUNDING_REJECTION_STAGE = "grounding_map"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FindingRecord:
    """One original MAP finding, regardless of grounding-filter outcome."""

    pmcid: str
    chunk_id: str | None
    finding_index: int
    grounding_score: float | None
    kept_at_producer_threshold: bool
    producer_threshold: float | None
    category: str | None


@dataclass(frozen=True)
class SweepMetadata:
    sweep_name: str
    input_dir: str
    out_csv: str
    out_md: str
    run_ids: tuple[str, ...]
    pipeline_config_hashes: tuple[str, ...]
    pmcids: tuple[str, ...]
    thresholds: tuple[float, ...]
    created_at: str
    git_commit: str | None
    schema_version: str = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Artifact loaders
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


# ---------------------------------------------------------------------------
# Finding extraction
# ---------------------------------------------------------------------------


def _parse_score(value: Any) -> float | None:
    """Tolerant float parser. Returns ``None`` for missing/blank/bad input."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def gather_findings_with_grounding(
    summaries_dir: Path,
) -> tuple[list[FindingRecord], list[str]]:
    """Return ``(records, warnings)`` over every original MAP finding.

    Surviving findings come from ``audit_trail.map_chunks[*].findings[*]``;
    grounding-rejected findings come from
    ``rejection_summary.rejected[stage='grounding_map']``. The function
    verifies, per paper, that ``surviving + grounding-rejected ==
    rejection_summary.map_findings_total`` whenever ``map_findings_total``
    is present, and appends a warning when the invariant fails.
    """
    records: list[FindingRecord] = []
    warnings: list[str] = []

    for path in iter_summary_paths(summaries_dir):
        summary = load_summary(path)
        pmcid = summary.get("pmcid") or path.stem

        rejection_summary = summary.get("rejection_summary") or {}
        producer_threshold = rejection_summary.get("grounding_threshold")
        if producer_threshold is not None:
            try:
                producer_threshold = float(producer_threshold)
            except (TypeError, ValueError):
                producer_threshold = None

        # Surviving findings.
        surviving_count = 0
        map_chunks = (summary.get("audit_trail") or {}).get("map_chunks") or []
        for chunk in map_chunks:
            chunk_id = chunk.get("chunk_id")
            findings = chunk.get("findings") or []
            for idx, finding in enumerate(findings):
                records.append(
                    FindingRecord(
                        pmcid=pmcid,
                        chunk_id=chunk_id,
                        finding_index=idx,
                        grounding_score=_parse_score(finding.get("grounding_score")),
                        kept_at_producer_threshold=True,
                        producer_threshold=producer_threshold,
                        category=finding.get("category"),
                    )
                )
                surviving_count += 1

        # Grounding-stage rejections.
        rejected_count = 0
        rejected_entries = rejection_summary.get("rejected") or []
        for idx, entry in enumerate(rejected_entries):
            stage = (entry.get("stage") or "").lower()
            if stage != GROUNDING_REJECTION_STAGE:
                continue
            records.append(
                FindingRecord(
                    pmcid=pmcid,
                    chunk_id=entry.get("chunk_id"),
                    finding_index=idx,
                    grounding_score=_parse_score(entry.get("grounding_score")),
                    kept_at_producer_threshold=False,
                    producer_threshold=producer_threshold,
                    category=entry.get("category"),
                )
            )
            rejected_count += 1

        # Invariant check.
        map_findings_total = rejection_summary.get("map_findings_total")
        observed = surviving_count + rejected_count
        if map_findings_total is None:
            warnings.append(
                f"{pmcid}: rejection_summary.map_findings_total is absent; "
                f"sweep operates on {observed} records (surviving={surviving_count}, "
                f"grounding_rejected={rejected_count}). Counts cannot be "
                f"cross-checked against a producer-reported total."
            )
        else:
            try:
                expected = int(map_findings_total)
            except (TypeError, ValueError):
                expected = None
            if expected is not None and expected != observed:
                warnings.append(
                    f"{pmcid}: surviving + grounding_rejected = "
                    f"{surviving_count} + {rejected_count} = {observed} but "
                    f"rejection_summary.map_findings_total = {expected} — "
                    f"possible missing or duplicated records."
                )

    return records, warnings


# ---------------------------------------------------------------------------
# Sweep simulation
# ---------------------------------------------------------------------------


def simulate_threshold_sweep(
    records: list[FindingRecord], thresholds: list[float]
) -> list[dict[str, Any]]:
    """For each threshold ``t`` return a row with retention metrics.

    Column semantics (also documented in CSV/Markdown writers):

    - ``total_findings``: count of all records (including missing-score).
    - ``scored_findings``: records with a non-None grounding_score.
    - ``missing_score_count``: ``total_findings - scored_findings``.
    - ``missing_score_pct``: ``missing_score_count / total_findings * 100``
      (denominator is total, not scored).
    - ``kept_findings`` / ``rejected_findings``: simulated by the predicate
      ``grounding_score >= threshold`` over ``scored_findings`` only.
    - ``kept_pct_of_scored`` / ``rejected_pct_of_scored``: denominator is
      ``scored_findings`` — explicit in the name to avoid misreads.
    - ``mean_grounding_score_kept`` / ``mean_grounding_score_rejected``:
      arithmetic mean of the corresponding survivor / rejected score set;
      ``None`` when the set is empty.
    - ``papers_covered``: distinct pmcids contributing at least one scored
      record.
    """
    total_findings = len(records)
    scored = [r for r in records if r.grounding_score is not None]
    scored_findings = len(scored)
    missing_score_count = total_findings - scored_findings
    missing_score_pct = (
        (missing_score_count / total_findings * 100.0) if total_findings else 0.0
    )

    rows: list[dict[str, Any]] = []
    for t in thresholds:
        kept = [r for r in scored if r.grounding_score >= t]
        rejected = [r for r in scored if r.grounding_score < t]
        kept_n = len(kept)
        rejected_n = len(rejected)
        if scored_findings:
            kept_pct = kept_n / scored_findings * 100.0
            rejected_pct = rejected_n / scored_findings * 100.0
        else:
            kept_pct = 0.0
            rejected_pct = 0.0
        mean_kept = (
            sum(r.grounding_score for r in kept) / kept_n if kept_n else None
        )
        mean_rejected = (
            sum(r.grounding_score for r in rejected) / rejected_n
            if rejected_n
            else None
        )
        papers_covered = len({r.pmcid for r in scored})
        rows.append(
            {
                "threshold": float(t),
                "total_findings": total_findings,
                "scored_findings": scored_findings,
                "missing_score_count": missing_score_count,
                "missing_score_pct": missing_score_pct,
                "kept_findings": kept_n,
                "rejected_findings": rejected_n,
                "kept_pct_of_scored": kept_pct,
                "rejected_pct_of_scored": rejected_pct,
                "papers_covered": papers_covered,
                "mean_grounding_score_kept": mean_kept,
                "mean_grounding_score_rejected": mean_rejected,
            }
        )
    return rows


CSV_COLUMNS: list[str] = [
    "threshold",
    "total_findings",
    "scored_findings",
    "missing_score_count",
    "missing_score_pct",
    "kept_findings",
    "rejected_findings",
    "kept_pct_of_scored",
    "rejected_pct_of_scored",
    "papers_covered",
    "mean_grounding_score_kept",
    "mean_grounding_score_rejected",
]


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_sweep_csv(
    rows: list[dict[str, Any]],
    out_path: Path,
    metadata: SweepMetadata,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        # First emit metadata as `#` comment lines so the CSV stays readable
        # by stdlib csv readers (which skip leading `#` rows when configured)
        # and by humans grepping the file. The DictReader test in the suite
        # uses ``csv.DictReader(skip_initial_comment_lines)``-style handling
        # — see _read_csv_skipping_metadata in the test module.
        meta_dict = dataclasses.asdict(metadata)
        for k, v in meta_dict.items():
            f.write(f"# {k}: {json.dumps(v, default=str)}\n")
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _format_csv_value(row.get(c)) for c in CSV_COLUMNS})


def _format_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return value


DISCLAIMER = (
    "This report does NOT measure accuracy, precision, or recall unless "
    "joined with gold/silver labels. It is a threshold sensitivity / "
    "retention report computed over persisted NLI scores."
)


def write_sweep_markdown(
    rows: list[dict[str, Any]],
    out_path: Path,
    metadata: SweepMetadata,
    *,
    warnings: list[str] | None = None,
    per_paper_detail: list[dict[str, Any]] | None = None,
    producer_thresholds: list[float | None] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(
        f"# Threshold sensitivity / retention report — {metadata.sweep_name}"
    )
    lines.append("")
    lines.append(f"> **{DISCLAIMER}**")
    lines.append("")

    # Metadata block.
    lines.append("## Sweep metadata")
    lines.append("")
    lines.append(f"- schema_version: `{metadata.schema_version}`")
    lines.append(f"- sweep_name: `{metadata.sweep_name}`")
    lines.append(f"- created_at: `{metadata.created_at}`")
    lines.append(f"- git_commit: `{metadata.git_commit}`")
    lines.append(f"- input_dir: `{metadata.input_dir}`")
    lines.append(f"- out_csv: `{metadata.out_csv}`")
    lines.append(f"- out_md: `{metadata.out_md}`")
    lines.append(
        f"- pmcids ({len(metadata.pmcids)}): "
        + (", ".join(f"`{p}`" for p in metadata.pmcids) if metadata.pmcids else "_none_")
    )
    lines.append(
        f"- run_ids ({len(metadata.run_ids)}): "
        + (", ".join(f"`{r}`" for r in metadata.run_ids) if metadata.run_ids else "_none_")
    )
    lines.append(
        f"- pipeline_config_hashes ({len(metadata.pipeline_config_hashes)}): "
        + (", ".join(f"`{h}`" for h in metadata.pipeline_config_hashes) if metadata.pipeline_config_hashes else "_none_")
    )
    lines.append(
        f"- thresholds tested ({len(metadata.thresholds)}): "
        + ", ".join(f"`{t:g}`" for t in metadata.thresholds)
    )
    lines.append("")

    # Interpretation block.
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Higher `threshold` means stricter grounding (fewer findings kept).")
    lines.append(
        "- `kept` / `rejected` are simulated by the predicate "
        "`grounding_score >= threshold`."
    )
    lines.append(
        "- Records with missing `grounding_score` are excluded from the "
        "keep/reject decision and surfaced separately as "
        "`missing_score_count` / `missing_score_pct`."
    )
    lines.append(
        "- Percentages use **scored** records as denominator "
        "(`kept_pct_of_scored`, `rejected_pct_of_scored`). "
        "`missing_score_pct` uses **total** records as denominator."
    )
    lines.append("")

    # Warnings (only present when applicable).
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    # Main retention table.
    lines.append("## Retention by threshold")
    lines.append("")
    lines.append(
        "| threshold | total | scored | missing | missing_pct | kept | rejected "
        "| kept_pct_of_scored | mean_score_kept | mean_score_rejected |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    for row in rows:
        lines.append(
            "| "
            + _md_cell(row["threshold"], "{:.3g}")
            + " | "
            + _md_cell(row["total_findings"])
            + " | "
            + _md_cell(row["scored_findings"])
            + " | "
            + _md_cell(row["missing_score_count"])
            + " | "
            + _md_cell(row["missing_score_pct"], "{:.2f}")
            + " | "
            + _md_cell(row["kept_findings"])
            + " | "
            + _md_cell(row["rejected_findings"])
            + " | "
            + _md_cell(row["kept_pct_of_scored"], "{:.2f}")
            + " | "
            + _md_cell(row.get("mean_grounding_score_kept"), "{:.4f}")
            + " | "
            + _md_cell(row.get("mean_grounding_score_rejected"), "{:.4f}")
            + " |"
        )
    lines.append("")

    # Per-paper detail.
    if per_paper_detail is not None:
        unique_producer = sorted({pt for pt in (producer_thresholds or []) if pt is not None})
        if not unique_producer:
            header = "## Per-paper detail (producer threshold: not stamped)"
        elif len(unique_producer) == 1:
            header = f"## Per-paper detail (producer threshold = {unique_producer[0]:g})"
        else:
            header = (
                "## Per-paper detail (multiple producer thresholds: "
                + ", ".join(f"{t:g}" for t in unique_producer)
                + ")"
            )
        lines.append(header)
        lines.append("")
        lines.append(
            "| pmcid | total | scored | grounding_rejected_at_producer | producer_threshold |"
        )
        lines.append("|---|---|---|---|---|")
        for entry in per_paper_detail:
            lines.append(
                "| `"
                + str(entry.get("pmcid", ""))
                + "` | "
                + _md_cell(entry.get("total"))
                + " | "
                + _md_cell(entry.get("scored"))
                + " | "
                + _md_cell(entry.get("grounding_rejected"))
                + " | "
                + _md_cell(entry.get("producer_threshold"), "{:g}")
                + " |"
            )
        lines.append("")

    # Footer.
    lines.append("## Notes")
    lines.append("")
    lines.append("- Generated by `eval/sweeps/grounding.py` — Layer A, no API calls.")
    lines.append("- See `eval/sweeps/README.md` for the Layer A invariant.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _md_cell(value: Any, fmt: str = "") -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if fmt and isinstance(value, (int, float)):
        try:
            return fmt.format(value)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def build_sweep_metadata(
    *,
    sweep_name: str,
    summaries_dir: Path,
    input_dir: Path,
    out_csv: Path,
    out_md: Path,
    thresholds: list[float],
) -> SweepMetadata:
    """Collect run_ids / pipeline_config_hashes / pmcids from the input
    summaries, then stamp creation metadata. Stays cheap — one pass over the
    summaries, only top-level fields read."""
    run_ids: list[str] = []
    hashes: list[str] = []
    pmcids: list[str] = []
    for path in iter_summary_paths(summaries_dir):
        try:
            summary = load_summary(path)
        except (OSError, json.JSONDecodeError):
            continue
        pmcid = summary.get("pmcid") or path.stem
        pmcids.append(pmcid)
        rid = summary.get("run_id")
        if rid:
            run_ids.append(rid)
        h = summary.get("pipeline_config_hash")
        if h:
            hashes.append(h)
    return SweepMetadata(
        sweep_name=sweep_name,
        input_dir=str(input_dir),
        out_csv=str(out_csv),
        out_md=str(out_md),
        run_ids=tuple(sorted(set(run_ids))),
        pipeline_config_hashes=tuple(sorted(set(hashes))),
        pmcids=tuple(sorted(set(pmcids))),
        thresholds=tuple(float(t) for t in thresholds),
        created_at=now_iso_utc(),
        git_commit=get_git_commit(),
    )


# ---------------------------------------------------------------------------
# Generic helpers (duplicated from scripts/eval/_lib.py on purpose — keeps
# the two packages independent and removes a cross-directory import that
# would require sys.path manipulation).
# ---------------------------------------------------------------------------


def get_git_commit(repo_dir: Path | None = None) -> str | None:
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


def now_iso_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
    return value
