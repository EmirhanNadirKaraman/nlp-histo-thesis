"""
Generate human-readable inspection reports for silver vs pipeline matching.

Outputs (all timestamped):
  inspect_<ts>.md              — Markdown report, one section per case
  matched_pairs_<ts>.csv       — all matched pairs with scores
  unmatched_silver_<ts>.csv    — silver claims with no pipeline match
  unmatched_pipeline_<ts>.csv  — pipeline claims with no silver match
  field_mismatches_<ts>.csv    — field mismatches from matched pairs
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from eval.silver.data.schemas import MatchResult, PipelineCaseOutput, SilverCaseResult, SourceCase

logger = logging.getLogger(__name__)


# ── Markdown ───────────────────────────────────────────────────────────────────

def _fmt_silver_finding(i: int, f) -> str:
    lines = [
        f"{i}. **[{f.category}]** {f.claim}",
        f"   - subject: {f.subject_entity or '—'}  |  outcome: {f.outcome_entity or '—'}",
        f"   - relation: `{f.relation_type}`  |  direction: `{f.direction or '—'}`  "
        f"|  confidence: `{f.confidence}`",
        f'   - verbatim: *"{f.verbatim_support}"*',
    ]
    return "\n".join(lines)


def _fmt_pipeline_finding(i: int, f) -> str:
    lines = [
        f"{i}. **[{f.category}]** {f.claim}",
        f"   - subject: {f.subject_entity or '—'}  |  outcome: {f.outcome_entity or '—'}",
        f"   - relation: `{f.relation_type}`  |  direction: `{f.direction or '—'}`  "
        f"|  confidence: `{f.confidence}`",
        f"   - grounding_score: `{f.grounding_score}`",
    ]
    return "\n".join(lines)


def _md_case(
    result: MatchResult,
    silver: SilverCaseResult,
    pipeline: PipelineCaseOutput,
    source: SourceCase | None,
) -> str:
    parts: list[str] = []
    parts.append(f"## Case: `{result.case_id}`\n")

    if source:
        parts.append("### Source text\n")
        parts.append(f"> {source.text}\n")
        parts.append(f"*Section: {source.path_string}*\n")

    parts.append(f"### Silver findings ({len(silver.findings)})\n")
    if silver.findings:
        parts.extend(_fmt_silver_finding(i + 1, f) + "\n"
                     for i, f in enumerate(silver.findings))
    else:
        parts.append("*(none)*\n")

    parts.append(f"### Pipeline findings ({len(pipeline.findings)})\n")
    if pipeline.findings:
        parts.extend(_fmt_pipeline_finding(i + 1, f) + "\n"
                     for i, f in enumerate(pipeline.findings))
    else:
        parts.append("*(none)*\n")

    parts.append(f"### Matched pairs ({len(result.matched)})\n")
    if result.matched:
        header = "| Silver claim | Pipeline claim | Score | Field mismatches |"
        sep = "|---|---|---|---|"
        rows = []
        for p in result.matched:
            mismatches = "; ".join(
                f"{m.field}: `{m.silver}` → `{m.pipeline}`"
                for m in p.field_mismatches
            ) or "—"
            rows.append(
                f"| {p.silver_claim[:80]} | {p.pipeline_claim[:80]} "
                f"| {p.similarity:.3f} | {mismatches} |"
            )
        parts.append("\n".join([header, sep] + rows) + "\n")
    else:
        parts.append("*(none)*\n")

    parts.append(f"### Unmatched silver ({len(result.unmatched_silver)})\n")
    if result.unmatched_silver:
        parts.extend(f"- {c}\n" for c in result.unmatched_silver)
    else:
        parts.append("*(none)*\n")

    parts.append(f"### Unmatched pipeline ({len(result.unmatched_pipeline)})\n")
    if result.unmatched_pipeline:
        parts.extend(f"- {c}\n" for c in result.unmatched_pipeline)
    else:
        parts.append("*(none)*\n")

    return "\n".join(parts)


# ── CSV helpers ────────────────────────────────────────────────────────────────

def _write_matched_pairs_csv(
    match_results: list[MatchResult],
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "case_id", "silver_claim", "pipeline_claim",
            "similarity", "n_field_mismatches", "field_mismatch_detail",
        ])
        writer.writeheader()
        for r in match_results:
            for p in r.matched:
                detail = "; ".join(
                    f"{m.field}:{m.silver}->{m.pipeline}" for m in p.field_mismatches
                )
                writer.writerow({
                    "case_id": r.case_id,
                    "silver_claim": p.silver_claim,
                    "pipeline_claim": p.pipeline_claim,
                    "similarity": round(p.similarity, 4),
                    "n_field_mismatches": len(p.field_mismatches),
                    "field_mismatch_detail": detail,
                })


def _write_unmatched_silver_csv(
    match_results: list[MatchResult],
    silver_by_case: dict[str, SilverCaseResult],
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "case_id", "claim", "category", "relation_type", "direction",
            "subject_entity", "outcome_entity",
        ])
        writer.writeheader()
        unmatched_set: dict[str, set[str]] = {
            r.case_id: set(r.unmatched_silver) for r in match_results
        }
        for case_id, unmatched in unmatched_set.items():
            silver = silver_by_case.get(case_id)
            if not silver:
                continue
            for f in silver.findings:
                if f.claim in unmatched:
                    writer.writerow({
                        "case_id": case_id,
                        "claim": f.claim,
                        "category": f.category,
                        "relation_type": f.relation_type,
                        "direction": f.direction or "",
                        "subject_entity": f.subject_entity or "",
                        "outcome_entity": f.outcome_entity or "",
                    })


def _write_unmatched_pipeline_csv(
    match_results: list[MatchResult],
    pipeline_by_case: dict[str, PipelineCaseOutput],
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "case_id", "claim", "category", "relation_type", "direction",
            "subject_entity", "outcome_entity", "grounding_score",
        ])
        writer.writeheader()
        unmatched_set: dict[str, set[str]] = {
            r.case_id: set(r.unmatched_pipeline) for r in match_results
        }
        for case_id, unmatched in unmatched_set.items():
            pipeline = pipeline_by_case.get(case_id)
            if not pipeline:
                continue
            for f in pipeline.findings:
                if f.claim in unmatched:
                    writer.writerow({
                        "case_id": case_id,
                        "claim": f.claim,
                        "category": f.category,
                        "relation_type": f.relation_type,
                        "direction": f.direction or "",
                        "subject_entity": f.subject_entity or "",
                        "outcome_entity": f.outcome_entity or "",
                        "grounding_score": f.grounding_score,
                    })


def _write_field_mismatches_csv(
    match_results: list[MatchResult],
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "case_id", "silver_claim", "pipeline_claim",
            "similarity", "field", "silver_value", "pipeline_value",
        ])
        writer.writeheader()
        for r in match_results:
            for p in r.matched:
                for m in p.field_mismatches:
                    writer.writerow({
                        "case_id": r.case_id,
                        "silver_claim": p.silver_claim,
                        "pipeline_claim": p.pipeline_claim,
                        "similarity": round(p.similarity, 4),
                        "field": m.field,
                        "silver_value": m.silver or "",
                        "pipeline_value": m.pipeline or "",
                    })


# ── Main entry ─────────────────────────────────────────────────────────────────

def write_inspection_report(
    match_results: list[MatchResult],
    silver_by_case: dict[str, SilverCaseResult],
    pipeline_by_case: dict[str, PipelineCaseOutput],
    source_by_case: dict[str, SourceCase],
    reports_dir: Path,
    timestamp: str,
) -> None:
    """
    Write inspection Markdown + 4 CSVs to reports_dir.
    All filenames include timestamp for traceability.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Markdown
    md_path = reports_dir / f"inspect_{timestamp}.md"
    sections = ["# MAP Silver Eval — Inspection Report\n\n"]
    sections.append(f"Timestamp: `{timestamp}`  |  Cases: {len(match_results)}\n\n---\n\n")
    for result in match_results:
        silver = silver_by_case.get(result.case_id)
        pipeline = pipeline_by_case.get(result.case_id)
        source = source_by_case.get(result.case_id)
        if silver and pipeline:
            sections.append(_md_case(result, silver, pipeline, source))
            sections.append("\n---\n\n")
    md_path.write_text("".join(sections), encoding="utf-8")
    logger.info("Inspection report → %s", md_path)

    # CSVs
    mp_path = reports_dir / f"matched_pairs_{timestamp}.csv"
    _write_matched_pairs_csv(match_results, mp_path)
    logger.info("Matched pairs CSV → %s", mp_path)

    us_path = reports_dir / f"unmatched_silver_{timestamp}.csv"
    _write_unmatched_silver_csv(match_results, silver_by_case, us_path)
    logger.info("Unmatched silver CSV → %s", us_path)

    up_path = reports_dir / f"unmatched_pipeline_{timestamp}.csv"
    _write_unmatched_pipeline_csv(match_results, pipeline_by_case, up_path)
    logger.info("Unmatched pipeline CSV → %s", up_path)

    fm_path = reports_dir / f"field_mismatches_{timestamp}.csv"
    _write_field_mismatches_csv(match_results, fm_path)
    logger.info("Field mismatches CSV → %s", fm_path)
