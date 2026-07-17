"""Export pipeline findings from sum_map_findings for sampled cases."""
from __future__ import annotations

import logging
from pathlib import Path

from .jsonl_utils import read_jsonl, write_jsonl
from .schemas import PipelineCaseOutput, PipelineFinding, SourceCase

logger = logging.getLogger(__name__)


def export_pipeline_outputs(
    source_cases_path: Path,
    output_path: Path,
) -> None:
    """
    For each case in source_cases.jsonl, collect all pipeline findings whose
    evidence_refs contain a ref matching the case's te_id.

    Uses the most recent successful pipeline run per pmcid.
    Writes pipeline_findings.jsonl — one line per case.
    """
    from nlp_histo.database import get_db_connection
    from nlp_histo.database.models import PipelineRun, SumMapFinding

    cases = list(read_jsonl(source_cases_path, SourceCase))
    logger.info("Exporting pipeline findings for %d cases", len(cases))

    db = get_db_connection()

    with db.session_scope() as session:
        # Latest successful run per pmcid
        pmcids = list({c.pmcid for c in cases})
        latest_runs: dict[str, PipelineRun] = {}
        for pmcid in pmcids:
            run = (
                session.query(PipelineRun)
                .filter(PipelineRun.pmcid == pmcid, PipelineRun.status == "success")
                .order_by(PipelineRun.id.desc())
                .first()
            )
            if run:
                latest_runs[pmcid] = run
            else:
                logger.warning("No successful pipeline run found for %s", pmcid)

        results: list[PipelineCaseOutput] = []
        for case in cases:
            run = latest_runs.get(case.pmcid)
            if not run:
                logger.warning("Skipping %s — no pipeline run", case.case_id)
                continue

            # Find all findings whose evidence_refs mention this te_id
            # evidence_refs format: ["S{i}|{pmcid}|{te_id}", ...]
            te_id_str = str(case.te_id)
            all_findings = (
                session.query(SumMapFinding)
                .filter(SumMapFinding.pipeline_run_id == run.id)
                .all()
            )

            matched_findings = []
            for f in all_findings:
                refs = f.evidence_refs or []
                if any(r.split("|")[-1] == te_id_str for r in refs):
                    matched_findings.append(
                        PipelineFinding(
                            pipeline_run_id=run.id,
                            run_id=run.run_id,
                            pmcid=f.pmcid,
                            chunk_id=f.chunk_id,
                            category=f.category,
                            claim=f.claim,
                            subject_entity=f.subject_entity,
                            outcome_entity=f.outcome_entity,
                            relation_type=f.relation_type or "unclear",
                            direction=f.direction,
                            confidence=f.confidence or "medium",
                            verbatim_support=f.verbatim_support or "",
                            grounding_score=f.grounding_score,
                            scope_disease_subtype=f.scope_disease_subtype,
                            scope_cohort_n=f.scope_cohort_n,
                            scope_assay_method=f.scope_assay_method,
                            scope_biomarker_cutoff=f.scope_biomarker_cutoff,
                            scope_tissue_site=f.scope_tissue_site,
                            scope_treatment_context=f.scope_treatment_context,
                            scope_endpoint=f.scope_endpoint,
                            scope_study_design=f.scope_study_design,
                        )
                    )

            results.append(
                PipelineCaseOutput(
                    case_id=case.case_id,
                    pmcid=case.pmcid,
                    te_id=case.te_id,
                    run_id=run.run_id,
                    pipeline_run_id=run.id,
                    findings=matched_findings,
                )
            )
            logger.info("  %s — %d finding(s)", case.case_id, len(matched_findings))

    write_jsonl(output_path, results)
    logger.info("Wrote %d case outputs to %s", len(results), output_path)
