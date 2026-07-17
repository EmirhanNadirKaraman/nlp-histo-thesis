"""
inspect_pipeline_output.py
──────────────────────────
Generate a self-contained HTML inspector from one (or two) pipeline output JSON files.

Usage
-----
Single-run mode:
    PYTHONPATH=. python scripts/inspect/inspect_pipeline_output.py out/summaries/summaries/PMC10047158.json
    PYTHONPATH=. python scripts/inspect/inspect_pipeline_output.py out/summaries/summaries/PMC10047158.json -o out/inspector/PMC10047158.html
    PYTHONPATH=. python scripts/inspect/inspect_pipeline_output.py out/summaries/summaries/PMC10047158.json --export-flagged-csv out/inspector/PMC10047158_flagged.csv

Cross-run diff mode:
    PYTHONPATH=. python scripts/inspect/inspect_pipeline_output.py run_a.json --compare run_b.json
    PYTHONPATH=. python scripts/inspect/inspect_pipeline_output.py run_a.json --compare run_b.json -o out/inspector/PMC10047158_diff.html

Batch mode:
    PYTHONPATH=. python scripts/inspect/inspect_pipeline_output.py --batch-dir out/summaries/summaries/
    PYTHONPATH=. python scripts/inspect/inspect_pipeline_output.py --batch-dir out/summaries/summaries/ -o out/inspector/

The output HTML file(s) can be opened directly in any browser — no server needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

# Bootstrap repo root onto sys.path so `from database import get_db_connection`
# resolves when this script is invoked directly (``scripts/`` has no
# ``__init__.py`` by project convention). Without this the lazy
# DB-backed imports in build_corpus_index_from_db and _lookup_paragraphs
# silently fail with ModuleNotFoundError and the inspector renders without
# DB-derived enrichments — a latent bug uncovered while wiring paragraph
# context display.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    print("ERROR: jinja2 is required.  pip install jinja2", file=sys.stderr)
    sys.exit(1)


# ── Suspicious-entity heuristics ──────────────────────────────────────────────

# Biomedical taxonomy strings that often leak into subject_entity due to UMLS
# normalization overshooting (CEAN → Cetacea, etc.)
_TAXONOMY_RE = re.compile(
    r"\b(cetacea|chiroptera|rodentia|mammalia|reptilia|aves|amphibia|"
    r"actinopterygii|insecta|arachnida|bacteria|fungi|archaea|viridae)\b",
    re.I,
)
# Procedure / technique strings that shouldn't be subject entities
_PROCEDURE_RE = re.compile(
    r"\b(electroretinography|immunohistochemistry|pcr|elisa|western blot|"
    r"flow cytometry|sequencing|radiography|tomography|colonoscopy|endoscopy)\b",
    re.I,
)
# Generic / vacuous entity strings
_GENERIC_RE = re.compile(
    r"^\s*(patient|patients|case|cases|subject|subjects|individual|individuals|"
    r"sample|samples|group|groups|cohort|cohorts|cell|cells|tissue|tissues|"
    r"tumor|tumour|lesion|lesions|the patient)\s*$",
    re.I,
)
# Polarity words that imply the direction shouldn't be "unclear"
_POLARITY_RE = re.compile(
    r"\b(not|no |absent|negative|negated|present|expressed|increased|decreased|"
    r"elevated|reduced|no evidence|lack of|without)\b",
    re.I,
)


def _load_corpus_relations(path: Path) -> dict | None:
    """Load corpus_relations.json and return parsed dict, or None on failure."""
    if not path or not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not load corpus relations from {path}: {exc}", file=sys.stderr)
        return None


def build_corpus_index(corpus_data: dict) -> dict[str, list[dict]]:
    """
    Build a canonical_id → list[relation dict] index for per-paper lookups.
    Only cross-paper relations are indexed (intra-paper are shown in per-paper
    Relations section already).
    """
    index: dict[str, list[dict]] = {}
    for rel in corpus_data.get("relations", []):
        if rel.get("comparison_scope") != "cross_paper":
            continue
        for cid_key in ("rule_id_a", "rule_id_b"):
            cid = rel.get(cid_key)
            if cid:
                index.setdefault(cid, []).append(rel)
    return index


def build_corpus_index_from_db(session) -> dict[str, list[dict]]:
    """
    Build a canonical_id → list[relation dict] index from the DB.

    Reads the most recent corpus_run_id from sum_corpus_relations and indexes
    only cross-paper relations (same logic as build_corpus_index).
    Returns an empty dict if no rows exist.
    """
    try:
        from nlp_histo.database.models import SumCorpusRelation  # noqa: PLC0415

        rows = (
            session.query(SumCorpusRelation)
            .filter_by(comparison_scope="cross_paper")
            .all()
        )

        index: dict[str, list[dict]] = {}
        for row in rows:
            rel = {
                "relation_id":              row.relation_id,
                "comparison_scope":         row.comparison_scope,
                "same_paper":               row.same_paper,
                "rule_id_a":                row.rule_id_a,
                "rule_id_b":                row.rule_id_b,
                "pmcid_a":                  row.pmcid_a,
                "pmcid_b":                  row.pmcid_b,
                "relation_type":            row.relation_type,
                "nli_score_a_to_b":         row.nli_score_a_to_b,
                "nli_score_b_to_a":         row.nli_score_b_to_a,
                "subject_entity":           row.subject_entity,
                "outcome_entity":           row.outcome_entity,
                "category":                 row.category,
                "relation_type_structural": row.relation_type_structural,
                "direction_a":              row.direction_a,
                "direction_b":              row.direction_b,
                "predicate_a":              row.predicate_a,
                "predicate_b":              row.predicate_b,
                "mean_grounding_a":         row.mean_grounding_a,
                "mean_grounding_b":         row.mean_grounding_b,
                "finding_count_a":          row.finding_count_a,
                "finding_count_b":          row.finding_count_b,
                "supporting_pmcids_a":      row.supporting_pmcids_a or [],
                "supporting_pmcids_b":      row.supporting_pmcids_b or [],
                "is_conflicted_a":          row.is_conflicted_a,
                "study_coverage_a":         row.study_coverage_a,
                "is_conflicted_b":          row.is_conflicted_b,
                "study_coverage_b":         row.study_coverage_b,
                "scope_check_result":       row.scope_check_result,
                "scope_note":               row.scope_note,
            }
            for cid_key in ("rule_id_a", "rule_id_b"):
                cid = rel.get(cid_key)
                if cid:
                    index.setdefault(cid, []).append(rel)

        return index

    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not load corpus index from DB: {exc}", file=sys.stderr)
        return {}


def _entity_flags(entity: str | None) -> list[str]:
    """Return warning labels for a suspicious entity string."""
    if not entity:
        return ["empty-entity"]
    flags: list[str] = []
    if _TAXONOMY_RE.search(entity):
        flags.append("taxonomy-leak")
    if _PROCEDURE_RE.search(entity):
        flags.append("procedure-as-entity")
    if _GENERIC_RE.match(entity):
        flags.append("generic-entity")
    return flags


def _compute_flags(obj: dict, low_gs_threshold: float = 0.4) -> list[str]:
    """Return all warning flags for a finding / rule dict."""
    flags: list[str] = []
    flags.extend(_entity_flags(obj.get("subject_entity")))
    outcome = obj.get("outcome_entity") or ""
    if not outcome.strip():
        flags.append("empty-outcome")
    # direction vs. claim polarity mismatch
    direction = obj.get("direction", "")
    claim     = obj.get("claim", "") or obj.get("predicate_text", "")
    if direction == "unclear" and _POLARITY_RE.search(claim):
        flags.append("polarity-mismatch")
    # low grounding score
    gs = obj.get("grounding_score") or obj.get("mean_grounding_score")
    if gs is not None and gs < low_gs_threshold:
        flags.append("low-grounding")
    return flags


def _is_low_grounding(obj: dict, threshold: float = 0.4) -> bool:
    gs = obj.get("grounding_score") or obj.get("mean_grounding_score")
    return gs is not None and gs < threshold


# ── Sentence text lookup ───────────────────────────────────────────────────────

def _build_sentence_lookup(data: dict) -> dict[str, str]:
    """
    Build a mapping from sentence reference token → source text.

    Evidence tokens look like: "S2|PMC10047158|5"
    We mine three sources (without guessing or hallucinating text):
      1. evidence_text_map: direct TextElement.text_content keyed by token
         (only present when data was built from DB via build_data_from_db).
      2. MAP findings: verbatim_support is paired with each evidence token.
      3. rules_provenance.rules[].evidence_chain[]: sentence_id + pmcid +
         text_element_id together identify the token, verbatim is the text.

    Source 1 has highest priority — it is the raw source paragraph, not the
    LLM's extracted quote.  Sources 2 and 3 fill gaps when 1 is absent.
    Only tokens that actually appear in evidence lists are populated.
    """
    lookup: dict[str, str] = {}

    # Source 1: direct TextElement content (DB mode — evidence_text_map)
    for token, text in (data.get("evidence_text_map") or {}).items():
        if token not in lookup and text:
            lookup[token] = text

    audit = data.get("audit_trail") or {}

    # Source 2: MAP finding verbatim_support paired with evidence tokens
    for chunk in audit.get("map_chunks", []):
        for f in chunk.get("findings", []):
            vs = (f.get("verbatim_support") or "").strip()
            if not vs:
                continue
            for ev in f.get("evidence") or []:
                ev = ev.strip()
                if ev and ev not in lookup:
                    lookup[ev] = vs

    # Source 3: rules_provenance evidence_chain verbatim
    rp = audit.get("rules_provenance") or {}
    for rule in rp.get("rules") or []:
        for ec in rule.get("evidence_chain") or []:
            sid = ec.get("sentence_id", "")
            pmcid = ec.get("pmcid", "")
            teid = ec.get("text_element_id")
            verbatim = (ec.get("verbatim") or "").strip()
            if not verbatim:
                continue
            if sid and pmcid and teid is not None:
                token = f"{sid}|{pmcid}|{teid}"
                if token not in lookup:
                    lookup[token] = verbatim

    return lookup


# ── Lineage builder ────────────────────────────────────────────────────────────

def _build_lineage_index(data: dict) -> dict[str, Any]:
    """
    Return a lookup dict:  canonical_id → list[map_finding dicts]

    Strategy:
      1. canonical_rule.member_normal_ids  (direct link when available)
         — we don't store NormalFindings in the JSON, so use as string labels only
      2. Fuzzy match: for each canonical rule, scan all MAP findings for those
         with the same (subject_entity, outcome_entity) — useful when normal_ids
         don't appear verbatim in MAP findings.
    """
    # Build a flat list of MAP findings with chunk_id attached
    map_findings_flat: list[dict] = []
    for chunk in (data.get("audit_trail") or {}).get("map_chunks", []):
        chunk_id = chunk.get("chunk_id", "?")
        for f in chunk.get("findings", []):
            map_findings_flat.append({**f, "chunk_id": chunk_id})

    # Index: (subject_entity_lower, outcome_entity_lower) → [findings]
    by_entity: dict[tuple, list[dict]] = {}
    for f in map_findings_flat:
        key = (
            (f.get("subject_entity") or "").lower().strip(),
            (f.get("outcome_entity") or "").lower().strip(),
        )
        by_entity.setdefault(key, []).append(f)

    lineage: dict[str, dict] = {}
    for cr in data.get("canonical_rules", []):
        cid = cr.get("canonical_id", "")
        subj = (cr.get("subject_entity") or "").lower().strip()
        outc = (cr.get("outcome_entity") or "").lower().strip()

        matched: list[dict] = []
        # Exact (subject, outcome) match
        exact = by_entity.get((subj, outc), [])
        matched.extend(exact)

        # Subject-only fallback if exact gives nothing
        if not matched:
            for (s, o), flist in by_entity.items():
                if s == subj:
                    matched.extend(flist)

        lineage[cid] = {"map_findings": matched[:8]}  # cap to avoid bloat

    return lineage


# ── DB → data dict reconstruction ─────────────────────────────────────────────

def build_data_from_db(pmcid: str, run_id: str, session) -> dict | None:
    """
    Reconstruct the pipeline output data dict from DB rows.

    Returns a dict with the same shape as the JSON files written by runner.py,
    or None if the run_id is not found.  Fields not stored in the DB
    (rules_provenance, master_summary) are omitted — the remaining content
    is sufficient for build_context() to produce a full inspector page.
    """
    from nlp_histo.database.models import (  # noqa: PLC0415
        PipelineRun, SumMapFinding, SumCanonicalRule,
        SumRelation, SumFinalRule, SumNormalFinding, SumNormalFindingSpan,
        TextElement,
    )
    from sqlalchemy.orm import aliased  # noqa: PLC0415

    run: PipelineRun | None = (
        session.query(PipelineRun)
        .filter_by(run_id=run_id, pmcid=pmcid)
        .first()
    )
    if run is None:
        return None

    db_id = run.id

    # ── Final rules (joined with canonical rules for missing fields) ──────────
    CRAlias = aliased(SumCanonicalRule)
    fr_rows = (
        session.query(SumFinalRule, CRAlias)
        .outerjoin(CRAlias, SumFinalRule.canonical_rule_id == CRAlias.id)
        .filter(SumFinalRule.pipeline_run_id == db_id)
        .order_by(SumFinalRule.final_score.desc())
        .all()
    )
    final_rules = []
    for fr, cr in fr_rows:
        final_rules.append({
            "final_id":           fr.final_id,
            "canonical_id":       fr.canonical_id,
            "subject_entity":     fr.subject_entity,
            "outcome_entity":     fr.outcome_entity,
            "relation_type":      fr.relation_type,
            "direction":          fr.direction,
            "predicate_text":     fr.predicate_text,
            "category":           fr.category,
            "final_score":        fr.final_score,
            "support_count":      fr.support_count,
            "contradict_count":   fr.contradict_count,
            "scope_qualify_count": fr.scope_qualify_count,
            "is_contradicted":    fr.is_contradicted,
            "contradicted_by":    list(fr.contradicted_by or []),
            # From SumCanonicalRule (may be None if FK was null)
            "mean_grounding_score": cr.mean_grounding_score if cr else None,
            "finding_count":        cr.finding_count if cr else None,
            "member_normal_ids":    list(cr.member_normal_ids or []) if cr else [],
            "is_conflicted":        cr.is_conflicted if cr else None,
            "study_coverage":       cr.study_coverage if cr else None,
            "supporting_pmcids":    list(cr.pmcids or []) if cr else [],
            "group_id":             cr.group_id if cr else None,
        })

    # ── Canonical rules ───────────────────────────────────────────────────────
    cr_rows = (
        session.query(SumCanonicalRule)
        .filter_by(pipeline_run_id=db_id)
        .order_by(SumCanonicalRule.mean_grounding_score.desc())
        .all()
    )
    canonical_rules = [
        {
            "canonical_id":       cr.canonical_id,
            "group_id":           cr.group_id,
            "subject_entity":     cr.subject_entity,
            "outcome_entity":     cr.outcome_entity,
            "relation_type":      cr.relation_type,
            "direction":          cr.direction,
            "predicate_text":     cr.predicate_text,
            "is_conflicted":      cr.is_conflicted,
            "study_coverage":     cr.study_coverage,
            "category":           cr.category,
            "supporting_pmcids":  list(cr.pmcids or []),
            "member_normal_ids":  list(cr.member_normal_ids or []),
            "mean_grounding_score": cr.mean_grounding_score,
            "finding_count":      cr.finding_count,
        }
        for cr in cr_rows
    ]

    # ── Relations ─────────────────────────────────────────────────────────────
    rel_rows = (
        session.query(SumRelation)
        .filter_by(pipeline_run_id=db_id)
        .all()
    )
    relations = [
        {
            "rule_id_a":       r.rule_id_a,
            "rule_id_b":       r.rule_id_b,
            "relation_type":   r.relation_type,
            "nli_score_a_to_b": r.nli_score_a_to_b,
            "nli_score_b_to_a": r.nli_score_b_to_a,
        }
        for r in rel_rows
    ]

    # ── MAP findings (grouped into chunks) ────────────────────────────────────
    mf_rows = (
        session.query(SumMapFinding)
        .filter_by(pipeline_run_id=db_id)
        .order_by(SumMapFinding.chunk_id, SumMapFinding.position_in_chunk)
        .all()
    )
    chunks_by_id: dict[str, list[dict]] = {}
    for mf in mf_rows:
        finding = {
            "category":        mf.category,
            "claim":           mf.claim,
            "confidence":      mf.confidence,
            "verbatim_support": mf.verbatim_support,
            "subject_entity":  mf.subject_entity,
            "outcome_entity":  mf.outcome_entity,
            "relation_type":   mf.relation_type,
            "direction":       mf.direction,
            "grounding_score": mf.grounding_score,
            "evidence":        list(mf.evidence_refs or []),
            "scope": {
                "disease_subtype":  mf.scope_disease_subtype,
                "cohort_n":         mf.scope_cohort_n,
                "assay_method":     mf.scope_assay_method,
                "biomarker_cutoff": mf.scope_biomarker_cutoff,
                "tissue_site":      mf.scope_tissue_site,
                "treatment_context": mf.scope_treatment_context,
                "endpoint":         mf.scope_endpoint,
                "study_design":     mf.scope_study_design,
            },
        }
        chunks_by_id.setdefault(mf.chunk_id, []).append(finding)

    map_chunks = [
        {"chunk_id": chunk_id, "findings": findings}
        for chunk_id, findings in sorted(
            chunks_by_id.items(),
            key=lambda kv: int(kv[0][1:]) if kv[0][1:].isdigit() else kv[0],
        )
    ]

    # ── Normal findings + spans ───────────────────────────────────────────────
    nf_rows = (
        session.query(SumNormalFinding)
        .filter_by(pipeline_run_id=db_id)
        .all()
    )
    nf_ids = [nf.id for nf in nf_rows]
    span_rows = (
        session.query(SumNormalFindingSpan)
        .filter(SumNormalFindingSpan.normal_finding_id.in_(nf_ids))
        .all()
    ) if nf_ids else []

    # ── Collect all text_element_ids for a single batch query ─────────────
    # From normal finding spans
    te_ids_spans = {sp.text_element_id for sp in span_rows if sp.text_element_id}

    # From MAP finding evidence tokens (format: S{sid}|{pmcid}|{te_id})
    ev_token_to_te_id: dict[str, int] = {}
    for mf in mf_rows:
        for tok in (mf.evidence_refs or []):
            parts = tok.split("|")
            if len(parts) == 3 and parts[2].isdigit():
                ev_token_to_te_id[tok] = int(parts[2])

    all_te_ids = te_ids_spans | set(ev_token_to_te_id.values())
    te_content: dict[int, str] = {}
    if all_te_ids:
        te_rows = session.query(TextElement.id, TextElement.text_content).filter(
            TextElement.id.in_(all_te_ids)
        ).all()
        te_content = {row.id: row.text_content for row in te_rows}

    # evidence_text_map: token → raw paragraph from source TextElement
    evidence_text_map: dict[str, str] = {}
    for tok, tid in ev_token_to_te_id.items():
        text = te_content.get(tid, "")
        if text:
            evidence_text_map[tok] = text

    spans_by_nf_id: dict[int, list[dict]] = {}
    for sp in span_rows:
        spans_by_nf_id.setdefault(sp.normal_finding_id, []).append({
            "verbatim":         sp.verbatim,
            "sentence_id":      sp.sentence_id,
            "text_element_id":  sp.text_element_id,
            "paragraph":        te_content.get(sp.text_element_id, "") if sp.text_element_id else "",
        })

    normal_findings_lookup: dict[str, dict] = {}
    for nf in nf_rows:
        normal_findings_lookup[nf.normal_id] = {
            "predicate_text": nf.predicate_text,
            "spans": spans_by_nf_id.get(nf.id, []),
        }

    return {
        "pmcid":   pmcid,
        "run_id":  run.run_id,
        "status":  run.status,
        "summary": run.narrative_summary or "",
        "final_rules":      final_rules,
        "canonical_rules":  canonical_rules,
        "relations":        relations,
        "audit_trail": {
            "map_chunks": map_chunks,
        },
        "normal_findings_lookup": normal_findings_lookup,
        "evidence_text_map":      evidence_text_map,
    }


def build_context_from_db(
    pmcid: str,
    run_id: str,
    session,
    low_gs_threshold: float = 0.4,
    corpus_connections: dict[str, list[dict]] | None = None,
) -> dict | None:
    """
    Build the template context dict directly from DB rows.

    Returns None if the run is not found.  Otherwise identical to
    build_context() called on the equivalent JSON file.
    """
    data = build_data_from_db(pmcid, run_id, session)
    if data is None:
        return None
    return build_context(data, low_gs_threshold=low_gs_threshold,
                         corpus_connections=corpus_connections)


# ── Template context builders ──────────────────────────────────────────────────

def _enrich_final_rule(
    fr: dict,
    lineage_index: dict,
    low_gs_threshold: float = 0.4,
    normal_findings_lookup: dict | None = None,
    paragraph_by_canonical_id: dict[str, str | None] | None = None,
) -> dict:
    cid = fr.get("canonical_id", "")
    lin = dict(lineage_index.get(cid, {}))
    # Attach source spans for each member NormalFinding
    if normal_findings_lookup:
        normal_with_spans = []
        for nid in (fr.get("member_normal_ids") or []):
            nf_info = normal_findings_lookup.get(nid, {})
            normal_with_spans.append({
                "normal_id": nid,
                "predicate_text": nf_info.get("predicate_text", ""),
                "spans": nf_info.get("spans", []),
            })
        lin["normal_findings"] = normal_with_spans
    paragraph = (paragraph_by_canonical_id or {}).get(cid)
    return {
        **fr,
        "flags": _compute_flags(fr, low_gs_threshold),
        "low_grounding": _is_low_grounding(fr, low_gs_threshold),
        "lineage": lin,
        "raw_json": json.dumps(fr, indent=2, ensure_ascii=False),
        # Same NLI-input projections used on canonical rules — final_rules
        # share the predicate_text + scope + representative_verbatim shape,
        # so the relate-audit story is consistent across both tables.
        "nli_input_scope_predicate": _build_nli_input_text(
            fr, scope_aware=True, with_verbatim=False,
        ),
        "nli_input_scope_verbatim":  _build_nli_input_text(
            fr, scope_aware=True, with_verbatim=True,
        ),
        # Full paragraph from text_elements.text_content — DB-resolved via
        # representative_text_element_id (see _lookup_paragraphs).
        "representative_paragraph": paragraph,
    }


# ── Paragraph context lookup (DB-backed, best-effort) ────────────────────────
#
# The rule card wants to surface the *full paragraph* containing the
# representative verbatim sentence — that's ``text_elements.text_content``
# joined by ``CanonicalRule.representative_text_element_id``. We do this
# once per ``build_context`` invocation (batched + dedup'd) so the HTML
# only pays one DB round-trip per paper instead of N per rule.
#
# Best-effort: if the DB connection fails, or if the text_element_id rows
# are missing, the dict is empty / partial and the rule cards silently
# render without a "full paragraph" block. Verbatim is already on the
# rule itself, so the audit story still works without DB access.


def _lookup_paragraphs(rules: list[dict]) -> dict[str, str | None]:
    """Resolve representative_text_element_id → full paragraph text.

    Returns ``{canonical_id → paragraph text}``; missing entries (DB
    unavailable, pointer is None, row not found) silently absent so
    callers can ``dict.get(canonical_id) or ""``.
    """
    te_ids = {
        int(r.get("representative_text_element_id"))
        for r in rules
        if r.get("representative_text_element_id") is not None
    }
    if not te_ids:
        return {}
    try:
        from nlp_histo.database import get_db_connection  # noqa: PLC0415
        from sqlalchemy import text  # noqa: PLC0415
        db = get_db_connection()
    except Exception as exc:  # noqa: BLE001 — DB missing is non-fatal
        print(
            f"WARNING: paragraph lookup skipped (DB unavailable): {exc}",
            file=sys.stderr,
        )
        return {}

    try:
        with db.session_scope() as session:
            rows = session.execute(
                text(
                    "SELECT id, text_content FROM text_elements "
                    "WHERE id = ANY(:ids)"
                ),
                {"ids": list(te_ids)},
            ).fetchall()
            by_id: dict[int, str] = {row.id: row.text_content for row in rows}
    except Exception as exc:  # noqa: BLE001
        print(
            f"WARNING: paragraph lookup query failed: {exc}",
            file=sys.stderr,
        )
        return {}

    out: dict[str, str | None] = {}
    for r in rules:
        cid = r.get("canonical_id")
        te_id = r.get("representative_text_element_id")
        if cid and te_id is not None:
            out[cid] = by_id.get(int(te_id))
    return out


# ── NLI input reconstruction ──────────────────────────────────────────────────
#
# Mirrors ``pipeline/stages/knowledge_extraction/stages/relate_stage.py::
# _build_nli_text`` so the inspector can show the *exact text* that the
# RELATE / corpus_relate NLI saw for each rule. The full pipeline records
# ``predicate_text`` on the rule (always) plus the newer ``scope`` and
# ``representative_verbatim`` fields (when present); this helper combines
# them into the same shape the stage feeds the NLI model.
#
# Default rendering policy mirrors ``RelateConfig`` defaults:
#   scope_aware_nli=True   → prepend "[scope: ...]" when scope has any
#                            non-None field
#   use_verbatim_for_nli=False → keep the abstracted predicate_text
#
# A second variant (``with_verbatim=True``) is also exposed so the HTML
# can show what NLI would have seen if the verbatim path were enabled.

_SCOPE_FIELD_LABELS: list[tuple[str, str]] = [
    ("disease",   "disease_subtype"),
    ("tissue",    "tissue_site"),
    ("assay",     "assay_method"),
    ("treatment", "treatment_context"),
    ("endpoint",  "endpoint"),
    ("design",    "study_design"),
    ("cutoff",    "biomarker_cutoff"),
    ("cohort_n",  "cohort_n"),
]


def _build_nli_input_text(
    rule: dict, *, scope_aware: bool = True, with_verbatim: bool = False,
) -> str:
    """Construct the NLI input text for one canonical rule, mirroring
    ``RelateStage._build_nli_text``.

    Returns an empty string when neither predicate nor verbatim is set.
    """
    predicate = (rule.get("predicate_text") or "").strip()
    verbatim  = (rule.get("representative_verbatim") or "").strip()

    base = verbatim if (with_verbatim and verbatim) else predicate
    if not base:
        return ""

    if not scope_aware:
        return base

    scope = rule.get("scope") or {}
    if not isinstance(scope, dict):
        return base

    parts: list[str] = []
    for label, field in _SCOPE_FIELD_LABELS:
        value = scope.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            parts.append(f"{label}={text}")
    if not parts:
        return base
    return f"[scope: {' | '.join(parts)}] {base}"


def _enrich_canonical_rule(
    cr: dict,
    low_gs_threshold: float = 0.4,
    paragraph_by_canonical_id: dict[str, str | None] | None = None,
) -> dict:
    cid = cr.get("canonical_id", "")
    paragraph = (paragraph_by_canonical_id or {}).get(cid)
    return {
        **cr,
        "flags": _compute_flags(cr, low_gs_threshold),
        "low_grounding": _is_low_grounding(cr, low_gs_threshold),
        "raw_json": json.dumps(cr, indent=2, ensure_ascii=False),
        # NLI-input text the RELATE stage would build for this rule under
        # the current RelateConfig defaults (scope-aware predicate). The
        # *_verbatim variant exposes what NLI would see if the verbatim
        # path were enabled — useful for thesis-grade "what changes" audit.
        "nli_input_scope_predicate": _build_nli_input_text(
            cr, scope_aware=True, with_verbatim=False,
        ),
        "nli_input_scope_verbatim":  _build_nli_input_text(
            cr, scope_aware=True, with_verbatim=True,
        ),
        # Full paragraph from text_elements.text_content — DB-resolved via
        # representative_text_element_id (see _lookup_paragraphs). None if
        # the rule pre-dates the field or the DB is unavailable; template
        # hides the row in that case.
        "representative_paragraph": paragraph,
    }


def _enrich_finding(f: dict, low_gs_threshold: float = 0.4) -> dict:
    return {
        **f,
        "flags": _compute_flags(f, low_gs_threshold),
        "low_grounding": _is_low_grounding(f, low_gs_threshold),
        "raw_json": json.dumps(f, indent=2, ensure_ascii=False),
    }


def _unique_sorted(values: list[str | None]) -> list[str]:
    return sorted({v for v in values if v})


def build_context(
    data: dict,
    low_gs_threshold: float = 0.4,
    corpus_connections: dict[str, list[dict]] | None = None,
) -> dict:
    pmcid   = data.get("pmcid", "unknown")
    run_id  = data.get("run_id", "")
    status  = data.get("status", "unknown")
    summary = data.get("summary", "")

    lineage_index = _build_lineage_index(data)
    sentence_lookup = _build_sentence_lookup(data)
    normal_findings_lookup = data.get("normal_findings_lookup") or {}

    raw_final_rules    = data.get("final_rules", []) or []
    raw_canonical_rules = data.get("canonical_rules", []) or []
    raw_relations       = data.get("relations", []) or []

    audit = data.get("audit_trail") or {}
    raw_chunks = audit.get("map_chunks", []) or []

    # Resolve representative_text_element_id → full paragraph text for the
    # rule cards. The pointer was baked into CanonicalRule at canonicalize
    # time (see paragraph_lookup.py); we look up the actual content from
    # the text_elements table here so the HTML can show "verbatim source"
    # alongside "full paragraph context". DB-best-effort: a missing DB
    # connection or a missing text_element_id silently leaves the field
    # blank (the verbatim sentence already covers most audit needs).
    paragraph_by_canonical_id: dict[str, str | None] = _lookup_paragraphs(
        raw_canonical_rules + raw_final_rules,
    )

    # Enrich each layer
    final_rules     = [
        _enrich_final_rule(
            fr, lineage_index, low_gs_threshold, normal_findings_lookup,
            paragraph_by_canonical_id=paragraph_by_canonical_id,
        )
        for fr in raw_final_rules
    ]
    canonical_rules = [
        _enrich_canonical_rule(
            cr, low_gs_threshold,
            paragraph_by_canonical_id=paragraph_by_canonical_id,
        )
        for cr in raw_canonical_rules
    ]

    # Inject cross-paper connections from corpus_relations.json (if provided)
    cross_paper_total = 0
    for cr in canonical_rules:
        cid = cr.get("canonical_id", "")
        conns = (corpus_connections or {}).get(cid, [])
        cr["corpus_connections"] = conns
        cross_paper_total += len(conns)

    enriched_chunks = []
    all_map_findings: list[dict] = []
    for chunk in raw_chunks:
        findings = [_enrich_finding(f, low_gs_threshold) for f in (chunk.get("findings") or [])]
        all_map_findings.extend(findings)
        enriched_chunks.append({**chunk, "findings": findings})

    total_map_findings = len(all_map_findings)
    flagged_count      = sum(1 for fr in final_rules if fr["flags"])
    contradicted_count = sum(1 for fr in final_rules if fr.get("is_contradicted"))
    low_grounding_count = sum(1 for fr in final_rules if fr["low_grounding"])

    # Category / relation_type option lists for filter dropdowns
    fr_categories      = _unique_sorted([fr.get("category") for fr in final_rules])
    fr_relation_types  = _unique_sorted([fr.get("relation_type") for fr in final_rules])
    cr_categories      = _unique_sorted([cr.get("category") for cr in canonical_rules])
    cr_relation_types  = _unique_sorted([cr.get("relation_type") for cr in canonical_rules])
    mf_categories      = _unique_sorted([f.get("category") for f in all_map_findings])
    mf_relation_types  = _unique_sorted([f.get("relation_type") for f in all_map_findings])

    # Sentence lookup serialised for embedding in HTML as JS object
    sentence_lookup_json = json.dumps(sentence_lookup, ensure_ascii=False)

    return {
        "pmcid":               pmcid,
        "run_id":              run_id,
        "status":              status,
        "summary":             summary,
        "final_rules":         final_rules,
        "canonical_rules":     canonical_rules,
        "relations":           raw_relations,
        "map_chunks":          enriched_chunks,
        "total_map_findings":  total_map_findings,
        "flagged_count":       flagged_count,
        "contradicted_count":  contradicted_count,
        "low_grounding_count": low_grounding_count,
        "fr_categories":       fr_categories,
        "fr_relation_types":   fr_relation_types,
        "cr_categories":       cr_categories,
        "cr_relation_types":   cr_relation_types,
        "mf_categories":       mf_categories,
        "mf_relation_types":   mf_relation_types,
        "sentence_lookup_json": sentence_lookup_json,
        "cross_paper_total":   cross_paper_total,
    }


# ── Cross-run diff helpers ────────────────────────────────────────────────────

def _stable_key(rule: dict) -> tuple:
    """Deterministic match key for a final rule or canonical rule."""
    return (
        (rule.get("category") or "").strip().lower(),
        (rule.get("relation_type") or "").strip().lower(),
        (rule.get("subject_entity") or "").strip().lower(),
        (rule.get("outcome_entity") or "").strip().lower(),
        (rule.get("predicate_text") or "").strip().lower(),
    )


def _relation_key(rel: dict) -> tuple:
    return (
        (rel.get("relation_type") or ""),
        (rel.get("rule_id_a") or ""),
        (rel.get("rule_id_b") or ""),
    )


def _finding_key(f: dict) -> tuple:
    return (
        (f.get("category") or "").strip().lower(),
        (f.get("relation_type") or "").strip().lower(),
        (f.get("subject_entity") or "").strip().lower(),
        (f.get("outcome_entity") or "").strip().lower(),
        (f.get("claim") or "").strip().lower(),
    )


_RULE_DIFF_FIELDS = [
    "final_score", "support_count", "contradict_count",
    "is_contradicted", "mean_grounding_score", "direction",
    "category", "relation_type",
]

_CANONICAL_DIFF_FIELDS = [
    "direction", "category", "relation_type",
    "mean_grounding_score", "finding_count", "is_conflicted", "study_coverage",
]


def _diff_items(
    a_list: list[dict],
    b_list: list[dict],
    key_fn,
    diff_fields: list[str],
) -> dict:
    """
    Compute additions, removals, and changes between two lists of dicts.

    Returns:
      {
        "added":   [items in b not in a],
        "removed": [items in a not in b],
        "changed": [{"a": ..., "b": ..., "field_diffs": {field: (val_a, val_b)}}],
      }
    """
    a_by_key: dict[tuple, dict] = {}
    for item in a_list:
        k = key_fn(item)
        a_by_key[k] = item

    b_by_key: dict[tuple, dict] = {}
    for item in b_list:
        k = key_fn(item)
        b_by_key[k] = item

    a_keys = set(a_by_key)
    b_keys = set(b_by_key)

    added   = [b_by_key[k] for k in sorted(b_keys - a_keys, key=str)]
    removed = [a_by_key[k] for k in sorted(a_keys - b_keys, key=str)]

    changed = []
    for k in sorted(a_keys & b_keys, key=str):
        ia, ib = a_by_key[k], b_by_key[k]
        field_diffs: dict[str, tuple] = {}
        for field in diff_fields:
            va, vb = ia.get(field), ib.get(field)
            if va != vb:
                field_diffs[field] = (va, vb)
        if field_diffs:
            changed.append({
                "a": ia,
                "b": ib,
                "field_diffs": field_diffs,
                "field_diffs_json": json.dumps(field_diffs, ensure_ascii=False),
            })

    return {"added": added, "removed": removed, "changed": changed}


def build_diff_context(
    data_a: dict,
    data_b: dict,
    low_gs_threshold: float = 0.4,
) -> dict:
    """Build the template context for cross-run diff view."""
    pmcid   = data_a.get("pmcid", "unknown")
    run_a   = data_a.get("run_id") or "run-A"
    run_b   = data_b.get("run_id") or "run-B"

    # Diffs for each section
    final_diff     = _diff_items(
        data_a.get("final_rules", []) or [],
        data_b.get("final_rules", []) or [],
        _stable_key, _RULE_DIFF_FIELDS,
    )
    canonical_diff = _diff_items(
        data_a.get("canonical_rules", []) or [],
        data_b.get("canonical_rules", []) or [],
        _stable_key, _CANONICAL_DIFF_FIELDS,
    )
    relations_diff = _diff_items(
        data_a.get("relations", []) or [],
        data_b.get("relations", []) or [],
        _relation_key, ["nli_score_a_to_b", "nli_score_b_to_a"],
    )

    # MAP findings flat
    def _flat_findings(data: dict) -> list[dict]:
        out = []
        for chunk in (data.get("audit_trail") or {}).get("map_chunks", []):
            for f in chunk.get("findings", []):
                out.append(f)
        return out

    findings_diff = _diff_items(
        _flat_findings(data_a),
        _flat_findings(data_b),
        _finding_key, ["grounding_score", "direction"],
    )

    # Summary stats
    def _counts(data: dict) -> dict:
        final = data.get("final_rules", []) or []
        flagged = sum(1 for fr in final if _compute_flags(fr, low_gs_threshold))
        contradicted = sum(1 for fr in final if fr.get("is_contradicted"))
        findings_count = sum(
            len(chunk.get("findings", []))
            for chunk in (data.get("audit_trail") or {}).get("map_chunks", [])
        )
        return {
            "final_rules": len(final),
            "canonical_rules": len(data.get("canonical_rules", []) or []),
            "relations": len(data.get("relations", []) or []),
            "map_findings": findings_count,
            "flagged": flagged,
            "contradicted": contradicted,
            "status": data.get("status", "unknown"),
        }

    # Diff summary numbers
    diff_summary = {
        "final_added":    len(final_diff["added"]),
        "final_removed":  len(final_diff["removed"]),
        "final_changed":  len(final_diff["changed"]),
        "rel_added":      len(relations_diff["added"]),
        "rel_removed":    len(relations_diff["removed"]),
        "rel_changed":    len(relations_diff["changed"]),
        "findings_added":   len(findings_diff["added"]),
        "findings_removed": len(findings_diff["removed"]),
        "findings_changed": len(findings_diff["changed"]),
    }

    # Sentence lookup (merge both runs)
    sent_a = _build_sentence_lookup(data_a)
    sent_b = _build_sentence_lookup(data_b)
    sentence_lookup = {**sent_a, **sent_b}  # b overrides a on collision

    return {
        "pmcid":              pmcid,
        "run_a":              run_a,
        "run_b":              run_b,
        "stats_a":            _counts(data_a),
        "stats_b":            _counts(data_b),
        "diff_summary":       diff_summary,
        "final_diff":         final_diff,
        "canonical_diff":     canonical_diff,
        "relations_diff":     relations_diff,
        "findings_diff":      findings_diff,
        "sentence_lookup_json": json.dumps(sentence_lookup, ensure_ascii=False),
    }


# ── Flagged CSV export ────────────────────────────────────────────────────────

_CSV_COLUMNS = [
    "pmcid",
    "run_id",
    "stage",
    "item_type",
    "item_id",
    "stable_key",
    "category",
    "relation_type",
    "subject_entity",
    "outcome_entity",
    "predicate_text",
    "direction",
    "grounding_score",
    "mean_grounding_score",
    "final_score",
    "flags",
    "evidence_refs",
    "verbatim_summary",
]


def _rule_row(
    item: dict,
    pmcid: str,
    run_id: str,
    stage: str,
    item_type: str,
    flags: list[str],
    low_gs_threshold: float,
) -> dict:
    key_parts = _stable_key(item)
    stable_key = "|".join(str(p) for p in key_parts)

    # Evidence from lineage MAP findings if available
    evidence_refs: list[str] = []
    verbatim_parts: list[str] = []
    lin = item.get("lineage") or {}
    for mf in lin.get("map_findings") or []:
        for ev in mf.get("evidence") or []:
            evidence_refs.append(ev)
        vs = (mf.get("verbatim_support") or "").strip()
        if vs and vs not in verbatim_parts:
            verbatim_parts.append(vs[:120])

    gs = item.get("grounding_score")
    mgs = item.get("mean_grounding_score")
    fs = item.get("final_score")

    return {
        "pmcid": pmcid,
        "run_id": run_id,
        "stage": stage,
        "item_type": item_type,
        "item_id": item.get("final_id") or item.get("canonical_id") or item.get("group_id") or "",
        "stable_key": stable_key,
        "category": item.get("category") or "",
        "relation_type": item.get("relation_type") or "",
        "subject_entity": item.get("subject_entity") or "",
        "outcome_entity": item.get("outcome_entity") or "",
        "predicate_text": (item.get("predicate_text") or item.get("claim") or "").replace("\n", " "),
        "direction": item.get("direction") or "",
        "grounding_score": f"{gs:.4f}" if gs is not None else "",
        "mean_grounding_score": f"{mgs:.4f}" if mgs is not None else "",
        "final_score": f"{fs:.4f}" if fs is not None else "",
        "flags": "|".join(flags),
        "evidence_refs": "|".join(evidence_refs),
        "verbatim_summary": " // ".join(verbatim_parts)[:300],
    }


def export_flagged_csv(
    ctx: dict,
    data: dict,
    output_path: Path,
    low_gs_threshold: float = 0.4,
) -> int:
    """
    Write a CSV of flagged items to output_path.
    Returns number of rows written.
    """
    pmcid  = ctx["pmcid"]
    run_id = ctx["run_id"]
    rows: list[dict] = []

    # Final rules with flags
    for fr in ctx.get("final_rules", []):
        flags = fr.get("flags") or []
        if flags:
            rows.append(_rule_row(fr, pmcid, run_id, "FINAL", "final_rule", flags, low_gs_threshold))

    # Canonical rules with flags (only those not already covered by a final rule with same key)
    final_keys = {_stable_key(fr) for fr in ctx.get("final_rules", [])}
    for cr in ctx.get("canonical_rules", []):
        flags = cr.get("flags") or []
        if flags and _stable_key(cr) not in final_keys:
            rows.append(_rule_row(cr, pmcid, run_id, "CANONICAL", "canonical_rule", flags, low_gs_threshold))

    # MAP findings with flags
    for chunk in ctx.get("map_chunks", []):
        for f in chunk.get("findings", []):
            flags = f.get("flags") or []
            if flags:
                row = _rule_row(f, pmcid, run_id, "MAP", "map_finding", flags, low_gs_threshold)
                # MAP findings have direct evidence refs
                evs = [ev.strip() for ev in (f.get("evidence") or [])]
                row["evidence_refs"] = "|".join(evs)
                vs = (f.get("verbatim_support") or "").strip()
                row["verbatim_summary"] = vs[:300] if vs else ""
                rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


# ── Renderer ───────────────────────────────────────────────────────────────────

def _make_env() -> Environment:
    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,  # we control the data; raw JSON is already escaped in <pre>
    )
    # Custom filter: format float with fallback
    env.filters["format_score"] = lambda v, fmt=".3f": format(v, fmt) if v is not None else "—"
    return env


def render(
    json_path: Path,
    output_path: Path,
    low_gs_threshold: float = 0.4,
    export_csv_path: Path | None = None,
    corpus_connections: dict[str, list[dict]] | None = None,
) -> dict:
    """Render single-run inspector. Returns context dict."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    ctx = build_context(data, low_gs_threshold, corpus_connections=corpus_connections)

    env = _make_env()
    template = env.get_template("pipeline_inspector.html.jinja2")
    html = template.render(**ctx, diff_mode=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Inspector written to: {output_path}")
    print(f"  {ctx['total_map_findings']} MAP findings  |  "
          f"{len(ctx['canonical_rules'])} canonical rules  |  "
          f"{len(ctx['final_rules'])} final rules  |  "
          f"{ctx['flagged_count']} flagged  |  "
          f"{ctx['contradicted_count']} contradicted")

    if export_csv_path is not None:
        n = export_flagged_csv(ctx, data, export_csv_path, low_gs_threshold)
        print(f"Flagged CSV written to: {export_csv_path}  ({n} rows)")

    return ctx


def render_diff(
    json_path_a: Path,
    json_path_b: Path,
    output_path: Path,
    low_gs_threshold: float = 0.4,
) -> None:
    """Render cross-run diff view."""
    with open(json_path_a, encoding="utf-8") as f:
        data_a = json.load(f)
    with open(json_path_b, encoding="utf-8") as f:
        data_b = json.load(f)

    pmcid_a = data_a.get("pmcid", "")
    pmcid_b = data_b.get("pmcid", "")
    if pmcid_a != pmcid_b:
        print(
            f"ERROR: PMCID mismatch — run A is '{pmcid_a}', run B is '{pmcid_b}'. "
            "Cross-run diff only works for the same PMCID.",
            file=sys.stderr,
        )
        sys.exit(1)

    ctx = build_diff_context(data_a, data_b, low_gs_threshold)

    env = _make_env()
    template = env.get_template("pipeline_diff.html.jinja2")
    html = template.render(**ctx)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Diff inspector written to: {output_path}")
    ds = ctx["diff_summary"]
    print(
        f"  Final rules: +{ds['final_added']} -{ds['final_removed']} ~{ds['final_changed']}  |  "
        f"Relations: +{ds['rel_added']} -{ds['rel_removed']} ~{ds['rel_changed']}  |  "
        f"Findings: +{ds['findings_added']} -{ds['findings_removed']} ~{ds['findings_changed']}"
    )


# ── Batch mode ────────────────────────────────────────────────────────────────

def render_batch(
    source_dir: Path,
    output_dir: Path,
    low_gs_threshold: float = 0.4,
    corpus_relations_path: Path | None = None,
) -> None:
    """Generate one inspector HTML per JSON in source_dir, plus an index page."""
    json_files = sorted(source_dir.glob("*.json"))
    if not json_files:
        print(f"WARNING: No JSON files found in {source_dir}", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Auto-detect corpus_relations.json ─────────────────────────────────────
    corpus_data: dict | None = None
    corpus_index: dict[str, list[dict]] = {}
    if corpus_relations_path is None:
        for candidate in [
            source_dir.parent / "corpus_relations.json",
            source_dir / "corpus_relations.json",
        ]:
            if candidate.exists():
                corpus_relations_path = candidate
                break
    if corpus_relations_path:
        corpus_data = _load_corpus_relations(corpus_relations_path)
        if corpus_data:
            corpus_index = build_corpus_index(corpus_data)
            print(
                f"Corpus relations loaded: "
                f"{corpus_data.get('total_relations', 0)} relations "
                f"({corpus_data.get('cross_paper_count', 0)} cross-paper, "
                f"{corpus_data.get('intra_paper_count', 0)} intra-paper) "
                f"from {corpus_relations_path}"
            )

    index_rows: list[dict] = []

    # Build a global canonical_id → enrichment map across all 15 papers so the
    # batch-index corpus-relations table can show scope tags, NLI input
    # strings, verbatim quotes, and paragraph context for *both sides* of a
    # cross-paper pair — not just the local one. Populated incrementally
    # below as each paper's context is built.
    global_rule_lookup: dict[str, dict] = {}

    for json_path in json_files:
        try:
            with open(json_path, encoding="utf-8") as fh:
                data = json.load(fh)

            pmcid = data.get("pmcid") or json_path.stem
            ctx = build_context(data, low_gs_threshold, corpus_connections=corpus_index)

            # Snapshot the audit-relevant fields per canonical rule for the
            # cross-paper lookup. We do this BEFORE writing HTML so any
            # later paper's index render can reference earlier papers'
            # rules (index render happens once at the end, after all
            # papers have streamed through).
            for cr in ctx["canonical_rules"]:
                cid = cr.get("canonical_id")
                if not cid:
                    continue
                global_rule_lookup[cid] = {
                    "scope":                          cr.get("scope") or {},
                    "representative_verbatim":        cr.get("representative_verbatim"),
                    "representative_paragraph":       cr.get("representative_paragraph"),
                    "representative_text_element_id": cr.get("representative_text_element_id"),
                    "nli_input_scope_predicate":      cr.get("nli_input_scope_predicate"),
                    "nli_input_scope_verbatim":       cr.get("nli_input_scope_verbatim"),
                }

            out_html = output_dir / f"{json_path.stem}.html"
            env = _make_env()
            template = env.get_template("pipeline_inspector.html.jinja2")
            html = template.render(**ctx, diff_mode=False)
            out_html.write_text(html, encoding="utf-8")

            # Collect warning badge summary
            warnings: list[str] = []
            if ctx["flagged_count"]:
                warnings.append(f"{ctx['flagged_count']} flagged")
            if ctx["contradicted_count"]:
                warnings.append(f"{ctx['contradicted_count']} contradicted")
            if ctx["low_grounding_count"]:
                warnings.append(f"{ctx['low_grounding_count']} low-gs")

            index_rows.append({
                "pmcid":        pmcid,
                "run_id":       ctx.get("run_id", ""),
                "html_path":    out_html.name,
                "status":       ctx.get("status", ""),
                "map_findings": ctx["total_map_findings"],
                "canonical":    len(ctx["canonical_rules"]),
                "final":        len(ctx["final_rules"]),
                "relations":    len(ctx["relations"]),
                "flagged":      ctx["flagged_count"],
                "contradicted": ctx["contradicted_count"],
                "low_gs":       ctx["low_grounding_count"],
                "warnings":     " · ".join(warnings),
            })
            print(f"  ✓ {pmcid}  →  {out_html.name}")

        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: skipping {json_path.name}: {exc}", file=sys.stderr)
            continue

    # Enrich corpus_data.relations with per-side scope / verbatim / paragraph
    # / NLI-input pulled from the global rule lookup (built per-paper above).
    # Lets the batch-index template render the same audit context the
    # per-paper inspector shows, on both sides of every cross-paper pair.
    if corpus_data:
        enriched_rels = []
        for rel in corpus_data.get("relations", []) or []:
            r = dict(rel)
            a = global_rule_lookup.get(rel.get("rule_id_a") or "", {})
            b = global_rule_lookup.get(rel.get("rule_id_b") or "", {})
            for side_letter, side in (("a", a), ("b", b)):
                r[f"scope_{side_letter}"]                    = side.get("scope") or {}
                r[f"verbatim_{side_letter}"]                 = side.get("representative_verbatim")
                r[f"paragraph_{side_letter}"]                = side.get("representative_paragraph")
                r[f"text_element_id_{side_letter}"]          = side.get("representative_text_element_id")
                r[f"nli_input_{side_letter}"]                = side.get("nli_input_scope_predicate")
                r[f"nli_input_verbatim_{side_letter}"]       = side.get("nli_input_scope_verbatim")
            enriched_rels.append(r)
        corpus_data = {**corpus_data, "relations": enriched_rels}

    # Render index page
    env = _make_env()
    index_template = env.get_template("pipeline_batch_index.html.jinja2")
    index_html = index_template.render(
        rows=index_rows,
        source_dir=str(source_dir),
        total=len(index_rows),
        corpus_data=corpus_data,
    )
    index_path = output_dir / "index.html"
    index_path.write_text(index_html, encoding="utf-8")
    print(f"\nBatch index written to: {index_path}  ({len(index_rows)} papers)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a static HTML inspector from a pipeline output JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Positional argument — optional in batch mode
    parser.add_argument(
        "json_path", type=Path, nargs="?",
        help="Path to pipeline output JSON file (omit when using --batch-dir)",
    )

    # Output
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help=(
            "Single/diff mode: output HTML path (default: out/inspector/<pmcid>.html). "
            "Batch mode: output directory (default: out/inspector/)."
        ),
    )

    # Cross-run diff
    parser.add_argument(
        "--compare", type=Path, default=None, metavar="OTHER_JSON",
        help="Second JSON file for cross-run diff view. Both must share the same PMCID.",
    )

    # CSV export
    parser.add_argument(
        "--export-flagged-csv", type=Path, default=None, metavar="OUTPUT_CSV",
        help="Export flagged findings to CSV. Only works in single-run mode.",
    )

    # Batch
    parser.add_argument(
        "--batch-dir", type=Path, default=None, metavar="DIR",
        help="Generate one inspector HTML per *.json file in DIR, plus an index page.",
    )

    parser.add_argument(
        "--corpus-relations", type=Path, default=None, metavar="CORPUS_JSON",
        help=(
            "Path to corpus_relations.json.  In batch mode, auto-detected from "
            "the source directory parent if omitted.  Adds a cross-paper relations "
            "section to the batch index and per-paper inspectors."
        ),
    )

    parser.add_argument(
        "--low-gs-threshold", type=float, default=0.4,
        help="Grounding score below which a finding is flagged as low-grounding (default: 0.4)",
    )

    args = parser.parse_args()

    # ── Batch mode ──────────────────────────────────────────────────────────────
    if args.batch_dir is not None:
        if args.json_path or args.compare or args.export_flagged_csv:
            print(
                "ERROR: --batch-dir cannot be combined with json_path, --compare, "
                "or --export-flagged-csv",
                file=sys.stderr,
            )
            sys.exit(1)

        source_dir = args.batch_dir
        if not source_dir.is_dir():
            print(f"ERROR: not a directory: {source_dir}", file=sys.stderr)
            sys.exit(1)

        output_dir = args.output or Path("out/inspector")
        print(f"Batch mode: scanning {source_dir}")
        render_batch(
            source_dir, output_dir, args.low_gs_threshold,
            corpus_relations_path=args.corpus_relations,
        )
        return

    # ── Single / diff mode — require json_path ──────────────────────────────────
    if args.json_path is None:
        parser.error("json_path is required unless using --batch-dir")

    json_path: Path = args.json_path
    if not json_path.exists():
        print(f"ERROR: file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    # ── Cross-run diff mode ─────────────────────────────────────────────────────
    if args.compare is not None:
        if args.export_flagged_csv:
            print(
                "ERROR: --export-flagged-csv is not supported in diff mode. "
                "Run single-run mode on each JSON separately.",
                file=sys.stderr,
            )
            sys.exit(1)

        compare_path: Path = args.compare
        if not compare_path.exists():
            print(f"ERROR: file not found: {compare_path}", file=sys.stderr)
            sys.exit(1)

        if args.output:
            output_path = args.output
        else:
            stem = json_path.stem
            output_path = Path("out/inspector") / f"{stem}_diff.html"

        render_diff(json_path, compare_path, output_path, args.low_gs_threshold)
        return

    # ── Single-run mode ─────────────────────────────────────────────────────────
    if args.output:
        output_path = args.output
    else:
        stem = json_path.stem
        output_path = Path("out/inspector") / f"{stem}.html"

    render(json_path, output_path, args.low_gs_threshold, args.export_flagged_csv)


if __name__ == "__main__":
    main()
