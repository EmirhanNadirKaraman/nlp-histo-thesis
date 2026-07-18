"""
Corpus-level relation stage: pool CanonicalRules from all per-paper JSONs and
run NLI-based pairwise comparison over the full corpus.

Each output relation carries a comparison_scope label:
  - "intra_paper"  — both rules come from the same paper's processing run
  - "cross_paper"  — rules come from different papers

All pairs go through the same _should_compare gate and NLI pipeline as the
per-paper RELATE stage.  Intra-paper pairs are included so the corpus graph is
complete — a rule's full neighborhood (both within-paper support and cross-paper
contradictions) is visible in one artifact.

This stage is entirely post-hoc.  It does NOT modify any per-paper output, does
NOT affect ResolveStage scoring, and does NOT alter per-paper `relations` keys.
Per-paper `relations` remain the authoritative input for ResolveStage.

Run-selection policy
--------------------
The current pipeline writes exactly one JSON per PMCID (``{pmcid}.json``),
always overwriting it with the latest successful result.  That makes single-file-
per-PMCID the common case.  However, when a corpus directory was assembled from
multiple batch runs (or when ``run_selection="all"`` is requested), the stage
must decide which runs to include.

  "latest_per_pmcid" (default)
      Group all files by PMCID.  Discard files whose status is not "success" or
      whose canonical_rules list is empty.  Among the remaining candidates for
      each PMCID, pick the one with the lexicographically largest run_id
      timestamp suffix (``{pmcid}_{YYYYMMDDTHHmmss}``).  Fall back to file mtime
      when run_id is absent or unparseable.  Emit a WARNING if no usable run is
      found for a PMCID.

  "all"
      Include every file that has status="success" and non-empty canonical_rules.
      If a PMCID has multiple runs all of them are pooled, which means the same
      claim may appear more than once in the corpus graph.  Use only when you
      specifically need to compare runs.

  manifest (list[Path])
      Ignore source_dir.  Load exactly the provided files without any filtering.
      Intended for testing and version-pinned workflows.

Usage
-----
Standalone (no runner required):
    from pathlib import Path
    from nlp_histo.pipeline.stages.knowledge_extraction.corpus_relate import CorpusRelateStage

    CorpusRelateStage().relate_from_dir(
        source_dir=Path("out/summaries/summaries/"),
        output_path=Path("out/summaries/corpus_relations.json"),
    )

Via KnowledgeExtractionRunner (after process_batch):
    results = runner.process_batch(papers)
    runner.corpus_relate(
        source_dir=runner._summaries_dir,
        output_path=runner._output_dir / "corpus_relations.json",
    )
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ..models import (
    CanonicalRule,
    CorpusRelation,
    NON_POLARITY_DIRS,
    Relation,
    direction_value,
)
from .relate_stage import (
    RelateStage,
    _norm_outcome,
    _norm_outcome_expression,
    _norm_subject,
    _should_compare,
)

logger = logging.getLogger(__name__)

# Matches the timestamp suffix in run_id: "{pmcid}_{YYYYMMDDTHHmmss}"
_TS_RE = re.compile(r"_(\d{8}T\d{6})$")


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _run_id_ts(run_id: str) -> str:
    """Extract the YYYYMMDDTHHmmss suffix from run_id, or "" if absent."""
    m = _TS_RE.search(run_id or "")
    return m.group(1) if m else ""


def _should_compare_cross_paper(
    a: CanonicalRule, b: CanonicalRule
) -> tuple[bool, str]:
    """
    Gate for cross-paper comparison.

    Entity names for the same concept differ across papers ("MGA" vs
    "microglandular adenosis"), so CUI match is preferred over exact strings.
    But when either side lacks a CUI we still require the normalized
    string to match — skipping the gate floods NLI with unrelated pairs
    (e.g. two rules whose subject was generically normalized to "Body
    tissue" are not actually about the same thing).

    Priority:
      1. category must match exactly.
      2. relation_type must match exactly.
      3. subject: prefer CUI match; otherwise fall back to normalized
         subject string match. A pair must satisfy one or the other.
      4. outcome: same fallback rule applied for every relation_type
         (previously only `expression` rules had an outcome gate, which
         let unrelated outcomes through for has_feature / causes / etc.).
    """
    from ..models import RelationTypeEnum  # noqa: PLC0415

    # B-049: skip any pair where either side carries a non-polarity direction
    # (unclear / no_direction). Cross-paper rules can arrive from JSON-loaded
    # metadata where direction is a raw string, or from legacy rows where
    # direction is None — `direction_value` normalises both paths.
    if (
        direction_value(a.direction) in NON_POLARITY_DIRS
        or direction_value(b.direction) in NON_POLARITY_DIRS
    ):
        return False, "non_polarity_direction"

    if a.category != b.category:
        return False, "category_mismatch"
    if a.relation_type != b.relation_type:
        return False, "relation_type_mismatch"

    # Subject gate — CUI when both present, normalized string fallback otherwise
    if a.subject_cui and b.subject_cui:
        if a.subject_cui != b.subject_cui:
            return False, "subject_cui_mismatch"
    else:
        if _norm_subject(a.subject_entity) != _norm_subject(b.subject_entity):
            return False, "subject_mismatch"

    # Outcome gate — applied to all relation types. expression rules use the
    # stricter expression-specific normalization (strips " expression"/
    # " positivity" suffixes so CD31 ≠ CD34); others use the general
    # synonym-aware normalizer.
    if a.outcome_cui and b.outcome_cui:
        if a.outcome_cui != b.outcome_cui:
            return False, "outcome_cui_incompatible"
    elif a.relation_type == RelationTypeEnum.expression:
        if _norm_outcome_expression(a.outcome_entity) != _norm_outcome_expression(b.outcome_entity):
            return False, "outcome_incompatible"
    else:
        if _norm_outcome(a.outcome_entity) != _norm_outcome(b.outcome_entity):
            return False, "outcome_incompatible"

    return True, ""


class CorpusRelateStage:
    """
    Corpus-level NLI relation stage.

    Reuses RelateStage.relate() verbatim — same comparability gate (_should_compare),
    same NLI model singleton, same direction-polarity guard (_classify_pair).
    The additions are:
      - run-selection: one representative run per PMCID by default
      - source_pmcid tracking per canonical_id (built while loading JSONs)
      - post-hoc enrichment of each Relation with corpus-level fields
      - JSON output with comparison_scope and same_paper labels

    Parameters
    ----------
    entailment_threshold:
        Forwarded to RelateStage.  Default 0.50 matches KnowledgeExtractionRunner default.
    contradiction_threshold:
        Forwarded to RelateStage.  Default 0.50 matches KnowledgeExtractionRunner default.
    """

    def __init__(
        self,
        entailment_threshold: float = 0.50,
        contradiction_threshold: float = 0.50,
        db=None,
        scope_aware_nli: bool = True,
        use_verbatim_for_nli: bool = False,
    ) -> None:
        self._relate = RelateStage(
            entailment_threshold=entailment_threshold,
            contradiction_threshold=contradiction_threshold,
            scope_aware_nli=scope_aware_nli,
            use_verbatim_for_nli=use_verbatim_for_nli,
        )
        self._db = db  # optional DatabaseConnection for DB persistence

    # Public API

    def relate_from_dir(
        self,
        source_dir: Path | None,
        output_path: Path,
        run_selection: str = "latest_per_pmcid",
        manifest: list[Path] | None = None,
    ) -> list[CorpusRelation]:
        """
        Pool CanonicalRules from per-paper JSONs, run RelateStage, write output.

        Parameters
        ----------
        source_dir:
            Directory containing per-paper JSON files (*.json).  Ignored when
            manifest is provided.  Required when manifest is None.
        output_path:
            Destination for corpus_relations.json.  Parent directories are
            created if absent.
        run_selection:
            "latest_per_pmcid" (default) — one representative run per PMCID.
            "all" — every valid file (status=success, non-empty canonical_rules).
            Ignored when manifest is provided.
        manifest:
            Explicit list of JSON paths to load.  When given, source_dir and
            run_selection are both ignored.  No status/rules filtering is applied
            (the caller is responsible for selecting valid files).

        Returns
        -------
        List of CorpusRelation objects written to output_path.
        """
        # Resolve the file list
        if manifest is not None:
            selected: dict[str, tuple[Path, dict]] = self._load_manifest(manifest)
        else:
            if source_dir is None:
                raise ValueError("source_dir is required when manifest is not provided")
            json_files = sorted(source_dir.glob("*.json"))
            if not json_files:
                logger.warning("CORPUS RELATE: no JSON files found in %s", source_dir)
                return []
            selected = self._select_runs(json_files, run_selection)

        if not selected:
            logger.warning("CORPUS RELATE: no usable runs found — nothing to compare")
            return []

        logger.info(
            "CORPUS RELATE: selected %d papers (run_selection=%r)",
            len(selected),
            "manifest" if manifest is not None else run_selection,
        )

        # Build canonical rule pool + lookup indexes
        all_rules: list[CanonicalRule] = []
        id_to_pmcid: dict[str, str] = {}     # canonical_id → source PMCID
        rule_meta: dict[str, dict] = {}      # canonical_id → raw rule dict
        pmcids_loaded: list[str] = []

        for pmcid, (_, data) in selected.items():
            raw_rules = data.get("canonical_rules") or []
            pmcids_loaded.append(pmcid)
            for rd in raw_rules:
                try:
                    rule = CanonicalRule.model_validate(rd)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "CORPUS RELATE: [%s] could not parse canonical rule %r — %s",
                        pmcid, rd.get("canonical_id", "?"), exc,
                    )
                    continue
                all_rules.append(rule)
                id_to_pmcid[rule.canonical_id] = pmcid
                rule_meta[rule.canonical_id] = rd

        logger.info(
            "CORPUS RELATE: loaded %d canonical rules from %d papers",
            len(all_rules), len(pmcids_loaded),
        )

        # Run RelateStage on the full pool
        if len(all_rules) < 2:
            logger.warning(
                "CORPUS RELATE: fewer than 2 rules across the corpus — nothing to compare"
            )
            corpus_relations: list[CorpusRelation] = []
            raw_pairs_corpus: list = []
        else:
            raw_relations, raw_pairs_corpus, _ = self._relate.relate(
                all_rules, pmcid="corpus", gate=_should_compare_cross_paper,
            )
            logger.info(
                "CORPUS RELATE: %d raw relations from pool of %d rules",
                len(raw_relations), len(all_rules),
            )
            corpus_relations = self._enrich(raw_relations, id_to_pmcid, rule_meta)

        # Write output
        intra_count = sum(1 for r in corpus_relations if r.same_paper)
        cross_count = len(corpus_relations) - intra_count

        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(source_dir) if source_dir else None,
            "run_selection": "manifest" if manifest is not None else run_selection,
            "pmcids_loaded": pmcids_loaded,
            "canonical_rules_loaded": len(all_rules),
            "total_relations": len(corpus_relations),
            "intra_paper_count": intra_count,
            "cross_paper_count": cross_count,
            "relations": [r.model_dump() for r in corpus_relations],
            "raw_nli_pairs": [
                {**p.model_dump(), "pmcid_a": id_to_pmcid.get(p.rule_id_a, ""),
                 "pmcid_b": id_to_pmcid.get(p.rule_id_b, "")}
                for p in raw_pairs_corpus
            ],
        }
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "CORPUS RELATE: written %d relations "
            "(%d intra-paper, %d cross-paper) → %s",
            len(corpus_relations), intra_count, cross_count, output_path,
        )

        if self._db is not None:
            self._persist_to_db(corpus_relations)

        return corpus_relations

    # Incremental API

    def relate_incremental(
        self,
        new_pmcid: str,
        new_rules: list[CanonicalRule],
        db,
    ) -> list[CorpusRelation]:
        """
        Update corpus relations for a newly processed paper without a full re-run.

        Loads canonical rules for all other PMCIDs from the DB (latest run per
        PMCID), compares them against new_rules using the cross-paper gate and
        NLI, then replaces all corpus relations touching new_pmcid in DB.

        No-ops (returns []) if new_rules is empty or no other PMCIDs exist in DB.
        """
        if not new_rules:
            logger.info(
                "CORPUS RELATE [%s]: no canonical rules — incremental skipped", new_pmcid,
            )
            return []

        existing_rules, id_to_pmcid, rule_meta = self._load_rules_from_db(new_pmcid, db)

        if not existing_rules:
            logger.info(
                "CORPUS RELATE [%s]: no other papers in DB — incremental skipped", new_pmcid,
            )
            return []

        # Register new paper's rules in the lookup maps
        for rule in new_rules:
            id_to_pmcid[rule.canonical_id] = new_pmcid
            rule_meta[rule.canonical_id] = rule.model_dump()

        all_rules = new_rules + existing_rules

        # Gate: only cross-paper pairs where exactly one rule is from new_pmcid
        def _incremental_gate(
            a: CanonicalRule, b: CanonicalRule
        ) -> tuple[bool, str]:
            a_new = id_to_pmcid.get(a.canonical_id) == new_pmcid
            b_new = id_to_pmcid.get(b.canonical_id) == new_pmcid
            if not (a_new ^ b_new):
                return False, "not_incremental_pair"
            return _should_compare_cross_paper(a, b)

        logger.info(
            "CORPUS RELATE [%s]: comparing %d new rules × %d existing rules",
            new_pmcid, len(new_rules), len(existing_rules),
        )
        raw_cross, _raw_pairs_cross, _ = self._relate.relate(
            all_rules, pmcid="corpus", gate=_incremental_gate,
        )

        # Second pass: intra-paper pairs from new_rules only.
        # The XOR gate above blocks same-paper pairs, so they are never computed
        # on reprocess.  Run RelateStage directly on new_rules (all same PMCID)
        # to produce the intra-paper neighborhood for the reprocessed paper.
        raw_intra: list[Relation] = []
        if len(new_rules) >= 2:
            raw_intra, _raw_pairs_intra, _ = self._relate.relate(
                new_rules, pmcid="corpus_intra", gate=_should_compare,
            )

        logger.info(
            "CORPUS RELATE [%s]: %d cross-paper + %d intra-paper relations found",
            new_pmcid, len(raw_cross), len(raw_intra),
        )

        corpus_relations = self._enrich(raw_cross + raw_intra, id_to_pmcid, rule_meta)
        self._replace_for_pmcid(new_pmcid, corpus_relations, db)
        return corpus_relations

    def _load_rules_from_db(
        self,
        exclude_pmcid: str,
        db,
    ) -> tuple[list[CanonicalRule], dict[str, str], dict[str, dict]]:
        """
        Load canonical rules from DB for all PMCIDs except exclude_pmcid.
        Uses the latest pipeline_run_id per PMCID.
        Returns (rules, id_to_pmcid, rule_meta).
        """
        from ..models import (  # noqa: PLC0415
            DirectionEnum, RelationTypeEnum,
        )

        rules: list[CanonicalRule] = []
        id_to_pmcid: dict[str, str] = {}
        rule_meta: dict[str, dict] = {}

        try:
            from nlp_histo.database.models import SumCanonicalRule  # noqa: PLC0415
            from sqlalchemy import func  # noqa: PLC0415

            with db.session_scope() as session:
                latest_run_subq = (
                    session.query(
                        SumCanonicalRule.pmcid,
                        func.max(SumCanonicalRule.pipeline_run_id).label("max_run"),
                    )
                    .filter(SumCanonicalRule.pmcid != exclude_pmcid)
                    .group_by(SumCanonicalRule.pmcid)
                    .subquery()
                )
                rows = (
                    session.query(SumCanonicalRule)
                    .join(
                        latest_run_subq,
                        (SumCanonicalRule.pmcid == latest_run_subq.c.pmcid)
                        & (SumCanonicalRule.pipeline_run_id == latest_run_subq.c.max_run),
                    )
                    .all()
                )
                for row in rows:
                    try:
                        rule = CanonicalRule(
                            canonical_id         = row.canonical_id,
                            group_id             = row.group_id,
                            subject_entity       = row.subject_entity,
                            outcome_entity       = row.outcome_entity,
                            relation_type        = RelationTypeEnum(row.relation_type),
                            direction            = DirectionEnum(row.direction) if row.direction else None,
                            predicate_text       = row.predicate_text,
                            is_conflicted        = row.is_conflicted,
                            study_coverage       = row.study_coverage,
                            category             = row.category,
                            supporting_pmcids    = row.pmcids or [],
                            member_normal_ids    = row.member_normal_ids or [],
                            mean_grounding_score = row.mean_grounding_score,
                            finding_count        = row.finding_count,
                            subject_cui          = row.subject_cui,
                            outcome_cui          = row.outcome_cui,
                        )
                        rules.append(rule)
                        id_to_pmcid[rule.canonical_id] = row.pmcid
                        rule_meta[rule.canonical_id] = rule.model_dump()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "CORPUS RELATE: could not reconstruct rule %r from DB — %s",
                            row.canonical_id, exc,
                        )
        except Exception as exc:  # noqa: BLE001
            logger.warning("CORPUS RELATE: failed to load rules from DB: %s", exc)

        return rules, id_to_pmcid, rule_meta

    def _replace_for_pmcid(
        self,
        pmcid: str,
        corpus_relations: list[CorpusRelation],
        db,
    ) -> None:
        """Delete all corpus relations touching pmcid, then insert the new ones."""
        try:
            from nlp_histo.database.models import SumCorpusRelation  # noqa: PLC0415
            from sqlalchemy import or_  # noqa: PLC0415

            corpus_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            rows = self._to_db_rows(corpus_relations, corpus_run_id)
            with db.session_scope() as session:
                session.query(SumCorpusRelation).filter(
                    or_(
                        SumCorpusRelation.pmcid_a == pmcid,
                        SumCorpusRelation.pmcid_b == pmcid,
                    )
                ).delete(synchronize_session=False)
                session.bulk_save_objects(rows)
            logger.info(
                "CORPUS RELATE [%s]: %d relations written to DB (corpus_run_id=%s)",
                pmcid, len(rows), corpus_run_id,
            )
        except Exception as exc:
            logger.warning(
                "CORPUS RELATE [%s]: DB replace failed: %s", pmcid, exc, exc_info=True,
            )

    def _to_db_rows(
        self, corpus_relations: list[CorpusRelation], corpus_run_id: str
    ) -> list:
        """Convert CorpusRelation objects to SumCorpusRelation ORM rows."""
        from nlp_histo.database.models import SumCorpusRelation  # noqa: PLC0415
        return [
            SumCorpusRelation(
                corpus_run_id            = corpus_run_id,
                relation_id              = r.relation_id,
                comparison_scope         = r.comparison_scope,
                same_paper               = r.same_paper,
                rule_id_a                = r.rule_id_a,
                rule_id_b                = r.rule_id_b,
                pmcid_a                  = r.pmcid_a,
                pmcid_b                  = r.pmcid_b,
                relation_type            = r.relation_type.value,
                nli_score_a_to_b         = r.nli_score_a_to_b,
                nli_score_b_to_a         = r.nli_score_b_to_a,
                subject_entity           = r.subject_entity,
                outcome_entity           = r.outcome_entity,
                category                 = r.category,
                relation_type_structural = r.relation_type_structural,
                direction_a              = r.direction_a,
                direction_b              = r.direction_b,
                predicate_a              = r.predicate_a,
                predicate_b              = r.predicate_b,
                mean_grounding_a         = r.mean_grounding_a,
                mean_grounding_b         = r.mean_grounding_b,
                finding_count_a          = r.finding_count_a,
                finding_count_b          = r.finding_count_b,
                supporting_pmcids_a      = list(r.supporting_pmcids_a),
                supporting_pmcids_b      = list(r.supporting_pmcids_b),
                is_conflicted_a          = r.is_conflicted_a,
                study_coverage_a         = r.study_coverage_a,
                is_conflicted_b          = r.is_conflicted_b,
                study_coverage_b         = r.study_coverage_b,
                scope_check_result       = r.scope_check_result,
                scope_note               = r.scope_note,
            )
            for r in corpus_relations
        ]

    def _persist_to_db(self, corpus_relations: list[CorpusRelation]) -> None:
        """Replace ALL corpus relations in the DB (batch mode)."""
        try:
            from nlp_histo.database.models import SumCorpusRelation  # noqa: PLC0415
            corpus_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            rows = self._to_db_rows(corpus_relations, corpus_run_id)
            with self._db.session_scope() as session:
                session.query(SumCorpusRelation).delete()
                session.bulk_save_objects(rows)
            logger.info(
                "CORPUS RELATE: persisted %d relations to DB (corpus_run_id=%s)",
                len(rows), corpus_run_id,
            )
        except Exception as exc:
            logger.warning("CORPUS RELATE: DB persist failed: %s", exc, exc_info=True)

    # Run-selection

    def _select_runs(
        self,
        json_files: list[Path],
        run_selection: str,
    ) -> dict[str, tuple[Path, dict]]:
        """
        Parse all json_files and return a selected subset.

        Returns
        -------
        Dict of pmcid → (path, parsed_data) for the selected runs.
        Files that fail to parse are skipped with a WARNING.
        Files with status != "success" or empty canonical_rules are silently
        discarded (not a warning — this is expected for error-status results).
        """
        # Parse every file once
        parsed: list[tuple[Path, dict]] = []
        for p in json_files:
            try:
                with open(p, encoding="utf-8") as fh:
                    data = json.load(fh)
                parsed.append((p, data))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CORPUS RELATE: skipping %s — could not parse JSON: %s",
                    p.name, exc,
                )

        if run_selection == "all":
            return self._filter_valid(parsed)

        # Default: latest_per_pmcid
        return self._latest_per_pmcid(parsed)

    def _filter_valid(
        self,
        parsed: list[tuple[Path, dict]],
    ) -> dict[str, tuple[Path, dict]]:
        """
        Keep every file that has status="success" and non-empty canonical_rules.
        When a PMCID appears more than once, both entries are kept under distinct
        keys (pmcid and pmcid_{run_id} for duplicates).
        """
        result: dict[str, tuple[Path, dict]] = {}
        seen_counts: dict[str, int] = defaultdict(int)

        for p, data in parsed:
            if data.get("status") != "success":
                continue
            if not data.get("canonical_rules"):
                continue
            pmcid = data.get("pmcid") or p.stem
            seen_counts[pmcid] += 1
            # First occurrence uses plain pmcid key; duplicates get a suffix
            key = pmcid if seen_counts[pmcid] == 1 else f"{pmcid}__{seen_counts[pmcid]}"
            result[key] = (p, data)

        return result

    def _latest_per_pmcid(
        self,
        parsed: list[tuple[Path, dict]],
    ) -> dict[str, tuple[Path, dict]]:
        """
        For each PMCID, select the latest successful run with non-empty
        canonical_rules.

        Sort key priority:
          1. run_id timestamp suffix (YYYYMMDDTHHmmss) — present in all runner-
             generated files since _make_run_id was introduced.
          2. file mtime — fallback when run_id is absent or has no timestamp.

        A WARNING is emitted for any PMCID where no usable run is found (i.e.,
        all candidates were either failed or had empty canonical_rules).
        """
        # Group by PMCID, collecting all (path, data) candidates
        by_pmcid: dict[str, list[tuple[Path, dict]]] = defaultdict(list)

        for p, data in parsed:
            pmcid = data.get("pmcid") or p.stem
            by_pmcid[pmcid].append((p, data))

        result: dict[str, tuple[Path, dict]] = {}

        for pmcid, candidates in by_pmcid.items():
            # Filter to usable runs
            usable = [
                (p, d) for p, d in candidates
                if d.get("status") == "success" and d.get("canonical_rules")
            ]
            if not usable:
                logger.warning(
                    "CORPUS RELATE: [%s] no usable run found "
                    "(all %d candidate(s) failed or had empty canonical_rules)",
                    pmcid, len(candidates),
                )
                continue

            # Sort descending: prefer run_id timestamp, fall back to mtime
            def _sort_key(item: tuple[Path, dict]) -> tuple[str, float]:
                path, data = item
                ts = _run_id_ts(data.get("run_id") or "")
                mtime = path.stat().st_mtime if path.exists() else 0.0
                return (ts, mtime)

            best_path, best_data = max(usable, key=_sort_key)
            result[pmcid] = (best_path, best_data)

        return result

    def _load_manifest(
        self,
        manifest: list[Path],
    ) -> dict[str, tuple[Path, dict]]:
        """
        Load exactly the files in manifest.  No status/rules filtering.
        PMCID key collisions (same pmcid in multiple files) are resolved by
        appending a counter suffix, matching _filter_valid behavior.
        """
        result: dict[str, tuple[Path, dict]] = {}
        seen_counts: dict[str, int] = defaultdict(int)

        for p in manifest:
            try:
                with open(p, encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CORPUS RELATE: manifest entry %s could not be parsed — %s",
                    p.name, exc,
                )
                continue
            pmcid = data.get("pmcid") or p.stem
            seen_counts[pmcid] += 1
            key = pmcid if seen_counts[pmcid] == 1 else f"{pmcid}__{seen_counts[pmcid]}"
            result[key] = (p, data)

        return result

    # Enrichment

    def _enrich(
        self,
        raw_relations: list,
        id_to_pmcid: dict[str, str],
        rule_meta: dict[str, dict],
    ) -> list[CorpusRelation]:
        """
        Convert raw Relation objects → CorpusRelation by adding provenance and
        human-readable fields.

        relation_id is a deterministic 8-char hash of the sorted canonical_id
        pair so it is stable across re-runs (same JSON files → same IDs).
        """
        result: list[CorpusRelation] = []

        for rel in raw_relations:
            cid_a = rel.rule_id_a
            cid_b = rel.rule_id_b
            pmcid_a = id_to_pmcid.get(cid_a, "unknown")
            pmcid_b = id_to_pmcid.get(cid_b, "unknown")
            same_paper = pmcid_a == pmcid_b

            # Sort IDs for a deterministic hash regardless of which rule was A vs B
            sorted_pair = "|".join(sorted([cid_a, cid_b]))
            relation_id = f"COR_{_sha8(sorted_pair)}"

            meta_a = rule_meta.get(cid_a, {})
            meta_b = rule_meta.get(cid_b, {})

            corpus_rel = CorpusRelation(
                # inherited Relation fields
                rule_id_a=cid_a,
                rule_id_b=cid_b,
                relation_type=rel.relation_type,
                nli_score_a_to_b=rel.nli_score_a_to_b,
                nli_score_b_to_a=rel.nli_score_b_to_a,
                # corpus-level provenance
                relation_id=relation_id,
                comparison_scope="intra_paper" if same_paper else "cross_paper",
                same_paper=same_paper,
                pmcid_a=pmcid_a,
                pmcid_b=pmcid_b,
                # structural metadata (from rule A; gate ensures A and B share these)
                subject_entity=meta_a.get("subject_entity") or "",
                outcome_entity=meta_a.get("outcome_entity") or "",
                category=meta_a.get("category") or "",
                relation_type_structural=meta_a.get("relation_type") or "",
                # per-rule fields
                direction_a=meta_a.get("direction"),
                direction_b=meta_b.get("direction"),
                predicate_a=meta_a.get("predicate_text") or "",
                predicate_b=meta_b.get("predicate_text") or "",
                mean_grounding_a=meta_a.get("mean_grounding_score"),
                mean_grounding_b=meta_b.get("mean_grounding_score"),
                finding_count_a=meta_a.get("finding_count") or 0,
                finding_count_b=meta_b.get("finding_count") or 0,
                supporting_pmcids_a=meta_a.get("supporting_pmcids") or [],
                supporting_pmcids_b=meta_b.get("supporting_pmcids") or [],
                is_conflicted_a=meta_a.get("is_conflicted") or False,
                study_coverage_a=meta_a.get("study_coverage") or "unknown",
                is_conflicted_b=meta_b.get("is_conflicted") or False,
                study_coverage_b=meta_b.get("study_coverage") or "unknown",
                # scope qualifier check (v1: not yet implemented)
                scope_check_result="scope_unknown",
                scope_note="",
            )
            result.append(corpus_rel)

        return result


# CLI entry point

def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Run corpus-level NLI relation stage over per-paper JSON summaries.",
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        default="out/summaries/summaries",
        help="Directory containing per-paper JSON files (default: out/summaries/summaries)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path for corpus_relations.json "
             "(default: <source_dir>/../corpus_relations.json)",
    )
    parser.add_argument(
        "--run-selection",
        choices=["latest_per_pmcid", "all"],
        default="latest_per_pmcid",
        help="Which runs to include when a PMCID has multiple files (default: latest_per_pmcid)",
    )
    parser.add_argument(
        "--entailment-threshold", type=float, default=0.50,
        help="NLI entailment score threshold (default: 0.50)",
    )
    parser.add_argument(
        "--contradiction-threshold", type=float, default=0.50,
        help="NLI contradiction score threshold (default: 0.50)",
    )
    parser.add_argument(
        "--no-save-to-db", action="store_true",
        help="Skip PostgreSQL persistence (sum_corpus_relations table). "
             "By default, all previous rows are deleted and replaced.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        print(f"ERROR: source_dir {source_dir!r} does not exist", file=sys.stderr)
        sys.exit(1)

    output_path = (
        Path(args.output) if args.output
        else source_dir.parent / "corpus_relations.json"
    )

    db = None
    if not args.no_save_to_db:
        try:
            from nlp_histo.database import get_db_connection  # noqa: PLC0415
            db = get_db_connection()
            print("DB connection established for corpus relation persistence.")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: could not connect to DB: {exc}", file=sys.stderr)
            sys.exit(1)

    stage = CorpusRelateStage(
        entailment_threshold=args.entailment_threshold,
        contradiction_threshold=args.contradiction_threshold,
        db=db,
    )
    relations = stage.relate_from_dir(
        source_dir=source_dir,
        output_path=output_path,
        run_selection=args.run_selection,
    )
    print(f"Done. {len(relations)} relations written to {output_path}")
    if db is not None:
        print("Corpus relations persisted to PostgreSQL.")
    else:
        print("DB persistence skipped (--no-save-to-db).")


if __name__ == "__main__":
    _main()
