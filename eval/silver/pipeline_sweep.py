"""
Pipeline threshold sweep harness.

Modes:
  prime       Run pipeline on dev sample, save raw result JSONs (run once).
              Uses grounding.threshold=0.0 so every finding is scored but nothing dropped.
  grounding   Offline grounding-threshold sweep → eval/reports/grounding_sweep_*.csv
  relate      Offline RELATE-threshold sweep   → eval/reports/relate_sweep_*.csv
  all         prime + grounding + relate in sequence

Usage:
  python -m eval.silver.pipeline_sweep prime --source eval/data/source_cases_related15.jsonl
  python -m eval.silver.pipeline_sweep grounding --source eval/data/source_cases_related15.jsonl --silver eval/data/silver_findings_related15.jsonl --embedder gemini
  python -m eval.silver.pipeline_sweep relate --source eval/data/source_cases_related15.jsonl --silver eval/data/silver_findings_related15.jsonl
  python -m eval.silver.pipeline_sweep all --source eval/data/source_cases_related15.jsonl --silver eval/data/silver_findings_related15.jsonl --embedder gemini

Notes:
  - prime must run before grounding / relate.
  - grounding uses the fixed eval-matching threshold (--sim-threshold, default 0.55).
    Run eval.silver.sweep first to calibrate that threshold.
  - relate produces relation-count statistics (no silver comparison — RELATE output
    is at FinalRule level, not at the individual Finding level that silver labels target).
  - MAP theta sweep is not included here (requires a full ABC cascade re-run per
    theta value). Run scripts/run_paper_single_model.py with different configs manually.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

from eval.silver.embedders import GeminiEmbedder, OpenAIEmbedder
from eval.silver.jsonl_utils import read_jsonl
from eval.silver.matcher import (
    DEFAULT_CACHE_PATH,
    DEFAULT_GEMINI_CACHE_PATH,
    EMBEDDING_MODEL,
    GEMINI_EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD,
    EmbeddingCache,
    compute_sim_matrix,
    greedy_match,
    make_embedding_cache,
    match_from_matrix,
    optimal_match,
)
from eval.silver.schemas import (
    PipelineCaseOutput,
    PipelineFinding,
    SilverCaseResult,
    SourceCase,
)
from eval.silver.split import filter_by_split

PRIME_DIR   = Path("eval/data/pipeline_raw")
REPORTS_DIR = Path("eval/reports")
SOURCE_PATH = Path("eval/data/source_cases_related15.jsonl")
SILVER_PATH = Path("eval/data/silver_findings_related15.jsonl")

# Sweep ranges
GROUNDING_THRESHOLDS: list[float | None] = [None, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                                              0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
RELATE_ENT_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
RELATE_CON_THRESHOLDS = [0.40, 0.50, 0.60, 0.70]


# ── LLM / runner factory (mirrors run_paper_single_model.py) ──────────────────

def _build_llm():
    from pipeline.stages.knowledge_extraction.llm_providers import gemini_direct_chat
    return gemini_direct_chat("gemini-2.5-flash-lite", max_tokens=16384, request_timeout=60)


def _build_runner(config, output_dir: Path, force_rerun: bool = False):
    from pipeline.stages.knowledge_extraction import KnowledgeExtractionRunner
    llm = _build_llm()
    return KnowledgeExtractionRunner(
        voter_llms=[llm],
        level2_voter_llms=[llm],
        escalation_llm=llm,
        config=config,
        output_dir=output_dir,
        db=None,        # no DB writes — sweep only
        run_ner=False,
        force_rerun=force_rerun,
    )


# ── file_data builder from SourceCase ─────────────────────────────────────────

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_sci_sm")
    return _nlp


def case_to_file_data(case: SourceCase) -> dict:
    """Build a runner-compatible file_data dict from a SourceCase."""
    nlp = _get_nlp()
    sentences = []
    for sent in nlp(case.text).sents:
        text = sent.text.strip()
        if text:
            sentences.append({
                "pmcid": case.pmcid,
                "text_element_id": case.te_id,
                "sentence": text,
            })
    return {"pmcid": case.pmcid, "sentences_with_provenance": sentences}


# ── Result → PipelineCaseOutput ───────────────────────────────────────────────

def result_to_case_output(result: dict, case: SourceCase) -> PipelineCaseOutput:
    """Extract MAP findings from a runner result dict and build PipelineCaseOutput."""
    findings: list[PipelineFinding] = []
    for chunk in result.get("audit_trail", {}).get("map_chunks", []):
        chunk_id = chunk.get("chunk_id", "")
        for f in chunk.get("findings", []):
            scope = f.get("scope") or {}
            findings.append(PipelineFinding(
                pipeline_run_id=0,
                run_id=result.get("run_id", "sweep"),
                pmcid=case.pmcid,
                chunk_id=chunk_id,
                category=f.get("category", ""),
                claim=f.get("claim", ""),
                subject_entity=f.get("subject_entity"),
                outcome_entity=f.get("outcome_entity"),
                relation_type=str(f.get("relation_type", "unclear")),
                direction=str(f["direction"]) if f.get("direction") else None,
                confidence=str(f.get("confidence", "medium")),
                verbatim_support=f.get("verbatim_support", ""),
                grounding_score=f.get("grounding_score"),
                scope_disease_subtype=scope.get("disease_subtype"),
                scope_cohort_n=scope.get("cohort_n"),
                scope_assay_method=scope.get("assay_method"),
                scope_biomarker_cutoff=scope.get("biomarker_cutoff"),
                scope_tissue_site=scope.get("tissue_site"),
                scope_treatment_context=scope.get("treatment_context"),
                scope_endpoint=scope.get("endpoint"),
                scope_study_design=scope.get("study_design"),
            ))
    return PipelineCaseOutput(
        case_id=case.case_id,
        pmcid=case.pmcid,
        te_id=case.te_id,
        run_id=result.get("run_id", "sweep"),
        pipeline_run_id=0,
        findings=findings,
    )


# ── Grounding threshold filter ────────────────────────────────────────────────

def _apply_grounding_threshold(
    co: PipelineCaseOutput,
    threshold: float | None,
) -> PipelineCaseOutput:
    if threshold is None:
        return co
    kept = [f for f in co.findings
            if f.grounding_score is None or f.grounding_score >= threshold]
    return co.model_copy(update={"findings": kept})


# ── RELATE offline re-classification ─────────────────────────────────────────

def _classify_pair_offline(
    pair: dict,
    ent_t: float,
    con_t: float,
) -> str:
    """Re-classify a RawNLIPair dict using new thresholds (simplified: no direction check)."""
    if pair["con_a_to_b"] >= con_t and pair["con_b_to_a"] >= con_t:
        return "CONTRADICT"
    if pair["ent_a_to_b"] >= ent_t and pair["ent_b_to_a"] >= ent_t:
        return "SUPPORT"
    if pair["ent_a_to_b"] >= ent_t or pair["ent_b_to_a"] >= ent_t:
        return "SCOPE_QUALIFY"
    return "UNRELATED"


# ── Evaluation helper ─────────────────────────────────────────────────────────

def _evaluate_outputs(
    case_outputs: list[PipelineCaseOutput],
    silver_by_case: dict[str, SilverCaseResult],
    embedder: object,
    embed_cache: EmbeddingCache,
    sim_threshold: float,
) -> dict[str, Any]:
    """Compute P/R/F1 + strict-F1 vs silver under BOTH one-to-one matchers.

    The similarity matrix is computed once per case; the (cheap) matching is run
    twice — ``greedy`` and ``optimal`` (Hungarian via
    ``scipy.optimize.linear_sum_assignment``) — so the caller gets both without a
    second embedding pass.

    NAMING: ``*_optimal`` is the globally optimal one-to-one assignment **by
    embedding similarity**. It does NOT mean a globally optimal *strict-F1*
    assignment — both matchers match on similarity, and the three categorical
    fields (category / relation_type / direction) are scored afterward.

    Returns explicit ``*_greedy`` and ``*_optimal`` metric sets. The legacy
    ``precision``/``recall``/``f1``/``strict_f1``/``n_matched`` keys ALIAS the
    **greedy** matcher (the historical default) for backward compatibility.
    """
    STRICT_FIELDS = {"category", "relation_type", "direction"}
    n_silver = n_pipeline = 0
    matchers = {"greedy": greedy_match, "optimal": optimal_match}
    acc = {"greedy": [0, 0.0], "optimal": [0, 0.0]}   # name → [n_matched, strict_tp]

    for co in case_outputs:
        silver = silver_by_case.get(co.case_id)
        if silver is None:
            continue
        sim, _, _ = compute_sim_matrix(silver, co, embedder, embed_cache)
        n_silver += len(silver.findings)
        n_pipeline += len(co.findings)
        for name, fn in matchers.items():
            result = match_from_matrix(silver, co, sim, sim_threshold, match_fn=fn)
            acc[name][0] += len(result.matched)
            for pair in result.matched:
                strict_mismatch = any(m.field in STRICT_FIELDS for m in pair.field_mismatches)
                acc[name][1] += 0.5 if strict_mismatch else 1.0

    def _prf(n_matched: int, strict_tp: float) -> tuple[float, float, float, float]:
        p = n_matched / n_pipeline if n_pipeline else 0.0
        r = n_matched / n_silver if n_silver else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        sp = strict_tp / n_pipeline if n_pipeline else 0.0
        sr = strict_tp / n_silver if n_silver else 0.0
        sf1 = (2 * sp * sr / (sp + sr)) if (sp + sr) else 0.0
        return p, r, f1, sf1

    gp, gr, gf1, gsf1 = _prf(*acc["greedy"])
    op, orr, of1, osf1 = _prf(*acc["optimal"])

    return {
        # legacy keys == greedy matcher (historical default; kept for back-compat)
        "precision": round(gp, 4), "recall": round(gr, 4),
        "f1": round(gf1, 4), "strict_f1": round(gsf1, 4),
        "n_matched": acc["greedy"][0], "n_silver": n_silver, "n_pipeline": n_pipeline,
        # explicit greedy (sensitivity / diagnostic)
        "precision_greedy": round(gp, 4), "recall_greedy": round(gr, 4),
        "f1_greedy": round(gf1, 4), "strict_f1_greedy": round(gsf1, 4),
        "n_matched_greedy": acc["greedy"][0],
        # explicit optimal / Hungarian (PRIMARY selection matcher)
        "precision_optimal": round(op, 4), "recall_optimal": round(orr, 4),
        "f1_optimal": round(of1, 4), "strict_f1_optimal": round(osf1, 4),
        "n_matched_optimal": acc["optimal"][0],
    }


# ── PRIME mode ────────────────────────────────────────────────────────────────

def run_prime(cases: list[SourceCase], prime_dir: Path,
              force_reprime: bool = False) -> None:
    """
    Run the pipeline on each case and save raw result JSONs to prime_dir.

    Uses grounding.threshold=0.0 so every finding is scored but nothing dropped.
    MAP LLM calls are cached under prime_dir/llm_cache/.
    """
    from pipeline.stages.knowledge_extraction.config import (
        KnowledgeExtractionConfig, MapConfig, GroundingConfig,
    )

    prime_dir.mkdir(parents=True, exist_ok=True)
    llm_cache_dir = prime_dir / "llm_cache"
    llm_cache_dir.mkdir(parents=True, exist_ok=True)

    prime_config = KnowledgeExtractionConfig(
        map=MapConfig(theta=0.0, reject_theta=-1.0),
        grounding=GroundingConfig(threshold=0.0),  # score all, drop nothing
    )

    runner = _build_runner(prime_config, llm_cache_dir, force_rerun=True)

    from tqdm.auto import tqdm  # noqa: PLC0415

    n_done = n_skip = n_fail = 0

    for case in tqdm(cases, desc="Priming cases", unit="case"):
        safe_id = case.case_id.replace("|", "_")
        out_path = prime_dir / f"{safe_id}.json"

        if out_path.exists() and not force_reprime:
            n_skip += 1
            continue

        logger.info("Priming %s…", case.case_id)
        try:
            file_data = case_to_file_data(case)
            result = runner.process(file_data)
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                encoding="utf-8")
            n_findings = sum(
                len(chunk.get("findings", []))
                for chunk in result.get("audit_trail", {}).get("map_chunks", [])
            )
            logger.info("  → %d findings, %d raw NLI pairs, saved %s",
                        n_findings,
                        len(result.get("relate_raw_pairs", [])),
                        out_path.name)
            n_done += 1
        except Exception as exc:
            logger.error("[%s] prime failed: %s", case.case_id, exc)
            n_fail += 1

    logger.info("Prime done. done=%d  skipped=%d  failed=%d", n_done, n_skip, n_fail)


# ── GROUNDING sweep mode ──────────────────────────────────────────────────────

def run_grounding_sweep(
    cases: list[SourceCase],
    silver_by_case: dict[str, SilverCaseResult],
    embedder: object,
    embed_cache: EmbeddingCache,
    sim_threshold: float,
    thresholds: list[float | None],
    prime_dir: Path,
    split: str,
    seed: int,
    dev_fraction: float,
) -> list[dict]:
    """
    Offline grounding-threshold sweep.

    For each threshold T:
    1. Load primed result from prime_dir.
    2. Filter findings to grounding_score >= T (None = keep all).
    3. Evaluate against silver → P/R/F1.
    """
    # Load primed case outputs
    primed: dict[str, PipelineCaseOutput] = {}
    missing = []
    for case in cases:
        safe_id = case.case_id.replace("|", "_")
        path = prime_dir / f"{safe_id}.json"
        if not path.exists():
            missing.append(case.case_id)
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("status") != "success":
            logger.warning("%s — primed result status=%s, skipping", case.case_id, raw.get("status"))
            continue
        primed[case.case_id] = result_to_case_output(raw, case)

    if missing:
        logger.warning("%d case(s) missing from prime dir — run prime first: %s",
                       len(missing), missing[:5])
    if not primed:
        logger.error("No primed results found. Run `prime` mode first.")
        return []

    cases_available = [c for c in cases if c.case_id in primed]
    logger.info("Grounding sweep: %d cases  %d thresholds", len(cases_available), len(thresholds))

    rows = []
    for t in thresholds:
        filtered = [_apply_grounding_threshold(primed[c.case_id], t)
                    for c in cases_available]
        metrics = _evaluate_outputs(filtered, silver_by_case, embedder, embed_cache, sim_threshold)
        t_label = "None" if t is None else f"{t:.2f}"
        row = {
            "grounding_threshold": t_label,
            "split": split,
            "seed": seed,
            "dev_fraction": dev_fraction,
            "sim_threshold": sim_threshold,
            **metrics,
        }
        rows.append(row)
        logger.info("  grounding=%-5s  P=%.3f  R=%.3f  F1=%.3f  strict_F1=%.3f  "
                    "matched=%d / silver=%d / pipeline=%d",
                    t_label,
                    metrics["precision"], metrics["recall"],
                    metrics["f1"], metrics["strict_f1"],
                    metrics["n_matched"], metrics["n_silver"], metrics["n_pipeline"])

    return rows


# ── RELATE: corpus-level NLI pair generation ──────────────────────────────────

def _build_corpus_nli_pairs(
    prime_dir: Path,
    cache_path: Path,
    force: bool = False,
) -> list[dict]:
    """
    Pool canonical_rules from all primed per-case JSONs, run the cross-paper
    RELATE gate + NLI (threshold=0 to capture every eligible pair), and return
    the raw NLI pair dicts.

    Results are cached in cache_path so subsequent sweep runs re-use scores
    without re-loading the NLI model.  Pass force=True to recompute.

    Per-paper relate_raw_pairs are always empty (structural: GROUP/CANONICALIZE
    ensure uniqueness of (category, relation_type, subject, outcome) within a
    paper, which is exactly the per-paper gate condition).  The corpus-level gate
    is looser — subject matching via CUI when available — so cross-paper pairs
    with the same entity concept but different surface forms do reach NLI.
    """
    if cache_path.exists() and not force:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        pairs = data.get("raw_nli_pairs", [])
        logger.info("Loaded %d cached corpus NLI pairs from %s", len(pairs), cache_path)
        return pairs

    from pipeline.stages.knowledge_extraction.models import CanonicalRule
    from pipeline.stages.knowledge_extraction.stages.relate_stage import RelateStage
    from pipeline.stages.knowledge_extraction.stages.corpus_relate import _should_compare_cross_paper

    all_rules: list[CanonicalRule] = []
    id_to_pmcid: dict[str, str] = {}

    for path in sorted(prime_dir.glob("*.json")):
        if path.is_dir():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not load %s: %s", path, exc)
            continue
        if raw.get("status") != "success":
            continue
        pmcid = raw.get("pmcid", path.stem)
        for rd in raw.get("canonical_rules") or []:
            try:
                rule = CanonicalRule.model_validate(rd)
                all_rules.append(rule)
                id_to_pmcid[rule.canonical_id] = pmcid
            except Exception as exc:
                logger.debug("Skipping invalid canonical rule in %s: %s", path.name, exc)

    logger.info(
        "Corpus NLI prime: %d canonical rules loaded from %s", len(all_rules), prime_dir
    )
    if len(all_rules) < 2:
        logger.warning("Fewer than 2 canonical rules across all primed JSONs — no pairs")
        return []

    # threshold=0 so every eligible pair is recorded in raw_pairs regardless of label
    stage = RelateStage(entailment_threshold=0.0, contradiction_threshold=0.0)
    _relations, raw_pairs = stage.relate(
        all_rules, pmcid="corpus_sweep", gate=_should_compare_cross_paper,
    )
    logger.info("Corpus NLI prime: %d eligible pairs compared", len(raw_pairs))

    pairs_dicts = []
    for p in raw_pairs:
        d = p.model_dump()
        d["pmcid_a"] = id_to_pmcid.get(p.rule_id_a, "")
        d["pmcid_b"] = id_to_pmcid.get(p.rule_id_b, "")
        pairs_dicts.append(d)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"raw_nli_pairs": pairs_dicts}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved %d corpus NLI pairs → %s", len(pairs_dicts), cache_path)
    return pairs_dicts


# ── RELATE sweep mode ─────────────────────────────────────────────────────────

def run_relate_sweep(
    all_pairs: list[dict],
    ent_thresholds: list[float],
    con_thresholds: list[float],
) -> list[dict]:
    """
    Offline RELATE-threshold sweep over pre-computed corpus NLI pair scores.

    Reports relation-count statistics per (entailment_threshold, contradiction_threshold)
    combo.  Call _build_corpus_nli_pairs() first to obtain all_pairs.
    No silver comparison — RELATE output is at FinalRule level, not Finding level.
    """
    logger.info(
        "Relate sweep: %d total NLI pairs  entailment grid=%s  contradiction grid=%s",
        len(all_pairs), ent_thresholds, con_thresholds,
    )

    rows = []
    for ent_t in ent_thresholds:
        for con_t in con_thresholds:
            counts = {"SUPPORT": 0, "CONTRADICT": 0, "SCOPE_QUALIFY": 0, "UNRELATED": 0}
            for pair in all_pairs:
                label = _classify_pair_offline(pair, ent_t, con_t)
                counts[label] += 1
            row = {
                "entailment_threshold":    round(ent_t, 3),
                "contradiction_threshold": round(con_t, 3),
                "n_support":      counts["SUPPORT"],
                "n_contradict":   counts["CONTRADICT"],
                "n_scope_qualify": counts["SCOPE_QUALIFY"],
                "n_unrelated":    counts["UNRELATED"],
                "n_total_pairs":  len(all_pairs),
            }
            rows.append(row)
            logger.info(
                "  ent=%.2f  con=%.2f  → support=%d  contradict=%d  "
                "scope_qualify=%d  unrelated=%d",
                ent_t, con_t,
                counts["SUPPORT"], counts["CONTRADICT"],
                counts["SCOPE_QUALIFY"], counts["UNRELATED"],
            )

    return rows


# ── CSV writers ───────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows → %s", len(rows), path)


def _print_grounding_table(rows: list[dict]) -> None:
    if not rows:
        return
    best = max(rows, key=lambda r: float(r["f1"]))
    print(f"\n{'─'*76}")
    print("Grounding Threshold Sweep")
    print(f"{'─'*76}")
    print(f"{'Threshold':>10}  {'Precision':>9}  {'Recall':>7}  "
          f"{'F1':>7}  {'Strict F1':>9}  {'Pipeline':>8}")
    print(f"{'─'*76}")
    for r in rows:
        marker = " ← best F1" if r["grounding_threshold"] == best["grounding_threshold"] else ""
        print(f"{r['grounding_threshold']:>10}  "
              f"{float(r['precision']):>9.3f}  {float(r['recall']):>7.3f}  "
              f"{float(r['f1']):>7.3f}  {float(r['strict_f1']):>9.3f}  "
              f"{int(r['n_pipeline']):>8}{marker}")
    print(f"{'─'*76}")
    print(f"\nBest F1: grounding_threshold={best['grounding_threshold']}  "
          f"F1={float(best['f1']):.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline threshold sweep harness"
    )
    parser.add_argument("mode", choices=["prime", "grounding", "relate", "all"],
                        help="Which sweep to run")
    parser.add_argument("--source",       required=True,
                        help="Source-cases JSONL (REQUIRED — every mode reads it; no default).")
    parser.add_argument("--silver",       default=None,
                        help="Silver findings JSONL (REQUIRED for grounding/relate/all; no default).")
    parser.add_argument("--reports",      default=str(REPORTS_DIR))
    parser.add_argument("--prime-dir",    default=str(PRIME_DIR))
    parser.add_argument("--embedder",     default="gemini", choices=["openai", "gemini"])
    parser.add_argument("--embed-cache",  default=None,
                        help="Override embedding cache path")
    parser.add_argument("--sim-threshold", type=float, default=SIMILARITY_THRESHOLD,
                        help="Fixed eval-matching threshold (calibrate with eval.silver.sweep first)")
    parser.add_argument("--split",        default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--dev-fraction", type=float, default=0.8)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--force-reprime", action="store_true",
                        help="Re-run priming even if cached JSONs exist")
    args = parser.parse_args()

    # No silver default — force the dataset choice for the eval modes (B-069 guard).
    if args.mode in ("grounding", "relate", "all") and not args.silver:
        parser.error(f"--silver is required for mode '{args.mode}' (no default — pick the set explicitly)")

    prime_dir   = Path(args.prime_dir)
    reports_dir = Path(args.reports)
    source_path = Path(args.source)
    silver_path = Path(args.silver) if args.silver else None
    timestamp   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    # ── Load source cases and apply split ──
    if not source_path.exists():
        print(f"source_cases.jsonl not found: {source_path}", file=sys.stderr)
        sys.exit(1)
    all_cases = list(read_jsonl(source_path, SourceCase))
    filtered_cases = filter_by_split(all_cases, args.split,
                                     dev_fraction=args.dev_fraction, seed=args.seed)
    logger.info("Split=%s  cases=%d  (seed=%d, dev_fraction=%.2f)",
                args.split, len(filtered_cases), args.seed, args.dev_fraction)

    do_prime     = args.mode in ("prime", "all")
    do_grounding = args.mode in ("grounding", "all")
    do_relate    = args.mode in ("relate", "all")

    # ── PRIME ──
    if do_prime:
        run_prime(filtered_cases, prime_dir, force_reprime=args.force_reprime)

    # ── Embedder (needed for grounding sweep) ──
    if do_grounding:
        if not silver_path.exists():
            print(f"silver_findings.jsonl not found: {silver_path}", file=sys.stderr)
            sys.exit(1)

        silver_by_case: dict[str, SilverCaseResult] = {}
        for rec in read_jsonl(silver_path, SilverCaseResult):
            silver_by_case[rec.case_id] = rec

        if args.embedder == "gemini":
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                print("GOOGLE_API_KEY not set", file=sys.stderr)
                sys.exit(1)
            embedder = GeminiEmbedder(api_key)
            cache_path = Path(args.embed_cache) if args.embed_cache else DEFAULT_GEMINI_CACHE_PATH
            embed_cache = make_embedding_cache(cache_path, GEMINI_EMBEDDING_MODEL)
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("OPENAI_API_KEY not set", file=sys.stderr)
                sys.exit(1)
            embedder = OpenAIEmbedder(api_key)
            cache_path = Path(args.embed_cache) if args.embed_cache else DEFAULT_CACHE_PATH
            embed_cache = make_embedding_cache(cache_path, EMBEDDING_MODEL)

        grounding_rows = run_grounding_sweep(
            filtered_cases, silver_by_case, embedder, embed_cache,
            sim_threshold=args.sim_threshold,
            thresholds=GROUNDING_THRESHOLDS,
            prime_dir=prime_dir,
            split=args.split, seed=args.seed, dev_fraction=args.dev_fraction,
        )
        if grounding_rows:
            csv_path = reports_dir / f"grounding_sweep_{timestamp}.csv"
            _write_csv(csv_path, grounding_rows, fieldnames=[
                "grounding_threshold", "precision", "recall", "f1", "strict_f1",
                "n_matched", "n_silver", "n_pipeline",
                "split", "seed", "dev_fraction", "sim_threshold",
            ])
            _print_grounding_table(grounding_rows)

    # ── RELATE sweep ──
    if do_relate:
        corpus_pairs_cache = prime_dir / "corpus_nli_pairs.json"
        all_pairs = _build_corpus_nli_pairs(
            prime_dir,
            cache_path=corpus_pairs_cache,
            force=args.force_reprime,
        )
        relate_rows = run_relate_sweep(
            all_pairs=all_pairs,
            ent_thresholds=RELATE_ENT_THRESHOLDS,
            con_thresholds=RELATE_CON_THRESHOLDS,
        )
        if relate_rows:
            csv_path = reports_dir / f"relate_sweep_{timestamp}.csv"
            _write_csv(csv_path, relate_rows, fieldnames=[
                "entailment_threshold", "contradiction_threshold",
                "n_support", "n_contradict", "n_scope_qualify",
                "n_unrelated", "n_total_pairs",
            ])
            print(f"\nRelate sweep CSV → {csv_path}")


if __name__ == "__main__":
    main()
