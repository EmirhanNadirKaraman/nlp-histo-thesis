"""
MAP cascade calibration sweep harness.

Submits all L1/L2/L3 batch requests for every dev source case simultaneously,
caches the raw voter outputs, then replays the agreement cascade offline across
a grid of (scorer × theta × reject_theta) — no additional API calls needed.

Calibration axes
----------------
- scorer:        SemanticAgreementScorer over EmbeddingSimilarityStrategy vs.
                 HybridStructuredSimilarity (both honour AgreementConfig / H-EMB-01
                 alignment weights, so weight variants are added as extra specs).
- theta:         escalation threshold (deferral score >= theta → KEEP).
- reject_theta:  hard-reject threshold (deferral score <= reject_theta → REJECT);
                 only pairs with reject_theta < theta are run.

Beyond P/R/F1 vs silver, each row reports deferral safety: early_accept_rate
(fraction of chunks KEPT at L1/L2 without escalating) and early_accept_precision
(silver precision restricted to those early-accepted chunks) — the pair that
tells you whether a low theta is prematurely accepting bad chunks.

CAVEAT: the replay drives the LEGACY AgreementChecker path, NOT MapOutputRouter
(production runs this same legacy L1->L2->L3 cascade — enable_router=false; the
MapOutputRouter L1->L3-skip path is an opt-in experiment, see B-062).
Every row is stamped cascade_path="legacy_agreement_checker"; re-validate any
chosen config against the router before promoting.

Modes
-----
  prime     Build chunk maps, submit 7 batch jobs (one per provider×model),
            save primer state to eval/data/map_primer/primer.json.
  collect   Check job statuses; polls internally (sleeps --poll-interval s
            between checks) until all complete, then builds voter_cache.json
            and prints "Voter cache written."
  sweep     Load voter_cache.json, replay theta grid, emit CSV to eval/reports/.
  all       prime → poll loop (30 s sleep, --poll-interval to override) → collect → sweep in one shot.

Usage
-----
  python -m eval.silver.map_theta_sweep all --source <cases.jsonl> --silver <silver.jsonl>
  python -m eval.silver.map_theta_sweep prime --source <cases.jsonl>
  python -m eval.silver.map_theta_sweep collect
  python -m eval.silver.map_theta_sweep sweep --silver <silver.jsonl> --embedder gemini

Notes
-----
- All three levels (L1, L2, L3) are submitted simultaneously for every case and
  every chunk so that we can replay any (theta, reject_theta, scorer) combination
  without additional API calls. The cache MUST have L3 populated — a sweep run
  warns loudly if it doesn't, because escalated chunks would contribute no findings.
- The agreement embedder uses OpenAI text-embedding-3-small by default; pass
  --embedder gemini to use the Gemini embedding model instead.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

from eval.silver.embedders import GeminiEmbedder
from eval.silver.jsonl_utils import read_jsonl
from eval.silver.matcher import (
    DEFAULT_GEMINI_CACHE_PATH,
    GEMINI_EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD,
    EmbeddingCache,
    make_embedding_cache,
)
from eval.silver.pipeline_sweep import case_to_file_data, _evaluate_outputs
from eval.silver.schemas import (
    PipelineCaseOutput,
    PipelineFinding,
    SilverCaseResult,
    SourceCase,
)
from eval.silver.split import filter_by_split

# ── Paths ─────────────────────────────────────────────────────────────────────

PRIMER_DIR   = Path("eval/data/map_primer")
PRIMER_PATH  = PRIMER_DIR / "primer.json"
CACHE_PATH   = PRIMER_DIR / "voter_cache.json"
REPORTS_DIR  = Path("eval/reports")
SOURCE_PATH  = Path("eval/data/source_cases_related15.jsonl")
SILVER_PATH  = Path("eval/data/silver_findings_related15.jsonl")

# ── Voter configs (production setup) ─────────────────────────────────────────

def _make_voters():
    from pipeline.stages.summarization.batch.voter_configs import (
        make_l1_voters, make_l2_voters, make_l3_voter,
    )
    return make_l1_voters(), make_l2_voters(), make_l3_voter()

THETA_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
REJECT_THETA_GRID = [0.0, 0.1, 0.2]   # 0.0 = "no rejection" baseline (scorer floors scores at 0,
                                      # so rej=-1.0 was a no-op duplicate of 0.0 — dropped 2026-05-26
                                      # after EXP_1 confirmed identical metrics across the two cells).
                                      # Pairs with theta where reject_theta < theta.

# legacy_single_voter_policy — AgreementChecker N=1 handling on the legacy path.
# Default is single-element ("keep") to preserve historical sweep behaviour: every
# row reproduces the silent N=1 → KEEP that AgreementChecker did before the policy
# was added. Pass --legacy-single-voter-policy escalate (or both) on the CLI to
# enable the comparison. Production default mirrors this list's head.
LEGACY_SINGLE_VOTER_POLICY_GRID = ("keep",)

# force_escalate_on_polarity_conflict — B-051 hard-fail policy. Default is
# single-element (True) to preserve historical sweep behaviour: every row
# reproduces the B-051 override (comparable positive/negative findings →
# ESCALATE regardless of scorer output). Pass --force-escalate-on-polarity-conflict
# false (or both) on the CLI to enable the ablation. Production default = True.
FORCE_ESCALATE_ON_POLARITY_CONFLICT_GRID: tuple[bool, ...] = (True,)

# Constant stamped into every sweep row. The replay drives AgreementChecker.compute
# directly — the LEGACY L1->L2->L3 cascade — which IS the production path
# (enable_router=false; decided in B-062, 2026-05-23). The MapOutputRouter
# L1->L3-skip path is an opt-in experiment; if it is ever adopted, re-validate the
# chosen (theta, reject_theta, scorer) on it before promotion. See THESIS.md ABC-P1
# TODO + B-062.
CASCADE_PATH = "legacy_agreement_checker"


@dataclass
class ScorerSpec:
    """One agreement scorer configuration to sweep.

    ``kind`` selects the similarity strategy; ``weights`` is an ``AgreementConfig``
    (H-EMB-01 soft-alignment knobs). Both the embedding and hybrid strategies read
    the weights via ``_align``, so a single spec list can compare scorers *and*
    weight variants. Append entries to vary weights, e.g.
    ``ScorerSpec("embedding_tau030", "embedding", AgreementConfig(tau=0.30))``.
    """
    name: str
    kind: str           # "embedding" | "hybrid"
    weights: Any        # AgreementConfig


def _default_scorer_specs() -> list[ScorerSpec]:
    from pipeline.stages.summarization.config import AgreementConfig
    return [
        ScorerSpec("embedding_default", "embedding", AgreementConfig()),
        ScorerSpec("hybrid_default",    "hybrid",    AgreementConfig()),
    ]


def _build_scorer(spec: ScorerSpec, embed_fn):
    """Build a SemanticAgreementScorer for ``spec`` (theta deferred to AgreementChecker)."""
    from pipeline.stages.summarization.agreement import (
        EmbeddingSimilarityStrategy,
        SemanticAgreementScorer,
    )
    from pipeline.stages.summarization.agreement.hybrid_structured import (
        HybridStructuredSimilarity,
    )
    if spec.kind == "embedding":
        strategy = EmbeddingSimilarityStrategy.from_config(spec.weights, embed_fn=embed_fn)
    elif spec.kind == "hybrid":
        strategy = HybridStructuredSimilarity.from_config(spec.weights, embed_fn=embed_fn)
    else:
        raise ValueError(f"Unknown scorer kind: {spec.kind!r} (expected 'embedding' or 'hybrid')")
    return SemanticAgreementScorer(strategy=strategy)

# ── PrimerHandle ──────────────────────────────────────────────────────────────

@dataclass
class PrimerHandle:
    """Serialisable state for the multi-case batch primer."""

    phase: str                           # "submitted" | "complete"
    case_ids: list[str]                  # original case_ids (contain "|")

    # safe_id → {chunk_id → [sentence dicts]}
    chunk_maps: dict[str, dict[str, list[dict]]] = field(default_factory=dict)

    # safe_id → {chunk_id → formatted source text string}
    source_texts: dict[str, dict[str, str]] = field(default_factory=dict)

    # safe_id → {case_id, pmcid, te_id}
    case_meta: dict[str, dict] = field(default_factory=dict)

    # submitted batch jobs
    jobs: list[dict] = field(default_factory=list)  # ProviderJob.to_dict()

    # strip_thinking flags per (level, voter_idx) — stored so retrieve knows how to parse
    l1_strip: list[bool] = field(default_factory=list)
    l2_strip: list[bool] = field(default_factory=list)
    l3_strip: bool = False

    # custom_id → raw content string (accumulated during collect)
    raw: dict[str, str] = field(default_factory=dict)

    # job_ids already retrieved (so we don't re-download)
    retrieved_job_ids: list[str] = field(default_factory=list)

    def save(self, path: Path = PRIMER_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _to_dict(self) -> dict:
        return {
            "phase":              self.phase,
            "case_ids":           self.case_ids,
            "chunk_maps":         self.chunk_maps,
            "source_texts":       self.source_texts,
            "case_meta":          self.case_meta,
            "jobs":               self.jobs,
            "l1_strip":           self.l1_strip,
            "l2_strip":           self.l2_strip,
            "l3_strip":           self.l3_strip,
            "raw":                self.raw,
            "retrieved_job_ids":  self.retrieved_job_ids,
        }

    @classmethod
    def load(cls, path: Path = PRIMER_PATH) -> PrimerHandle:
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            phase=d["phase"],
            case_ids=d["case_ids"],
            chunk_maps=d.get("chunk_maps", {}),
            source_texts=d.get("source_texts", {}),
            case_meta=d.get("case_meta", {}),
            jobs=d.get("jobs", []),
            l1_strip=d.get("l1_strip", []),
            l2_strip=d.get("l2_strip", []),
            l3_strip=d.get("l3_strip", False),
            raw=d.get("raw", {}),
            retrieved_job_ids=d.get("retrieved_job_ids", []),
        )


# ── PRIME ─────────────────────────────────────────────────────────────────────

def run_prime(cases: list[SourceCase], primer_path: Path = PRIMER_PATH) -> PrimerHandle:
    """Build chunk maps for all cases, submit all L1+L2+L3 batch jobs, save primer."""
    from pipeline.stages.summarization.batch.dispatch import (
        build_requests,
        build_providers,
        OPENAI_MAP_TOOL,
    )
    from pipeline.stages.summarization.stages.map_stage import _format_sentences
    from pipeline.stages.summarization.config import MapConfig

    L1, L2, L3 = _make_voters()

    chunk_size = MapConfig().chunk_size  # default 10

    # Build chunk maps and source texts for every case
    chunk_maps: dict[str, dict[str, list[dict]]] = {}
    source_texts: dict[str, dict[str, str]] = {}
    case_meta: dict[str, dict] = {}

    for case in cases:
        safe_id = case.case_id.replace("|", "_")
        file_data = case_to_file_data(case)
        sentences = file_data["sentences_with_provenance"]

        cmap: dict[str, list[dict]] = {}
        for i in range(0, max(len(sentences), 1), chunk_size):
            cid = f"C{i // chunk_size + 1}"
            cmap[cid] = sentences[i : i + chunk_size]
        if not cmap:
            cmap["C1"] = []

        stexts = {cid: _format_sentences(sents) for cid, sents in cmap.items()}
        chunk_maps[safe_id] = cmap
        source_texts[safe_id] = stexts
        case_meta[safe_id] = {
            "case_id": case.case_id,
            "pmcid":   case.pmcid,
            "te_id":   case.te_id,
        }

    logger.info("Built chunk maps for %d cases", len(cases))

    # Build all requests grouped by (provider, model)
    from pipeline.stages.summarization.batch.models import BatchRequest
    by_prov_model: dict[tuple[str, str], list[BatchRequest]] = {}

    for safe_id, cmap in chunk_maps.items():
        for level, voters in [("l1", L1), ("l2", L2), ("l3", [L3])]:
            reqs = build_requests(cmap, safe_id, voters, level=level)
            for req in reqs:
                key = (req.provider, req.model)
                by_prov_model.setdefault(key, []).append(req)

    total_reqs = sum(len(v) for v in by_prov_model.values())
    logger.info(
        "Total requests: %d across %d (provider, model) groups",
        total_reqs, len(by_prov_model),
    )
    for (prov, model), reqs in by_prov_model.items():
        logger.info("  %s / %s  → %d requests", prov, model, len(reqs))

    # Build handle first so we can save incrementally after each submission.
    handle = PrimerHandle(
        phase="submitted",
        case_ids=[c.case_id for c in cases],
        chunk_maps=chunk_maps,
        source_texts=source_texts,
        case_meta=case_meta,
        l1_strip=[v.strip_thinking for v in L1],
        l2_strip=[v.strip_thinking for v in L2],
        l3_strip=L3.strip_thinking,
    )
    handle.save(primer_path)  # write skeleton so partial progress survives a crash

    providers = build_providers({k[0] for k in by_prov_model})

    for (prov, model), reqs in by_prov_model.items():
        job = providers[prov].submit(reqs, OPENAI_MAP_TOOL)
        logger.info("Submitted %s/%s: job_id=%s", prov, model, job.job_id)
        handle.jobs.append(job.to_dict())
        handle.save(primer_path)  # persist after each successful submission
    handle.save(primer_path)
    logger.info("Primer state saved → %s", primer_path)
    return handle


# ── RETRY FAILED ──────────────────────────────────────────────────────────────

def run_retry_failed(primer_path: Path = PRIMER_PATH) -> PrimerHandle:
    """Resubmit only the failed batch jobs, preserving all already-collected raw results.

    Identifies failed jobs by (provider, model), rebuilds the exact same requests
    (preserving voter indices so custom_ids are identical to the originals), and
    replaces the failed job entries in primer.json with the new job IDs.
    Run ``collect`` afterwards to retrieve results and rebuild the voter cache.
    """
    from pipeline.stages.summarization.batch.dispatch import build_requests, build_providers, OPENAI_MAP_TOOL
    from pipeline.stages.summarization.batch.models import ProviderJob

    handle = PrimerHandle.load(primer_path)
    L1, L2, L3 = _make_voters()

    # Build (provider, model) → (level, full voter list) so voter indices are preserved
    level_voter_map: dict[tuple[str, str], tuple[str, list]] = {}
    for v in L1:
        level_voter_map[(v.provider, v.model)] = ("l1", L1)
    for v in L2:
        level_voter_map[(v.provider, v.model)] = ("l2", L2)
    level_voter_map[(L3.provider, L3.model)] = ("l3", [L3])

    failed_jobs = [ProviderJob.from_dict(j) for j in handle.jobs if j["status"] == "failed"]
    if not failed_jobs:
        logger.info("No failed jobs in primer — nothing to retry.")
        return handle

    logger.info("Retrying %d failed job(s):", len(failed_jobs))
    for j in failed_jobs:
        logger.info("  %s / %s  job_id=%s", j.provider, j.model, j.job_id)

    # Build requests for each failed (provider, model), passing the full voter list
    # so custom_id voter indices (e.g. __l1__1) match the original submission exactly.
    by_key: dict[tuple[str, str], list] = {}
    for job in failed_jobs:
        key = (job.provider, job.model)
        if key not in level_voter_map:
            logger.warning(
                "Cannot determine level for %s/%s — skipping (add to level_voter_map if needed)",
                job.provider, job.model,
            )
            continue
        level, voters = level_voter_map[key]
        for safe_id, cmap in handle.chunk_maps.items():
            all_reqs = build_requests(cmap, safe_id, voters, level=level)
            # Keep only requests belonging to this specific (provider, model)
            by_key.setdefault(key, []).extend(
                r for r in all_reqs if r.provider == job.provider and r.model == job.model
            )

    if not by_key:
        logger.warning("No requests built — aborting retry.")
        return handle

    for (prov, model), reqs in by_key.items():
        logger.info("  %s / %s  → %d requests to resubmit", prov, model, len(reqs))

    # Remove failed entries from handle; clear them from retrieved_job_ids too
    failed_ids = {j.job_id for j in failed_jobs}
    handle.jobs = [j for j in handle.jobs if j["job_id"] not in failed_ids]
    handle.retrieved_job_ids = [jid for jid in handle.retrieved_job_ids if jid not in failed_ids]
    handle.phase = "submitted"

    providers = build_providers({k[0] for k in by_key})
    for (prov, model), reqs in by_key.items():
        job = providers[prov].submit(reqs, OPENAI_MAP_TOOL)
        logger.info("Resubmitted %s/%s: job_id=%s (%d requests)", prov, model, job.job_id, len(reqs))
        handle.jobs.append(job.to_dict())
        handle.save(primer_path)  # persist after each submission so a crash loses at most one job

    handle.save(primer_path)
    logger.info("Primer updated → %s  (run 'collect' to retrieve results)", primer_path)
    return handle


# ── COLLECT ───────────────────────────────────────────────────────────────────

def _strip_flag(custom_id: str, handle: PrimerHandle) -> bool:
    """Determine strip_thinking flag from custom_id level and voter index."""
    parts = custom_id.split("__")
    if len(parts) < 4:
        return False
    level, vi_str = parts[-2], parts[-1]
    vi = int(vi_str) if vi_str.isdigit() else 0
    if level == "l1":
        return handle.l1_strip[vi] if vi < len(handle.l1_strip) else False
    if level == "l2":
        return handle.l2_strip[vi] if vi < len(handle.l2_strip) else False
    return handle.l3_strip


def run_collect(
    handle: PrimerHandle,
    primer_path: Path = PRIMER_PATH,
    cache_path: Path = CACHE_PATH,
) -> tuple[PrimerHandle, bool]:
    """
    Check all jobs; retrieve results for completed jobs.
    Returns (updated_handle, is_complete).
    """
    from pipeline.stages.summarization.batch.dispatch import build_providers
    from pipeline.stages.summarization.batch.models import ProviderJob

    jobs = [ProviderJob.from_dict(d) for d in handle.jobs]
    providers_needed = {j.provider for j in jobs}
    providers = build_providers(providers_needed)

    # Refresh statuses
    for job in jobs:
        if job.status in ("completed", "failed"):
            continue
        p = providers.get(job.provider)
        if p:
            p.check(job)

    # Retrieve results for newly completed jobs
    for job in jobs:
        if job.job_id in handle.retrieved_job_ids:
            continue
        if job.status == "completed":
            p = providers.get(job.provider)
            if p:
                results = p.retrieve(job)
                for res in results:
                    if res.content:
                        handle.raw[res.custom_id] = res.content
                handle.retrieved_job_ids.append(job.job_id)
                logger.info("Retrieved %d results from job %s", len(results), job.job_id)
        elif job.status == "failed":
            logger.warning("Job %s failed", job.job_id)
            handle.retrieved_job_ids.append(job.job_id)  # mark as done to avoid re-checking

    # Update serialised job list with refreshed statuses
    handle.jobs = [j.to_dict() for j in jobs]

    pending = [j for j in jobs if j.status not in ("completed", "failed")]
    done    = [j for j in jobs if j.status in ("completed", "failed")]
    logger.info("%d/%d jobs complete", len(done), len(jobs))

    all_done = len(pending) == 0
    if all_done:
        handle.phase = "complete"
        handle.save(primer_path)  # raw content persisted before cache build
        _build_voter_cache(handle, cache_path)
    else:
        handle.save(primer_path)

    return handle, all_done


def rebuild_cache_from_primer(primer_path: Path = PRIMER_PATH,
                               cache_path: Path = CACHE_PATH) -> None:
    """Rebuild voter_cache.json from a saved primer without any API calls.

    Use this after a parse failure in collect — the raw content is already
    in primer.json, so no jobs need to be re-submitted or re-retrieved.
    """
    handle = PrimerHandle.load(primer_path)
    _build_voter_cache(handle, cache_path)


def _build_voter_cache(handle: PrimerHandle, cache_path: Path = CACHE_PATH) -> None:
    """Parse handle.raw into a per-case voter cache and write to CACHE_PATH."""
    from pipeline.stages.summarization.batch.dispatch import parse_result
    from pipeline.stages.summarization.batch.models import BatchResult

    cache: dict[str, dict] = {}

    for safe_id, meta in handle.case_meta.items():
        cmap = handle.chunk_maps.get(safe_id, {})
        stexts = handle.source_texts.get(safe_id, {})
        entry: dict[str, Any] = {
            "case_id":      meta["case_id"],
            "pmcid":        meta["pmcid"],
            "te_id":        meta["te_id"],
            "source_texts": stexts,
            "chunk_map":    cmap,
            "l1":           {cid: [] for cid in cmap},
            "l2":           {cid: [] for cid in cmap},
            "l3":           {cid: None for cid in cmap},
        }
        cache[safe_id] = entry

    # Parse every raw result into the appropriate slot
    for custom_id, content in handle.raw.items():
        parts = custom_id.split("__")
        if len(parts) != 4:
            logger.debug("Skipping malformed custom_id: %s", custom_id)
            continue
        safe_id, chunk_id, level, vi_str = parts
        if safe_id not in cache:
            logger.debug("Unknown safe_id in custom_id: %s", custom_id)
            continue

        strip = _strip_flag(custom_id, handle)
        parsed = parse_result(
            BatchResult(custom_id=custom_id, content=content),
            strip_thinking=strip,
        )
        if parsed is None:
            logger.warning("Failed to parse %s", custom_id)
            continue

        vi = int(vi_str) if vi_str.isdigit() else 0
        entry = cache[safe_id]

        if level == "l3":
            entry["l3"][chunk_id] = parsed.model_dump()
        elif level in ("l1", "l2"):
            slot: list = entry[level].setdefault(chunk_id, [])
            # Expand list to hold slot for voter vi
            while len(slot) <= vi:
                slot.append(None)
            slot[vi] = parsed.model_dump()

    # Count parsed outputs
    n_l1 = sum(
        sum(1 for v in cmap.values() for x in (v if isinstance(v, list) else []) if x)
        for cmap in [entry["l1"] for entry in cache.values()]
    )
    logger.info("Voter cache: %d cases, %d L1 outputs parsed", len(cache), n_l1)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Voter cache written → %s", cache_path)


# ── SWEEP ─────────────────────────────────────────────────────────────────────

def _ev(x):
    """Enum → its ``.value`` (a plain string); anything else unchanged.

    ``AuditableSummary.model_dump()`` yields Enum OBJECTS for relation_type /
    direction / confidence / category, so a bare ``str(enum)`` produces
    ``'RelationTypeEnum.demographic'`` — which never matches the silver string
    ``'demographic'`` and silently strict-penalises every matched finding (B-074).
    Raw cached dicts already hold the value string, so this is a no-op for them.
    """
    return getattr(x, "value", x)


def _finding_to_pipeline(f: dict, pmcid: str, chunk_id: str, theta: float) -> PipelineFinding:
    scope = f.get("scope") or {}
    return PipelineFinding(
        pipeline_run_id=0,
        run_id=f"map_theta_{theta:.2f}",
        pmcid=pmcid,
        chunk_id=chunk_id,
        category=str(_ev(f.get("category", ""))),
        claim=f.get("claim", ""),
        subject_entity=f.get("subject_entity"),
        outcome_entity=f.get("outcome_entity"),
        relation_type=str(_ev(f.get("relation_type", "unclear"))),
        direction=str(_ev(f["direction"])) if f.get("direction") else None,
        confidence=str(_ev(f.get("confidence", "medium"))),
        verbatim_support=f.get("verbatim_support", ""),
        evidence=f.get("evidence") or [],
        grounding_score=f.get("grounding_score"),
        scope_disease_subtype=scope.get("disease_subtype"),
        scope_cohort_n=scope.get("cohort_n"),
        scope_assay_method=scope.get("assay_method"),
        scope_biomarker_cutoff=scope.get("biomarker_cutoff"),
        scope_tissue_site=scope.get("tissue_site"),
        scope_treatment_context=scope.get("treatment_context"),
        scope_endpoint=scope.get("endpoint"),
        scope_study_design=scope.get("study_design"),
    )


def _prewarm_agreement_cache(
    voter_cache: dict[str, dict],
    embed_fn,
    disk_cache: "EmbeddingCache",
    batch_size: int = 100,
) -> None:
    """Extract every unique claim string from the voter cache and embed any misses.

    Results are written into disk_cache (persistent JSON file) so subsequent
    sweep runs skip this step entirely.  disk_cache.save() is called after each
    batch so progress survives a crash.
    """
    from pipeline.stages.summarization.agreement.embedding import _claims
    from pipeline.stages.summarization.models import AuditableSummary

    texts: set[str] = set()
    for entry in voter_cache.values():
        for level in ("l1", "l2"):
            for data in entry.get(level, {}).values():
                for d in (data or []):
                    if d:
                        try:
                            s = AuditableSummary.model_validate(d)
                            texts.update(_claims(s))
                        except Exception:
                            pass
        for d in entry.get("l3", {}).values():
            if d:
                try:
                    s = AuditableSummary.model_validate(d)
                    texts.update(_claims(s))
                except Exception:
                    pass

    misses = [t for t in sorted(texts) if disk_cache.get(t) is None]
    logger.info(
        "Agreement embed pre-warm: %d unique claims, %d cache misses",
        len(texts), len(misses),
    )
    for i in range(0, len(misses), batch_size):
        batch = misses[i : i + batch_size]
        embs = embed_fn(batch)
        for t, e in zip(batch, embs):
            disk_cache.set(t, e)
        disk_cache.save()
        logger.info("  embedded %d / %d", min(i + batch_size, len(misses)), len(misses))
    logger.info("Agreement embed pre-warm complete")


def _make_cached_embed_fn(disk_cache: "EmbeddingCache", embed_fn):
    """Wrap embed_fn with disk_cache so any remaining misses are persisted."""
    def _fn(texts: list[str]) -> list[list[float]]:
        result: list[list[float] | None] = [None] * len(texts)
        miss_idx: list[int] = []
        miss_texts: list[str] = []
        for i, t in enumerate(texts):
            cached = disk_cache.get(t)
            if cached is not None:
                result[i] = cached
            else:
                miss_idx.append(i)
                miss_texts.append(t)
        if miss_texts:
            new_embs = embed_fn(miss_texts)
            for idx, t, e in zip(miss_idx, miss_texts, new_embs):
                disk_cache.set(t, e)
                result[idx] = e
            disk_cache.save()
        return result  # type: ignore[return-value]
    return _fn


def _replay(
    voter_cache: dict[str, dict],
    scorer,
    theta: float,
    reject_theta: float,
    single_voter_policy: str = "keep",
    force_escalate_on_polarity_conflict: bool = True,
    citation_enabled: bool = True,
    citation_check_verbatim: bool = False,
    citation_fabricated_threshold: float = 0.25,
) -> tuple[list[PipelineCaseOutput], dict[str, int], dict[str, set[str]], int, dict[str, int]]:
    """Replay the legacy L1→L2→L3 cascade for one cell of the sweep grid.

    ``single_voter_policy`` mirrors ``RoutingConfig.legacy_single_voter_policy``:
    ``"keep"`` (legacy default) accepts a single surviving voter at N=1 with
    confidence=1.0; ``"escalate"`` treats N=1 as low-evidence and routes the
    chunk up the cascade (L1→L2, L2→L3). Replay-only — no API calls.

    ``force_escalate_on_polarity_conflict`` mirrors
    ``AgreementConfig.force_escalate_on_polarity_conflict``: ``True`` (legacy
    default, B-051 safety guard) overrides the scorer decision to ESCALATE
    when a comparable positive/negative pair is detected. ``False`` records
    the conflict in ``score_details`` but lets the scorer's natural decision
    stand — see the docstring on ``AgreementChecker``.

    Returns ``(case_outputs, accept_counts, early_chunks, n_polarity_conflicts, invoked)``:
      * ``accept_counts`` buckets every chunk by where it RESOLVED —
        ``l1`` / ``l2`` (early KEEP), ``l3`` (escalated), ``reject``, ``no_data``.
      * ``early_chunks`` maps ``case_id → {chunk_id}`` for chunks accepted early
        (L1 or L2 KEEP), used to compute deferral-safety precision.
      * ``n_polarity_conflicts`` counts chunks where ``score_details["hard_fail_reason"]``
        equalled ``"polarity_conflict"`` — independent of whether the flag
        forced escalation, so the count is comparable across both flag values.
      * ``invoked`` counts per-tier INVOCATIONS (a tier actually ran for the chunk),
        distinct from where it RESOLVED: ``l1`` runs on every chunk with L1 data;
        ``l2`` only when L1 escalated; ``l3`` only when L2 escalated (or had no data).
        Needed for cost (cost ∝ invocations × voters/tier); ``early_accept_rate``
        sums l1+l2 *accepts* and so can't recover the L1↔L2 split on its own.
        Invariant: ``invoked["l3"] == accept_counts["l3"]``.
    """
    from pipeline.stages.summarization.agreement import AgreementChecker
    from pipeline.stages.summarization.interfaces.scoring import ChunkDecision
    from pipeline.stages.summarization.models import AuditableSummary
    from pipeline.stages.summarization.helpers.citation_filter import citation_drop_indices

    checker = AgreementChecker(
        scorer,
        theta=theta,
        reject_theta=reject_theta,
        single_voter_policy=single_voter_policy,  # type: ignore[arg-type]
        force_escalate_on_polarity_conflict=force_escalate_on_polarity_conflict,
    )
    outputs: list[PipelineCaseOutput] = []
    accept_counts = {"l1": 0, "l2": 0, "l3": 0, "reject": 0, "no_data": 0}
    # Per-tier INVOCATION counts (a tier actually ran), distinct from where the chunk
    # RESOLVED above. Cost ∝ invocations × voters/tier; see the return docstring.
    invoked = {"l1": 0, "l2": 0, "l3": 0}
    early_chunks: dict[str, set[str]] = {}
    # Count chunks where the polarity-conflict marker fired. Counted once per
    # chunk (at L1) — the same chunk re-evaluated at L2 would double-count.
    # Independent of the flag value: with True the chunk escalates, with False
    # it doesn't, but the marker is recorded in both cases.
    n_polarity_conflicts = 0

    def _is_polarity_conflict(bundle) -> bool:
        det = bundle.score_details if bundle is not None else None
        return bool(det and det.get("hard_fail_reason") == "polarity_conflict")

    for _safe_id, entry in voter_cache.items():
        case_id  = entry["case_id"]
        pmcid    = entry["pmcid"]
        te_id    = entry["te_id"]
        findings: list[PipelineFinding] = []
        early_chunks.setdefault(case_id, set())

        for chunk_id in entry.get("chunk_map", {}):
            source_text = entry.get("source_texts", {}).get(chunk_id, "")

            l1_raw = entry.get("l1", {}).get(chunk_id) or []
            l1_out = [AuditableSummary.model_validate(d) for d in l1_raw if d]

            selected: dict | None = None
            accept_level = "no_data"
            chunk_had_polarity_conflict = False

            if l1_out:
                invoked["l1"] += 1
                b1 = checker.compute(l1_out, source_text=source_text)
                chunk_had_polarity_conflict = _is_polarity_conflict(b1)
                if b1.decision == ChunkDecision.KEEP:
                    selected = checker.best(l1_out, b1).model_dump()
                    accept_level = "l1"
                elif b1.decision == ChunkDecision.REJECT:
                    accept_level = "reject"
                # else ESCALATE → fall through to L2

            if selected is None and accept_level == "no_data":
                # Escalation path: L1 escalated, or no L1 data.
                l2_raw = entry.get("l2", {}).get(chunk_id) or []
                l2_out = [AuditableSummary.model_validate(d) for d in l2_raw if d]
                if l2_out:
                    invoked["l2"] += 1
                    b2 = checker.compute(l2_out, source_text=source_text)
                    # Count L2 polarity conflicts only when L1 didn't already
                    # surface one for this chunk — keeps the count "per chunk",
                    # not "per agreement-gate call".
                    if not chunk_had_polarity_conflict and _is_polarity_conflict(b2):
                        chunk_had_polarity_conflict = True
                    if b2.decision == ChunkDecision.KEEP:
                        selected = checker.best(l2_out, b2).model_dump()
                        accept_level = "l2"
                    elif b2.decision == ChunkDecision.REJECT:
                        accept_level = "reject"
                    else:  # ESCALATE → L3
                        invoked["l3"] += 1
                        l3_raw = entry.get("l3", {}).get(chunk_id)
                        if l3_raw:
                            selected = l3_raw
                        accept_level = "l3"
                else:  # no L2 data → L3
                    invoked["l3"] += 1
                    l3_raw = entry.get("l3", {}).get(chunk_id)
                    if l3_raw:
                        selected = l3_raw
                    accept_level = "l3"

            accept_counts[accept_level] += 1
            if accept_level in ("l1", "l2"):
                early_chunks[case_id].add(chunk_id)
            if chunk_had_polarity_conflict:
                n_polarity_conflicts += 1

            if selected is not None:
                raw_findings = selected.get("findings", [])
                # Citation filter (B-080) — mirror production: drop findings whose
                # citation fails structural validation against the chunk source
                # index. Index-filter the RAW cached dicts so kept findings stay
                # byte-identical (the off/on delta is purely the citation effect).
                if citation_enabled and raw_findings:
                    chunk = entry.get("chunk_map", {}).get(chunk_id) or []
                    summary = AuditableSummary.model_validate(selected)
                    if len(summary.findings) == len(raw_findings):
                        drop = citation_drop_indices(
                            summary, chunk, pmcid,
                            check_verbatim=citation_check_verbatim,
                            fabricated_threshold=citation_fabricated_threshold,
                        )
                        if drop:
                            raw_findings = [
                                f for i, f in enumerate(raw_findings) if i not in drop
                            ]
                for f in raw_findings:
                    findings.append(_finding_to_pipeline(f, pmcid, chunk_id, theta))

        outputs.append(PipelineCaseOutput(
            case_id=case_id,
            pmcid=pmcid,
            te_id=te_id,
            run_id=f"map_t{theta:.2f}_r{reject_theta:.2f}",
            pipeline_run_id=0,
            findings=findings,
        ))

    return outputs, accept_counts, early_chunks, n_polarity_conflicts, invoked


def run_sweep(
    voter_cache: dict[str, dict],
    silver_by_case: dict[str, SilverCaseResult],
    embedder: object,
    embed_cache: EmbeddingCache,
    sim_threshold: float,
    scorer_specs: list[ScorerSpec],
    thetas: list[float],
    reject_thetas: list[float],
    split: str,
    seed: int,
    dev_fraction: float,
    agreement_embed_fn=None,
    single_voter_policies: list[str] | tuple[str, ...] = LEGACY_SINGLE_VOTER_POLICY_GRID,
    force_escalate_on_polarity_conflict_grid: list[bool] | tuple[bool, ...] = (
        FORCE_ESCALATE_ON_POLARITY_CONFLICT_GRID
    ),
) -> list[dict]:
    """Joint sweep over scorer × theta × reject_theta × legacy_single_voter_policy
    × force_escalate_on_polarity_conflict.

    ``single_voter_policies`` defaults to ``("keep",)`` and
    ``force_escalate_on_polarity_conflict_grid`` defaults to ``(True,)`` so
    existing callers reproduce historical behaviour. Pass two-element grids to
    compare policies side-by-side; every cell in the inner product runs
    independently.
    """
    rows: list[dict] = []
    n_skipped = 0
    for spec in scorer_specs:
        scorer = _build_scorer(spec, agreement_embed_fn)  # built once, reused across grid
        for theta in thetas:
            for reject_theta in reject_thetas:
                if reject_theta >= theta:
                    n_skipped += 1
                    continue
                for policy in single_voter_policies:
                    for force_escalate_polarity in force_escalate_on_polarity_conflict_grid:
                        case_outputs, accept, early_chunks, n_polarity, invoked = _replay(
                            voter_cache, scorer, theta, reject_theta,
                            single_voter_policy=policy,
                            force_escalate_on_polarity_conflict=force_escalate_polarity,
                        )
                        # Keep only cases present in silver (dev split filtered at load time).
                        case_outputs = [co for co in case_outputs if co.case_id in silver_by_case]
                        metrics = _evaluate_outputs(
                            case_outputs, silver_by_case, embedder, embed_cache, sim_threshold,
                        )

                        # Deferral safety: precision restricted to findings from chunks the
                        # cascade accepted EARLY (L1/L2 KEEP, not escalated to L3).
                        early_outputs = [
                            PipelineCaseOutput(
                                case_id=co.case_id, pmcid=co.pmcid, te_id=co.te_id,
                                run_id=co.run_id, pipeline_run_id=0,
                                findings=[
                                    f for f in co.findings
                                    if f.chunk_id in early_chunks.get(co.case_id, set())
                                ],
                            )
                            for co in case_outputs
                        ]
                        early_metrics = _evaluate_outputs(
                            early_outputs, silver_by_case, embedder, embed_cache, sim_threshold,
                        )
                        total_chunks = sum(accept.values()) or 1

                        # Hybrid blend weights live on AgreementConfig.hybrid
                        # (added 2026-05-26). Always emit the 4 + sum so any
                        # variant in Stage 1b (or future hybrid-blend sweeps)
                        # is auditable in the CSV. For non-hybrid rows, they
                        # show the AgreementConfig defaults — meaningful only
                        # to confirm the embedding scorer ignores them.
                        _h = getattr(spec.weights, "hybrid", None)
                        wc = _h.w_category if _h is not None else None
                        we = _h.w_embedding if _h is not None else None
                        wn = _h.w_entity if _h is not None else None
                        wv = _h.w_evidence if _h is not None else None
                        wsum = (
                            round(wc + we + wn + wv, 4)
                            if None not in (wc, we, wn, wv) else None
                        )
                        row = {
                            "scorer":                spec.name,
                            "tau":                   spec.weights.tau,
                            "count_alpha":           spec.weights.count_alpha,
                            "reuse_weight":          spec.weights.reuse_weight,
                            "contradiction_weight":  spec.weights.contradiction_weight,
                            "w_category":            wc,
                            "w_embedding":           we,
                            "w_entity":              wn,
                            "w_evidence":            wv,
                            "weights_sum":           wsum,
                            "theta":                 round(theta, 2),
                            "reject_theta":          round(reject_theta, 2),
                            "legacy_single_voter_policy": policy,
                            "force_escalate_on_polarity_conflict": str(bool(force_escalate_polarity)).lower(),
                            "cascade_path":          CASCADE_PATH,
                            "early_accept_rate":     round((accept["l1"] + accept["l2"]) / total_chunks, 4),
                            "escalate_rate":         round(accept["l3"] / total_chunks, 4),
                            # Per-tier invocation counts (cost ∝ invocations × voters/tier).
                            # l1/l2 ACCEPTS are summed in early_accept_rate, so the L1↔L2
                            # split — hence L2 cost — is only recoverable from these.
                            "n_chunks":              total_chunks,
                            "n_l1_invoked":          invoked["l1"],
                            "n_l2_invoked":          invoked["l2"],
                            "n_l3_invoked":          invoked["l3"],
                            "early_accept_precision": early_metrics["precision"],
                            "n_polarity_conflict_chunks": int(n_polarity),
                            "polarity_conflict_rate":    round(n_polarity / total_chunks, 4),
                            "split":                 split,
                            "seed":                  seed,
                            "dev_fraction":          dev_fraction,
                            "sim_threshold":         sim_threshold,
                            **metrics,
                        }
                        rows.append(row)
                        logger.info(
                            "  %-18s theta=%.2f rej=%.2f  pol=%-8s pol_fail=%-5s "
                            "F1=%.3f strictF1=%.3f  earlyP=%.3f earlyRate=%.2f "
                            "esc=%.2f  n_pol=%d  (pipeline=%d)",
                            spec.name, theta, reject_theta, policy,
                            str(bool(force_escalate_polarity)).lower(),
                            metrics["f1"], metrics["strict_f1"],
                            early_metrics["precision"], row["early_accept_rate"],
                            row["escalate_rate"], n_polarity, metrics["n_pipeline"],
                        )
    if n_skipped:
        logger.info("Skipped %d (theta, reject_theta) pairs where reject_theta >= theta", n_skipped)
    return rows


# ── Reporting ─────────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows → %s", len(rows), path)


def _print_table(rows: list[dict]) -> None:
    if not rows:
        return
    best = max(rows, key=lambda r: float(r["f1"]))
    width = 114
    print(f"\n{'─'*width}")
    print("MAP cascade calibration sweep  (path: legacy AgreementChecker — re-validate vs router)")
    print(f"{'─'*width}")
    print(f"{'Scorer':>18}  {'Theta':>5}  {'RejT':>5}  {'Pol':>8}  {'F1':>6}  {'StrictF1':>8}  "
          f"{'EarlyP':>6}  {'EarlyR':>6}  {'Esc':>5}  {'Pipe':>5}")
    print(f"{'─'*width}")
    for r in rows:
        is_best = (r["scorer"] == best["scorer"] and r["theta"] == best["theta"]
                   and r["reject_theta"] == best["reject_theta"]
                   and r.get("legacy_single_voter_policy") == best.get("legacy_single_voter_policy"))
        marker = " ← best F1" if is_best else ""
        pol = str(r.get("legacy_single_voter_policy", "keep"))
        print(f"{str(r['scorer']):>18}  {float(r['theta']):>5.2f}  {float(r['reject_theta']):>5.2f}  "
              f"{pol:>8}  "
              f"{float(r['f1']):>6.3f}  {float(r['strict_f1']):>8.3f}  "
              f"{float(r['early_accept_precision']):>6.3f}  {float(r['early_accept_rate']):>6.2f}  "
              f"{float(r['escalate_rate']):>5.2f}  {int(r['n_pipeline']):>5}{marker}")
    print(f"{'─'*width}")
    print(f"\nBest F1: scorer={best['scorer']} theta={best['theta']:.2f} "
          f"reject_theta={best['reject_theta']:.2f} "
          f"legacy_single_voter_policy={best.get('legacy_single_voter_policy', 'keep')} "
          f" F1={float(best['f1']):.3f}  "
          f"early_accept_precision={float(best['early_accept_precision']):.3f}")
    print("EarlyP/EarlyR = precision/rate of chunks accepted at L1-L2 (deferral safety); "
          "Esc = fraction escalated to L3. Pol = legacy_single_voter_policy.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MAP theta sweep harness")
    parser.add_argument("mode", choices=["prime", "collect", "rebuild-cache", "retry-failed", "sweep", "all"])
    parser.add_argument("--source",        default=None,
                        help="Source-cases JSONL (REQUIRED for prime/all; no default — "
                             "pick the set explicitly).")
    parser.add_argument("--silver",        default=None,
                        help="Silver findings JSONL (REQUIRED for sweep/all; no default — "
                             "pick the set explicitly).")
    parser.add_argument("--reports",       default=str(REPORTS_DIR))
    parser.add_argument("--embedder",      default="openai", choices=["openai", "gemini", "both"],
                        help="Embedding model for BOTH agreement scoring and silver matching. "
                             "'both' runs the sweep under each model and writes one CSV with an "
                             "`embedder` column (--embed-cache is ignored in that mode).")
    parser.add_argument("--embed-cache",   default=None)
    parser.add_argument("--sim-threshold", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--split",         default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--dev-fraction",  type=float, default=0.8)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Seconds between status polls in collect/all mode")
    parser.add_argument("--n-cases", type=int, default=None,
                        help="Limit to first N cases (for smoke testing)")
    parser.add_argument("--primer-dir", default=None,
                        help="Override primer/cache directory (default: eval/data/map_primer)")
    parser.add_argument(
        "--legacy-single-voter-policy",
        default="keep",
        choices=["keep", "escalate", "both"],
        help=(
            "Sweep axis for RoutingConfig.legacy_single_voter_policy. "
            "'keep' (default) reproduces historical behaviour — every cell uses "
            "the silent N=1 → KEEP. 'escalate' sweeps only the new branch "
            "(N=1 → ESCALATE). 'both' produces a side-by-side comparison "
            "(double the row count). No API calls; pure offline replay."
        ),
    )
    parser.add_argument(
        "--force-escalate-on-polarity-conflict",
        default="true",
        choices=["true", "false", "both"],
        help=(
            "Sweep axis for AgreementConfig.force_escalate_on_polarity_conflict. "
            "'true' (default) preserves the B-051 safety guard — comparable "
            "positive/negative findings override the scorer decision to ESCALATE. "
            "'false' records the marker in score_details but lets the scorer "
            "decision stand (ablation only — polarity overrides exist for "
            "semantic safety). 'both' produces a side-by-side comparison "
            "(double the row count). No API calls; pure offline replay."
        ),
    )
    args = parser.parse_args()

    # No path defaults — force the dataset choice so a sweep can't silently run on
    # the calibration set when the held-out set was intended (B-069 guard).
    if args.mode in ("prime", "all") and not args.source:
        parser.error(f"--source is required for mode '{args.mode}' (no default — pick the set explicitly)")
    if args.mode in ("sweep", "all") and not args.silver:
        parser.error(f"--silver is required for mode '{args.mode}' (no default — pick the set explicitly)")

    source_path = Path(args.source) if args.source else None
    silver_path = Path(args.silver) if args.silver else None
    reports_dir = Path(args.reports)
    timestamp   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    primer_dir       = Path(args.primer_dir) if args.primer_dir else PRIMER_DIR
    primer_path      = primer_dir / "primer.json"
    voter_cache_path = primer_dir / "voter_cache.json"

    # ── PRIME ──
    # Only prime/all consume the source-case split. Loading it for
    # collect/sweep is wasteful AND the old "cases=N" log line misled a
    # user into thinking `collect` had submitted N cases. collect/sweep operate
    # off primer.json / voter_cache.json, never the case list — so the source
    # split is loaded only when we actually prime.
    if args.mode in ("prime", "all"):
        if primer_path.exists():
            logger.info("Primer already exists at %s — loading", primer_path)
            handle = PrimerHandle.load(primer_path)
        else:
            if not source_path.exists():
                print(f"source_cases.jsonl not found: {source_path}", file=sys.stderr)
                sys.exit(1)
            all_cases = list(read_jsonl(source_path, SourceCase))
            filtered_cases = filter_by_split(
                all_cases, args.split, dev_fraction=args.dev_fraction, seed=args.seed,
            )
            if args.n_cases is not None:
                filtered_cases = filtered_cases[: args.n_cases]
            logger.info("Priming Split=%s  cases=%d  (seed=%d, dev_fraction=%.2f)",
                        args.split, len(filtered_cases), args.seed, args.dev_fraction)
            handle = run_prime(filtered_cases, primer_path)

    # ── REBUILD CACHE (no API calls) ──
    if args.mode == "rebuild-cache":
        if not primer_path.exists():
            print(f"No primer found at {primer_path}.", file=sys.stderr)
            sys.exit(1)
        rebuild_cache_from_primer(primer_path, voter_cache_path)
        print("Voter cache rebuilt from saved primer. Ready to sweep.")
        return

    # ── RETRY FAILED ──
    if args.mode == "retry-failed":
        if not primer_path.exists():
            print(f"No primer found at {primer_path}. Run `prime` first.", file=sys.stderr)
            sys.exit(1)
        run_retry_failed(primer_path)
        print("Failed jobs resubmitted. Run `collect` to retrieve results.")
        return

    # ── COLLECT / poll loop ──
    if args.mode == "collect":
        if not primer_path.exists():
            print("No primer found. Run `prime` first.", file=sys.stderr)
            sys.exit(1)
        handle = PrimerHandle.load(primer_path)
        while True:
            handle, done = run_collect(handle, primer_path, voter_cache_path)
            if done:
                print("Voter cache written. Ready to sweep.")
                break
            from pipeline.stages.summarization.batch.models import ProviderJob
            for jd in handle.jobs:
                j = ProviderJob.from_dict(jd)
                logger.info("  job %s  provider=%s  status=%s", j.job_id, j.provider, j.status)
            logger.info("Jobs still running — sleeping %ds…", args.poll_interval)
            time.sleep(args.poll_interval)

    if args.mode == "all":
        while handle.phase != "complete":
            logger.info("Polling jobs… (sleeping %ds)", args.poll_interval)
            time.sleep(args.poll_interval)
            handle, done = run_collect(handle, primer_path, voter_cache_path)
            if done:
                logger.info("All jobs complete. Voter cache ready.")
                break
            # Show job statuses
            from pipeline.stages.summarization.batch.models import ProviderJob
            for jd in handle.jobs:
                j = ProviderJob.from_dict(jd)
                logger.info("  job %s  provider=%s  status=%s", j.job_id, j.provider, j.status)

    # ── SWEEP ──
    if args.mode in ("sweep", "all"):
        if not voter_cache_path.exists():
            print(f"Voter cache not found: {voter_cache_path}. Run prime + collect first.", file=sys.stderr)
            sys.exit(1)
        if not silver_path.exists():
            print(f"silver_findings.jsonl not found: {silver_path}", file=sys.stderr)
            sys.exit(1)

        voter_cache = json.loads(voter_cache_path.read_text(encoding="utf-8"))

        # Methodology note: the replay drives the LEGACY AgreementChecker L1->L2->L3
        # path, which IS production (enable_router=false, B-062). The MapOutputRouter
        # L1->L3-skip path is an opt-in experiment, not the production default.
        logger.info(
            "CASCADE PATH = %s (production path; enable_router=false). The MapOutputRouter "
            "L1->L3-skip path is opt-in — re-validate the chosen (theta, reject_theta, "
            "scorer) on it only if that experiment is adopted (B-062).", CASCADE_PATH,
        )
        # L3-population guard: escalated chunks contribute no findings without L3.
        l3_pop = sum(
            1 for e in voter_cache.values() for cid in e.get("chunk_map", {})
            if e.get("l3", {}).get(cid)
        )
        if l3_pop == 0:
            logger.warning(
                "voter_cache has ZERO L3 outputs — every escalated chunk contributes no "
                "findings, so escalation-sensitive metrics (escalate_rate, recall at low "
                "theta) are unreliable. Re-run `prime` + `collect` with L3 before trusting them.",
            )

        silver_by_case: dict[str, SilverCaseResult] = {}
        for rec in read_jsonl(silver_path, SilverCaseResult):
            silver_by_case[rec.case_id] = rec

        # Resolve --legacy-single-voter-policy {keep|escalate|both} into the
        # axis the sweep iterates over. 'both' = compare keep vs escalate side
        # by side; useful for answering "does silently keeping N=1 survivors
        # hurt precision?". Other values restrict to a single policy. These axes
        # are embedder-independent, so resolve them once before the embedder loop.
        if args.legacy_single_voter_policy == "both":
            policy_grid: tuple[str, ...] = ("keep", "escalate")
        else:
            policy_grid = (args.legacy_single_voter_policy,)
        # Same shape for --force-escalate-on-polarity-conflict. 'both' enables
        # the B-051 ablation; 'true' (default) preserves historical sweep rows.
        if args.force_escalate_on_polarity_conflict == "both":
            polarity_grid: tuple[bool, ...] = (True, False)
        else:
            polarity_grid = (args.force_escalate_on_polarity_conflict == "true",)
        logger.info(
            "legacy_single_voter_policy axis = %s (%d cell%s)",
            policy_grid, len(policy_grid), "" if len(policy_grid) == 1 else "s",
        )
        logger.info(
            "force_escalate_on_polarity_conflict axis = %s (%d cell%s)",
            polarity_grid, len(polarity_grid), "" if len(polarity_grid) == 1 else "s",
        )

        # --embedder both runs the sweep under each embedding model and writes one
        # combined CSV (tagged with an `embedder` column). The embedder is BOTH the
        # agreement-scorer model AND the silver-matching model, so each pass is a
        # full end-to-end run. --embed-cache override is ignored for 'both' (each
        # model needs its own cache).
        embedders_to_run = ["gemini", "openai"] if args.embedder == "both" else [args.embedder]
        if args.embedder == "both" and args.embed_cache:
            logger.warning("--embed-cache ignored with --embedder both; using each embedder's default cache")

        def _build_sweep_embedder(name: str):
            """Return (embedder, embed_cache, raw_agreement_fn) for 'gemini' or 'openai'.

            Mirrors the per-provider construction used for eval matching + the
            EmbeddingScorer agreement fn; honours --embed-cache only for a single
            embedder (None ⇒ provider default cache).
            """
            override = args.embed_cache if (args.embedder != "both" and args.embed_cache) else None
            if name == "gemini":
                key = os.environ.get("GOOGLE_API_KEY")
                if not key:
                    print("GOOGLE_API_KEY not set", file=sys.stderr)
                    sys.exit(1)
                from pipeline.stages.summarization.agreement.providers import GeminiEmbedder as _AgGem
                cache = make_embedding_cache(
                    Path(override) if override else DEFAULT_GEMINI_CACHE_PATH, GEMINI_EMBEDDING_MODEL)
                return GeminiEmbedder(key), cache, _AgGem()
            from eval.silver.embedders import OpenAIEmbedder
            from eval.silver.matcher import DEFAULT_CACHE_PATH, EMBEDDING_MODEL
            from pipeline.stages.summarization.agreement.providers import OpenAIEmbedder as _AgOAI
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                print("OPENAI_API_KEY not set", file=sys.stderr)
                sys.exit(1)
            cache = make_embedding_cache(
                Path(override) if override else DEFAULT_CACHE_PATH, EMBEDDING_MODEL)
            return OpenAIEmbedder(key), cache, _AgOAI()

        all_rows: list[dict] = []
        for emb_name in embedders_to_run:
            if len(embedders_to_run) > 1:
                logger.info("===== SWEEP embedder=%s =====", emb_name)
            embedder, embed_cache, _raw_agreement_fn = _build_sweep_embedder(emb_name)

            # Pre-compute all agreement embeddings upfront so the theta replay loop
            # makes zero API calls (same texts are used for all theta values).
            # Reuses the same disk cache as eval matching — keys are sha256(model+text)
            # so agreement claims and eval finding strings coexist without collision.
            _prewarm_agreement_cache(voter_cache, _raw_agreement_fn, embed_cache)
            agreement_embed_fn = _make_cached_embed_fn(embed_cache, _raw_agreement_fn)

            sweep_rows = run_sweep(
                voter_cache=voter_cache,
                silver_by_case=silver_by_case,
                embedder=embedder,
                embed_cache=embed_cache,
                sim_threshold=args.sim_threshold,
                scorer_specs=_default_scorer_specs(),
                thetas=THETA_GRID,
                reject_thetas=REJECT_THETA_GRID,
                split=args.split,
                seed=args.seed,
                dev_fraction=args.dev_fraction,
                agreement_embed_fn=agreement_embed_fn,
                single_voter_policies=policy_grid,
                force_escalate_on_polarity_conflict_grid=polarity_grid,
            )
            for r in sweep_rows:
                r["embedder"] = emb_name
            if sweep_rows:
                if len(embedders_to_run) > 1:
                    print(f"\n───── embedder = {emb_name} ─────")
                _print_table(sweep_rows)
            all_rows.extend(sweep_rows)

        if all_rows:
            csv_path = reports_dir / f"map_cascade_sweep_{timestamp}.csv"
            _write_csv(csv_path, all_rows, fieldnames=[
                "embedder",
                "scorer", "tau", "count_alpha", "reuse_weight", "contradiction_weight",
                "w_category", "w_embedding", "w_entity", "w_evidence", "weights_sum",
                "theta", "reject_theta",
                "legacy_single_voter_policy", "force_escalate_on_polarity_conflict",
                "cascade_path",
                "precision", "recall", "f1", "strict_f1",
                "early_accept_rate", "escalate_rate", "early_accept_precision",
                "n_chunks", "n_l1_invoked", "n_l2_invoked", "n_l3_invoked",
                "n_polarity_conflict_chunks", "polarity_conflict_rate",
                "n_matched", "n_silver", "n_pipeline",
                "split", "seed", "dev_fraction", "sim_threshold",
            ])


if __name__ == "__main__":
    main()
