"""
Estimate end-to-end knowledge_extraction-pipeline cost for P80/P90 papers
ordered by text-element count.

Outputs a markdown report to stdout (and to ``out/cost_percentile_report.md``).

Reproducibility
---------------
- Percentile method: nearest-rank, ``index = ceil(p * n) - 1`` over the set of
  papers with at least one text element in the ``text_elements`` table.
- Per-chunk token baseline is taken from the only existing trace we have on
  disk (PMC10047158, escalation_report_20260511T234134.json) — one L3 chunk
  with input=4733, output=2521 tokens for claude-sonnet-4-6.  All other
  voters are assumed to consume similar input tokens (same prompt) and
  similar output tokens (same structured AuditableSummary schema).
- Sentence count per paper is *estimated* from total word count using the
  observed ratio from PMC10047158 (78 sentences / 1335 words ≈ 0.0584).
- Chunk count = ceil(n_sentences / stride) with stride = chunk_size - overlap.

Cost coverage
-------------
The active 6-stage cascade only spends LLM API tokens in **MAP**.  GROUNDING
and RELATE run a local NLI model; NORMALIZE, GROUP, CANONICALIZE, and
RESOLVE are deterministic UMLS / embedding stages.  Their LLM API cost is
$0.  Compute time / GPU is out of scope here.
"""
from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sqlalchemy import func, distinct

from database import get_db_connection
from database.models import Document, TextElement, Figure, Table
from pipeline.stages.knowledge_extraction.config import MapConfig
from pipeline.stages.knowledge_extraction.costing import PriceBook
from pipeline.stages.knowledge_extraction.batch.voter_configs import (
    CascadeProfile, get_profile, list_profiles,
)


# ─────────────────────────────────────────────────────────────────────────────
# Reference numbers
# ─────────────────────────────────────────────────────────────────────────────

# Observed per-chunk tokens (from PMC10047158 escalation_report L3 call).
# Used as the canonical "one LLM call on one MAP chunk" estimate for every
# tier (L1, L2, L3).  Conservative: structured-output overhead is similar
# across providers.
OBS_INPUT_TOK_PER_CHUNK = 4733
OBS_OUTPUT_TOK_PER_CHUNK = 2521

# Sentence-from-words ratio observed in PMC10047158.
OBS_SENT_PER_WORD = 78 / 1335  # ≈ 0.0584

# MAP chunking sourced from the single source of truth (`MapConfig` defaults)
# so a change in production config cannot silently desync the estimate.
_MAP_DEFAULTS = MapConfig()
CHUNK_SIZE = _MAP_DEFAULTS.chunk_size
CHUNK_OVERLAP = _MAP_DEFAULTS.chunk_overlap
STRIDE = CHUNK_SIZE - CHUNK_OVERLAP


# Profile model lists are pulled from voter_configs at runtime — no local
# copy. `cheap` is structurally a 2-tier cascade (L3 mirrors L2); `real` is
# the production 3-tier cascade. See
# pipeline/stages/knowledge_extraction/batch/voter_configs.py for the source of truth.
def _profile_models(p: CascadeProfile) -> dict[str, list[str] | str]:
    return {
        "l1": [v.model for v in p.l1_voters],
        "l2": [v.model for v in p.l2_voters],
        "l3": p.l3_voter.model,
    }


# Which cascade profiles get cost-projected in the report. Explicit allowlist
# rather than `list_profiles()` so a future re-introduction of `default` (or any
# experimental profile) in `voter_configs.py` doesn't silently leak into the
# thesis budget report. To add a profile to the report, add it here.
_REPORTED_PROFILES: tuple[str, ...] = ("cheap", "real")


PROFILES: dict[str, dict[str, list[str] | str]] = {
    name: _profile_models(get_profile(name))
    for name in _REPORTED_PROFILES
    if name in list_profiles()
}

# Escalation rate scenarios (fraction of chunks that escalate past each tier).
# `expected` is the load-bearing one — used by per-paper and cumulative tables.
# Default values are loose hedges over the single early observation
# (PMC10047158: 1/8 reached L3, 0 stopped at L2). At runtime, `main()` calls
# `load_observed_escalation_rates()` and replaces this `expected` row with
# chunk-weighted means over any `out/summaries/reports/escalation_report_*.json`
# files present. Best / worst stay as anchor scenarios for the sensitivity table.
SCENARIOS = {
    "best":     {"l2_rate": 0.05, "l3_rate": 0.01},
    "expected": {"l2_rate": 0.20, "l3_rate": 0.10},
    "worst":    {"l2_rate": 1.00, "l3_rate": 1.00},  # every chunk reaches L3
}


REPORTS_DIR = Path("out/summaries/reports")


@dataclass
class ObservedRates:
    """Chunk-weighted escalation fractions aggregated across run reports."""
    l2_rate: float
    l3_rate: float
    n_reports: int
    total_chunks: int
    total_l2_escalated: int
    total_l3_escalated: int
    sources: list[str] = field(default_factory=list)


def load_observed_escalation_rates(
    reports_dir: Path = REPORTS_DIR,
) -> ObservedRates | None:
    """Return chunk-weighted L2/L3 escalation fractions across all escalation
    reports in ``reports_dir`` — or ``None`` when there is nothing to learn from.

    Chunk-weighted = `sum(l2_escalated) / sum(total_chunks)` across reports,
    not the mean of per-report rates. Each chunk gets equal vote regardless
    of which paper / report it came from. A single small paper can't pull the
    mean around.

    Skips reports that are unreadable, malformed, or carry `total_chunks <= 0`
    (and logs the skip). Reads `totals` first; falls back to summing the
    per-paper `papers[]` array if `totals` is absent.
    """
    if not reports_dir.exists():
        return None
    paths = sorted(reports_dir.glob("escalation_report_*.json"))
    if not paths:
        return None

    total_chunks = 0
    total_l2 = 0
    total_l3 = 0
    sources: list[str] = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("escalation report unreadable: %s (%s)", path.name, exc)
            continue

        n, l2, l3 = _extract_chunk_counts(data)
        if n <= 0:
            logger.debug("escalation report has zero chunks: %s", path.name)
            continue
        total_chunks += n
        total_l2 += l2
        total_l3 += l3
        sources.append(path.name)

    if total_chunks == 0:
        return None
    return ObservedRates(
        l2_rate=total_l2 / total_chunks,
        l3_rate=total_l3 / total_chunks,
        n_reports=len(sources),
        total_chunks=total_chunks,
        total_l2_escalated=total_l2,
        total_l3_escalated=total_l3,
        sources=sources,
    )


def _extract_chunk_counts(report: dict) -> tuple[int, int, int]:
    """Pull `(total_chunks, l2_escalated, l3_escalated)` out of one report.

    Prefers the `totals` block (writer-emitted aggregate). Falls back to
    summing the per-paper `papers[]` list when an older report omits totals.
    """
    totals = report.get("totals")
    if isinstance(totals, dict) and totals.get("total_chunks") is not None:
        return (
            int(totals.get("total_chunks", 0) or 0),
            int(totals.get("l2_escalated", 0) or 0),
            int(totals.get("l3_escalated", 0) or 0),
        )
    papers = report.get("papers") or []
    n = l2 = l3 = 0
    for p in papers:
        if not isinstance(p, dict):
            continue
        n += int(p.get("total_chunks", 0) or 0)
        l2 += int(p.get("l2_escalated", 0) or 0)
        l3 += int(p.get("l3_escalated", 0) or 0)
    return n, l2, l3


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PaperMeta:
    pmcid: str
    title: str | None
    n_te: int
    n_words: int
    n_chars: int
    n_figures: int
    n_tables: int
    n_distinct_paths: int

    @property
    def est_sentences(self) -> int:
        return max(1, int(round(self.n_words * OBS_SENT_PER_WORD)))

    @property
    def est_chunks(self) -> int:
        if self.est_sentences <= 0:
            return 0
        # Mirrors `len(range(0, n_sentences, stride))` from `MapStage._make_chunks`.
        return max(1, math.ceil(self.est_sentences / STRIDE))


def rank_papers() -> list[PaperMeta]:
    db = get_db_connection()
    out: list[PaperMeta] = []
    with db.session_scope() as s:
        rows = (
            s.query(
                Document.pmcid,
                Document.title,
                func.count(TextElement.id).label("n_te"),
                func.coalesce(
                    func.sum(func.length(TextElement.text_content)), 0
                ).label("n_chars"),
            )
            .join(TextElement, TextElement.document_id == Document.id)
            .group_by(Document.id)
            .order_by(func.count(TextElement.id))
            .all()
        )
        # We need word counts and figure/table counts. Issue a second pass
        # only for the percentile rows — but it's cheap to compute words from
        # chars via avg-word-length 5.6 (English).  We will refine for the
        # selected papers below.
        for r in rows:
            out.append(PaperMeta(
                pmcid=r.pmcid, title=r.title,
                n_te=r.n_te, n_words=int(round(r.n_chars / 5.6)),
                n_chars=r.n_chars,
                n_figures=0, n_tables=0, n_distinct_paths=0,
            ))
    return out


def enrich(paper: PaperMeta) -> PaperMeta:
    db = get_db_connection()
    with db.session_scope() as s:
        doc = s.query(Document).filter_by(pmcid=paper.pmcid).first()
        if doc is None:
            return paper
        figs = s.query(func.count(Figure.id)).filter_by(document_id=doc.id).scalar() or 0
        tabs = s.query(func.count(Table.id)).filter_by(document_id=doc.id).scalar() or 0
        n_paths = (
            s.query(func.count(distinct(TextElement.path_string)))
            .filter_by(document_id=doc.id)
            .scalar() or 0
        )
        # Exact word count
        tes = s.query(TextElement).filter_by(document_id=doc.id).all()
        n_words = sum(len((t.text_content or "").split()) for t in tes)
        return PaperMeta(
            pmcid=paper.pmcid, title=doc.title,
            n_te=paper.n_te, n_words=n_words, n_chars=paper.n_chars,
            n_figures=figs, n_tables=tabs, n_distinct_paths=n_paths,
        )


def pick_percentile(papers: list[PaperMeta], p: float) -> tuple[int, PaperMeta]:
    n = len(papers)
    if n == 0:
        raise ValueError("pick_percentile: empty paper list — no candidates to rank")
    if not 0.0 < p <= 1.0:
        raise ValueError(f"pick_percentile: p must be in (0, 1]; got {p}")
    idx = math.ceil(p * n) - 1
    return idx, papers[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Cost estimation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StageCost:
    stage: str
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    input_price_per_1m: float | None
    output_price_per_1m: float | None
    cost_usd: float | None

    batch_cost_usd: float | None = None

    @property
    def batch_cost(self) -> float | None:
        return self.batch_cost_usd


def _price_or_none(book: PriceBook, model: str) -> tuple[float | None, float | None]:
    p = book.get(model)
    if p is None:
        return None, None
    return p.input_per_1m, p.output_per_1m


def _cost(book: PriceBook, model: str, in_tok: int, out_tok: int) -> float | None:
    return book.cost(model, in_tok, out_tok, batch=False)


def _batch_cost(book: PriceBook, model: str, in_tok: int, out_tok: int) -> float | None:
    return book.cost(model, in_tok, out_tok, batch=True)


def estimate_map(
    paper: PaperMeta,
    book: PriceBook,
    profile: dict,
    l2_rate: float,
    l3_rate: float,
) -> list[StageCost]:
    """Per-tier cost rows for a given cascade profile.

    Number of chunks reaching each tier follows the rates. L1 voters fire on
    every chunk. L2 voters fire on `n_chunks × l2_rate`. The single L3 model
    fires on `n_chunks × l3_rate`.
    """
    n_chunks = paper.est_chunks
    rows: list[StageCost] = []

    for m in profile["l1"]:
        in_tok = n_chunks * OBS_INPUT_TOK_PER_CHUNK
        out_tok = n_chunks * OBS_OUTPUT_TOK_PER_CHUNK
        ip, op = _price_or_none(book, m)
        rows.append(StageCost(
            stage="MAP/L1", model=m, calls=n_chunks,
            input_tokens=in_tok, output_tokens=out_tok,
            input_price_per_1m=ip, output_price_per_1m=op,
            cost_usd=_cost(book, m, in_tok, out_tok),
            batch_cost_usd=_batch_cost(book, m, in_tok, out_tok),
        ))

    l2_chunks = int(round(n_chunks * l2_rate))
    for m in profile["l2"]:
        in_tok = l2_chunks * OBS_INPUT_TOK_PER_CHUNK
        out_tok = l2_chunks * OBS_OUTPUT_TOK_PER_CHUNK
        ip, op = _price_or_none(book, m)
        rows.append(StageCost(
            stage="MAP/L2", model=m, calls=l2_chunks,
            input_tokens=in_tok, output_tokens=out_tok,
            input_price_per_1m=ip, output_price_per_1m=op,
            cost_usd=_cost(book, m, in_tok, out_tok),
            batch_cost_usd=_batch_cost(book, m, in_tok, out_tok),
        ))

    l3_chunks = int(round(n_chunks * l3_rate))
    l3_model = profile["l3"]
    in_tok = l3_chunks * OBS_INPUT_TOK_PER_CHUNK
    out_tok = l3_chunks * OBS_OUTPUT_TOK_PER_CHUNK
    ip, op = _price_or_none(book, l3_model)
    rows.append(StageCost(
        stage="MAP/L3", model=l3_model, calls=l3_chunks,
        input_tokens=in_tok, output_tokens=out_tok,
        input_price_per_1m=ip, output_price_per_1m=op,
        cost_usd=_cost(book, l3_model, in_tok, out_tok),
        batch_cost_usd=_batch_cost(book, l3_model, in_tok, out_tok),
    ))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Cumulative spend across the smallest P% of papers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CumulativePoint:
    cut: float            # in (0, 1]
    n_papers: int         # papers[:n_papers] included
    total_chunks: int
    normal_cost: float
    batch_cost: float


def cumulative_cost_by_cut(
    papers: list[PaperMeta],
    book: PriceBook,
    profile: dict,
    l2_rate: float,
    l3_rate: float,
    cuts: list[float],
) -> list[CumulativePoint]:
    """Running sum of per-paper MAP cost from rank 1 → ceil(cut × N).

    Answers "what if I run only the smallest cut % of papers?". Papers
    are taken in the order given (this script feeds them sorted ascending
    by `n_te`).  Each paper's per-tier cost comes from `estimate_map`.

    Cuts outside (0, 1] are rejected.  Missing prices propagate as 0 —
    matches the per-paper tables, which render them as `n/a`; a global
    cumulative number can't carry "partial unknown" cleanly without
    misleading the reader.
    """
    n = len(papers)
    if n == 0:
        return []
    for p in cuts:
        if not 0.0 < p <= 1.0:
            raise ValueError(f"cumulative_cost_by_cut: cut must be in (0, 1]; got {p}")

    per_paper: list[tuple[int, float, float]] = []  # (chunks, normal, batch)
    for paper in papers:
        rows = estimate_map(paper, book, profile, l2_rate, l3_rate)
        normal = _sum_cost(rows) or 0.0
        batch = _sum_batch_cost(rows) or 0.0
        per_paper.append((paper.est_chunks, normal, batch))

    out: list[CumulativePoint] = []
    for cut in cuts:
        idx = math.ceil(cut * n)
        slice_ = per_paper[:idx]
        out.append(CumulativePoint(
            cut=cut,
            n_papers=idx,
            total_chunks=sum(c for c, _, _ in slice_),
            normal_cost=sum(n_c for _, n_c, _ in slice_),
            batch_cost=sum(b for _, _, b in slice_),
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Markdown rendering
# ─────────────────────────────────────────────────────────────────────────────

def _money(v: float | None) -> str:
    return "n/a" if v is None else f"${v:,.4f}"


def _ppm(v: float | None) -> str:
    return "n/a" if v is None else f"${v:.2f}"


def _sum_cost(rows: Iterable[StageCost]) -> float | None:
    total = 0.0
    seen_unknown = False
    for r in rows:
        if r.cost_usd is None:
            seen_unknown = True
        else:
            total += r.cost_usd
    return None if seen_unknown else total


def _sum_batch_cost(rows: Iterable[StageCost]) -> float | None:
    total = 0.0
    seen_unknown = False
    for r in rows:
        if r.batch_cost is None:
            seen_unknown = True
        else:
            total += r.batch_cost
    return None if seen_unknown else total


def _cascade_rates_assumption_line(observed: ObservedRates | None) -> str:
    """Single bullet describing where the `expected` cascade rates came from.

    Falls into two branches: observed (chunk-weighted from on-disk reports)
    or the static hedge fallback when nothing is on disk yet.
    """
    if observed is None:
        return (
            "- **Cascade rates** are scenario inputs, not measurements. No "
            "`out/summaries/reports/escalation_report_*.json` files found, so "
            "the `expected` row falls back to the static hedge defaults "
            "(20 % / 10 %). Run the pipeline on ≥3–5 papers and re-run this "
            "script to switch to calibrated rates automatically."
        )
    return (
        "- **Cascade rates (calibrated)**: `expected` row reads from "
        f"{observed.n_reports} escalation report(s) covering "
        f"{observed.total_chunks:,} chunks. Chunk-weighted means: "
        f"l2 = {observed.l2_rate:.1%} "
        f"({observed.total_l2_escalated:,}/{observed.total_chunks:,}), "
        f"l3 = {observed.l3_rate:.1%} "
        f"({observed.total_l3_escalated:,}/{observed.total_chunks:,}). "
        "Best / worst rows remain anchor scenarios."
    )


def main() -> None:
    book = PriceBook.load()
    papers = rank_papers()
    n = len(papers)

    # Calibrate `expected` cascade rates from observed escalation reports,
    # if any exist. Falls back to the static hedge defaults otherwise.
    observed = load_observed_escalation_rates()
    if observed is not None:
        SCENARIOS["expected"] = {
            "l2_rate": observed.l2_rate,
            "l3_rate": observed.l3_rate,
        }

    # Median, P80, P90 nearest-rank
    selections: dict[str, tuple[int, PaperMeta]] = {
        "P50": pick_percentile(papers, 0.50),
        "P80": pick_percentile(papers, 0.80),
        "P90": pick_percentile(papers, 0.90),
    }
    enriched = {k: (idx, enrich(p)) for k, (idx, p) in selections.items()}

    lines: list[str] = []
    lines += ["# Pipeline Cost Estimate for P80 / P90 Papers", ""]
    lines += [
        "## Dataset",
        "",
        f"- papers with ≥1 text element: **{n}**",
        "- ordering: number of text elements (ascending)",
        "- percentile method: nearest-rank — `idx = ceil(p × n) − 1`",
        f"- price book: `configs/model_prices.json` ({len(book.known_models())} models)",
        f"- batch_discount_multiplier: **{book.batch_discount_multiplier}**",
        "",
        "## Selected papers",
        "",
        "| pct | rank | PMCID | title | text elements | words | figs | tables | sections | est. sentences | est. MAP chunks |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, (idx, paper) in enriched.items():
        title = paper.title or "—"
        lines.append(
            f"| {label} | {idx + 1}/{n} | {paper.pmcid} | {title} | "
            f"{paper.n_te} | {paper.n_words:,} | {paper.n_figures} | {paper.n_tables} | "
            f"{paper.n_distinct_paths} | {paper.est_sentences} | {paper.est_chunks} |"
        )
    lines.append("")

    # Per-profile stage tables at the expected cascade rates.
    expected = SCENARIOS["expected"]
    lines += [
        "## Stage-level cost — expected cascade rates, per profile",
        "",
        f"Expected cascade rates: L2 escalation = {expected['l2_rate']:.1%}, "
        f"L3 escalation = {expected['l3_rate']:.1%} of chunks "
        + (
            f"(chunk-weighted mean over {observed.n_reports} escalation "
            f"report(s), {observed.total_chunks:,} chunks)."
            if observed is not None else
            "(static hedge — no escalation reports on disk yet)."
        ),
        "",
        "Profiles imported live from "
        "`pipeline/stages/knowledge_extraction/batch/voter_configs.py`:",
        "",
    ]
    for pname in PROFILES:
        prof = get_profile(pname)
        l1 = ", ".join(v.model for v in prof.l1_voters)
        l2 = ", ".join(v.model for v in prof.l2_voters)
        l3 = prof.l3_voter.model
        lines.append(f"- **{pname}** — L1: {l1}; L2: {l2}; L3: {l3}")
    lines += [
        "",
        "All non-MAP stages spend **zero LLM API tokens**: GROUNDING and RELATE "
        "run a local NLI model (`cross-encoder/nli-deberta-v3-large`); "
        "NORMALIZE / GROUP / CANONICALIZE / RESOLVE are deterministic UMLS + "
        "embedding stages.",
        "",
    ]
    for label in ("P80", "P90"):
        idx, paper = enriched[label]
        lines.append(f"### {label}: `{paper.pmcid}`")
        lines.append("")
        lines.append(
            f"- est. sentences: **{paper.est_sentences}** "
            f"(from {paper.n_words} words × {OBS_SENT_PER_WORD:.4f} sent/word)"
        )
        lines.append(
            f"- est. MAP chunks: **{paper.est_chunks}** "
            f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, stride={STRIDE})"
        )
        lines.append("")
        for pname, profile in PROFILES.items():
            rows = estimate_map(
                paper, book, profile, expected["l2_rate"], expected["l3_rate"]
            )
            lines.append(f"#### profile = `{pname}`")
            lines.append("")
            lines.append(
                "| stage | model | calls | input tok | output tok | "
                "in $/1M | out $/1M | normal | batch (×0.5) |"
            )
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
            for r in rows:
                lines.append(
                    f"| {r.stage} | `{r.model}` | {r.calls} | "
                    f"{r.input_tokens:,} | {r.output_tokens:,} | "
                    f"{_ppm(r.input_price_per_1m)} | {_ppm(r.output_price_per_1m)} | "
                    f"{_money(r.cost_usd)} | {_money(r.batch_cost)} |"
                )
            t = _sum_cost(rows)
            bt = _sum_batch_cost(rows)
            lines.append(
                f"| **MAP total** | | | | | | | "
                f"**{_money(t)}** | **{_money(bt)}** |"
            )
            lines.append("")

    # Cascade scenario summary across all profiles.
    lines += [
        "## Cascade scenarios (MAP-only) — all profiles",
        "",
        "| paper | profile | scenario | L2 rate | L3 rate | normal $ | batch $ |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for label in ("P80", "P90"):
        idx, paper = enriched[label]
        for pname, profile in PROFILES.items():
            for sname, rates in SCENARIOS.items():
                rows = estimate_map(
                    paper, book, profile, rates["l2_rate"], rates["l3_rate"]
                )
                t = _sum_cost(rows)
                bt = _sum_batch_cost(rows)
                lines.append(
                    f"| {label} | {pname} | {sname} | "
                    f"{rates['l2_rate']:.0%} | {rates['l3_rate']:.0%} | "
                    f"{_money(t)} | {_money(bt)} |"
                )
    lines.append("")

    # Missing-price flag — surface any profile model that the price book lacks.
    missing = sorted({
        m
        for profile in PROFILES.values()
        for m in [*profile["l1"], *profile["l2"], profile["l3"]]
        if not book.has(m)
    })
    if missing:
        lines += [
            "> ⚠️  **Missing prices** for: " + ", ".join(f"`{m}`" for m in missing) + ". "
            "Cost cells render as `n/a`. Add entries to `configs/model_prices.json` and re-run.",
            "",
        ]

    # Cumulative spend across the smallest P% of papers — answers "what
    # if I run only the first 90% of the corpus?". Uses the same expected
    # cascade rates as the per-paper section above.
    cuts = [0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 1.00]
    lines += [
        "## Cumulative spend — running the smallest P% of papers",
        "",
        f"Running sum of per-paper MAP cost from rank 1 (smallest by `n_te`) "
        f"up to `ceil(P × {n})`. Same expected cascade rates as above "
        f"(L2 = {expected['l2_rate']:.1%}, L3 = {expected['l3_rate']:.1%}). "
        "Answers 'what if I cap the run at the smallest P% of the corpus?'.",
        "",
    ]
    # Pre-compute per-profile cumulative series so each profile is rendered
    # in its own table (one combined table would have 6 cost columns × 4
    # profiles → unreadable).
    cumulative_by_profile = {
        pname: cumulative_cost_by_cut(
            papers, book, profile,
            expected["l2_rate"], expected["l3_rate"], cuts,
        )
        for pname, profile in PROFILES.items()
    }
    for pname, series in cumulative_by_profile.items():
        lines.append(f"#### profile = `{pname}`")
        lines.append("")
        lines.append(
            "| cut | n_papers | total chunks | normal $ | batch (×0.5) $ |"
        )
        lines.append("|---:|---:|---:|---:|---:|")
        for pt in series:
            lines.append(
                f"| {pt.cut:>4.0%} | {pt.n_papers:,} | {pt.total_chunks:,} | "
                f"{_money(pt.normal_cost)} | {_money(pt.batch_cost)} |"
            )
        lines.append("")

    # Cold vs warm cache
    lines += [
        "## Cold-run vs cache-warm",
        "",
        "MAP results are cached in `PipelineCache` keyed by chunk text + cascade "
        "metadata. A re-run of the *same paper with unchanged sentences and "
        "unchanged cascade config* hits the cache for every chunk → ~$0 LLM cost. "
        "The numbers in the tables above are **cold-run** estimates. Warm-run "
        "cost ≈ 0 (cache hit ratio observed = 100 % on the only existing trace).",
        "",
    ]

    # Assumptions
    lines += [
        "## Assumptions & uncertainty",
        "",
        "- **Per-chunk tokens** taken from the single observed L3 call "
        f"on PMC10047158: input={OBS_INPUT_TOK_PER_CHUNK}, "
        f"output={OBS_OUTPUT_TOK_PER_CHUNK}. Applied uniformly to every voter "
        "at every tier (same prompt; same structured-output schema). "
        "Real L1/L2 outputs are typically smaller — these estimates are "
        "**conservative / upper-bound** for the cheap voters.",
        f"- **Sentence count** estimated as words × {OBS_SENT_PER_WORD:.4f} "
        "from the PMC10047158 ratio (78 sentences / 1335 words). A 25 % swing "
        "either way is plausible — multiply final cost by 0.75–1.25 for "
        "a sensitivity band.",
        f"- **Chunk count** = `ceil(n_sentences / {STRIDE})` (chunk_size="
        f"{CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).",
        _cascade_rates_assumption_line(observed),
        "- **Prices** loaded verbatim from `configs/model_prices.json`. "
        "Models present: " + ", ".join(book.known_models()) + ".",
        "- **Cache** assumed cold for the tables; the warm-run section reports "
        "the trivial-cost case separately.",
        "- **Batch discount** = 0.5× across all providers, taken from the price "
        "book. Provider-specific batch availability is not verified here.",
        "- **Non-MAP stages**: GROUNDING and RELATE use a local NLI model — "
        "no API tokens billed, but GPU / wall-time cost is non-zero (out of "
        "scope). UMLS-driven stages issue no LLM calls.",
        "",
    ]

    out_md = "\n".join(lines) + "\n"
    out_path = Path("out") / "cost_percentile_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_md, encoding="utf-8")
    print(out_md)
    print(f"# wrote {out_path}")


if __name__ == "__main__":
    main()
