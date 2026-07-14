"""Count tokens and project ABC-cascade cost for a paper-selection YAML.

Reads the YAML produced by ``eval.paper_selection.run_select``, loads each
paper's sentences from the database (the exact same path the MAP stage uses
via ``KnowledgeExtractionRunner.load_paper_from_db``), computes chunk counts at the
production MapConfig defaults, tokenises every sentence with the
``cl100k_base`` tokenizer, and prints:

  - per-paper table:  pmcid | bucket | n_sentences | n_chunks | n_tokens
  - totals
  - projected MAP cost at four cascade-escalation scenarios
    (optimistic, middle, pessimistic, and a 95%-escalation worst case)

Usage
-----
    python scripts/estimate_selection_cost.py \\
        --from-selection configs/paper_selection/runA.yaml \\
        --profile real

Or pass PMCIDs explicitly:

    python scripts/estimate_selection_cost.py PMC10047158 PMC222

Or estimate CALIBRATION cost (silver prime + Opus judge) over a source-cases
JSONL — prints the prime (all voters, no escalation) and Opus silver tables:

    python scripts/estimate_selection_cost.py \
        --source-cases eval/data/source_cases_related15.jsonl --profile real

Notes
-----
- Token counts use ``tiktoken`` (cl100k_base). Anthropic's tokenizer differs
  by ~10 %, fine for budget estimates.
- Cost projections cover the MAP stage only. CANONICALIZE/RESOLVE add a
  smaller, more variable amount (typically ~10–20 % of MAP); see the
  printed footer.
- Escalation rates are coarse guesses. Override with ``--l2-rate`` and
  ``--l3-rate`` once you have real data from a canary run.
"""
from __future__ import annotations

import argparse
import logging
import sys
from math import ceil
from dataclasses import dataclass
from pathlib import Path

# Bootstrap repo root onto sys.path so `from pipeline...` / `from database...`
# resolve when invoked as `python scripts/estimate_selection_cost.py` (Python
# puts the script's own dir on sys.path, not the CWD). Bare `sys.path.insert`
# (not a guarded _REPO_ROOT assignment) keeps ruff's E402 sys.path exemption —
# matches scripts/run_paper.py. (B-063)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("estimate_selection_cost")


# ── MAP defaults (mirror pipeline/stages/knowledge_extraction/config.py:MapConfig) ─

DEFAULT_CHUNK_SIZE = 10
DEFAULT_CHUNK_OVERLAP = 2
# Prompt overhead per MAP call (system + JSON schema + voter instructions).
# Conservative middle estimate; real prompts are 1.2–1.8 K tokens.
PROMPT_OVERHEAD_TOKENS = 1500
# Output JSON per chunk (AuditableSummary findings array). Measured over the
# primed voter cache (2117 real voter outputs): mean ~1100, median ~900,
# p90 ~2200 tokens. 1500 is a conservative (~p75) choice so the worst case
# leans high rather than low — output drives the L3/Sonnet cost ($15/1M out).
OUTPUT_TOKENS_PER_CHUNK = 1500

# ── Calibration (silver) defaults ───────────────────────────────────────────
# The map_theta_sweep PRIME runs EVERY voter on EVERY chunk (no escalation) so
# the sweep can replay any theta — priced as project_map_cost at l2_rate=
# l3_rate=1.0. The Opus SILVER step (eval.silver.generation.generate) is one judge call
# per source case.
DEFAULT_SILVER_MODEL = "claude-opus-4-7"   # = eval/silver/generation/generator.py DEFAULT_MODEL
# Opus findings-JSON tokens per case. Hard cap is max_tokens=8192
# (generator.py); a single paragraph realistically emits far fewer. Conservative
# middle estimate (> the 800 voter output — judging is more verbose); override
# with --silver-output-tokens.
SILVER_OUTPUT_TOKENS = 1500


# ── Cost matrix (per million tokens) ────────────────────────────────────────
# Sourced live from
#   pipeline/stages/knowledge_extraction/batch/voter_configs.py  (cascade structure)
# and
#   configs/model_prices.json via PriceBook  (per-model prices).
# Pick the cascade with ``--profile {cheap|real}`` (required —
# no implicit default).

@dataclass
class VoterPricing:
    name: str
    input_per_mtok: float
    output_per_mtok: float


@dataclass
class TierPricing:
    voters: list[VoterPricing]

    @property
    def n_voters(self) -> int:
        return len(self.voters)

    @property
    def total_input_per_mtok(self) -> float:
        return sum(v.input_per_mtok for v in self.voters)

    @property
    def total_output_per_mtok(self) -> float:
        return sum(v.output_per_mtok for v in self.voters)


def build_pricing(profile_name: str | None,
                  book) -> tuple[dict[str, TierPricing], str]:
    """Build per-tier pricing tables from the live cascade profile.

    Returns ``(pricing, resolved_profile_name)``. Raises if any voter model
    is missing from the price book.
    """
    # Local imports — keep the module importable even when the optional
    # voter_configs deps are unavailable in test contexts.
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from pipeline.stages.knowledge_extraction.batch.voter_configs import get_profile

    prof = get_profile(profile_name)
    tier_to_voters = {
        "L1": prof.l1_voters,
        "L2": prof.l2_voters,
        "L3": [prof.l3_voter],
    }
    pricing: dict[str, TierPricing] = {}
    missing: list[str] = []
    for tier, voters in tier_to_voters.items():
        rows: list[VoterPricing] = []
        for v in voters:
            p = book.get(v.model)
            if p is None:
                missing.append(v.model)
                continue
            rows.append(VoterPricing(v.model, p.input_per_1m, p.output_per_1m))
        pricing[tier] = TierPricing(voters=rows)
    if missing:
        raise SystemExit(
            f"Missing prices in configs/model_prices.json for: "
            f"{sorted(set(missing))}. Add them and re-run."
        )
    return pricing, prof.name


# ── YAML loader ─────────────────────────────────────────────────────────────

def load_selection_yaml(path: Path) -> dict[str, str]:
    """Return ``{pmcid: bucket}`` in related → diverse → hard order, deduped."""
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path}: expected a top-level mapping with a version key")
    version_key = next(iter(data))
    buckets = data[version_key] or {}
    if not isinstance(buckets, dict):
        raise ValueError(f"{path}: version '{version_key}' is not a mapping")

    out: dict[str, str] = {}
    for bucket in ("related", "diverse", "hard"):
        for pmcid in buckets.get(bucket) or []:
            out.setdefault(pmcid, bucket)
    if not out:
        raise ValueError(f"{path}: no PMCIDs found under related/diverse/hard")
    return out


# ── Chunk + token math ──────────────────────────────────────────────────────

def count_chunks(n_sentences: int, chunk_size: int, chunk_overlap: int) -> int:
    """Mirror MapStage._make_chunks chunk count exactly."""
    if n_sentences <= 0:
        return 0
    stride = chunk_size - chunk_overlap
    return len(list(range(0, n_sentences, stride)))


def make_tokenizer():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(tokenizer, sentences: list[str]) -> int:
    return sum(len(tokenizer.encode(s)) for s in sentences)


def per_chunk_input_tokens(text_tokens: int, n_chunks: int,
                           chunk_size: int, chunk_overlap: int,
                           n_sentences: int) -> int:
    """Approximate average input tokens for a single MAP call.

    Mirrors `MapStage._make_chunks` semantics exactly: each chunk is
    `sentences[i : i+chunk_size]` for `i` stepping by `stride = chunk_size
    - chunk_overlap`. The trailing chunk is truncated if it would run past
    `n_sentences`. Average sentences per chunk = total sentence-occurrences
    across all chunks / n_chunks — this is what the LLM actually sees,
    overlap counted as duplicates by construction.

    Budget estimates use conservative `ceil` rounding so the printed
    cost is never lower than the realised cost given the modelled inputs.
    """
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap must satisfy 0 <= overlap < chunk_size; "
            f"got chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )
    if n_chunks == 0 or n_sentences == 0:
        return PROMPT_OVERHEAD_TOKENS
    stride = chunk_size - chunk_overlap
    total_sentence_occurrences = sum(
        min(chunk_size, n_sentences - start)
        for start in range(0, n_sentences, stride)
    )
    avg_sentences_per_chunk = total_sentence_occurrences / n_chunks
    chunk_text_tokens = text_tokens * avg_sentences_per_chunk / n_sentences
    return PROMPT_OVERHEAD_TOKENS + ceil(chunk_text_tokens)


# ── Paper loading ───────────────────────────────────────────────────────────

@dataclass
class PaperStats:
    pmcid: str
    bucket: str
    n_text_elements: int
    n_sentences: int
    n_chunks: int
    text_tokens: int            # tokens across all sentence text
    avg_chunk_in_tokens: int    # mean input tokens per MAP call


def load_paper_stats(pmcid: str, bucket: str, *,
                     tokenizer, chunk_size: int, chunk_overlap: int,
                     nlp) -> PaperStats | None:
    """Mirror `KnowledgeExtractionRunner.load_paper_from_db` (TextElement →
    spaCy sentencize).

    Sort by `TextElement.id` to match the production load path
    (`pipeline/stages/knowledge_extraction/runner.py:load_paper_from_db`).
    `id` is autoincrement and `db_ingester` writes rows in document order
    via per-row `session.flush()`, so `.order_by(id)` preserves document
    order. Sorting by `position_in_section` instead — as this script did
    pre-fix — interleaves sections (the B-039 bug).
    """
    import time as _time
    from nlp_histo.database import get_db_connection, Document, TextElement

    t0 = _time.perf_counter()
    db = get_db_connection()
    with db.session_scope() as session:
        doc = session.query(Document).filter_by(pmcid=pmcid).first()
        if doc is None:
            logger.warning("PMCID %s not in database — skipping", pmcid)
            return None
        rows = (
            session.query(TextElement)
            .filter_by(document_id=doc.id)
            .order_by(TextElement.id)
            .all()
        )
        n_text_elements = len(rows)
        logger.debug("[%s] DB fetched %d text elements in %.2fs",
                     pmcid, n_text_elements, _time.perf_counter() - t0)

        t_sent = _time.perf_counter()
        sentences: list[str] = []
        texts = [te.text_content for te in rows if te.text_content]
        for doc_obj in nlp.pipe(texts, batch_size=64):
            for sent in doc_obj.sents:
                text = sent.text.strip()
                if text:
                    sentences.append(text)
        logger.debug("[%s] sentencized %d paragraphs → %d sentences in %.2fs",
                     pmcid, len(texts), len(sentences),
                     _time.perf_counter() - t_sent)

    t_tok = _time.perf_counter()
    n_sentences = len(sentences)
    n_chunks = count_chunks(n_sentences, chunk_size, chunk_overlap)
    text_tokens = count_tokens(tokenizer, sentences)
    avg_in = per_chunk_input_tokens(text_tokens, n_chunks, chunk_size,
                                    chunk_overlap, n_sentences)
    logger.debug("[%s] tokenized in %.2fs (%d tokens)",
                 pmcid, _time.perf_counter() - t_tok, text_tokens)

    logger.info("[%s] %s: %d sentences, %d chunks, %d tokens (%.2fs)",
                pmcid, bucket, n_sentences, n_chunks, text_tokens,
                _time.perf_counter() - t0)

    return PaperStats(
        pmcid=pmcid, bucket=bucket,
        n_text_elements=n_text_elements,
        n_sentences=n_sentences, n_chunks=n_chunks,
        text_tokens=text_tokens, avg_chunk_in_tokens=avg_in,
    )


# ── Cost projection ─────────────────────────────────────────────────────────

@dataclass
class Scenario:
    label: str
    l2_rate: float   # fraction of chunks escalated to L2
    l3_rate: float   # fraction of chunks escalated to L3


def project_map_cost(total_chunks: int, avg_in_tokens: int,
                     l2_rate: float, l3_rate: float,
                     pricing: dict[str, TierPricing]) -> dict[str, float]:
    """Return per-tier and total MAP cost in USD.

    Each tier sums per-voter costs: one chunk → ``len(tier.voters)`` API
    calls, each charged at that voter's own input/output price. No averaging.
    """
    out_tokens = OUTPUT_TOKENS_PER_CHUNK
    cost = {"L1": 0.0, "L2": 0.0, "L3": 0.0}

    def tier_cost(tier: str, n_chunks: float) -> float:
        p = pricing[tier]
        in_cost  = n_chunks * avg_in_tokens / 1_000_000 * p.total_input_per_mtok
        out_cost = n_chunks * out_tokens     / 1_000_000 * p.total_output_per_mtok
        return in_cost + out_cost

    cost["L1"] = tier_cost("L1", total_chunks)
    cost["L2"] = tier_cost("L2", total_chunks * l2_rate)
    cost["L3"] = tier_cost("L3", total_chunks * l3_rate)
    cost["total"] = cost["L1"] + cost["L2"] + cost["L3"]
    return cost


# ── Calibration (source-case) math ──────────────────────────────────────────

def summarize_cases(cases, *, tokenizer, chunk_size: int, chunk_overlap: int,
                    nlp) -> tuple[int, int, int]:
    """Aggregate chunk + token stats over a list of SourceCase.

    Each case's ``text`` is sentencized independently — mirrors
    ``map_theta_sweep.run_prime``, which builds one chunk map per case. Returns
    ``(total_chunks, weighted_avg_input_tokens, total_text_tokens)``.
    """
    texts = [(c.text or "") for c in cases]
    total_chunks = 0
    weighted_in = 0.0
    total_text_tokens = 0
    for doc in nlp.pipe(texts, batch_size=64):
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        n_chunks = count_chunks(len(sents), chunk_size, chunk_overlap)
        ttok = count_tokens(tokenizer, sents)
        avg_in = per_chunk_input_tokens(ttok, n_chunks, chunk_size,
                                        chunk_overlap, len(sents))
        total_chunks += n_chunks
        weighted_in += avg_in * n_chunks
        total_text_tokens += ttok
    avg_in_tokens = int(round(weighted_in / max(total_chunks, 1)))
    return total_chunks, avg_in_tokens, total_text_tokens


def silver_token_totals(cases, tokenizer,
                        out_tokens_per_case: int) -> tuple[int, int]:
    """Opus-silver input/output token totals over ALL cases.

    Mirrors ``eval.silver.generation.generator``: one Opus call per case =
    ``SYSTEM_PROMPT`` + tool schema + ``make_user_prompt(text, path)``. The
    prompt is measured live so the estimate tracks ``PROMPT_VERSION``.
    """
    import json as _json
    from eval.silver.generation.prompts import (
        SYSTEM_PROMPT, EXTRACT_FINDINGS_TOOL, make_user_prompt,
    )
    fixed = (len(tokenizer.encode(SYSTEM_PROMPT))
             + len(tokenizer.encode(_json.dumps(EXTRACT_FINDINGS_TOOL))))
    in_tok = 0
    for c in cases:
        in_tok += fixed + len(tokenizer.encode(
            make_user_prompt(c.text or "", c.path_string or "")))
    return in_tok, len(cases) * out_tokens_per_case


# ── Estimators (one per mode) ────────────────────────────────────────────────

def run_cascade_estimate(args, *, book, pricing, resolved_profile, discount,
                         nlp, tokenizer) -> int:
    """Production MAP cascade projection over a paper selection / PMCID list."""
    if args.from_selection:
        sel_map = load_selection_yaml(Path(args.from_selection))
    else:
        sel_map = {p: "explicit" for p in args.pmcid}

    if args.l2_rate or args.l3_rate:
        l2 = args.l2_rate or []
        l3 = args.l3_rate or []
        if len(l2) != len(l3):
            raise SystemExit("--l2-rate and --l3-rate must be paired (same count)")
        scenarios = [Scenario(f"custom_{i+1}", l2[i], l3[i]) for i in range(len(l2))]
    else:
        scenarios = [
            Scenario("optimistic", l2_rate=0.10, l3_rate=0.02),
            Scenario("middle",     l2_rate=0.25, l3_rate=0.08),
            Scenario("pessimistic", l2_rate=0.50, l3_rate=0.20),
            # Worst case: 95% of chunks fail agreement and escalate through
            # every tier (L1 → L2 → L3), so L2 and L3 each run on 95% of chunks.
            Scenario("worst_case", l2_rate=0.95, l3_rate=0.95),
        ]

    logger.info("Loading %d papers from DB…", len(sel_map))
    import time as _time
    t_all = _time.perf_counter()
    stats: list[PaperStats] = []
    for i, (pmcid, bucket) in enumerate(sel_map.items(), start=1):
        logger.info("(%d/%d) loading %s [%s]…", i, len(sel_map), pmcid, bucket)
        s = load_paper_stats(
            pmcid, bucket, tokenizer=tokenizer,
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, nlp=nlp,
        )
        if s is not None:
            stats.append(s)
    logger.info("Loaded %d/%d papers in %.1fs", len(stats), len(sel_map),
                _time.perf_counter() - t_all)
    if not stats:
        logger.error("No papers loaded — aborting")
        return 2

    print()
    print(f"{'pmcid':<48} {'bucket':<10} {'tx_el':>6} {'sents':>6} "
          f"{'chunks':>7} {'tokens':>9} {'in/chunk':>9}")
    print("-" * 100)
    for s in stats:
        print(f"{s.pmcid:<48} {s.bucket:<10} {s.n_text_elements:>6d} "
              f"{s.n_sentences:>6d} {s.n_chunks:>7d} {s.text_tokens:>9d} "
              f"{s.avg_chunk_in_tokens:>9d}")
    print("-" * 100)
    total_chunks = sum(s.n_chunks for s in stats)
    total_tokens = sum(s.text_tokens for s in stats)
    total_sents  = sum(s.n_sentences for s in stats)
    weighted_avg_in = (
        sum(s.avg_chunk_in_tokens * s.n_chunks for s in stats) / max(total_chunks, 1)
    )
    print(f"{'TOTAL':<48} {'':<10} {'':>6} {total_sents:>6d} "
          f"{total_chunks:>7d} {total_tokens:>9d} "
          f"{int(round(weighted_avg_in)):>9d}")

    print()
    discount_note = (
        f"  (batch discount ×{book.batch_discount_multiplier} applied)"
        if args.batch else ""
    )
    print(f"MAP cost projection — profile = {resolved_profile} "
          f"({total_chunks} chunks, "
          f"avg {int(round(weighted_avg_in))} input tokens, "
          f"{OUTPUT_TOKENS_PER_CHUNK} output tokens/chunk){discount_note}")
    print()
    print(f"{'scenario':<14} {'L2 esc':>7} {'L3 esc':>7} "
          f"{'L1 $':>8} {'L2 $':>8} {'L3 $':>8} {'TOTAL $':>10}")
    print("-" * 70)
    for sc in scenarios:
        c = project_map_cost(total_chunks, int(round(weighted_avg_in)),
                             sc.l2_rate, sc.l3_rate, pricing)
        print(f"{sc.label:<14} {sc.l2_rate:>6.0%} {sc.l3_rate:>7.0%} "
              f"{c['L1'] * discount:>8.2f} {c['L2'] * discount:>8.2f} "
              f"{c['L3'] * discount:>8.2f} {c['total'] * discount:>10.2f}")
    print()
    print("MAP only. CANONICALIZE + RESOLVE typically add ~10–20 % "
          "(Sonnet calls on dense subsets of findings).")
    print(f"Cascade configuration (profile = {resolved_profile}, "
          f"loaded from voter_configs.get_profile):")
    for tier, p in pricing.items():
        print(f"  {tier} ({p.n_voters} voter{'s' if p.n_voters != 1 else ''}):")
        for v in p.voters:
            print(f"    {v.name:<36} ${v.input_per_mtok:>5.2f} in / "
                  f"${v.output_per_mtok:>5.2f} out  per MTok")
    print()
    return 0


def run_calibration_estimate(args, *, book, nlp, tokenizer) -> int:
    """Calibration cost: map_theta_sweep PRIME (all voters) + Opus SILVER."""
    from eval.silver.data.jsonl_utils import read_jsonl
    from eval.silver.data.schemas import SourceCase
    from eval.silver.data.split import filter_by_split

    src_path = Path(args.source_cases)
    if not src_path.exists():
        logger.error("source cases not found: %s", src_path)
        return 2
    all_cases = list(read_jsonl(src_path, SourceCase))
    logger.info("Loaded %d source cases from %s", len(all_cases), src_path)
    if not all_cases:
        logger.error("no source cases — aborting calibration estimate")
        return 2

    # PRIME runs the dev split and always uses the REAL-profile voters
    # (map_theta_sweep._make_voters → make_l{1,2,3}_voters), independent of
    # --profile — so price it against the real cascade.
    prime_cases = filter_by_split(all_cases, args.split,
                                  dev_fraction=args.dev_fraction, seed=args.seed)
    real_pricing, _ = build_pricing("real", book)
    p_chunks, p_avg_in, _ = summarize_cases(
        prime_cases, tokenizer=tokenizer,
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, nlp=nlp)
    prime = project_map_cost(p_chunks, p_avg_in, 1.0, 1.0, real_pricing)

    # OPUS SILVER runs over ALL cases (generate.py applies no split).
    opus_p = book.get(args.silver_model)
    if opus_p is None:
        raise SystemExit(
            f"Missing price for silver model {args.silver_model!r} in "
            f"configs/model_prices.json. Add it and re-run.")
    in_tok, out_tok = silver_token_totals(all_cases, tokenizer,
                                          args.silver_output_tokens)
    silver = (in_tok / 1_000_000 * opus_p.input_per_1m
              + out_tok / 1_000_000 * opus_p.output_per_1m)
    bd = book.batch_discount_multiplier
    n1, n2, n3 = (real_pricing["L1"].n_voters, real_pricing["L2"].n_voters,
                  real_pricing["L3"].n_voters)

    print()
    print("=" * 72)
    print(f"CALIBRATION cost — source_cases = {src_path}")
    print("=" * 72)
    print(f"  cases total : {len(all_cases)}")
    print(f"  PRIME       : split={args.split} (dev_fraction={args.dev_fraction}, "
          f"seed={args.seed}) → {len(prime_cases)} cases, {p_chunks} chunks, "
          f"avg {p_avg_in} in-tok/chunk")
    print(f"  SILVER      : all {len(all_cases)} cases, {args.silver_model}, "
          f"~{args.silver_output_tokens} out-tok/case (cap 8192)")
    print()
    print(f"{'step':<30} {'unit cost':<20} {'sync $':>9} {'batch $':>9}")
    print("-" * 72)
    print(f"{'PRIME (map_theta_sweep)':<30} "
          f"{f'{n1}+{n2}+{n3} voters/chunk':<20} "
          f"{prime['total']:>9.2f} {prime['total'] * bd:>9.2f}")
    print(f"{'OPUS SILVER (generate.py)':<30} {'1 judge/case':<20} "
          f"{silver:>9.2f} {silver * bd:>9.2f}")
    print("-" * 72)
    total = prime["total"] + silver
    print(f"{'CALIBRATION TOTAL':<30} {'':<20} {total:>9.2f} {total * bd:>9.2f}")
    print()
    print(f"PRIME = all real-profile voters (L1×{n1}+L2×{n2}+L3×{n3}) on every "
          f"chunk, no escalation (project_map_cost @ 100%). The primer always "
          f"runs the real cascade regardless of --profile.")
    print(f"SILVER = one {args.silver_model} call per source case over ALL "
          f"{len(all_cases)} cases (generate.py has no dev/test split); silver "
          f"prompt measured live (PROMPT_VERSION-tracked).")
    print()
    return 0


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pmcid", nargs="*",
                        help="Explicit PMCIDs (alternative to --from-selection)")
    parser.add_argument("--from-selection", metavar="PATH",
                        help="Path to a paper-selection YAML (production cascade estimate)")
    parser.add_argument("--source-cases", nargs="?",
                        const="eval/data/source_cases_related15.jsonl", default=None,
                        metavar="PATH",
                        help="Estimate CALIBRATION cost (silver prime + Opus judge) "
                             "over a source-cases JSONL. Bare flag uses "
                             "eval/data/source_cases_related15.jsonl. Combinable with "
                             "--from-selection (both sections print).")
    parser.add_argument("--chunk-size",    type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--l2-rate", type=float, action="append", default=None,
                        metavar="FRAC",
                        help="L2 escalation fraction for one scenario row "
                             "(repeatable; pair with --l3-rate). "
                             "If unset, three baked-in scenarios are shown.")
    parser.add_argument("--l3-rate", type=float, action="append", default=None,
                        metavar="FRAC", help="L3 escalation fraction "
                                              "(repeatable; matched to --l2-rate).")
    parser.add_argument("--profile", required=True,
                        choices=("cheap", "real"),
                        help="Cascade profile name. Required — no implicit "
                             "default. `default` profile retired 2026-05-16. "
                             "(The calibration PRIME is always priced at `real`.)")
    parser.add_argument("--silver-model", default=DEFAULT_SILVER_MODEL,
                        help=f"Silver judge model for the Opus estimate "
                             f"(default {DEFAULT_SILVER_MODEL}).")
    parser.add_argument("--silver-output-tokens", type=int,
                        default=SILVER_OUTPUT_TOKENS,
                        help=f"Estimated Opus findings-JSON tokens per case "
                             f"(default {SILVER_OUTPUT_TOKENS}; hard cap 8192).")
    parser.add_argument("--split", default="dev", choices=("dev", "test", "all"),
                        help="Split the PRIME processes (default dev — matches "
                             "map_theta_sweep). Opus silver always covers all cases.")
    parser.add_argument("--dev-fraction", type=float, default=0.8,
                        help="Dev fraction for the prime split (default 0.8).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Split seed (default 42 — matches map_theta_sweep).")
    parser.add_argument("--prices", default=None,
                        help="Path to model_prices.json. "
                             "Defaults to configs/model_prices.json.")
    parser.add_argument("--batch", action="store_true",
                        help="Apply the price-book batch discount to the cascade "
                             "totals (the calibration table always shows sync AND batch).")
    args = parser.parse_args()

    if args.from_selection and args.pmcid:
        parser.error("pass either positional PMCIDs OR --from-selection, not both")

    cascade_mode = bool(args.from_selection or args.pmcid)
    calib_mode = args.source_cases is not None
    if not (cascade_mode or calib_mode):
        parser.print_help()
        return 1

    # ── Shared setup (price book + sentencizer + tokenizer) ─────────────────
    logger.info("Loading price book + cascade profile…")
    from pipeline.stages.knowledge_extraction.costing import PriceBook
    book = PriceBook.load(args.prices)
    pricing, resolved_profile = build_pricing(args.profile, book)
    discount = book.batch_discount_multiplier if args.batch else 1.0

    logger.info("Loading spaCy en_core_sci_sm (sentencizer only)…")
    import spacy
    nlp = spacy.load(
        "en_core_sci_sm",
        disable=["tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
    )
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    logger.info("Loading tiktoken cl100k_base…")
    tokenizer = make_tokenizer()

    rc = 0
    if cascade_mode:
        rc = run_cascade_estimate(
            args, book=book, pricing=pricing, resolved_profile=resolved_profile,
            discount=discount, nlp=nlp, tokenizer=tokenizer) or rc
    if calib_mode:
        rc = run_calibration_estimate(
            args, book=book, nlp=nlp, tokenizer=tokenizer) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
