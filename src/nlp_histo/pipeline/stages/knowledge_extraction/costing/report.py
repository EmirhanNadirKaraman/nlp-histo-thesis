"""
Cost report builder.

Reads a list of ``LLMUsageRecord``s (or a JSONL path) and writes
``cost_report.md`` + ``cost_report.json``.

Reports are strictly *derived* from the records.  Records are the source of
truth — both the runtime path and ``scripts/estimate_cost_savings.py``
produce identical reports from the same JSONL.

Scenario semantics
------------------
A: no batching, no cascade. Every cascade item goes to the strong (L3) model.
B: batching only. Same as A but with the batch discount applied.
C: cascade only. Actual cascade routing as observed, no batch discount.
D: actual run. Whatever happened — batched and/or cascaded.

Baseline (Scenario A) output tokens are *counterfactual* — we estimate them
using, in order:
  1. observed L3-call output tokens for the same stage (if any),
  2. average voter output tokens for the same stage,
  3. 0 (with explicit "n/a" marker) when no signal exists.

Low/central/high columns wrap the central estimate at 0.75× / 1.0× / 1.25×
output tokens to expose the uncertainty.
"""
from __future__ import annotations

import json
import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .models import LLMUsageRecord, load_records
from .pricing import PriceBook

logger = logging.getLogger(__name__)


# Aggregation primitives


@dataclass
class Aggregate:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cost_usd: float | None = None        # None → at least one missing price
    missing_price_for: list[str] | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.cost_usd is not None:
            out["cost_usd"] = round(self.cost_usd, 6)
        if self.missing_price_for:
            out["missing_price_for"] = sorted(set(self.missing_price_for))
        return out


def _bucket(records: Iterable[LLMUsageRecord], key) -> dict[str, list[LLMUsageRecord]]:
    out: dict[str, list[LLMUsageRecord]] = defaultdict(list)
    for r in records:
        out[str(key(r))].append(r)
    return out


def _sum_cost(records: Iterable[LLMUsageRecord], book: PriceBook) -> Aggregate:
    agg = Aggregate()
    cost = 0.0
    cost_known = True
    missing: list[str] = []
    for r in records:
        agg.calls += 1
        agg.input_tokens += r.input_tokens
        agg.output_tokens += r.output_tokens
        if r.cache_hit or r.call_status in {"cached", "skipped", "failed"}:
            continue
        c = book.cost(
            r.model,
            r.input_tokens,
            r.output_tokens,
            batch=r.batch_used,
            cached_input_tokens=r.cached_tokens or 0,
        )
        if c is None:
            cost_known = False
            missing.append(r.model)
        else:
            cost += c
    agg.cost_usd = cost if cost_known else None
    agg.missing_price_for = missing or None
    return agg


# Scenario building


@dataclass
class ScenarioRow:
    name: str
    usage_type: str          # "measured" | "estimated" | "mixed"
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_low: float | None
    cost_central: float | None
    cost_high: float | None
    cost_per_paper: float | None
    savings_vs_A: float | None
    percent_savings_vs_A: float | None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "scenario": self.name,
            "usage_type": self.usage_type,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
        for k in (
            "cost_low", "cost_central", "cost_high", "cost_per_paper",
            "savings_vs_A", "percent_savings_vs_A",
        ):
            v = getattr(self, k)
            if v is not None:
                d[k] = round(v, 6)
        return d


def _paper_count(records: list[LLMUsageRecord]) -> int:
    papers = {r.paper_id for r in records if r.paper_id}
    return max(len(papers), 1)


def _strong_model(records: list[LLMUsageRecord]) -> str | None:
    """Pick the observed L3 model (most common). Used for Scenario A."""
    l3 = [r for r in records if r.cascade_level == 3]
    if not l3:
        return None
    counts: dict[str, int] = defaultdict(int)
    for r in l3:
        counts[r.model] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _estimate_baseline_output_tokens(
    records: list[LLMUsageRecord],
    stage: str,
) -> tuple[float, str]:
    """Estimate per-cascade-item output tokens under "always strong" baseline.

    Returns (mean_output_tokens, source) where source explains the choice.
    """
    l3 = [r for r in records if r.stage == stage and r.cascade_level == 3 and r.output_tokens > 0]
    if l3:
        return statistics.mean(r.output_tokens for r in l3), "observed_L3"
    voters = [r for r in records if r.stage == stage and r.output_tokens > 0]
    if voters:
        return statistics.mean(r.output_tokens for r in voters), "stage_average"
    return 0.0, "no_signal"


def _scenario_a_b(
    records: list[LLMUsageRecord],
    book: PriceBook,
    *,
    batched: bool,
) -> tuple[ScenarioRow, dict[str, Any]]:
    """No-cascade baseline: every cascade item goes directly to the strong model.

    Cascade items are identified as MAP records with a chunk-level item_id.
    Each unique (stage, item_id) counts as one baseline call.
    """
    strong = _strong_model(records)
    items: dict[tuple[str, str], LLMUsageRecord] = {}
    for r in records:
        if r.cascade_level is None:
            continue
        key = (r.stage, r.item_id or "")
        if key not in items:
            items[key] = r
    if not items:
        return ScenarioRow(
            name=("B" if batched else "A"),
            usage_type="estimated",
            input_tokens=0, output_tokens=0, total_tokens=0,
            cost_low=None, cost_central=None, cost_high=None,
            cost_per_paper=None, savings_vs_A=None, percent_savings_vs_A=None,
        ), {"note": "no cascade items observed"}

    input_tokens_total = 0
    output_central = 0.0
    estimate_sources: set[str] = set()

    for (stage, _item), rep in items.items():
        # Input tokens — use the actual prompt tokens already paid (any voter
        # at this cascade item) since the prompt under the baseline is the
        # same prompt.
        input_tokens_total += rep.input_tokens
        mean_out, src = _estimate_baseline_output_tokens(records, stage)
        output_central += mean_out
        estimate_sources.add(src)

    output_low = output_central * 0.75
    output_high = output_central * 1.25

    if strong is None or not book.has(strong):
        return ScenarioRow(
            name=("B" if batched else "A"),
            usage_type="estimated",
            input_tokens=int(input_tokens_total),
            output_tokens=int(output_central),
            total_tokens=int(input_tokens_total + output_central),
            cost_low=None, cost_central=None, cost_high=None,
            cost_per_paper=None, savings_vs_A=None, percent_savings_vs_A=None,
        ), {
            "strong_model": strong,
            "estimate_sources": sorted(estimate_sources),
            "item_count": len(items),
            "note": "no price for strong model; cost omitted",
        }

    def _cost(out_tokens: float) -> float | None:
        return book.cost(strong, int(input_tokens_total), int(out_tokens), batch=batched)

    return ScenarioRow(
        name=("B" if batched else "A"),
        usage_type="estimated",
        input_tokens=int(input_tokens_total),
        output_tokens=int(output_central),
        total_tokens=int(input_tokens_total + output_central),
        cost_low=_cost(output_low),
        cost_central=_cost(output_central),
        cost_high=_cost(output_high),
        cost_per_paper=None,
        savings_vs_A=None,
        percent_savings_vs_A=None,
    ), {
        "strong_model": strong,
        "estimate_sources": sorted(estimate_sources),
        "item_count": len(items),
        "batch_discount_applied": batched,
    }


def _scenario_c(records: list[LLMUsageRecord], book: PriceBook) -> ScenarioRow:
    """Cascade as observed, but with the batch discount stripped."""
    input_t = 0
    output_t = 0
    cost = 0.0
    cost_known = True
    for r in records:
        if r.cache_hit or r.call_status in {"cached", "skipped", "failed"}:
            continue
        input_t += r.input_tokens
        output_t += r.output_tokens
        c = book.cost(r.model, r.input_tokens, r.output_tokens, batch=False)
        if c is None:
            cost_known = False
        else:
            cost += c
    return ScenarioRow(
        name="C", usage_type="mixed",
        input_tokens=input_t, output_tokens=output_t, total_tokens=input_t + output_t,
        cost_low=cost if cost_known else None,
        cost_central=cost if cost_known else None,
        cost_high=cost if cost_known else None,
        cost_per_paper=None, savings_vs_A=None, percent_savings_vs_A=None,
    )


def _scenario_d(records: list[LLMUsageRecord], book: PriceBook) -> ScenarioRow:
    agg = _sum_cost(records, book)
    return ScenarioRow(
        name="D", usage_type="measured",
        input_tokens=agg.input_tokens, output_tokens=agg.output_tokens,
        total_tokens=agg.total_tokens,
        cost_low=agg.cost_usd, cost_central=agg.cost_usd, cost_high=agg.cost_usd,
        cost_per_paper=None, savings_vs_A=None, percent_savings_vs_A=None,
    )


def _annotate_scenarios(rows: list[ScenarioRow], n_papers: int) -> None:
    a_central = next((r.cost_central for r in rows if r.name == "A"), None)
    for r in rows:
        if r.cost_central is not None and n_papers:
            r.cost_per_paper = r.cost_central / n_papers
        if a_central is not None and r.cost_central is not None:
            r.savings_vs_A = a_central - r.cost_central
            r.percent_savings_vs_A = (
                100.0 * r.savings_vs_A / a_central if a_central > 0 else 0.0
            )


# Cascade & batch savings


def _cascade_savings(
    records: list[LLMUsageRecord], book: PriceBook,
) -> dict[str, Any]:
    cascade_records = [r for r in records if r.cascade_level is not None]
    if not cascade_records:
        return {"note": "no cascade records present"}

    items: dict[tuple[str, str], list[LLMUsageRecord]] = defaultdict(list)
    for r in cascade_records:
        items[(r.stage, r.item_id or "")].append(r)

    total_items = len(items)
    escalated = sum(
        1 for recs in items.values()
        if any(r.cascade_level and r.cascade_level >= 2 for r in recs)
    )
    accepted_without_escalation = total_items - escalated

    cheap_cost = 0.0
    cheap_known = True
    escalator_cost = 0.0
    escalator_known = True
    for r in cascade_records:
        c = book.cost(r.model, r.input_tokens, r.output_tokens, batch=r.batch_used)
        if c is None:
            if r.cascade_level == 3:
                escalator_known = False
            else:
                cheap_known = False
            continue
        if r.cascade_level == 3:
            escalator_cost += c
        else:
            cheap_cost += c

    return {
        "total_items": total_items,
        "accepted_without_escalation": accepted_without_escalation,
        "escalated": escalated,
        "escalation_rate": (escalated / total_items) if total_items else 0.0,
        "cheap_voter_cost_usd": cheap_cost if cheap_known else None,
        "escalator_cost_usd": escalator_cost if escalator_known else None,
        "judge_overhead_usd": _judge_overhead(records, book),
    }


def _judge_overhead(records: list[LLMUsageRecord], book: PriceBook) -> float | None:
    judge_records = [r for r in records if r.role == "judge"]
    if not judge_records:
        return 0.0
    cost = 0.0
    known = True
    for r in judge_records:
        c = book.cost(r.model, r.input_tokens, r.output_tokens, batch=r.batch_used)
        if c is None:
            known = False
        else:
            cost += c
    return cost if known else None


def _batch_savings(
    records: list[LLMUsageRecord], book: PriceBook,
) -> dict[str, Any]:
    batched = [r for r in records if r.batch_used and not r.cache_hit]
    if not batched:
        return {"note": "no batched calls"}
    regular_cost = 0.0
    discounted_cost = 0.0
    known = True
    for r in batched:
        full = book.cost(r.model, r.input_tokens, r.output_tokens, batch=False)
        disc = book.cost(r.model, r.input_tokens, r.output_tokens, batch=True)
        if full is None or disc is None:
            known = False
            continue
        regular_cost += full
        discounted_cost += disc
    if not known:
        return {
            "batched_calls": len(batched),
            "note": "missing prices for some batched models — savings omitted",
        }
    abs_savings = regular_cost - discounted_cost
    pct = (100.0 * abs_savings / regular_cost) if regular_cost > 0 else 0.0
    return {
        "batched_calls": len(batched),
        "regular_cost_usd": round(regular_cost, 6),
        "discounted_cost_usd": round(discounted_cost, 6),
        "absolute_savings_usd": round(abs_savings, 6),
        "percent_savings": round(pct, 3),
    }


# Public API


def build_report_from_records(
    records: list[LLMUsageRecord],
    book: PriceBook,
) -> dict[str, Any]:
    """Build the full structured report dict. Used by both runtime + offline."""
    if not records:
        return {"empty": True, "note": "no usage records collected"}

    n_papers = _paper_count(records)

    actual = _sum_cost(records, book)

    by_stage = {
        k: _sum_cost(v, book).to_dict()
        for k, v in _bucket(records, lambda r: r.stage).items()
    }
    by_model = {
        k: _sum_cost(v, book).to_dict()
        for k, v in _bucket(records, lambda r: r.model or "unknown").items()
    }
    by_role = {
        k: _sum_cost(v, book).to_dict()
        for k, v in _bucket(records, lambda r: r.role).items()
    }
    by_paper = {
        k: _sum_cost(v, book).to_dict()
        for k, v in _bucket(records, lambda r: r.paper_id or "unknown").items()
    }

    cache_hits = sum(1 for r in records if r.cache_hit)

    a_row, a_meta = _scenario_a_b(records, book, batched=False)
    b_row, _ = _scenario_a_b(records, book, batched=True)
    c_row = _scenario_c(records, book)
    d_row = _scenario_d(records, book)
    rows = [a_row, b_row, c_row, d_row]
    _annotate_scenarios(rows, n_papers)

    return {
        "schema_version": 1,
        "n_papers": n_papers,
        "price_book_path": str(book.source_path) if book.source_path else None,
        "price_book_models_loaded": len(book.known_models()),
        "actual_usage": {
            "total": actual.to_dict(),
            "by_stage": by_stage,
            "by_model": by_model,
            "by_role": by_role,
            "by_paper": by_paper,
            "cache_hits": cache_hits,
        },
        "batch_savings": _batch_savings(records, book),
        "cascade_savings": _cascade_savings(records, book),
        "counterfactual_baseline": a_meta,
        "scenarios": [r.to_dict() for r in rows],
    }


def write_cost_reports(
    records: list[LLMUsageRecord],
    book: PriceBook,
    output_dir: Path,
    *,
    write_json: bool = True,
    write_markdown: bool = True,
) -> dict[str, Path]:
    """Write the JSON + Markdown reports into ``output_dir``. Returns written paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report_from_records(records, book)
    out: dict[str, Path] = {}
    if write_json:
        p = output_dir / "cost_report.json"
        p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        out["json"] = p
    if write_markdown:
        p = output_dir / "cost_report.md"
        p.write_text(_render_markdown(report), encoding="utf-8")
        out["markdown"] = p
    return out


# Markdown rendering


def _money(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"${x:,.4f}"


def _pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.2f}%"


def _render_markdown(report: dict[str, Any]) -> str:
    if report.get("empty"):
        return "# Cost Report\n\nNo usage records collected.\n"

    lines: list[str] = []
    lines.append("# Cost Report")
    lines.append("")
    lines.append(f"- papers processed: **{report['n_papers']}**")
    lines.append(f"- price book: `{report.get('price_book_path')}` "
                 f"({report.get('price_book_models_loaded', 0)} models)")
    actual = report["actual_usage"]["total"]
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- actual tokens: **{actual['total_tokens']:,}** "
                 f"({actual['input_tokens']:,} in / {actual['output_tokens']:,} out)")
    if "cost_usd" in actual:
        lines.append(f"- actual cost (measured): **{_money(actual['cost_usd'])}**")
    else:
        lines.append(
            f"- actual cost: n/a — missing prices for "
            f"{actual.get('missing_price_for', [])}"
        )

    scenarios = report["scenarios"]
    a = next((s for s in scenarios if s["scenario"] == "A"), None)
    d = next((s for s in scenarios if s["scenario"] == "D"), None)
    if a and d and "cost_central" in a and "cost_central" in d:
        lines.append(
            f"- baseline (A, estimated): {_money(a['cost_central'])} "
            f"vs actual (D): {_money(d['cost_central'])}"
        )
        if "percent_savings_vs_A" in d:
            lines.append(
                f"- savings vs baseline: **{_money(d.get('savings_vs_A'))} "
                f"({_pct(d['percent_savings_vs_A'])})**"
            )

    batch = report["batch_savings"]
    cas = report["cascade_savings"]
    if "absolute_savings_usd" in batch:
        lines.append(
            f"- batch discount savings: {_money(batch['absolute_savings_usd'])} "
            f"({_pct(batch.get('percent_savings'))})"
        )
    if "escalation_rate" in cas:
        lines.append(
            f"- cascade escalation rate: {_pct(100.0 * cas['escalation_rate'])} "
            f"({cas['escalated']}/{cas['total_items']} items)"
        )

    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- Scenario A (no-batch, no-cascade) is **estimated**. Output tokens for "
        "items that never ran through the strong model are inferred from observed "
        "L3 calls (preferred), the stage-average, or text length (last resort)."
    )
    lines.append(
        "- Cache hits are counted as zero paid cost and excluded from per-model "
        "billing aggregates."
    )
    lines.append("- All prices come from the configured price book; missing "
                 "models are reported as `n/a` rather than estimated.")

    # Per-stage table
    lines.append("")
    lines.append("## Actual Usage")
    lines.append("")
    lines.append("### Per stage")
    lines.append("")
    lines.append("| stage | calls | input | output | total | cost |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for stage, agg in sorted(report["actual_usage"]["by_stage"].items()):
        lines.append(
            f"| {stage} | {agg['calls']} | {agg['input_tokens']:,} | "
            f"{agg['output_tokens']:,} | {agg['total_tokens']:,} | "
            f"{_money(agg.get('cost_usd'))} |"
        )

    lines.append("")
    lines.append("### Per model")
    lines.append("")
    lines.append("| model | calls | input | output | total | cost |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model, agg in sorted(report["actual_usage"]["by_model"].items()):
        lines.append(
            f"| {model} | {agg['calls']} | {agg['input_tokens']:,} | "
            f"{agg['output_tokens']:,} | {agg['total_tokens']:,} | "
            f"{_money(agg.get('cost_usd'))} |"
        )

    lines.append("")
    lines.append("### Per role")
    lines.append("")
    lines.append("| role | calls | input | output | total | cost |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for role, agg in sorted(report["actual_usage"]["by_role"].items()):
        lines.append(
            f"| {role} | {agg['calls']} | {agg['input_tokens']:,} | "
            f"{agg['output_tokens']:,} | {agg['total_tokens']:,} | "
            f"{_money(agg.get('cost_usd'))} |"
        )

    if report["actual_usage"]["by_paper"] and len(report["actual_usage"]["by_paper"]) > 1:
        lines.append("")
        lines.append("### Per paper")
        lines.append("")
        lines.append("| paper | calls | input | output | total | cost |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for paper, agg in sorted(report["actual_usage"]["by_paper"].items()):
            lines.append(
                f"| {paper} | {agg['calls']} | {agg['input_tokens']:,} | "
                f"{agg['output_tokens']:,} | {agg['total_tokens']:,} | "
                f"{_money(agg.get('cost_usd'))} |"
            )

    lines.append("")
    lines.append("## Batch Savings")
    lines.append("")
    if "note" in batch and "regular_cost_usd" not in batch:
        lines.append(f"_{batch['note']}_")
    else:
        lines.append(f"- batched calls: {batch.get('batched_calls', 0)}")
        lines.append(f"- regular cost (no discount): "
                     f"{_money(batch.get('regular_cost_usd'))}")
        lines.append(f"- discounted cost: "
                     f"{_money(batch.get('discounted_cost_usd'))}")
        lines.append(f"- absolute savings: "
                     f"{_money(batch.get('absolute_savings_usd'))} "
                     f"({_pct(batch.get('percent_savings'))})")

    lines.append("")
    lines.append("## Agreement-Based Cascade Savings")
    lines.append("")
    if "note" in cas and "total_items" not in cas:
        lines.append(f"_{cas['note']}_")
    else:
        lines.append(f"- cascade items: {cas['total_items']}")
        lines.append(f"- accepted without escalation: "
                     f"{cas['accepted_without_escalation']}")
        lines.append(f"- escalated: {cas['escalated']} "
                     f"({_pct(100.0 * cas['escalation_rate'])})")
        lines.append(f"- cheap-voter cost: "
                     f"{_money(cas.get('cheap_voter_cost_usd'))}")
        lines.append(f"- judge / agreement overhead: "
                     f"{_money(cas.get('judge_overhead_usd'))}")
        lines.append(f"- escalator (L3) cost: "
                     f"{_money(cas.get('escalator_cost_usd'))}")

    lines.append("")
    lines.append("## Counterfactual Baseline")
    lines.append("")
    base = report["counterfactual_baseline"]
    if isinstance(base, dict) and "note" in base and "strong_model" not in base:
        lines.append(f"_{base['note']}_")
    else:
        lines.append(
            f"- strong model assumed for Scenario A: `{base.get('strong_model')}`"
        )
        lines.append(
            f"- cascade items counted: {base.get('item_count')}"
        )
        lines.append(
            f"- output-token estimate sources: "
            f"{', '.join(base.get('estimate_sources', []))}"
        )

    lines.append("")
    lines.append("## Scenario table")
    lines.append("")
    lines.append(
        "| scenario | usage_type | input | output | total | "
        "cost_low | cost_central | cost_high | $/paper | savings vs A | % savings |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in scenarios:
        lines.append(
            "| {scenario} | {usage_type} | {input_tokens:,} | {output_tokens:,} | "
            "{total_tokens:,} | {cost_low} | {cost_central} | {cost_high} | "
            "{per_paper} | {sv} | {pct} |".format(
                scenario=s["scenario"],
                usage_type=s["usage_type"],
                input_tokens=s["input_tokens"],
                output_tokens=s["output_tokens"],
                total_tokens=s["total_tokens"],
                cost_low=_money(s.get("cost_low")),
                cost_central=_money(s.get("cost_central")),
                cost_high=_money(s.get("cost_high")),
                per_paper=_money(s.get("cost_per_paper")),
                sv=_money(s.get("savings_vs_A")),
                pct=_pct(s.get("percent_savings_vs_A")),
            )
        )

    lines.append("")
    return "\n".join(lines) + "\n"


def write_cost_reports_from_jsonl(
    jsonl_path: Path,
    book: PriceBook,
    output_dir: Path,
) -> dict[str, Path]:
    return write_cost_reports(load_records(jsonl_path), book, output_dir)
