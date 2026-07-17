#!/usr/bin/env python3
"""
LLM-as-judge labeling for MAP-stage routing records.

Fills keep_ok, strong_judge_score, strong_judge_preferred, judge_reason_codes,
judge_explanation, judge_model, and judge_timestamp in each RoutingRecord.

Record types
------------
A — KEEP: quality evaluation of the voter (best eligible) output
B — ESCALATE/REJECT with best_eligible_output: voter vs escalation comparison
C — ESCALATE/REJECT without best_eligible_output: quality eval; keep_ok=None always

Usage
-----
python scripts/label_routing_records.py \\
    --records out/routing_records.jsonl \\
    --out out/routing_records_labeled.jsonl \\
    --judge-model claude-opus-4-6 \\
    [--skip-labeled] \\
    [--resume] \\
    [--dry-run]

After labeling, run the human audit step (required):
  Sample 15 boundary-region records from the labeled file.
  Review independently and compare keep_ok with judge labels.
  Target: Cohen's kappa > 0.70 before fitting theta.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)

# ── Reason code vocabularies ───────────────────────────────────────────────────

_QUALITY_CODES = [
    "CORRECT_GROUNDING",      # all claims correctly grounded in source
    "HALLUCINATED_EVIDENCE",  # citation doesn't match source sentence
    "OVERSTATED_FINDING",     # claim exceeds what the source states
    "WRONG_CATEGORY",         # IHC/IF/morphology label incorrect
    "INCOMPLETE",             # important finding in source not captured
    "MINOR_PARAPHRASE",       # slight rephrasing, still accurate
]

_COMPARISON_CODES = _QUALITY_CODES + [
    "VOTER_BETTER",           # voter output is superior
    "ESCALATION_BETTER",      # escalation output is superior
    "BOTH_ACCEPTABLE",        # both are fine; keep_ok=True
    "BOTH_UNACCEPTABLE",      # neither reliable; keep_ok=False
]

# ── Judge output schemas ───────────────────────────────────────────────────────

class QualityLabel(BaseModel):
    """Output schema for single-output quality evaluation (types A and C)."""
    keep_ok: bool
    strong_judge_score: float = Field(ge=0.0, le=1.0)
    reason_codes: list[str]
    explanation: str


class ComparisonLabel(BaseModel):
    """Output schema for voter-vs-escalation comparison (type B)."""
    keep_ok: bool
    strong_judge_score: float = Field(ge=0.0, le=1.0)
    strong_judge_preferred: Literal["output", "voter", "neither"]
    reason_codes: list[str]
    explanation: str


# ── Prompt templates ───────────────────────────────────────────────────────────

_QUALITY_SYSTEM = """\
You are a histopathology research assistant evaluating whether LLM-extracted findings \
from a source text are accurate and acceptable for inclusion in a research summary.

"Acceptable" means all of the following hold:
- Every evidence citation (format [S{{n}}|{{pmcid}}|{{te_id}}]) corresponds to a \
sentence that exists verbatim or near-verbatim in the source chunk
- Each claim accurately reflects the content of its cited sentence — not a \
generalization, extrapolation, or contradiction
- No finding states a measurement, percentage, marker, or diagnosis absent from the source
- Category labels (IHC, morphology, molecular_genetics, staging, treatment, prognosis) \
are appropriate for the content described

Be strict: a single hallucinated citation or fabricated claim makes the output \
unacceptable. Slight paraphrasing that preserves meaning is acceptable.

Use reason_codes only from: {codes}\
"""

_QUALITY_USER = """\
## Source chunk
{chunk_text}

## Extracted findings
{findings_text}

Evaluate and return a structured label. \
Set keep_ok=true only if ALL findings are grounded and accurate.\
"""

_COMPARISON_SYSTEM = """\
You are a histopathology research assistant comparing two LLM-generated summaries \
of the same source chunk: a voter output (cheap path) and an escalation output \
(expensive path).

Your two independent tasks:
1. Decide which output is better: "voter", "output" (escalation), or "neither" if \
both are unacceptable.
2. Decide whether the voter output alone was acceptable — meaning it would have been \
safe to use without escalation, regardless of which output is superior.

"Acceptable" means all findings are correctly grounded with no hallucinated citations, \
fabricated claims, or category errors.

keep_ok refers only to voter output acceptability, not to which is preferred.
strong_judge_score should reflect the quality of the preferred output \
(or the escalation if preferred="neither").

Use reason_codes only from: {codes}\
"""

_COMPARISON_USER = """\
## Source chunk
{chunk_text}

## Voter output (cheap path)
{voter_findings_text}

## Escalation output (expensive path)
{escalation_findings_text}

Compare and return a structured label.\
"""


# ── LLM + chain construction ───────────────────────────────────────────────────

def _make_llm(judge_model: str):
    if judge_model.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=judge_model, max_tokens=1024)
    if judge_model.startswith("gpt") or judge_model.startswith("o"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=judge_model, max_tokens=1024)
    raise ValueError(
        f"Unknown judge model prefix: {judge_model!r}. Expected 'claude-' or 'gpt-'."
    )


def _build_quality_chain(llm):
    system = _QUALITY_SYSTEM.format(codes=", ".join(_QUALITY_CODES))
    prompt = ChatPromptTemplate.from_messages([("system", system), ("user", _QUALITY_USER)])
    return prompt | llm.with_structured_output(QualityLabel, strict=True).with_retry(
        stop_after_attempt=3
    )


def _build_comparison_chain(llm):
    system = _COMPARISON_SYSTEM.format(codes=", ".join(_COMPARISON_CODES))
    prompt = ChatPromptTemplate.from_messages([("system", system), ("user", _COMPARISON_USER)])
    return prompt | llm.with_structured_output(ComparisonLabel, strict=True).with_retry(
        stop_after_attempt=3
    )


# ── Finding formatter ──────────────────────────────────────────────────────────

def _format_findings(output_dict: dict | None) -> str:
    if not output_dict:
        return "(no output)"
    findings = output_dict.get("findings", [])
    if not findings:
        return "(no findings extracted)"
    lines = []
    for i, f in enumerate(findings, 1):
        evidence = ", ".join(f.get("evidence", []))
        lines.append(
            f"Finding {i} [{f.get('category', '?')}, {f.get('confidence', '?')} confidence]\n"
            f"  Claim: {f.get('claim', '')}\n"
            f"  Evidence: {evidence}\n"
            f"  Verbatim: {f.get('verbatim_support', '')}"
        )
    return "\n\n".join(lines)


# ── Record type classifier ─────────────────────────────────────────────────────

def _record_type(rec: dict) -> Literal["A", "B", "C"]:
    if rec.get("decision") == "keep":
        return "A"
    if rec.get("best_eligible_exists"):
        return "B"
    return "C"


# ── Per-record labeling ────────────────────────────────────────────────────────

def _label_record(
    rec: dict,
    quality_chain,
    comparison_chain,
    judge_model: str,
) -> dict:
    """Invoke the appropriate chain and merge judge fields into rec (mutates in place)."""
    rtype = _record_type(rec)
    timestamp = datetime.now(timezone.utc).isoformat()

    if rtype == "B":
        label: ComparisonLabel = comparison_chain.invoke({
            "chunk_text": rec["chunk_text"],
            "voter_findings_text": _format_findings(rec.get("best_eligible_output")),
            "escalation_findings_text": _format_findings(rec.get("output")),
        })
        rec["keep_ok"] = label.keep_ok
        rec["strong_judge_score"] = label.strong_judge_score
        rec["strong_judge_preferred"] = label.strong_judge_preferred
        rec["judge_reason_codes"] = label.reason_codes
        rec["judge_explanation"] = label.explanation

    else:
        label: QualityLabel = quality_chain.invoke({
            "chunk_text": rec["chunk_text"],
            "findings_text": _format_findings(rec.get("output")),
        })
        # Type A: keep_ok from judge; type C: always None (no threshold signal)
        rec["keep_ok"] = label.keep_ok if rtype == "A" else None
        rec["strong_judge_score"] = label.strong_judge_score
        rec["strong_judge_preferred"] = None  # not a comparison
        rec["judge_reason_codes"] = label.reason_codes
        rec["judge_explanation"] = label.explanation

    rec["judge_model"] = judge_model
    rec["judge_timestamp"] = timestamp
    return rec


def _print_dry_run(rec: dict, label_type: str) -> None:
    rtype = _record_type(rec)
    print(f"\n{'=' * 64}")
    print(f"DRY RUN — type {rtype} ({label_type})")
    print(
        f"pmcid={rec.get('pmcid')}  chunk_id={rec.get('chunk_id')}  "
        f"decision={rec.get('decision')}  deferral_score={rec.get('deferral_score')}"
    )
    print(f"\n--- chunk_text (first 400 chars) ---\n{rec.get('chunk_text', '')[:400]}")
    if rtype == "B":
        print(f"\n--- voter findings ---\n{_format_findings(rec.get('best_eligible_output'))[:400]}")
        print(f"\n--- escalation findings ---\n{_format_findings(rec.get('output'))[:400]}")
    else:
        print(f"\n--- findings ---\n{_format_findings(rec.get('output'))[:400]}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--records", required=True, help="Input routing_records.jsonl")
    p.add_argument("--out", required=True, help="Output labeled JSONL path")
    p.add_argument(
        "--judge-model", default="claude-opus-4-6",
        help="Judge model (default: claude-opus-4-6). Use 'gpt-4o' for OpenAI.",
    )
    p.add_argument(
        "--skip-labeled", action="store_true",
        help="Skip input records where keep_ok is already set",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Append to existing output; skip (pmcid, chunk_id) pairs already present",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print first 3 prompts without calling the API",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build dedup set from existing output file (--resume)
    done: set[tuple[str, str]] = set()
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    done.add((obj.get("pmcid", ""), obj.get("chunk_id", "")))
        print(f"[resume]  {len(done)} already-labeled records in {out_path}")

    # Load input records
    raw_lines = [
        line for line in Path(args.records).read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    all_records = [json.loads(line) for line in raw_lines]

    # Filter to records that need processing
    to_process: list[dict] = []
    n_skipped = 0
    for rec in all_records:
        key = (rec.get("pmcid", ""), rec.get("chunk_id", ""))
        if key in done:
            n_skipped += 1
            continue
        if args.skip_labeled and rec.get("keep_ok") is not None:
            n_skipped += 1
            continue
        to_process.append(rec)

    type_counts = {"A": 0, "B": 0, "C": 0}
    for rec in to_process:
        type_counts[_record_type(rec)] += 1

    print("\n=== Input ===")
    print(f"  total          : {len(all_records)}")
    print(f"  skipped        : {n_skipped}")
    print(f"  to label       : {len(to_process)}")
    print(f"  type breakdown : A={type_counts['A']}  B={type_counts['B']}  C={type_counts['C']}")

    if not to_process:
        print("Nothing to do.")
        return

    # Build chains (not needed for dry-run — avoids requiring API key)
    if not args.dry_run:
        llm = _make_llm(args.judge_model)
        quality_chain = _build_quality_chain(llm)
        comparison_chain = _build_comparison_chain(llm)
    else:
        quality_chain = comparison_chain = None

    file_mode = "a" if args.resume else "w"
    dry_run_count = 0
    n_ok = 0
    n_failed = 0

    with out_path.open(file_mode, encoding="utf-8") as out_f:
        for i, rec in enumerate(to_process, 1):
            rtype = _record_type(rec)

            if args.dry_run:
                _print_dry_run(rec, {"A": "KEEP quality", "B": "comparison", "C": "escalation quality"}[rtype])
                dry_run_count += 1
                if dry_run_count >= 3:
                    print(f"\n[dry-run] Printed {dry_run_count} records. Stopping.")
                    break
                continue

            try:
                labeled = _label_record(rec, quality_chain, comparison_chain, args.judge_model)
                out_f.write(json.dumps(labeled) + "\n")
                out_f.flush()
                score = labeled.get("strong_judge_score")
                score_str = f"{score:.2f}" if score is not None else "—"
                print(
                    f"  [{i:>4}/{len(to_process)}] {labeled['pmcid']} {labeled['chunk_id']} "
                    f"type={rtype}  keep_ok={labeled.get('keep_ok')}  score={score_str}"
                )
                n_ok += 1
            except Exception:
                logger.exception(
                    "Failed to label %s / %s — skipping",
                    rec.get("pmcid"), rec.get("chunk_id"),
                )
                n_failed += 1

    if not args.dry_run:
        print("\n=== Done ===")
        print(f"  labeled  : {n_ok}")
        print(f"  failed   : {n_failed}")
        print(f"  output   : {out_path}")
        print()
        print("=" * 64)
        print("REQUIRED NEXT STEP: Human audit")
        print("=" * 64)
        print("  1. Sample 15 boundary-region records from the labeled file.")
        print("     (records where deferral_score is near your expected theta)")
        print("  2. Review chunk_text + output.findings independently")
        print("     WITHOUT looking at judge labels first.")
        print("  3. Compare your keep_ok labels against judge keep_ok.")
        print("  4. Compute Cohen's kappa:")
        print("       kappa > 0.70  → labels reliable, proceed to fit_routing_threshold.py")
        print("       kappa 0.50–0.70 → inspect disagreements, refine prompt if needed")
        print("       kappa < 0.50  → do not fit theta; redesign the judge prompt")


if __name__ == "__main__":
    main()
