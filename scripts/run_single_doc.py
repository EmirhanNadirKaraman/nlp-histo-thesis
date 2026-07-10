"""
Quick smoke-test for KnowledgeExtractionRunner on a single paper from the database.

Usage:
    cd /Users/emir/Documents/GitHub/nlp-histo
    python scripts/run_single_doc.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from pipeline.stages.knowledge_extraction import KnowledgeExtractionRunner

PMCID = "PMC10047158"

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s  %(name)s  %(message)s",
    stream=sys.stdout,
)
# Silence noisy third-party loggers
for noisy in ("httpx", "openai", "httpcore", "langchain", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ── load input from DB ─────────────────────────────────────────────────────────
load_dotenv()

file_data = KnowledgeExtractionRunner.load_paper_from_db(PMCID)

print(f"\nPaper    : {file_data['pmcid']}")
print(f"Sentences: {len(file_data['sentences_with_provenance'])}\n")

# ── output dirs ────────────────────────────────────────────────────────────────
OUTPUT_DIR        = Path(__file__).parent / "output"
MAP_DIR           = OUTPUT_DIR / "map"
REDUCE_DIR        = OUTPUT_DIR / "reduce"
RULES_DIR         = OUTPUT_DIR / "rules"
CONTRADICTION_DIR = OUTPUT_DIR / "contradictions"
for d in (MAP_DIR, REDUCE_DIR, RULES_DIR, CONTRADICTION_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── build runner ───────────────────────────────────────────────────────────────
voter_llms = [
    ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
    ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
]
escalation_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

runner = KnowledgeExtractionRunner(
    voter_llms=voter_llms,
    escalation_llm=escalation_llm,
    theta=0.6,
    chunk_size=10,
    output_dir=OUTPUT_DIR,
)

# ── run ────────────────────────────────────────────────────────────────────────
result = runner.process(file_data)

# ── save per-stage outputs ─────────────────────────────────────────────────────
pmcid = file_data["pmcid"]
if result["status"] in ("success", "skipped"):
    (MAP_DIR / f"{pmcid}.json").write_text(
        json.dumps(result["audit_trail"]["map_chunks"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (REDUCE_DIR / f"{pmcid}.json").write_text(
        json.dumps(result["audit_trail"]["master_summary"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (RULES_DIR / f"{pmcid}.json").write_text(
        json.dumps(result["audit_trail"]["rules_provenance"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if result["contradiction_report"]:
        (CONTRADICTION_DIR / f"{pmcid}.json").write_text(
            json.dumps(result["contradiction_report"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

# ── print result ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
if result["status"] in ("success", "skipped"):
    print("STATUS : success")
    print(f"PAPER  : {result['pmcid']}")
    print(f"SUMMARY:\n{result['summary']}\n")
    print(f"RULES  : {len(result['rules'])}")
    for r in result["rules"]:
        print(f"  [{r['rule_id']}] {r['type']}  conf={r['confidence']}")
        print(f"    {r['condition']}")
        print(f"    {r['action']}")
    print(f"\nOutputs saved to {OUTPUT_DIR}")
else:
    print(f"STATUS : error\n{result['error']}")
print("=" * 70)
