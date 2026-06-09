"""
Summarize a single paper end-to-end and print the result.

Usage (from project root):
    python scripts/summarize_paper.py
    python scripts/summarize_paper.py PMC10047158
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

from pipeline.stages.summarization import SummarizationRunner  # noqa: E402

PMCID = sys.argv[1] if len(sys.argv) > 1 else "PMC10047158"

print(f"Loading {PMCID} from database...")
file_data = SummarizationRunner.load_paper_from_db(PMCID)
print(f"  {len(file_data['sentences_with_provenance'])} sentences\n")

runner = SummarizationRunner(
    voter_llms=[
        ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
        ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
    ],
    escalation_llm=ChatOpenAI(model="gpt-4o", temperature=0.0),
    theta=0.6,
    grounding_threshold=None,
    contradiction_similarity_threshold=None,
    output_dir=Path("out/summaries"),
)

result = runner.process(file_data)

print("=" * 70)
if result["status"] in ("success", "skipped"):
    print(f"SUMMARY\n{result['summary']}\n")
    print(f"RULES ({len(result['rules'])})")
    for r in result["rules"]:
        print(f"  [{r['rule_id']}] {r['type']}  conf={r['confidence']}")
        print(f"    {r['condition']}")
        print(f"    {r['action']}")
else:
    print(f"ERROR: {result['error']}")
print("=" * 70)
