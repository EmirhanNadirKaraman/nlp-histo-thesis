"""
Export pipeline findings for sampled cases.

Reads eval/data/source_cases.jsonl, queries sum_map_findings for matching
evidence_refs, writes eval/data/pipeline_findings.jsonl.

Usage:
  python -m eval.silver.export_pipeline
  python -m eval.silver.export_pipeline --source eval/data/source_cases.jsonl
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

from eval.silver.exporter import export_pipeline_outputs

SOURCE_PATH = Path("eval/data/source_cases_related15.jsonl")
OUTPUT_PATH = Path("eval/data/pipeline_findings.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Export pipeline findings for sampled cases")
    parser.add_argument("--source",  default=str(SOURCE_PATH))
    parser.add_argument("--output",  default=str(OUTPUT_PATH))
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"Source file not found: {source}", file=sys.stderr)
        print("Run `python -m eval.silver.sample` first.", file=sys.stderr)
        sys.exit(1)

    export_pipeline_outputs(source_cases_path=source, output_path=Path(args.output))


if __name__ == "__main__":
    main()
