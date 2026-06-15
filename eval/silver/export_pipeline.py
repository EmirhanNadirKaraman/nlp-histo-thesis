"""
Export pipeline findings for sampled cases.

Reads eval/data/source_cases_related15.jsonl, queries sum_map_findings for matching
evidence_refs, writes eval/data/pipeline_findings_related15.jsonl.

Usage:
  python -m eval.silver.export_pipeline
  python -m eval.silver.export_pipeline --source eval/data/source_cases_heldout15.jsonl
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


def _derive_output(source: Path) -> Path:
    """Mirror generate.py's namespacing: source_cases_<stem>.jsonl routes to
    pipeline_findings_<stem>.jsonl, so each selection's pipeline findings land in
    their own file (source_cases_related15.jsonl → pipeline_findings_related15.jsonl).
    The ad-hoc scratch source_cases.jsonl → bare pipeline_findings.jsonl."""
    stem = source.stem
    if stem == "source_cases":
        return OUTPUT_PATH
    if stem.startswith("source_cases"):
        return OUTPUT_PATH.with_name(f"pipeline_findings{stem[len('source_cases'):]}.jsonl")
    return OUTPUT_PATH.with_name(f"pipeline_findings_{stem}.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Export pipeline findings for sampled cases")
    parser.add_argument("--source",  default=str(SOURCE_PATH))
    parser.add_argument("--output",  default=None,
                        help="Output JSONL. Default: pipeline_findings.jsonl for the primary "
                             "source_cases.jsonl; a namespaced source_cases_<stem>.jsonl "
                             "auto-routes to pipeline_findings_<stem>.jsonl.")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"Source file not found: {source}", file=sys.stderr)
        print("Run `python -m eval.silver.sample` first.", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output) if args.output is not None else _derive_output(source)
    logging.info("Output → %s", output)
    export_pipeline_outputs(source_cases_path=source, output_path=output)


if __name__ == "__main__":
    main()
