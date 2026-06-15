"""
Generate silver findings from source cases using Opus.

Reads eval/data/source_cases_related15.jsonl, writes eval/data/silver_findings_related15.jsonl.
Skips cases already in the output file (cache by case_id + prompt_version + model).

Usage:
  python -m eval.silver.generate
  python -m eval.silver.generate --model claude-opus-4-5
  python -m eval.silver.generate --source eval/data/source_cases.jsonl
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

from eval.silver.generator import SilverGenerator, DEFAULT_MODEL

SOURCE_PATH = Path("eval/data/source_cases_related15.jsonl")
OUTPUT_PATH = Path("eval/data/silver_findings.jsonl")


def _derive_output(source: Path) -> Path:
    """Mirror sample.py's namespacing: source_cases_<stem>.jsonl routes to
    silver_findings_<stem>.jsonl, so each selection's silver lands in its own
    file (source_cases_related15.jsonl → silver_findings_related15.jsonl).
    The ad-hoc scratch source_cases.jsonl → bare silver_findings.jsonl."""
    stem = source.stem  # e.g. "source_cases_related15"
    if stem == "source_cases":
        return OUTPUT_PATH
    if stem.startswith("source_cases"):
        return OUTPUT_PATH.with_name(f"silver_findings{stem[len('source_cases'):]}.jsonl")
    return OUTPUT_PATH.with_name(f"silver_findings_{stem}.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Generate silver labels via Opus")
    parser.add_argument("--source",  default=str(SOURCE_PATH))
    parser.add_argument("--output",  default=None,
                        help="Output JSONL. Default: silver_findings.jsonl for the primary "
                             "source_cases.jsonl; a namespaced source_cases_<stem>.jsonl "
                             "auto-routes to silver_findings_<stem>.jsonl.")
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--batch",   action="store_true",
                        help="Use Anthropic batch API (~50% cheaper). Re-run to check status.")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"Source file not found: {source}", file=sys.stderr)
        print("Run `python -m eval.silver.sample` first.", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output) if args.output is not None else _derive_output(source)
    logging.info("Output → %s", output)

    gen = SilverGenerator(
        source_cases_path=source,
        output_path=output,
        model=args.model,
    )
    if args.batch:
        gen.run_batch()
    else:
        gen.run()


if __name__ == "__main__":
    main()
