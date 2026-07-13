"""
Sample source cases from the TextElement table.

Default: all usable chunks of the related15 calibration cluster →
source_cases_related15.jsonl. Any other --from-selection writes
source_cases_<yaml-stem>.jsonl (held-out set, etc.).

Usage:
  python -m eval.silver.data.sample                                         # related15 → source_cases_related15.jsonl
  python -m eval.silver.data.sample --from-selection configs/paper_selection/heldout15.yaml
  python -m eval.silver.data.sample --from-selection '' --n 100 --seed 7    # ad-hoc random → source_cases.jsonl
  python -m eval.silver.data.sample --pmcids PMC1 PMC2
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

from eval.silver.data.sampler import sample_source_cases, sample_papers_source_cases
from eval.silver.data.schemas import SourceCase
from eval.silver.data.jsonl_utils import write_jsonl
from eval.paper_selection.loaders import load_pmcids_from_selection

OUTPUT_PATH = Path("eval/data/source_cases.jsonl")
DEFAULT_SELECTION = Path("configs/paper_selection/related15.yaml")


def main():
    parser = argparse.ArgumentParser(description="Sample source cases for silver-label generation")
    parser.add_argument("--n",        type=int, default=50, help="Number of individual TEs to sample")
    parser.add_argument("--n-papers", type=int, default=None,
                        help="Select N random papers and take ALL their text elements")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--pmcids",   nargs="*", help="Restrict to specific PMCIDs")
    parser.add_argument("--from-selection", default=str(DEFAULT_SELECTION), metavar="PATH",
                        help="Selection YAML to pull PMCIDs from (default: the related15 "
                             "ILP cluster). Takes ALL usable chunks of those papers. "
                             "Pass '' to disable and use random sampling / --pmcids.")
    parser.add_argument("--output",   default=None,
                        help="Output JSONL. Default: derived from --from-selection as "
                             "source_cases_<yaml-stem>.jsonl (e.g. related15.yaml → "
                             "source_cases_related15.jsonl; heldout15.yaml → "
                             "source_cases_heldout15.jsonl). Bare source_cases.jsonl is used "
                             "only for ad-hoc random sampling / --pmcids.")
    args = parser.parse_args()

    # Resolve output. Explicit --output wins. Every named selection is written to
    # source_cases_<yaml-stem>.jsonl so the calibration (related15) and held-out
    # (heldout15) sets never clobber each other; the bare source_cases.jsonl is
    # reserved for ad-hoc random sampling / --pmcids.
    if args.output is not None:
        output = Path(args.output)
    elif args.from_selection and not args.pmcids:
        output = OUTPUT_PATH.with_name(f"source_cases_{Path(args.from_selection).stem}.jsonl")
    else:
        output = OUTPUT_PATH
    logging.info("Output → %s", output)

    # Resolve the PMCID set: explicit --pmcids wins, else the selection YAML.
    pmcids = args.pmcids
    if not pmcids and args.from_selection:
        pmcids = load_pmcids_from_selection(Path(args.from_selection))
        logging.info("Loaded %d PMCIDs from %s", len(pmcids), args.from_selection)

    if pmcids and args.n_papers is None:
        # Cluster mode: every usable chunk of exactly these papers.
        cases_raw = sample_papers_source_cases(
            n_papers=len(pmcids), seed=args.seed, pmcids=pmcids
        )
        found = {c["pmcid"] for c in cases_raw}
        missing = [p for p in pmcids if p not in found]
        if missing:
            raise SystemExit(
                f"{len(missing)}/{len(pmcids)} selected paper(s) produced no cases "
                f"(not ingested, or no usable chunks): {missing}"
            )
    elif args.n_papers is not None:
        cases_raw = sample_papers_source_cases(
            n_papers=args.n_papers, seed=args.seed, pmcids=pmcids
        )
    else:
        cases_raw = sample_source_cases(n=args.n, seed=args.seed, pmcids=pmcids)

    cases = [SourceCase(**c) for c in cases_raw]

    write_jsonl(output, cases)
    print(f"Wrote {len(cases)} source cases → {output}")


if __name__ == "__main__":
    main()
