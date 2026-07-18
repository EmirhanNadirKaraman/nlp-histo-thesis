#!/usr/bin/env python3
"""Remove every annotation + ground-truth reference to a given pmcid from the
PDF-extraction eval set, so it is excluded from `score_pdf_variants.py`.

Strips, in place (writers mirror the repo's own tooling so git diffs stay to
the removed lines only):
  * eval/annotations/<variant>/json_{figures,tables_docling,tables_full}.json
    — drop keys `^<pmcid>_…` (annotate.py writer: json.dumps(indent=2)).
  * eval/annotations/share_map.json — drop keys `^<pmcid>_…`
    (build_share_map.py writer: json.dumps(indent=2, ensure_ascii=False)).
    Values reference variant names, never pmcids, so only keys are touched.
  * eval/ground_truth.csv — drop the row whose pmcid PMC-prefix matches.

Deliberately leaves eval/annotations/_archive_*/ untouched (historical
snapshots, not read by the scorer). See BUGS.md B-068.

Usage:
    python3 scripts/eval/drop_pmcid_from_annotations.py PMC11863705 [--dry-run]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ANN_ROOT = _REPO_ROOT / "eval" / "annotations"
_GT_CSV = _REPO_ROOT / "eval" / "ground_truth.csv"


def _pmc_prefix(stem: str) -> str:
    m = re.match(r"(PMC\d+)", stem)
    return m.group(1) if m else stem


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    dry = "--dry-run" in sys.argv
    if not args:
        print("usage: drop_pmcid_from_annotations.py PMCxxxxx [--dry-run]")
        return 2
    pmcid = args[0]
    key_prefix = pmcid + "_"  # crop ids are '<pmcid>_<file>_Figure_1_p1.png'

    total_keys = 0
    files_touched = 0

    # per-variant label files + active share_map.json
    targets = sorted(_ANN_ROOT.glob("*/json_*.json"))
    share_map = _ANN_ROOT / "share_map.json"
    if share_map.exists():
        targets.append(share_map)

    for path in targets:
        if "_archive" in str(path.relative_to(_ANN_ROOT)):
            continue  # leave historical snapshots alone
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  skip (not json): {path}")
            continue
        if not isinstance(data, dict):
            continue
        hits = [k for k in data if k.startswith(key_prefix)]
        if not hits:
            continue
        for k in hits:
            del data[k]
        total_keys += len(hits)
        files_touched += 1
        rel = path.relative_to(_REPO_ROOT)
        if dry:
            print(f"  [dry] would drop {len(hits):>2} keys from {rel}")
        else:
            # mirror the writer used by the tool that owns each file
            if path.name == "share_map.json":
                text = json.dumps(data, indent=2, ensure_ascii=False)
            else:
                text = json.dumps(data, indent=2)
            path.write_text(text)
            print(f"  dropped {len(hits):>2} keys from {rel}")

    # ground_truth.csv row
    gt_dropped = 0
    if _GT_CSV.exists():
        lines = _GT_CSV.read_text().splitlines(keepends=True)
        kept = []
        for i, line in enumerate(lines):
            stem = line.split(",", 1)[0].strip()
            if i > 0 and _pmc_prefix(stem) == pmcid:
                gt_dropped += 1
                if dry:
                    print(f"  [dry] would drop GT row: {line.strip()}")
                continue
            kept.append(line)
        if gt_dropped and not dry:
            _GT_CSV.write_text("".join(kept))
            print(f"  dropped {gt_dropped} ground_truth.csv row(s)")

    print(f"\n{'[dry-run] ' if dry else ''}Summary for {pmcid}: "
          f"{total_keys} annotation keys across {files_touched} files, "
          f"{gt_dropped} GT row(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
