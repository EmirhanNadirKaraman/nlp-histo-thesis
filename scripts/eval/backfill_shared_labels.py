"""backfill_shared_labels.py — copy labels between peer variants using share_map.

When a new sweep variant is added after some of its bbox-peers have already been
labelled, the per-variant label files for the new variant start empty.  The
``annotate.py`` propagation only writes forward (label → peers) at the moment a
label is added, so peer labels added before the new variant existed do not
backfill automatically.

This script reads ``eval/annotations/share_map.json`` plus every per-variant
``<variant>/<mode>.json`` and, for each bbox-group, copies the first-found label
to every variant in the group that doesn't have one yet.  Existing labels are
preserved — pass ``--force-overwrite`` only if you want existing entries
overwritten by the canonical-peer label (rarely useful).

Usage::

    python scripts/eval/backfill_shared_labels.py            # backfill all variants
    python scripts/eval/backfill_shared_labels.py --dry-run  # report counts only
    python scripts/eval/backfill_shared_labels.py --variant 15_best_twopass_off
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ANN_DIR = Path("eval/annotations")
SHARE_MAP_PATH = ANN_DIR / "share_map.json"
MODES = ("json_figures", "json_tables_full", "json_tables_docling")


def _load_labels() -> dict[str, dict[str, dict]]:
    labels: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    for vdir in ANN_DIR.iterdir():
        if not vdir.is_dir() or vdir.name.startswith("_"):
            continue
        for mode in MODES:
            p = vdir / f"{mode}.json"
            if not p.exists():
                continue
            try:
                labels[vdir.name][mode] = json.loads(p.read_text())
            except json.JSONDecodeError:
                print(f"WARN  skipping unreadable JSON: {p}", file=sys.stderr)
    return labels


def _backfill(
    labels: dict[str, dict[str, dict]],
    share_map: dict,
    *,
    target_variant: str | None,
    force_overwrite: bool,
) -> dict[tuple[str, str], int]:
    copies: dict[tuple[str, str], int] = defaultdict(int)
    for fname, groups in share_map.items():
        if not isinstance(groups, list):
            continue
        for g in groups:
            if not isinstance(g, dict) or "variants" not in g:
                continue
            variants = g["variants"]
            for mode in MODES:
                canon = None
                for v in variants:
                    lbl = labels[v][mode].get(fname)
                    if lbl:
                        canon = lbl
                        break
                if canon is None:
                    continue
                for v in variants:
                    if target_variant and v != target_variant:
                        continue
                    if fname in labels[v][mode] and not force_overwrite:
                        continue
                    labels[v][mode][fname] = canon
                    copies[(v, mode)] += 1
    return copies


def _write(labels: dict[str, dict[str, dict]], touched: set[tuple[str, str]]) -> None:
    for (v, mode) in touched:
        content = labels[v][mode]
        p = ANN_DIR / v / f"{mode}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default=None,
                        help="Only backfill into this variant (default: all).")
    parser.add_argument("--force-overwrite", action="store_true",
                        help="Overwrite existing per-variant labels with the canonical peer label.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report counts only; do not write files.")
    args = parser.parse_args()

    if not SHARE_MAP_PATH.exists():
        print(f"ERROR  share_map missing: {SHARE_MAP_PATH}", file=sys.stderr)
        print("       run `python scripts/eval/build_share_map.py` first.", file=sys.stderr)
        return 1

    share_map = json.loads(SHARE_MAP_PATH.read_text())
    labels = _load_labels()

    copies = _backfill(
        labels,
        share_map,
        target_variant=args.variant,
        force_overwrite=args.force_overwrite,
    )

    total = sum(copies.values())
    print(f"Backfilled {total} labels across {len(copies)} (variant, mode) buckets"
          + (" [dry-run]" if args.dry_run else ""))
    for (v, mode), n in sorted(copies.items()):
        print(f"  {v}/{mode}: +{n}")

    if not args.dry_run and copies:
        _write(labels, set(copies.keys()))

    return 0


if __name__ == "__main__":
    sys.exit(main())
