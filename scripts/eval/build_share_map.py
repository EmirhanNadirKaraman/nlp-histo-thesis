#!/usr/bin/env python3
"""
build_share_map.py — invert the crop-filename × sweep relation.

Scans every ``out/sweeps/<variant>/json/*_media.json`` and produces a
JSON file mapping each crop filename to the list of sweep variants that
emitted that crop:

    {
      "PMC10047158_..._Table_1_p1.png": ["E3a_tatr095", "E3b_tatr090"],
      "PMC10047158_..._Figure_3_p4.png": ["baseline", "detector_docling",
                                          "detector_tatr", "tatr_095",
                                          "tatr_090", "no_two_pass"],
      ...
    }

Output: ``eval/annotations/share_map.json``.

Used by the extended ``eval/annotate.py`` (with ``--sweep`` / ``--variant``
flags) so a label written for a crop in one variant's annotation file is
also propagated to every other variant's annotation file that emitted
the same crop.

Usage::

    python scripts/eval/build_share_map.py
    python scripts/eval/build_share_map.py --sweeps-root out/sweeps
    python scripts/eval/build_share_map.py --out eval/annotations/share_map.json
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SWEEPS_ROOT = _DEFAULT_REPO_ROOT / "out" / "sweeps"
_DEFAULT_OUT = _DEFAULT_REPO_ROOT / "eval" / "annotations" / "share_map.json"


def _collect(sweep_dir: Path) -> set[str]:
    """Return the set of crop filenames a sweep emitted (figures + tables)."""
    out: set[str] = set()
    for media in sweep_dir.rglob("*_media.json"):
        try:
            data = json.loads(media.read_text())
        except Exception:
            logger.warning("skipping unreadable media.json: %s", media)
            continue
        for kind in ("figures", "tables"):
            for m in data.get(kind, []) or []:
                image_path = m.get("image_path")
                if image_path:
                    out.add(Path(image_path).name)
    return out


def build(sweeps_root: Path) -> Dict[str, List[str]]:
    """Return {filename: [variants_that_emitted_it]}, alphabetically sorted."""
    inv: Dict[str, set] = defaultdict(set)
    if not sweeps_root.exists():
        raise FileNotFoundError(f"sweeps root not found: {sweeps_root}")
    variants = sorted(p for p in sweeps_root.iterdir() if p.is_dir())
    if not variants:
        raise RuntimeError(f"no sweep variants under {sweeps_root}")
    for variant in variants:
        json_dir = variant / "json"
        if not json_dir.exists():
            logger.warning("variant %s has no json/ subdir; skipping", variant.name)
            continue
        crops = _collect(json_dir)
        for c in crops:
            inv[c].add(variant.name)
        logger.info("variant %s: %d crops", variant.name, len(crops))
    return {k: sorted(v) for k, v in sorted(inv.items())}


def write(share_map: Dict[str, List[str]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(share_map, indent=2, ensure_ascii=False))
    tmp.replace(out)
    logger.info("wrote share_map: %d crops → %s", len(share_map), out)


def _summarize(share_map: Dict[str, List[str]]) -> None:
    by_count: Dict[int, int] = defaultdict(int)
    for variants in share_map.values():
        by_count[len(variants)] += 1
    print("\nShare-map summary:")
    print(f"  total unique crop filenames: {len(share_map)}")
    for n in sorted(by_count):
        print(f"  emitted by {n} variant(s): {by_count[n]} crops")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweeps-root", type=Path, default=_DEFAULT_SWEEPS_ROOT,
                   help="Directory containing per-variant sweep subdirs.")
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                   help="Path to write share_map.json.")
    p.add_argument("--quiet", action="store_true", help="Suppress per-variant log lines.")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        share_map = build(args.sweeps_root)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 2

    write(share_map, args.out)
    _summarize(share_map)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
