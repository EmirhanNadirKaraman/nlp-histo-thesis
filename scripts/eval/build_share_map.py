#!/usr/bin/env python3
"""
build_share_map.py — invert the crop-filename × sweep relation, keyed by
both filename AND bbox.

Two variants emitting the same crop FILENAME may still produce different
crop CONTENT — different detectors find different boundaries, and
post-detection cropping options (footnote expansion, threshold relaxation)
extend the bbox.  Propagating labels across variants whose bboxes differ
will corrupt extent-sensitive labels (``missing footnotes``, ``crop is too
big``, ``crop is too small``).

To prevent that, this script groups variants by (filename, quantized bbox)
— variants only share a group when both filename AND bbox match.  The
annotator's ``propagate_label`` looks up the matching group, so labels
only flow between variants whose crop content is the same.

Output structure (``eval/annotations/share_map.json``)::

    {
      "PMC..._Table_1_p2.png": [
        {
          "bbox":     [50.0, 532.0, 520.0, 402.0],
          "variants": ["baseline", "no_two_pass", "tatr_090"]
        },
        {
          "bbox":     [48.0, 540.0, 525.0, 380.0],
          "variants": ["baseline_footnote_relaxed"]
        }
      ],
      ...
    }

Bboxes are quantized to ``BBOX_TOLERANCE_PTS`` (default 1.0 pt) before
grouping, to absorb sub-pt floating-point variations across detector
runs that should otherwise be considered identical.

Usage::

    python scripts/eval/build_share_map.py
    python scripts/eval/build_share_map.py --bbox-tolerance 2.0
    python scripts/eval/build_share_map.py --out eval/annotations/share_map.json
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SWEEPS_ROOT = _DEFAULT_REPO_ROOT / "out" / "sweeps"
_DEFAULT_OUT = _DEFAULT_REPO_ROOT / "eval" / "annotations" / "share_map.json"
_DEFAULT_BBOX_TOLERANCE = 1.0  # pts


BBoxKey = Tuple[float, float, float, float]


def quantize_bbox(bbox: dict, tol: float) -> BBoxKey:
    """Round bbox corners to the nearest ``tol`` pts so cross-detector
    floating-point noise collapses into a single equivalence class."""
    def q(v):
        try:
            return round(float(v) / tol) * tol
        except (TypeError, ValueError):
            return 0.0
    return (q(bbox.get("x1")), q(bbox.get("y1")),
            q(bbox.get("x2")), q(bbox.get("y2")))


def _collect_with_bbox(
    sweep_dir: Path, *, tol: float
) -> List[Tuple[str, BBoxKey]]:
    """Return ``[(crop_filename, quantized_bbox)]`` for everything emitted
    by ``sweep_dir`` (figures + tables)."""
    out: List[Tuple[str, BBoxKey]] = []
    for media in sweep_dir.rglob("*_media.json"):
        try:
            data = json.loads(media.read_text())
        except Exception:
            logger.warning("skipping unreadable media.json: %s", media)
            continue
        for kind in ("figures", "tables"):
            for m in data.get(kind, []) or []:
                image_path = m.get("image_path")
                if not image_path:
                    continue
                bbox = m.get("bbox") or {}
                out.append((Path(image_path).name, quantize_bbox(bbox, tol)))
    return out


def build(
    sweeps_root: Path,
    *,
    bbox_tolerance: float = _DEFAULT_BBOX_TOLERANCE,
) -> Dict[str, List[Dict]]:
    """Return ``{filename: [{"bbox": [...], "variants": [...]}, ...]}``.

    Each filename maps to one entry per distinct bbox-equivalence class.
    Variants whose crop bbox quantizes to the same key share an entry —
    propagation between them is safe.  Variants whose bbox quantizes
    differently end up in separate entries — propagation between them is
    explicitly not supported by this map.
    """
    if not sweeps_root.exists():
        raise FileNotFoundError(f"sweeps root not found: {sweeps_root}")
    variants = sorted(p for p in sweeps_root.iterdir() if p.is_dir())
    if not variants:
        raise RuntimeError(f"no sweep variants under {sweeps_root}")

    groups: Dict[Tuple[str, BBoxKey], set] = defaultdict(set)
    for variant in variants:
        json_dir = variant / "json"
        if not json_dir.exists():
            logger.warning("variant %s has no json/ subdir; skipping", variant.name)
            continue
        emissions = _collect_with_bbox(json_dir, tol=bbox_tolerance)
        for filename, bbox_key in emissions:
            groups[(filename, bbox_key)].add(variant.name)
        logger.info("variant %s: %d crops", variant.name, len(emissions))

    by_filename: Dict[str, List[Dict]] = defaultdict(list)
    for (filename, bbox_key), variant_set in groups.items():
        by_filename[filename].append({
            "bbox":     list(bbox_key),
            "variants": sorted(variant_set),
        })
    out = {}
    for filename in sorted(by_filename):
        out[filename] = sorted(by_filename[filename],
                                key=lambda g: tuple(g["bbox"]))
    return out


def write(share_map: Dict[str, List[Dict]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(share_map, indent=2, ensure_ascii=False))
    tmp.replace(out)
    logger.info("wrote share_map: %d crops → %s", len(share_map), out)


def _summarize(share_map: Dict[str, List[Dict]]) -> None:
    total_filenames = len(share_map)
    total_groups = sum(len(g) for g in share_map.values())
    by_group_count: Dict[int, int] = defaultdict(int)
    by_variant_count: Dict[int, int] = defaultdict(int)
    for groups in share_map.values():
        by_group_count[len(groups)] += 1
        for g in groups:
            by_variant_count[len(g["variants"])] += 1
    print("\nShare-map summary:")
    print(f"  unique crop filenames: {total_filenames}")
    print(f"  total bbox-groups:     {total_groups} "
          f"(+{total_groups - total_filenames} split across multiple bbox-groups)")
    print("  filenames by group-count:")
    for n in sorted(by_group_count):
        print(f"    {n} bbox-group(s): {by_group_count[n]} files")
    print("  group sizes (#variants sharing the same bbox):")
    for n in sorted(by_variant_count):
        print(f"    {n} variant(s): {by_variant_count[n]} groups")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweeps-root", type=Path, default=_DEFAULT_SWEEPS_ROOT,
                   help="Directory containing per-variant sweep subdirs.")
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                   help="Path to write share_map.json.")
    p.add_argument("--bbox-tolerance", type=float, default=_DEFAULT_BBOX_TOLERANCE,
                   help="Bbox quantization step in pts.  Larger values collapse "
                        "near-duplicate bboxes into one group.  Default: 1.0 pt.")
    p.add_argument("--quiet", action="store_true", help="Suppress per-variant log lines.")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        share_map = build(args.sweeps_root, bbox_tolerance=args.bbox_tolerance)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 2

    write(share_map, args.out)
    _summarize(share_map)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
