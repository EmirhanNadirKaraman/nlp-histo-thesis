#!/usr/bin/env python3
"""
seed_variant_labels.py — pre-populate per-variant annotation files from the
legacy shared label files, so the `eval/annotate.py` --variant flow starts
with the existing labels already applied (and only prompts for *new* crops).

Logic per variant:

1. Find which crops that variant emitted (from
   ``out/sweeps/<variant>/json/*_media.json``).
2. Load the matching legacy label file
   (``eval/annotations/annotations_<mode>.json``).
3. Filter the legacy labels to just the crops this variant emitted.
4. Write the filtered set to ``eval/annotations/<variant>/<mode>.json``.

The mode-per-variant mapping mirrors what ``score_pdf_variants.detector_modes``
infers from each sweep's manifest:

* DOCLING-only sweeps → ``json_tables_docling``
* All other sweeps → ``json_tables_full``
* All sweeps (figures are detector-invariant) → ``json_figures``

By default the seeder refuses to overwrite an existing per-variant file
(use ``--force`` to clobber).  Per-variant labels you've already added by
hand are preserved.

Usage::

    python scripts/eval/seed_variant_labels.py
    python scripts/eval/seed_variant_labels.py --force
    python scripts/eval/seed_variant_labels.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Set, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SWEEPS_ROOT = _REPO_ROOT / "out" / "sweeps"
_ANN_ROOT = _REPO_ROOT / "eval" / "annotations"


def _load_legacy(mode: str) -> Dict[str, str]:
    p = _ANN_ROOT / f"annotations_{mode}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def _emitted_crops(sweep_dir: Path) -> Tuple[Set[str], Set[str]]:
    figures, tables = set(), set()
    for media in (sweep_dir / "json").rglob("*_media.json"):
        try:
            data = json.loads(media.read_text())
        except Exception:
            continue
        for m in data.get("figures", []) or []:
            if m.get("image_path"):
                figures.add(Path(m["image_path"]).name)
        for m in data.get("tables", []) or []:
            if m.get("image_path"):
                tables.add(Path(m["image_path"]).name)
    return figures, tables


def _table_mode(sweep_dir: Path) -> str:
    """HYBRID / TATR variants → json_tables_full; DOCLING → json_tables_docling."""
    manifests = sorted((sweep_dir / "run_metadata").glob("run_*.json"))
    detector = "HYBRID"
    if manifests:
        try:
            data = json.loads(manifests[-1].read_text())
            detector = data.get("config", {}).get("table_detector", "HYBRID")
        except Exception:
            pass
    if detector.upper() == "DOCLING":
        return "json_tables_docling"
    return "json_tables_full"


def seed_variant(
    sweep_dir: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> Dict[str, Dict[str, int]]:
    """Pre-seed per-variant label files for one sweep.  Returns a per-mode
    summary ``{mode: {emitted, legacy_total, seeded, would_overwrite}}``."""
    variant = sweep_dir.name
    figures, tables = _emitted_crops(sweep_dir)
    table_mode = _table_mode(sweep_dir)

    summary: Dict[str, Dict[str, int]] = {}

    for mode, crops in (("json_figures", figures), (table_mode, tables)):
        legacy = _load_legacy(mode)
        applicable = {k: v for k, v in legacy.items() if k in crops}
        out_path = _ANN_ROOT / variant / f"{mode}.json"

        existing = out_path.exists()
        if existing and not force:
            summary[mode] = {
                "emitted": len(crops),
                "legacy_total": len(legacy),
                "applicable": len(applicable),
                "seeded": 0,
                "skipped_already_exists": 1,
            }
            continue

        if not dry_run:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp.write_text(json.dumps(applicable, indent=2, ensure_ascii=False))
            tmp.replace(out_path)

        summary[mode] = {
            "emitted": len(crops),
            "legacy_total": len(legacy),
            "applicable": len(applicable),
            "seeded": len(applicable),
            "skipped_already_exists": 0,
        }
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweeps-root", type=Path, default=_SWEEPS_ROOT)
    p.add_argument("--variant", default=None,
                   help="Seed only this one variant (by sweep dir name).  "
                        "Default: seed every variant under --sweeps-root.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing per-variant label files.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be seeded but write nothing.")
    args = p.parse_args(argv)

    if not args.sweeps_root.exists():
        print(f"error: sweeps root not found: {args.sweeps_root}", file=sys.stderr)
        return 2

    targets = sorted(d for d in args.sweeps_root.iterdir() if d.is_dir())
    if args.variant:
        targets = [d for d in targets if d.name == args.variant]
        if not targets:
            print(f"error: variant {args.variant!r} not under {args.sweeps_root}",
                  file=sys.stderr)
            return 2

    if args.dry_run:
        print("(dry run — no files written)")

    for sweep_dir in targets:
        print(f"\n{sweep_dir.name}:")
        s = seed_variant(sweep_dir, force=args.force, dry_run=args.dry_run)
        for mode, stats in s.items():
            note = ""
            if stats["skipped_already_exists"]:
                note = "  (already exists — use --force to overwrite)"
            elif stats["seeded"] == 0 and stats["applicable"] == 0:
                note = "  (no applicable legacy labels)"
            print(f"  {mode:24s}  emitted={stats['emitted']:3d}  "
                  f"legacy={stats['legacy_total']:3d}  "
                  f"seeded={stats['seeded']:3d}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
