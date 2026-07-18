"""check_label_coverage.py — report emitted vs labelled vs unlabelled crops.

Reads each variant's emitted media JSONs (under
``out/sweeps/<variant>/json/*_media.json``) and cross-references against
the per-variant annotation files (``eval/annotations/<variant>/<mode>.json``)
to surface how many crops still need labelling.

Default: every variant directory under ``out/sweeps/``.  Restrict with
``--variant`` (repeatable) or by passing names as positional args.

Modes scanned per variant:
  * ``json_figures``          — figures (detector-invariant)
  * ``json_tables_docling``   — tables emitted by Docling-only variants
  * ``json_tables_full``      — tables emitted by TATR / Hybrid variants

By default the table mode is auto-detected from the variant's
``run_metadata/*_stats.json`` (config_used.table_detector) so docling
variants only show ``json_tables_docling`` and TATR/Hybrid variants
only show ``json_tables_full``.  Pass ``--all-modes`` to force every
mode regardless.

Usage::

    python scripts/eval/check_label_coverage.py
    python scripts/eval/check_label_coverage.py 02_tatr_090 03_tatr_095
    python scripts/eval/check_label_coverage.py --variant 18_best_drop_tables_in_figures
    python scripts/eval/check_label_coverage.py --unlabelled-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SWEEPS_DIR = _REPO_ROOT / "out" / "sweeps"
_ANN_DIR    = _REPO_ROOT / "eval" / "annotations"

_MODES = ("json_figures", "json_tables_docling", "json_tables_full")


def _emitted_crops(sweep_dir: Path) -> dict[str, set[str]]:
    """Return {figures: {filenames}, tables: {filenames}} for a variant."""
    figures, tables = set(), set()
    json_dir = sweep_dir / "json"
    if not json_dir.exists():
        return {"figures": figures, "tables": tables}
    for mj in sorted(json_dir.glob("*_media.json")):
        try:
            data = json.loads(mj.read_text())
        except json.JSONDecodeError:
            continue
        for f in data.get("figures", []):
            figures.add(Path(f["image_path"]).name)
        for t in data.get("tables", []):
            tables.add(Path(t["image_path"]).name)
    return {"figures": figures, "tables": tables}


def _labels_for(variant: str, mode: str) -> dict | None:
    p = _ANN_DIR / variant / f"{mode}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        print(f"WARN  unreadable: {p}", file=sys.stderr)
        return None


def _detect_table_mode(sweep_dir: Path) -> str:
    """Detect the primary table-annotation mode from variant run metadata.

    Returns ``"json_tables_docling"`` for docling-only variants and
    ``"json_tables_full"`` for TATR / Hybrid variants.  Falls back to a
    name-pattern guess when no run_metadata is present.
    """
    metadata_dir = sweep_dir / "run_metadata"
    if metadata_dir.exists():
        for stats in metadata_dir.glob("*_stats.json"):
            try:
                d = json.loads(stats.read_text())
            except json.JSONDecodeError:
                continue
            det = d.get("config_used", {}).get("table_detector")
            if det == "docling":
                return "json_tables_docling"
            if det in ("tatr", "hybrid"):
                return "json_tables_full"
    return "json_tables_docling" if "docling" in sweep_dir.name else "json_tables_full"


def _coverage_for(variant: str, all_modes: bool) -> list[tuple[str, int, int, int]]:
    """Return [(mode, emitted, labelled, unlabelled), ...] for the variant."""
    sweep_dir = _SWEEPS_DIR / variant
    crops = _emitted_crops(sweep_dir)
    primary_table_mode = _detect_table_mode(sweep_dir)

    rows: list[tuple[str, int, int, int]] = []
    for mode in _MODES:
        # Skip off-primary table modes unless --all-modes
        if not all_modes and mode != "json_figures" and mode != primary_table_mode:
            continue
        labels = _labels_for(variant, mode)
        if labels is None and not all_modes:
            continue
        labels = labels or {}
        emitted = crops["figures"] if mode == "json_figures" else crops["tables"]
        labelled = emitted & set(labels)
        unlabelled = emitted - set(labels)
        rows.append((mode, len(emitted), len(labelled), len(unlabelled)))
    return rows


def _resolve_variants(positional: list[str], flag_list: list[str]) -> list[str]:
    explicit = list(positional) + list(flag_list)
    if explicit:
        return explicit
    if not _SWEEPS_DIR.exists():
        return []
    return sorted(
        p.name for p in _SWEEPS_DIR.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", help="Variant names (positional, repeatable).")
    parser.add_argument("--variant", action="append", default=[],
                        help="Variant name (repeatable; equivalent to positional).")
    parser.add_argument("--all-modes", action="store_true",
                        help="Report every mode (including those with no annotation file).")
    parser.add_argument("--unlabelled-only", action="store_true",
                        help="Only print variants with unlabelled crops.")
    args = parser.parse_args()

    variants = _resolve_variants(args.variants, args.variant)
    if not variants:
        print(f"error: no variants found under {_SWEEPS_DIR}", file=sys.stderr)
        return 1

    total_unlabelled = 0
    n_printed = 0
    for variant in variants:
        rows = _coverage_for(variant, args.all_modes)
        v_unlabelled = sum(u for _, _, _, u in rows)
        if args.unlabelled_only and v_unlabelled == 0:
            continue
        n_printed += 1
        print(f"\n{variant}")
        if not rows:
            print("  (no annotation files; pass --all-modes to scan emissions only)")
            continue
        for mode, emitted, labelled, unlabelled in rows:
            flag = "  " if unlabelled == 0 else "!!"
            print(f"  {flag} {mode:24s} emitted={emitted:3d}  labelled={labelled:3d}  unlabelled={unlabelled:3d}")
        total_unlabelled += v_unlabelled

    print()
    print(f"Variants scanned: {len(variants)}  shown: {n_printed}  total unlabelled crops: {total_unlabelled}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
