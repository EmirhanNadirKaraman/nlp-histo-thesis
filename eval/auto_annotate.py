"""
eval/auto_annotate.py — Copy annotations between variants for identical tables.

For each table that appears in both the source and target variant JSON files
with exactly the same metadata (label, caption, page, bbox), the annotation
is copied from the source annotation file into the target annotation file.
Tables whose entries differ between variants are left unannotated.

Usage:
    python eval/auto_annotate.py <source_mode> <target_mode>

Example:
    python eval/auto_annotate.py json_tables_docling json_tables_docling_recon
    python eval/auto_annotate.py json_tables_docling_recon json_tables_full
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE     = Path(__file__).parent
OUT_DIR  = HERE / "out"
ANN_DIR  = HERE / "annotations"

VARIANT_DIRS = {
    "json_tables_full":          OUT_DIR / "json" / "full",
    "json_tables_docling":       OUT_DIR / "json" / "docling",
    "json_tables_docling_recon": OUT_DIR / "json" / "docling_recon",
}


def build_lookup(json_dir: Path) -> dict[str, dict]:
    """Map image filename -> table entry (excluding image_path and source)."""
    lookup: dict[str, dict] = {}
    for f in sorted(json_dir.glob("*_media.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        for t in data.get("tables", []):
            img = (t.get("image_path") or "").strip()
            fname = Path(img).name if img else None
            if fname:
                lookup[fname] = {k: v for k, v in t.items()
                                 if k not in ("image_path", "source")}
    return lookup


def ann_path(mode: str) -> Path:
    return ANN_DIR / f"annotations_{mode}.json"


def load_annotations(mode: str) -> dict[str, str]:
    p = ann_path(mode)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_annotations(mode: str, ann: dict[str, str]) -> None:
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    ann_path(mode).write_text(json.dumps(ann, indent=2))


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in VARIANT_DIRS or sys.argv[2] not in VARIANT_DIRS:
        print("Usage: python eval/auto_annotate.py <source_mode> <target_mode>")
        print(f"Modes: {', '.join(VARIANT_DIRS)}")
        sys.exit(1)

    src_mode, tgt_mode = sys.argv[1], sys.argv[2]
    src_lookup = build_lookup(VARIANT_DIRS[src_mode])
    tgt_lookup = build_lookup(VARIANT_DIRS[tgt_mode])

    src_ann = load_annotations(src_mode)
    tgt_ann = load_annotations(tgt_mode)

    copied = skipped_differs = skipped_missing = skipped_existing = 0
    for key, value in src_ann.items():
        if key in tgt_ann:
            skipped_existing += 1
            continue
        if key not in tgt_lookup:
            skipped_missing += 1
            continue
        if key not in src_lookup or src_lookup[key] != tgt_lookup[key]:
            print(f"  DIFFERS  {key}")
            skipped_differs += 1
            continue
        tgt_ann[key] = value
        copied += 1

    save_annotations(tgt_mode, tgt_ann)

    print(f"\nSource : {src_mode}  ({len(src_ann)} annotations)")
    print(f"Target : {tgt_mode}")
    print(f"Copied : {copied}")
    if skipped_differs:
        print(f"Skipped (bbox/caption differs) : {skipped_differs}")
    if skipped_missing:
        print(f"Skipped (not in target variant): {skipped_missing}")
    if skipped_existing:
        print(f"Skipped (already annotated)    : {skipped_existing}")
    print(f"\nSaved to {ann_path(tgt_mode)}")


if __name__ == "__main__":
    main()
