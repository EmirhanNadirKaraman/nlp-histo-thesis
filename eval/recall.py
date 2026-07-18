"""
eval/recall.py — Compute recall from docling_full annotations.

For each element labeled 'correct' in annotations_docling_full.json, checks
whether its text appears in the pipeline's text_raw output.

Usage:
    python eval/recall.py            # summary + per-doc breakdown
    python eval/recall.py --missing  # also print all missing elements
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE     = Path(__file__).parent
ANN_PATH = HERE / "annotations" / "annotations_docling_full.json"
FULL_DIR = HERE / "out" / "docling_full"
TEXT_DIR = HERE / "out" / "text_raw"

MATCH_LEN = 40   # characters used for substring match


def load_elements() -> dict[str, str]:
    """Return {ann_key: text} for all docling_full elements that have text."""
    elements: dict[str, str] = {}
    for f in FULL_DIR.glob("*_layout.json"):
        pmcid = f.stem.replace("_layout", "")
        data  = json.loads(f.read_text(encoding="utf-8"))
        for el in data["elements"]:
            text = (el.get("text") or "").strip()
            if not text:
                continue
            page = el.get("page", "?")
            bbox = el.get("bbox", {})
            x1   = bbox.get("x1", 0)
            y1   = bbox.get("y1", 0)
            key  = f"{pmcid}::p{page}_{el['type']}_{x1:.1f}_{y1:.1f}"
            elements[key] = text
    return elements


def load_pipeline_text() -> dict[str, str]:
    """Return {pmcid: full text} from text_raw output."""
    return {
        f.stem.replace("_raw", ""): f.read_text(encoding="utf-8")
        for f in TEXT_DIR.glob("*_raw.txt")
    }


def is_found(text: str, output: str) -> bool:
    snippet = text[:MATCH_LEN].strip()
    return bool(snippet) and snippet in output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--missing", action="store_true",
                        help="Print all missing elements")
    args = parser.parse_args()

    ann      = json.loads(ANN_PATH.read_text(encoding="utf-8"))
    elements = load_elements()
    pipeline = load_pipeline_text()

    correct_keys = [k for k, v in ann.items() if v == "correct"]

    # Per-doc tracking
    per_doc: dict[str, dict[str, int]] = defaultdict(lambda: {"found": 0, "total": 0})
    missing: list[tuple[str, str, str]] = []   # (pmcid, type, text)

    for key in correct_keys:
        pmcid    = key.split("::")[0]
        el_text  = elements.get(key, "")
        el_type  = key.split("_")[1] if "::" in key else "?"
        output   = pipeline.get(pmcid, "")

        per_doc[pmcid]["total"] += 1
        if is_found(el_text, output):
            per_doc[pmcid]["found"] += 1
        else:
            missing.append((pmcid, el_type, el_text))

    total = len(correct_keys)
    found = sum(d["found"] for d in per_doc.values())
    recall = found / total if total else 0.0

    # Summary
    print(f"{'─' * 52}")
    print(f"  Labeled 'correct' : {total}")
    print(f"  Found in pipeline : {found}")
    print(f"  Missing (FN)      : {total - found}")
    print(f"  Recall            : {recall:.1%}")
    print(f"{'─' * 52}")

    # Per-doc breakdown
    print(f"\n  {'PMCID':<48}  found / total   recall")
    print(f"  {'─' * 48}  {'─' * 13}   {'─' * 6}")
    for pmcid, counts in sorted(per_doc.items()):
        f_ = counts["found"]
        t_ = counts["total"]
        r_ = f_ / t_ if t_ else 0.0
        bar = "█" * int(r_ * 20) + "░" * (20 - int(r_ * 20))
        print(f"  {pmcid:<48}  {f_:>3} / {t_:<3}       [{bar}] {r_:.0%}")

    # Missing elements
    if args.missing:
        print(f"\n  Missing elements ({len(missing)}):")
        print(f"  {'─' * 52}")
        for pmcid, el_type, text in missing:
            print(f"  [{pmcid}] ({el_type})")
            print(f"    {text[:100]}")
            print()


if __name__ == "__main__":
    main()
