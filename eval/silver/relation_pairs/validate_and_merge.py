#!/usr/bin/env python3
"""E13 Part 2 — validate the 10 manual raw batches and merge into the final set.

Reads ``eval/data/relation_pairs/raw_batches/batch_01.jsonl … batch_10.jsonl``
(the user's pasted Opus-4.7 @ T=0 outputs), validates them, and — only when all
ten are present and every invariant holds — writes the committed dataset
``eval/data/relation_claim_pairs_300.jsonl`` plus ``…_300.meta.json``.

No API / LLM calls. Pure file validation.

  python -m eval.silver.relation_pairs.validate_and_merge            # validate + write
  python -m eval.silver.relation_pairs.validate_and_merge --check    # validate only, never write

Invariants enforced:
  * every line is valid JSON with EXACTLY the 9 required keys
  * gold_label ∈ {SUPPORTING, CONTRADICTING, UNRELATED}
  * difficulty ∈ {easy, medium, hard}
  * each present batch: 30 records, 10/10/10 per label, ids contiguous & in order
  * full set (all 10 present): 300 records, 100/100/100 per label,
    ids pair_0001…pair_0300 unique & contiguous
Warnings (non-fatal): duplicate / near-duplicate normalized claim strings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.silver.relation_pairs.api_common import (  # noqa: E402
    GENERATION_META,
    _load_json,
)
from eval.silver.relation_pairs.prepare_manual_batches import (  # noqa: E402
    N_BATCHES,
    PAIRS_PER_BATCH,
    PER_LABEL_PER_BATCH,
    TOTAL_PAIRS,
    prompt_hash,
)
from eval.silver.relation_pairs.prompt_spec import (  # noqa: E402
    DIFFICULTIES,
    GOLD_LABELS,
    RELATION_SUBTYPES,
    RELATION_SUBTYPES_BY_LABEL,
    SCHEMA_KEYS,
)

REQUIRED_KEYS = set(SCHEMA_KEYS)
LABELS = GOLD_LABELS

RAW_DIR = _REPO_ROOT / "eval" / "data" / "relation_pairs" / "raw_batches"
OUT_JSONL = _REPO_ROOT / "eval" / "data" / "relation_claim_pairs_300.jsonl"
OUT_META = _REPO_ROOT / "eval" / "data" / "relation_claim_pairs_300.meta.json"
OUT_CALIB = _REPO_ROOT / "eval" / "data" / "relation_claim_pairs_calibration.jsonl"
OUT_EVAL = _REPO_ROOT / "eval" / "data" / "relation_claim_pairs_evaluation.jsonl"
DEFAULT_CALIB_FRAC = 0.5
DEFAULT_SPLIT_SEED = 42

_PAIR_ID_RE = re.compile(r"^pair_(\d{4})$")


def normalize_claim(text: str) -> str:
    """Lowercase, strip punctuation/whitespace — for duplicate detection only."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def validate_record(rec: dict, line_no: int, expected_id: str | None = None) -> list[str]:
    """Return a list of error strings for one record (empty == valid)."""
    errs: list[str] = []
    if not isinstance(rec, dict):
        return [f"line {line_no}: not a JSON object"]
    keys = set(rec.keys())
    missing = REQUIRED_KEYS - keys
    extra = keys - REQUIRED_KEYS
    if missing:
        errs.append(f"line {line_no}: missing keys {sorted(missing)}")
    if extra:
        errs.append(f"line {line_no}: unexpected keys {sorted(extra)}")
    label = rec.get("gold_label")
    if label not in LABELS:
        errs.append(f"line {line_no}: gold_label={label!r} not in {LABELS}")
    if rec.get("difficulty") not in DIFFICULTIES:
        errs.append(f"line {line_no}: difficulty={rec.get('difficulty')!r} not in {DIFFICULTIES}")
    subtype = rec.get("relation_subtype")
    if subtype not in RELATION_SUBTYPES:
        errs.append(f"line {line_no}: relation_subtype={subtype!r} not in {RELATION_SUBTYPES}")
    elif label in RELATION_SUBTYPES_BY_LABEL and subtype not in RELATION_SUBTYPES_BY_LABEL[label]:
        errs.append(f"line {line_no}: relation_subtype={subtype!r} not compatible with "
                    f"gold_label={label!r} (allowed: {RELATION_SUBTYPES_BY_LABEL[label]})")
    pid = rec.get("id")
    if not (isinstance(pid, str) and _PAIR_ID_RE.match(pid)):
        errs.append(f"line {line_no}: id={pid!r} not of form pair_NNNN")
    elif expected_id is not None and pid != expected_id:
        errs.append(f"line {line_no}: id={pid!r} out of order (expected {expected_id})")
    for fld in ("claim_a", "claim_b", "topic", "disease_or_entity"):
        if not str(rec.get(fld, "")).strip():
            errs.append(f"line {line_no}: empty {fld}")
    return errs


def load_batch(path: Path, batch_number: int) -> tuple[list[dict], list[str]]:
    """Parse + validate one raw batch file. Returns (records, errors)."""
    errs: list[str] = []
    records: list[dict] = []
    start_n = PAIRS_PER_BATCH * (batch_number - 1) + 1
    raw_lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for i, line in enumerate(raw_lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            errs.append(f"{path.name} line {i + 1}: invalid JSON ({e})")
            continue
        expected_id = f"pair_{start_n + i:04d}" if i < PAIRS_PER_BATCH else None
        errs.extend(f"{path.name} {m}" for m in validate_record(rec, i + 1, expected_id))
        records.append(rec)

    if len(records) != PAIRS_PER_BATCH:
        errs.append(f"{path.name}: expected {PAIRS_PER_BATCH} records, found {len(records)}")
    label_counts = Counter(r.get("gold_label") for r in records)
    for lab in LABELS:
        if label_counts.get(lab, 0) != PER_LABEL_PER_BATCH:
            errs.append(
                f"{path.name}: gold_label={lab} count {label_counts.get(lab, 0)} "
                f"(expected {PER_LABEL_PER_BATCH})"
            )
    return records, errs


def validate_full(records: list[dict]) -> tuple[list[str], list[str], dict]:
    """Validate the merged 300-record set. Returns (errors, warnings, stats)."""
    errs: list[str] = []
    warnings: list[str] = []

    if len(records) != TOTAL_PAIRS:
        errs.append(f"full set: expected {TOTAL_PAIRS} records, found {len(records)}")

    ids = [r.get("id") for r in records]
    if len(set(ids)) != len(ids):
        dupes = [pid for pid, c in Counter(ids).items() if c > 1]
        errs.append(f"full set: duplicate ids {dupes[:10]}")
    expected_ids = [f"pair_{n:04d}" for n in range(1, TOTAL_PAIRS + 1)]
    if ids != expected_ids and set(ids) != set(expected_ids):
        errs.append("full set: id set is not exactly pair_0001..pair_0300")

    label_counts = Counter(r.get("gold_label") for r in records)
    per_label = TOTAL_PAIRS // len(LABELS)
    for lab in LABELS:
        if label_counts.get(lab, 0) != per_label:
            errs.append(f"full set: {lab} count {label_counts.get(lab, 0)} (expected {per_label})")

    # Near-duplicate claim warning (normalized string match across all claims).
    seen: dict[str, str] = {}
    for r in records:
        for fld in ("claim_a", "claim_b"):
            norm = normalize_claim(r.get(fld, ""))
            if norm and norm in seen:
                warnings.append(f"near-duplicate claim: {r.get('id')}.{fld} ≈ {seen[norm]}")
            elif norm:
                seen[norm] = f"{r.get('id')}.{fld}"

    stats = {
        "total": len(records),
        "label_counts": dict(label_counts),
        "difficulty_counts": dict(Counter(r.get("difficulty") for r in records)),
        "relation_subtype_counts": dict(Counter(r.get("relation_subtype") for r in records)),
        "n_unique_entities": len({str(r.get("disease_or_entity", "")).lower() for r in records}),
        "n_unique_topics": len({str(r.get("topic", "")).lower() for r in records}),
    }
    return errs, warnings, stats


def _content_hash(records: list[dict]) -> str:
    blob = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def split_records(records: list[dict], calib_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministic, label-stratified calibration/evaluation split.

    Within each gold_label, sort by id then shuffle with a fixed seed and take the
    first ``calib_frac`` for calibration, the rest for evaluation — so the two
    splits are reproducible and each is balanced across labels. The split exists so
    the threshold-sweep *sensitivity analysis* can be inspected on calibration
    while the primary report stays on evaluation; per the thesis, NO threshold is
    tuned on this set.
    """
    calib: list[dict] = []
    ev: list[dict] = []
    for lab in LABELS:
        grp = sorted([r for r in records if r.get("gold_label") == lab], key=lambda r: r.get("id", ""))
        random.Random(f"{seed}:{lab}").shuffle(grp)
        n_cal = round(len(grp) * calib_frac)
        calib.extend(grp[:n_cal])
        ev.extend(grp[n_cal:])
    calib.sort(key=lambda r: r.get("id", ""))
    ev.sort(key=lambda r: r.get("id", ""))
    return calib, ev


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate + merge the 10 relation-pair batches, write splits (E13).")
    ap.add_argument("--check", action="store_true", help="validate only; never write the merged dataset")
    ap.add_argument("--calib-frac", type=float, default=DEFAULT_CALIB_FRAC,
                    help=f"calibration fraction per label for the split (default {DEFAULT_CALIB_FRAC})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED,
                    help=f"seed for the stratified split (default {DEFAULT_SPLIT_SEED})")
    ap.add_argument("--no-split", action="store_true", help="do not write calibration/evaluation split files")
    args = ap.parse_args()

    present, missing = [], []
    for k in range(1, N_BATCHES + 1):
        p = RAW_DIR / f"batch_{k:02d}.jsonl"
        (present if p.exists() else missing).append((k, p))

    print(f"E13 validate+merge — raw batches in {RAW_DIR}")
    print(f"  present: {len(present)}/{N_BATCHES}   missing: {[k for k, _ in missing]}\n")

    all_errors: list[str] = []
    merged: list[dict] = []
    for k, p in present:
        records, errs = load_batch(p, k)
        merged.extend(records)
        status = "OK" if not errs else f"{len(errs)} error(s)"
        print(f"  batch {k:2d} ({p.name}): {len(records)} records — {status}")
        for e in errs:
            print(f"      ✗ {e}")
        all_errors.extend(errs)

    if missing:
        print(f"\nIncomplete: {len(missing)} batch file(s) missing from {RAW_DIR}. "
              "Produce them via the API path (prepare_api_batch_jsonl → submit_api_batch "
              "→ collect_api_batch) or the manual fallback (prepare_manual_batches).")
        return 1
    if all_errors:
        print(f"\nFAILED: {len(all_errors)} per-batch error(s). Fix the raw files and re-run.")
        return 1

    # Merge in canonical id order before full validation.
    merged.sort(key=lambda r: r.get("id", ""))
    full_errs, warnings, stats = validate_full(merged)
    print("\nFull-set validation:")
    print(f"  total={stats['total']}  labels={stats['label_counts']}")
    print(f"  difficulty={stats['difficulty_counts']}")
    print(f"  unique entities={stats['n_unique_entities']}  unique topics={stats['n_unique_topics']}")
    for w in warnings:
        print(f"  ! {w}")
    if full_errs:
        for e in full_errs:
            print(f"  ✗ {e}")
        print(f"\nFAILED: {len(full_errs)} full-set error(s). Not writing.")
        return 1

    if args.check:
        print("\n--check: all invariants hold. (Not writing — drop --check to materialize.)")
        return 0

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(OUT_JSONL, merged)

    # Truthful provenance: prefer the API generator's generation_meta.json when
    # present (workflow=api_batch, model, batch_ids, no fake temperature); fall
    # back to the manual workflow otherwise.
    gen = _load_json(GENERATION_META, None)
    provenance = gen if gen else {
        "workflow": "manual",
        "generator_model": "Opus 4.7 (manual paste)",
        "temperature": "n/a (manual run)",
    }

    split_meta = None
    if not args.no_split:
        calib, ev = split_records(merged, args.calib_frac, args.seed)
        _write_jsonl(OUT_CALIB, calib)
        _write_jsonl(OUT_EVAL, ev)
        split_meta = {
            "strategy": "label-stratified, seeded",
            "calib_frac": args.calib_frac,
            "seed": args.seed,
            "calibration_n": len(calib),
            "evaluation_n": len(ev),
            "calibration_labels": dict(Counter(r["gold_label"] for r in calib)),
            "evaluation_labels": dict(Counter(r["gold_label"] for r in ev)),
            "note": "diagnostic only; per the thesis no threshold is tuned on this set",
        }

    meta = {
        "dataset": "relation_claim_pairs_300",
        **provenance,
        "n_batches": N_BATCHES,
        "pairs_per_batch": PAIRS_PER_BATCH,
        "per_label_per_batch": PER_LABEL_PER_BATCH,
        "total": stats["total"],
        "labels": LABELS,
        "label_counts": stats["label_counts"],
        "difficulty_counts": stats["difficulty_counts"],
        "relation_subtype_counts": stats["relation_subtype_counts"],
        "n_unique_entities": stats["n_unique_entities"],
        "n_unique_topics": stats["n_unique_topics"],
        "prompt_template_hash": prompt_hash(),
        "content_hash": _content_hash(merged),
        "split": split_meta,
        "validated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_warnings": len(warnings),
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"\n✓ wrote {OUT_JSONL.relative_to(_REPO_ROOT)} ({len(merged)} pairs)  "
          f"[workflow={provenance.get('workflow')}]")
    print(f"✓ wrote {OUT_META.relative_to(_REPO_ROOT)}")
    if split_meta:
        print(f"✓ wrote split: {OUT_CALIB.name} ({split_meta['calibration_n']}) + "
              f"{OUT_EVAL.name} ({split_meta['evaluation_n']})  [seed={args.seed}, "
              f"calib_frac={args.calib_frac}]")
    print("Next: python -m eval.silver.relation_pairs.inspect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
