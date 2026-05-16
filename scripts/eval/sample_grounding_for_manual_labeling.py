"""Phase 2 — stratified grounding-finding sample for manual labeling.

Reads existing summarisation artifacts under ``--input`` (default
``out/summaries``), gathers every persisted MAP finding (surviving +
grounding-rejected) with its NLI ``grounding_score``, buckets each finding
by score, and emits a JSONL with one row per sampled finding plus a meta
header line. Sampling is deterministic from ``--seed`` and over-represents
the near-threshold buckets, because that's where threshold calibration
matters most.

No API calls, no NLI re-inference, no full-pipeline replay.

Usage::

    python scripts/eval/sample_grounding_for_manual_labeling.py \\
        [--input out/summaries] \\
        [--threshold 0.50] \\
        [--n 100] \\
        [--seed 42] \\
        [--out eval/results/grounding_manual_sample.jsonl] \\
        [--out-md eval/results/grounding_manual_sample.md]

See ``eval/sweeps/README.md`` for the broader Layer A invariant.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Re-use ``eval/sweeps/_lib.py`` from the repo for the finding gatherer
# (surviving + rejected with parsed grounding scores). Also pull in the
# small handful of stdlib helpers (git/iso) from the sibling
# ``scripts/eval/_lib.py``.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval.sweeps import _lib as sweeps_lib  # type: ignore  # noqa: E402
import _lib as proxy_lib  # type: ignore  # noqa: E402


SAMPLE_TYPE = "grounding"
SCHEMA_VERSION = "v1"
LABEL_OPTIONS: tuple[str, ...] = ("supported", "partial", "unsupported")

# Bucket definitions: ``[lower, upper)`` for the first five, ``[lower, upper]``
# for ``high`` so 1.00 lands somewhere. Order matters: it's also the priority
# order used by ``_allocate_targets`` to break rounding ties.
BUCKETS: list[tuple[str, float, float]] = [
    ("very_low",            0.00, 0.30),
    ("low",                 0.30, 0.45),
    ("near_threshold_low",  0.45, 0.50),
    ("near_threshold_high", 0.50, 0.55),
    ("medium",              0.55, 0.75),
    ("high",                0.75, 1.00),
]
BUCKET_ORDER: list[str] = [name for name, _l, _u in BUCKETS]
BUCKET_RANGES: dict[str, tuple[float, float]] = {
    name: (lo, hi) for name, lo, hi in BUCKETS
}

# Spill priorities — when a bucket has fewer items than its target, the
# deficit walks outward in this order. The list mirrors the spec.
SPILL_NEIGHBOURS: dict[str, list[str]] = {
    "very_low":             ["low", "near_threshold_low",
                             "near_threshold_high", "medium", "high"],
    "low":                  ["very_low", "near_threshold_low",
                             "near_threshold_high", "medium", "high"],
    "near_threshold_low":   ["low", "very_low",
                             "near_threshold_high", "medium", "high"],
    "near_threshold_high":  ["medium", "high",
                             "near_threshold_low", "low", "very_low"],
    "medium":               ["near_threshold_high", "high",
                             "near_threshold_low", "low", "very_low"],
    "high":                 ["medium", "near_threshold_high",
                             "near_threshold_low", "low", "very_low"],
}

# Default per-bucket weights (sum to 100). For other ``--n`` values these
# scale proportionally and any rounding remainder is allocated to the
# near-threshold buckets first.
DEFAULT_BUCKET_WEIGHTS: dict[str, int] = {
    "very_low":            10,
    "low":                 15,
    "near_threshold_low":  25,
    "near_threshold_high": 25,
    "medium":              15,
    "high":                10,
}

DEFAULT_INPUT = Path("out/summaries")
DEFAULT_OUT = Path("eval/results/grounding_manual_sample.jsonl")
DEFAULT_OUT_MD = Path("eval/results/grounding_manual_sample.md")

logger = logging.getLogger("eval.sample_grounding")


# ---------------------------------------------------------------------------
# Bucket assignment + target allocation
# ---------------------------------------------------------------------------


def assign_bucket(score: float | None) -> str:
    """Return the bucket name for a score. ``None`` → ``"missing"``."""
    if score is None:
        return "missing"
    if score < 0.0:
        # Clamp pathological values into very_low to avoid silent dropping.
        return "very_low"
    if score >= 1.0:
        return "high"
    for name, lo, hi in BUCKETS:
        if lo <= score < hi:
            return name
    return "high"


def allocate_targets(n: int) -> dict[str, int]:
    """Return target sample count per bucket, summing to ``n``.

    Scales :data:`DEFAULT_BUCKET_WEIGHTS` proportionally; any rounding
    remainder goes to ``near_threshold_low`` first, then ``near_threshold_high``,
    so the prioritisation invariant survives non-100 ``n``.
    """
    if n <= 0:
        return {name: 0 for name in BUCKET_ORDER}
    total_weight = sum(DEFAULT_BUCKET_WEIGHTS.values())  # 100
    raw = {name: n * w / total_weight for name, w in DEFAULT_BUCKET_WEIGHTS.items()}
    floors = {name: int(v) for name, v in raw.items()}
    remainder = n - sum(floors.values())
    # Allocate remainder to near-threshold buckets first (priority order).
    priority_for_remainder = [
        "near_threshold_low", "near_threshold_high",
        "low", "medium", "very_low", "high",
    ]
    i = 0
    while remainder > 0:
        floors[priority_for_remainder[i % len(priority_for_remainder)]] += 1
        remainder -= 1
        i += 1
    return floors


# ---------------------------------------------------------------------------
# Context-field extraction
# ---------------------------------------------------------------------------


_SENTENCE_ID_RE = re.compile(r"^S\d+\|[^|]+\|(\d+)$")


def _parse_text_element_ids(sentences_cited: Any) -> list[int]:
    if not isinstance(sentences_cited, list):
        return []
    out: list[int] = []
    for entry in sentences_cited:
        if not isinstance(entry, str):
            continue
        m = _SENTENCE_ID_RE.match(entry.strip())
        if not m:
            continue
        try:
            out.append(int(m.group(1)))
        except ValueError:
            continue
    return out


def _build_chunk_context_index(
    summary_path: Path,
) -> dict[str | None, dict[str, Any]]:
    """Map ``chunk_id -> {chunk_text, source_sentence_ids, text_element_ids,
    section_title}`` for one summary. Robust to missing fields.
    """
    try:
        summary = sweeps_lib.load_summary(summary_path)
    except (OSError, json.JSONDecodeError):
        return {}
    index: dict[str | None, dict[str, Any]] = {}
    map_chunks = (summary.get("audit_trail") or {}).get("map_chunks") or []
    for chunk in map_chunks:
        chunk_id = chunk.get("chunk_id")
        audit_md = chunk.get("audit_metadata") or {}
        sentences_cited = audit_md.get("sentences_cited") or []
        if not isinstance(sentences_cited, list):
            sentences_cited = []
        index[chunk_id] = {
            "chunk_text": chunk.get("summary_text"),
            "source_sentence_ids": list(sentences_cited),
            "text_element_ids": _parse_text_element_ids(sentences_cited),
            # section_title is not currently persisted in summary JSONs.
            "section_title": None,
        }
    return index


def _build_finding_lookup(
    summary_path: Path,
) -> dict[tuple[str | None, int], dict[str, Any]]:
    """``(chunk_id, finding_index) -> finding dict`` for one summary, so we
    can retrieve ``verbatim_support`` and other per-finding fields for
    surviving entries.
    """
    try:
        summary = sweeps_lib.load_summary(summary_path)
    except (OSError, json.JSONDecodeError):
        return {}
    lookup: dict[tuple[str | None, int], dict[str, Any]] = {}
    map_chunks = (summary.get("audit_trail") or {}).get("map_chunks") or []
    for chunk in map_chunks:
        chunk_id = chunk.get("chunk_id")
        for idx, finding in enumerate(chunk.get("findings") or []):
            lookup[(chunk_id, idx)] = finding
    return lookup


def _build_rejected_lookup(
    summary_path: Path,
) -> dict[int, dict[str, Any]]:
    """Map ``finding_index`` (the position in the ``rejected[]`` list filtered
    to ``stage='grounding_map'``) → original rejection record."""
    try:
        summary = sweeps_lib.load_summary(summary_path)
    except (OSError, json.JSONDecodeError):
        return {}
    rejection_summary = summary.get("rejection_summary") or {}
    rejected_entries = rejection_summary.get("rejected") or []
    out: dict[int, dict[str, Any]] = {}
    for idx, entry in enumerate(rejected_entries):
        stage = (entry.get("stage") or "").lower()
        if stage != sweeps_lib.GROUNDING_REJECTION_STAGE:
            continue
        out[idx] = entry
    return out


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _record_sort_key(record: sweeps_lib.FindingRecord) -> tuple:
    """Stable sort key for ordinal assignment."""
    return (
        record.pmcid,
        "kept" if record.kept_at_producer_threshold else "rejected",
        record.chunk_id if record.chunk_id is not None else "",
        record.finding_index,
    )


def _assign_ordinals(
    records: list[sweeps_lib.FindingRecord],
) -> dict[int, int]:
    """Return ``{id(record): ordinal}`` after sorting records by
    ``_record_sort_key``. Ordinals are 0-indexed and unique across the
    whole input."""
    ordered = sorted(records, key=_record_sort_key)
    return {id(r): idx for idx, r in enumerate(ordered)}


def _sample_id_for(
    record: sweeps_lib.FindingRecord, ordinal: int,
) -> str:
    source = "kept" if record.kept_at_producer_threshold else "rejected"
    chunk = record.chunk_id if record.chunk_id else "none"
    return f"grd-{record.pmcid}-{source}-{chunk}-{record.finding_index}-{ordinal:04d}"


def stratified_sample(
    records: list[sweeps_lib.FindingRecord],
    *,
    n: int,
    seed: int,
) -> tuple[list[sweeps_lib.FindingRecord], dict[str, int],
           dict[str, int], dict[str, int]]:
    """Stratified draw with spill-on-deficit.

    Returns ``(sampled_records, bucket_available, bucket_targets,
    bucket_sampled)``. Ordering within the returned list mirrors
    :data:`BUCKET_ORDER` for stability.
    """
    rng = random.Random(seed)
    by_bucket: dict[str, list[sweeps_lib.FindingRecord]] = defaultdict(list)
    for r in records:
        by_bucket[assign_bucket(r.grounding_score)].append(r)

    # Stable per-bucket order before random draw — sort by the ordinal sort
    # key so two runs with the same seed produce identical picks.
    for name in by_bucket:
        by_bucket[name].sort(key=_record_sort_key)

    bucket_available = {name: len(by_bucket.get(name, [])) for name in BUCKET_ORDER}
    bucket_available["missing"] = len(by_bucket.get("missing", []))
    targets = allocate_targets(n)
    selected: dict[str, list[sweeps_lib.FindingRecord]] = {n_: [] for n_ in BUCKET_ORDER}
    seen_ids: set[int] = set()

    def _take(bucket: str, k: int) -> int:
        """Pull up to ``k`` unseen records from ``bucket``, return count taken."""
        pool = [r for r in by_bucket.get(bucket, []) if id(r) not in seen_ids]
        if not pool or k <= 0:
            return 0
        k_actual = min(k, len(pool))
        picks = rng.sample(pool, k_actual)
        selected[bucket].extend(picks)
        for r in picks:
            seen_ids.add(id(r))
        return k_actual

    # Pass 1: draw the natural target from each bucket.
    deficits: dict[str, int] = {}
    for name in BUCKET_ORDER:
        got = _take(name, targets[name])
        deficits[name] = targets[name] - got

    # Pass 2: spill deficits outward.
    for name in BUCKET_ORDER:
        deficit = deficits[name]
        if deficit <= 0:
            continue
        for neighbour in SPILL_NEIGHBOURS[name]:
            if deficit <= 0:
                break
            got = _take(neighbour, deficit)
            deficit -= got
        deficits[name] = deficit  # leftover (truly no neighbour has supply)

    sampled: list[sweeps_lib.FindingRecord] = []
    for name in BUCKET_ORDER:
        sampled.extend(selected[name])

    bucket_sampled = {name: len(selected[name]) for name in BUCKET_ORDER}
    return sampled, bucket_available, targets, bucket_sampled


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------


def _build_per_summary_indexes(summaries_dir: Path):
    """Return ``{pmcid_or_stem: (path, chunk_index, finding_lookup,
    rejected_lookup)}``."""
    out: dict[str, tuple] = {}
    for path in sweeps_lib.iter_summary_paths(summaries_dir):
        summary = sweeps_lib.load_summary(path)
        pmcid = summary.get("pmcid") or path.stem
        out[pmcid] = (
            path,
            _build_chunk_context_index(path),
            _build_finding_lookup(path),
            _build_rejected_lookup(path),
        )
    return out


def _row_for(
    record: sweeps_lib.FindingRecord,
    *,
    ordinal: int,
    threshold: float,
    summary_index: dict[str, tuple],
) -> dict[str, Any]:
    pmcid = record.pmcid
    entry = summary_index.get(pmcid)
    source_artifact_path: str | None
    chunk_ctx: dict[str, Any]
    if entry is None:
        source_artifact_path = None
        chunk_ctx = {}
        finding_dict: dict[str, Any] | None = None
        rejected_dict: dict[str, Any] | None = None
    else:
        path, chunk_index, finding_lookup, rejected_lookup = entry
        source_artifact_path = str(path)
        chunk_ctx = chunk_index.get(record.chunk_id) or {}
        if record.kept_at_producer_threshold:
            finding_dict = finding_lookup.get((record.chunk_id, record.finding_index))
            rejected_dict = None
        else:
            finding_dict = None
            rejected_dict = rejected_lookup.get(record.finding_index)

    claim = None
    verbatim_support = None
    if finding_dict is not None:
        claim = finding_dict.get("claim")
        verbatim_support = finding_dict.get("verbatim_support")
    elif rejected_dict is not None:
        claim = rejected_dict.get("claim")
        verbatim_support = rejected_dict.get("verbatim_support")

    kept_at_threshold = (
        record.grounding_score is not None
        and record.grounding_score >= threshold
    )

    return {
        "sample_id": _sample_id_for(record, ordinal),
        "sample_type": SAMPLE_TYPE,
        "source": "kept" if record.kept_at_producer_threshold else "rejected",
        "pmcid": pmcid,
        "chunk_id": record.chunk_id,
        "finding_index": record.finding_index,
        "claim": claim,
        "verbatim_support": verbatim_support,
        "section_title": chunk_ctx.get("section_title"),
        "chunk_text": chunk_ctx.get("chunk_text"),
        "surrounding_text": None,
        "source_sentence_ids": list(chunk_ctx.get("source_sentence_ids") or []),
        "text_element_ids": list(chunk_ctx.get("text_element_ids") or []),
        "grounding_score": record.grounding_score,
        "producer_threshold": record.producer_threshold,
        "kept_at_producer_threshold": record.kept_at_producer_threshold,
        "kept_at_threshold": kept_at_threshold,
        "threshold": threshold,
        "score_bucket": assign_bucket(record.grounding_score),
        "label": None,
        "label_options": list(LABEL_OPTIONS),
        "notes": "",
        "source_artifact_path": source_artifact_path,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


DISCLAIMER = (
    "This is a labeling sample, not an accuracy report. It contains "
    "placeholder rows ready for hand-labeling. Until the `label` field is "
    "populated, none of these numbers measure accuracy, precision, or recall."
)


def _write_jsonl(
    out_path: Path,
    *,
    meta: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_markdown(
    out_path: Path,
    *,
    meta: dict[str, Any],
    bucket_available: dict[str, int],
    bucket_targets: dict[str, int],
    bucket_sampled: dict[str, int],
    rows: list[dict[str, Any]],
    threshold: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = sum(1 for r in rows if r["kept_at_threshold"])
    rejected = sum(
        1 for r in rows
        if r["grounding_score"] is not None and not r["kept_at_threshold"]
    )

    lines: list[str] = []
    lines.append("# Grounding manual-label sample")
    lines.append("")
    lines.append(f"> **{DISCLAIMER}**")
    lines.append("")

    lines.append("## Sweep metadata")
    lines.append("")
    for key in [
        "schema_version", "sample_type", "tool", "created_at", "git_commit",
        "input_dir", "seed", "threshold", "requested_n", "actual_n",
        "total_findings_scanned", "missing_score_count",
    ]:
        lines.append(f"- {key}: `{meta.get(key)}`")
    lines.append(
        "- pipeline_config_hashes: "
        + (", ".join(f"`{h}`" for h in meta.get("pipeline_config_hashes") or []) or "_none_")
    )
    lines.append(
        "- run_ids: "
        + (", ".join(f"`{r}`" for r in meta.get("run_ids") or []) or "_none_")
    )
    lines.append("")

    lines.append("## Bucket allocation")
    lines.append("")
    lines.append("| bucket | range | available | target | sampled |")
    lines.append("|---|---|---|---|---|")
    for name in BUCKET_ORDER:
        lo, hi = BUCKET_RANGES[name]
        rng = f"[{lo:.2f}, {hi:.2f}{']' if name == 'high' else ')'}"
        lines.append(
            f"| {name} | {rng} | {bucket_available.get(name, 0)} "
            f"| {bucket_targets.get(name, 0)} | {bucket_sampled.get(name, 0)} |"
        )
    lines.append(
        f"| missing | (no grounding_score) | {bucket_available.get('missing', 0)} | 0 | 0 |"
    )
    lines.append("")

    lines.append(f"## Kept vs rejected (at --threshold {threshold:g})")
    lines.append("")
    lines.append(f"- kept (`grounding_score >= {threshold:g}`): {kept}")
    lines.append(f"- rejected (`grounding_score <  {threshold:g}`): {rejected}")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        f"- Sampled rows live in `{meta.get('out_path')}` (first line is the "
        "meta header; subsequent lines are the samples)."
    )
    lines.append(
        "- To hand-label, set the `label` field on each row to one of "
        "`supported` / `partial` / `unsupported`. The schema is stable across "
        "re-runs with the same seed."
    )
    lines.append(
        "- Re-running the sampler with the same seed and same input "
        "produces an identical JSONL — safe to regenerate."
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    summaries_dir = args.input / "summaries"
    if not summaries_dir.exists():
        logger.error("Summaries directory not found: %s", summaries_dir)
        return 1

    records, gather_warnings = sweeps_lib.gather_findings_with_grounding(summaries_dir)
    if not records:
        logger.error("No findings gathered from %s — nothing to sample.", summaries_dir)
        return 1

    # Stable ordinals for collision-safe sample_id.
    ordinals = _assign_ordinals(records)

    # Stratified sample.
    sampled, bucket_available, bucket_targets, bucket_sampled = stratified_sample(
        records, n=args.n, seed=args.seed,
    )

    summary_index = _build_per_summary_indexes(summaries_dir)
    rows = [
        _row_for(
            r, ordinal=ordinals[id(r)],
            threshold=args.threshold, summary_index=summary_index,
        )
        for r in sampled
    ]

    # Meta header.
    run_ids = sorted({
        sweeps_lib.load_summary(p).get("run_id")
        for p in sweeps_lib.iter_summary_paths(summaries_dir)
        if sweeps_lib.load_summary(p).get("run_id")
    })
    hashes = sorted({
        sweeps_lib.load_summary(p).get("pipeline_config_hash")
        for p in sweeps_lib.iter_summary_paths(summaries_dir)
        if sweeps_lib.load_summary(p).get("pipeline_config_hash")
    })
    missing_score_count = sum(1 for r in records if r.grounding_score is None)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "sample_type": SAMPLE_TYPE,
        "tool": "scripts/eval/sample_grounding_for_manual_labeling.py",
        "created_at": proxy_lib.now_iso_utc(),
        "git_commit": proxy_lib.get_git_commit(),
        "input_dir": str(args.input),
        "out_path": str(args.out),
        "out_md_path": str(args.out_md),
        "seed": args.seed,
        "threshold": args.threshold,
        "requested_n": args.n,
        "actual_n": len(rows),
        "total_findings_scanned": len(records),
        "missing_score_count": missing_score_count,
        "bucket_targets": bucket_targets,
        "bucket_available": bucket_available,
        "bucket_sampled": bucket_sampled,
        "pipeline_config_hashes": hashes,
        "run_ids": run_ids,
        "gather_warnings": gather_warnings,
    }

    _write_jsonl(args.out, meta=meta, rows=rows)
    _write_markdown(
        args.out_md, meta=meta,
        bucket_available=bucket_available,
        bucket_targets=bucket_targets,
        bucket_sampled=bucket_sampled,
        rows=rows, threshold=args.threshold,
    )

    logger.info(
        "Sampled %d/%d findings (requested %d, missing-score %d) across %d "
        "papers, %d pipeline_config_hashes — wrote %s and %s",
        len(rows), len(records), args.n, missing_score_count,
        len(meta["pipeline_config_hashes"]) and len(set(
            r.pmcid for r in records
        )),
        len(meta["pipeline_config_hashes"]),
        args.out, args.out_md,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
