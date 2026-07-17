"""
eval.llm_judge.metrics — Summary computation from result rows.

Computes aggregate metrics for Q1, Q2, Q3, Q5 and writes summary.json.
"""
from __future__ import annotations

from collections import Counter
from typing import Any


def _score_buckets(
    rows: list[dict],
    score_key: str,
    label_key: str,
    positive_value: Any,
) -> dict:
    """Bin rows by score into 10 buckets; compute positive-label rate per bucket."""
    buckets: dict[str, list[bool]] = {}
    for r in rows:
        s = r.get(score_key)
        if s is None:
            continue
        bucket = f"{int(s * 10) / 10:.1f}"
        buckets.setdefault(bucket, []).append(r[label_key] == positive_value)
    return {
        k: {"n": len(v), "positive_rate": round(sum(v) / len(v), 3)}
        for k, v in sorted(buckets.items())
    }


def _field_accuracy(rows: list[dict], field_name: str) -> float | None:
    """Fraction of rows where the field was NOT in fields_changed."""
    if not rows:
        return None
    correct = sum(1 for r in rows if field_name not in r.get("fields_changed", []))
    return round(correct / len(rows), 4)


def summarize_q1(rows: list[dict]) -> dict:
    if not rows:
        return {}
    grounded = sum(1 for r in rows if r.get("is_grounded"))
    all_changed = [f for r in rows for f in r.get("fields_changed", [])]
    return {
        "total": len(rows),
        "grounded": grounded,
        "grounded_rate": round(grounded / len(rows), 4),
        "relation_type_accuracy": _field_accuracy(rows, "relation_type"),
        "direction_accuracy": _field_accuracy(rows, "direction"),
        "category_accuracy": _field_accuracy(rows, "category"),
        "entity_error_rate": round(
            sum(1 for r in rows if "subject_entity" in r.get("fields_changed", [])
                or "outcome_entity" in r.get("fields_changed", [])) / len(rows), 4
        ),
        "scope_error_rate": round(
            sum(1 for r in rows if any(
                f.startswith("scope_") for f in r.get("fields_changed", [])
            )) / len(rows), 4
        ),
        "field_error_counts": dict(Counter(all_changed)),
        "score_calibration": _score_buckets(
            rows, "pipeline_grounding_score", "is_grounded", True
        ),
    }


def summarize_q2(rows: list[dict]) -> dict:
    if not rows:
        return {}
    correct = sum(1 for r in rows if r.get("is_correct"))
    confusion: dict[str, dict[str, int]] = {}
    per_label: dict[str, int] = Counter()
    for r in rows:
        pl = r.get("pipeline_label", "?")
        cl = r.get("correct_label", "?")
        confusion.setdefault(pl, {}).setdefault(cl, 0)
        confusion[pl][cl] += 1
        per_label[cl] += 1

    # NLI score calibration
    nli_rows = []
    for r in rows:
        a2b = r.get("pipeline_nli_a_to_b")
        b2a = r.get("pipeline_nli_b_to_a")
        if a2b is not None and b2a is not None:
            nli_rows.append({**r, "_max_nli": max(a2b, b2a)})

    return {
        "total": len(rows),
        "correct": correct,
        "accuracy": round(correct / len(rows), 4),
        "confusion_matrix": confusion,
        "per_label_counts": dict(per_label),
        "nli_score_calibration": _score_buckets(
            nli_rows, "_max_nli", "is_correct", True
        ) if nli_rows else {},
        "note": (
            "Phase 1 measures precision of persisted relations only. "
            "UNRELATED/rejected candidate pairs are not persisted, "
            "so relation recall is not measured."
        ),
    }


def summarize_q3(rows: list[dict]) -> dict:
    if not rows:
        return {}
    with_gaps = sum(1 for r in rows if r.get("has_gaps"))
    total_missing = sum(len(r.get("missing_findings", [])) for r in rows)

    # By stratum
    strata: dict[str, list[dict]] = {}
    for r in rows:
        s = r.get("stratum", "unknown")
        strata.setdefault(s, []).append(r)

    gap_by_stratum = {}
    for s, srows in strata.items():
        s_gaps = sum(1 for r in srows if r.get("has_gaps"))
        gap_by_stratum[s] = {
            "n": len(srows),
            "with_gaps": s_gaps,
            "gap_rate": round(s_gaps / len(srows), 4) if srows else 0.0,
        }

    return {
        "total_paragraphs": len(rows),
        "paragraphs_with_gaps": with_gaps,
        "gap_rate_overall": round(with_gaps / len(rows), 4),
        "total_missing_findings": total_missing,
        "avg_missing_per_paragraph": round(total_missing / len(rows), 2),
        "gap_rate_by_stratum": gap_by_stratum,
    }


def summarize_q5(rows: list[dict]) -> dict:
    if not rows:
        return {}
    total_tp = sum(r.get("tp", 0) for r in rows)
    total_fp = sum(r.get("fp", 0) for r in rows)
    total_fn = sum(r.get("fn", 0) for r in rows)
    total_partial = sum(r.get("partial_match_count", 0) for r in rows)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "total_paragraphs": len(rows),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_partial_matches": total_partial,
        "micro_precision": round(precision, 4),
        "micro_recall": round(recall, 4),
        "micro_f1": round(f1, 4),
    }


def build_summary(
    q1_rows: list[dict],
    q2_rows: list[dict],
    q3_rows: list[dict],
    q5_rows: list[dict],
) -> dict:
    """Build the complete summary.json from all result rows."""
    out: dict[str, Any] = {}
    if q1_rows:
        out["q1_precision"] = summarize_q1(q1_rows)
    if q2_rows:
        out["q2_relations"] = summarize_q2(q2_rows)
    if q3_rows:
        out["q3_recall"] = summarize_q3(q3_rows)
    if q5_rows:
        out["q5_f1"] = summarize_q5(q5_rows)
    return out
