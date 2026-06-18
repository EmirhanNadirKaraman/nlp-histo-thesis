"""Tests for the E13 relation claim-pair workflow (no API, no model load).

Covers the load-bearing contracts of the manual, no-API pipeline:
  * the generation-prompt batch ID arithmetic (batch 10 → pair_0271..pair_0300);
  * per-record + per-batch + full-set validation (300 / 100-100-100 / 10-10-10 /
    contiguous ordered ids / label & difficulty domains), incl. rejecting a
    deliberately-bad fixture;
  * near-duplicate claim warning;
  * the rule-shim → `_build_nli_text` round-trip (bare claim vs scope-prefixed);
  * `_classify_pair` maps scripted entailment/contradiction scores to the
    expected label, and the synthetic shims (direction=None) keep the
    contradiction path open while a same-polarity guard still blocks it.

Importing the evaluator does NOT load transformers (the NLI pipe is lazy), so
these run fast and offline.
"""
from __future__ import annotations

import copy

import pytest

from eval.silver.experiments.E13_nli_ablation.evaluate import build_shim
from eval.silver.relation_pairs import prepare_manual_batches as prep
from eval.silver.relation_pairs.validate_and_merge import (
    load_batch,
    normalize_claim,
    validate_full,
    validate_record,
)
from pipeline.stages.summarization.current_stages.relate_stage import (
    _build_nli_text,
    _classify_pair,
)
from pipeline.stages.summarization.models import DirectionEnum, RelationTypeLabel


# ── prompt batch arithmetic ──────────────────────────────────────────────────
def test_batch_id_range():
    assert prep.batch_id_range(1) == ("0001", "0030")
    assert prep.batch_id_range(10) == ("0271", "0300")


def test_render_batch_contains_ids_and_no_verbatim():
    text = prep.render_batch(10)
    assert "pair_0271" in text and "pair_0300" in text
    assert "batch 10 of 10" in text
    assert "30 claim-pairs" in text
    # synthetic set must never request quoted source text: no verbatim KEY in the
    # requested JSON schema (the prompt only mentions "verbatim" to PROHIBIT it).
    assert "representative_verbatim" not in text
    schema_line = next(ln for ln in text.splitlines() if ln.strip().startswith('{"id"'))
    assert "verbatim" not in schema_line


# ── fixtures: a valid synthetic 300-record set ───────────────────────────────
def _make_records() -> list[dict]:
    recs = []
    cycle = [("SUPPORTING", "paraphrase"), ("CONTRADICTING", "opposite_direction"),
             ("UNRELATED", "different_entity")]
    for n in range(1, 301):
        label, subtype = cycle[(n - 1) % 3]
        recs.append({
            "id": f"pair_{n:04d}",
            "topic": f"topic {n % 17}",
            "disease_or_entity": f"entity {n % 23}",
            "claim_a": f"claim a number {n} about biomarker expression",
            "claim_b": f"claim b number {n} about biomarker expression",
            "gold_label": label,
            "difficulty": ["easy", "medium", "hard"][n % 3],
            "relation_subtype": subtype,
            "rationale": "because the relation is intended this way",
        })
    return recs


def _write_batch(tmp_path, batch_number: int, records: list[dict]):
    import json
    p = tmp_path / f"batch_{batch_number:02d}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return p


# ── per-record validation ────────────────────────────────────────────────────
def test_validate_record_accepts_good():
    rec = _make_records()[0]
    assert validate_record(rec, 1, expected_id="pair_0001") == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda r: r.pop("rationale"), "missing keys"),
    (lambda r: r.update(extra="x"), "unexpected keys"),
    (lambda r: r.update(gold_label="SUPPORT"), "gold_label"),     # canonical is SUPPORTING
    (lambda r: r.update(difficulty="trivial"), "difficulty"),
    (lambda r: r.update(id="p1"), "pair_NNNN"),
    (lambda r: r.update(claim_a="   "), "empty claim_a"),
])
def test_validate_record_rejects_bad(mutate, needle):
    rec = copy.deepcopy(_make_records()[0])
    mutate(rec)
    errs = validate_record(rec, 1, expected_id="pair_0001")
    assert any(needle in e for e in errs), errs


def test_validate_record_out_of_order_id():
    rec = _make_records()[0]
    errs = validate_record(rec, 1, expected_id="pair_0002")
    assert any("out of order" in e for e in errs)


# ── per-batch validation ─────────────────────────────────────────────────────
def test_load_batch_good(tmp_path):
    batch1 = _make_records()[:30]
    p = _write_batch(tmp_path, 1, batch1)
    records, errs = load_batch(p, 1)
    assert len(records) == 30
    assert errs == [], errs


def test_load_batch_bad_label_balance(tmp_path):
    batch1 = _make_records()[:30]
    # flip one CONTRADICTING → SUPPORTING: 11/9/10 breaks 10-10-10
    for r in batch1:
        if r["gold_label"] == "CONTRADICTING":
            r["gold_label"] = "SUPPORTING"
            break
    p = _write_batch(tmp_path, 1, batch1)
    _, errs = load_batch(p, 1)
    assert any("count" in e for e in errs), errs


def test_load_batch_wrong_count(tmp_path):
    p = _write_batch(tmp_path, 1, _make_records()[:29])
    _, errs = load_batch(p, 1)
    assert any("expected 30 records" in e for e in errs), errs


# ── full-set validation ──────────────────────────────────────────────────────
def test_validate_full_good():
    errs, warnings, stats = validate_full(_make_records())
    assert errs == [], errs
    assert stats["total"] == 300
    assert stats["label_counts"] == {"SUPPORTING": 100, "CONTRADICTING": 100, "UNRELATED": 100}


def test_validate_full_wrong_total():
    errs, _, _ = validate_full(_make_records()[:290])
    assert any("expected 300" in e for e in errs)


def test_validate_full_duplicate_id():
    recs = _make_records()
    recs[5]["id"] = recs[4]["id"]
    errs, _, _ = validate_full(recs)
    assert any("duplicate ids" in e for e in errs)


def test_validate_full_near_duplicate_claim_warns():
    recs = _make_records()
    recs[7]["claim_a"] = recs[3]["claim_a"].upper() + " !!"
    _, warnings, _ = validate_full(recs)
    assert any("near-duplicate" in w for w in warnings)


def test_normalize_claim():
    assert normalize_claim("Ki-67 HIGH!!") == normalize_claim("ki 67 high")


# ── rule-shim → _build_nli_text ──────────────────────────────────────────────
def test_shim_predicate_only_is_bare_claim():
    shim = build_shim("Ki-67 high predicts worse survival", "breast carcinoma")
    assert _build_nli_text(shim, scope_aware=False) == "Ki-67 high predicts worse survival"


def test_shim_scope_aware_prefixes_disease():
    shim = build_shim("Ki-67 high predicts worse survival", "breast carcinoma")
    out = _build_nli_text(shim, scope_aware=True)
    assert out == "[scope: disease=breast carcinoma] Ki-67 high predicts worse survival"


def test_shim_scope_aware_no_entity_is_bare():
    shim = build_shim("some claim", None)
    assert _build_nli_text(shim, scope_aware=True) == "some claim"


# ── _classify_pair label mapping ─────────────────────────────────────────────
def _scores(ent: float, con: float) -> dict[str, float]:
    return {"entailment": ent, "contradiction": con, "neutral": 1.0 - ent - con}


def test_classify_support():
    a, b = build_shim("x", None), build_shim("y", None)
    lab = _classify_pair(a, b, _scores(0.9, 0.0), _scores(0.9, 0.0), 0.5, 0.5)
    assert lab == RelationTypeLabel.SUPPORT


def test_classify_contradict_with_neutral_direction():
    # direction=None on both shims → same-polarity guard is open → contradiction allowed
    a, b = build_shim("x", None), build_shim("y", None)
    lab = _classify_pair(a, b, _scores(0.0, 0.9), _scores(0.0, 0.9), 0.5, 0.5)
    assert lab == RelationTypeLabel.CONTRADICT


def test_classify_unrelated():
    a, b = build_shim("x", None), build_shim("y", None)
    lab = _classify_pair(a, b, _scores(0.2, 0.2), _scores(0.2, 0.2), 0.5, 0.5)
    assert lab is None  # → UNRELATED in the evaluator


def test_same_polarity_guard_blocks_contradiction():
    a, b = build_shim("x", None), build_shim("y", None)
    a.direction = DirectionEnum.positive
    b.direction = DirectionEnum.positive
    lab = _classify_pair(a, b, _scores(0.0, 0.9), _scores(0.0, 0.9), 0.5, 0.5)
    assert lab is None  # two positive findings cannot contradict
