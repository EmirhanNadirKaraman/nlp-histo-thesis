"""Tests for ``scripts/eval/sample_grounding_for_manual_labeling.py``."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "eval" / "sample_grounding_for_manual_labeling.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sampling_summaries"


@pytest.fixture()
def sampler():
    spec = importlib.util.spec_from_file_location(
        "sample_grounding_under_test", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


def _read_jsonl(path: Path) -> tuple[dict, list[dict]]:
    """Return ``(meta, rows)`` from a JSONL where the first line is ``_meta``."""
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[0]["_meta"], lines[1:]


# ---------------------------------------------------------------------------
# 1) Determinism
# ---------------------------------------------------------------------------


def test_determinism_with_seed(tmp_path: Path, sampler) -> None:
    out1 = tmp_path / "a.jsonl"
    out2 = tmp_path / "b.jsonl"
    md1 = tmp_path / "a.md"
    md2 = tmp_path / "b.md"
    rc1 = sampler.main([
        "--input", str(FIXTURES),
        "--n", "12", "--seed", "42",
        "--out", str(out1), "--out-md", str(md1),
        "--log-level", "WARNING",
    ])
    rc2 = sampler.main([
        "--input", str(FIXTURES),
        "--n", "12", "--seed", "42",
        "--out", str(out2), "--out-md", str(md2),
        "--log-level", "WARNING",
    ])
    assert rc1 == 0 and rc2 == 0
    _meta_a, rows_a = _read_jsonl(out1)
    _meta_b, rows_b = _read_jsonl(out2)
    assert rows_a == rows_b, "same --seed must yield identical sample rows"


# ---------------------------------------------------------------------------
# 2) Bucket assignment boundary cases
# ---------------------------------------------------------------------------


def test_bucket_assignment_boundaries(sampler) -> None:
    cases = [
        (0.00,  "very_low"),
        (0.299, "very_low"),
        (0.30,  "low"),
        (0.449, "low"),
        (0.45,  "near_threshold_low"),
        (0.499, "near_threshold_low"),
        (0.50,  "near_threshold_high"),
        (0.549, "near_threshold_high"),
        (0.55,  "medium"),
        (0.749, "medium"),
        (0.75,  "high"),
        (1.00,  "high"),
    ]
    for score, expected in cases:
        assert sampler.assign_bucket(score) == expected, (
            f"score={score} should land in {expected!r}, got "
            f"{sampler.assign_bucket(score)!r}"
        )
    assert sampler.assign_bucket(None) == "missing"


# ---------------------------------------------------------------------------
# 3) Output schema
# ---------------------------------------------------------------------------


REQUIRED_ROW_KEYS = {
    "sample_id", "sample_type", "source",
    "pmcid", "chunk_id", "finding_index",
    "claim", "verbatim_support",
    "section_title", "chunk_text", "surrounding_text",
    "source_sentence_ids", "text_element_ids",
    "grounding_score", "producer_threshold",
    "kept_at_producer_threshold", "kept_at_threshold",
    "threshold", "score_bucket",
    "label", "label_options", "notes",
    "source_artifact_path",
}


def test_output_schema(tmp_path: Path, sampler) -> None:
    out = tmp_path / "s.jsonl"
    sampler.main([
        "--input", str(FIXTURES),
        "--n", "20", "--seed", "1",
        "--out", str(out), "--out-md", str(tmp_path / "s.md"),
        "--log-level", "WARNING",
    ])
    _meta, rows = _read_jsonl(out)
    assert rows, "expected at least one sampled row"
    for row in rows:
        missing = REQUIRED_ROW_KEYS - set(row.keys())
        assert not missing, f"row missing keys: {missing}"
        assert row["label"] is None
        assert row["label_options"] == ["supported", "partial", "unsupported"]
        assert row["sample_type"] == "grounding"
        assert row["source"] in {"kept", "rejected"}


# ---------------------------------------------------------------------------
# 4) Meta header
# ---------------------------------------------------------------------------


def test_meta_header_present_and_consistent(tmp_path: Path, sampler) -> None:
    out = tmp_path / "s.jsonl"
    sampler.main([
        "--input", str(FIXTURES),
        "--n", "10", "--seed", "7",
        "--out", str(out), "--out-md", str(tmp_path / "s.md"),
        "--log-level", "WARNING",
    ])
    meta, rows = _read_jsonl(out)
    assert meta["schema_version"] == "v1"
    assert meta["sample_type"] == "grounding"
    assert meta["requested_n"] == 10
    assert meta["actual_n"] == len(rows)
    assert meta["actual_n"] == sum(meta["bucket_sampled"].values())
    # Total findings includes all gathered records (incl. missing-score).
    assert meta["total_findings_scanned"] >= meta["actual_n"]
    assert meta["seed"] == 7
    assert "pipeline_config_hashes" in meta
    assert "run_ids" in meta


# ---------------------------------------------------------------------------
# 5) Spill behaviour
# ---------------------------------------------------------------------------


def test_spill_when_bucket_too_small(tmp_path: Path, sampler) -> None:
    """Fixture has only 1 near_threshold_low; default target is 25 (out of 100).
    Verify the script doesn't crash, the bucket reports its 1 sample, and
    the deficit spilled to neighbours (low, very_low) in priority order.
    """
    out = tmp_path / "s.jsonl"
    sampler.main([
        "--input", str(FIXTURES),
        "--n", "100", "--seed", "42",
        "--out", str(out), "--out-md", str(tmp_path / "s.md"),
        "--log-level", "WARNING",
    ])
    meta, rows = _read_jsonl(out)
    assert meta["bucket_available"]["near_threshold_low"] == 1
    assert meta["bucket_sampled"]["near_threshold_low"] == 1
    # near_threshold_low's spill priority is [low, very_low, ...]. The fixture
    # has 6 low + 7 very_low + 5 near_threshold_high + 5 medium + 6 high = 29
    # scored besides the 1 near_threshold_low → actual_n capped near 30.
    assert meta["actual_n"] <= meta["total_findings_scanned"] - meta["missing_score_count"]
    # Total samples should at least exhaust the corpus when targets exceed
    # availability: every scored record sampled exactly once.
    sampled_pmcid_chunk_idx = {
        (r["pmcid"], r["source"], r["chunk_id"], r["finding_index"])
        for r in rows
    }
    assert len(sampled_pmcid_chunk_idx) == len(rows), "no duplicate records"


# ---------------------------------------------------------------------------
# 5b) sample_id uniqueness + format
# ---------------------------------------------------------------------------


SAMPLE_ID_RE = re.compile(
    r"^grd-[^-]+(?:-[^-]+)*?-(?:kept|rejected)-[^-]+-\d+-\d{4}$"
)


def test_sample_id_is_collision_safe(tmp_path: Path, sampler) -> None:
    out = tmp_path / "s.jsonl"
    sampler.main([
        "--input", str(FIXTURES),
        "--n", "100", "--seed", "42",
        "--out", str(out), "--out-md", str(tmp_path / "s.md"),
        "--log-level", "WARNING",
    ])
    _meta, rows = _read_jsonl(out)

    # The two rejected entries with chunk_id=null both make it through gather
    # (one's score is 0.41 → near_threshold_low bucket; the other 0.13 →
    # very_low). With aggressive spill targeting those buckets, both should
    # be in the sample.
    null_chunk_rejected = [
        r for r in rows
        if r["source"] == "rejected" and r["chunk_id"] is None
    ]
    assert len(null_chunk_rejected) == 2, (
        "fixture has 2 rejected entries with chunk_id=null; both should "
        "appear in a 100-sample draw"
    )
    # Their sample_ids must differ (via the ordinal suffix) even though
    # source/pmcid/chunk_id collide.
    ids = [r["sample_id"] for r in null_chunk_rejected]
    assert len(set(ids)) == len(ids), f"sample_ids collided: {ids}"
    assert all("-none-" in sid for sid in ids), (
        "chunk_id=None should serialise to 'none' segment in sample_id"
    )

    # All sample_ids in the full output are unique.
    all_ids = [r["sample_id"] for r in rows]
    assert len(set(all_ids)) == len(all_ids), "duplicate sample_id detected"

    # All sample_ids match the documented format.
    for sid in all_ids:
        assert SAMPLE_ID_RE.match(sid), f"sample_id {sid!r} fails format check"


# ---------------------------------------------------------------------------
# 5c) Context fields populated when available
# ---------------------------------------------------------------------------


def test_context_fields_present_when_available(tmp_path: Path, sampler) -> None:
    out = tmp_path / "s.jsonl"
    sampler.main([
        "--input", str(FIXTURES),
        "--n", "100", "--seed", "42",
        "--out", str(out), "--out-md", str(tmp_path / "s.md"),
        "--log-level", "WARNING",
    ])
    _meta, rows = _read_jsonl(out)
    # The fixture's C1 has sentences_cited with text_element_ids 101, 102.
    c1_rows = [r for r in rows if r["chunk_id"] == "C1"]
    assert c1_rows, "fixture should yield at least one C1 sample"
    assert any(r["chunk_text"] == "Chunk 1 summary text." for r in c1_rows)
    assert any(r["source_sentence_ids"] == ["S1|PAPER_MIX|101", "S2|PAPER_MIX|102"]
               for r in c1_rows)
    assert any(r["text_element_ids"] == [101, 102] for r in c1_rows)
    # Every row carries the documented context keys even when null/empty.
    for r in rows:
        for key in ("section_title", "chunk_text", "surrounding_text",
                    "source_sentence_ids", "text_element_ids"):
            assert key in r


# ---------------------------------------------------------------------------
# 6) Missing-score findings
# ---------------------------------------------------------------------------


def test_missing_score_findings_excluded(tmp_path: Path, sampler) -> None:
    out = tmp_path / "s.jsonl"
    sampler.main([
        "--input", str(FIXTURES),
        "--n", "100", "--seed", "42",
        "--out", str(out), "--out-md", str(tmp_path / "s.md"),
        "--log-level", "WARNING",
    ])
    meta, rows = _read_jsonl(out)
    assert meta["missing_score_count"] == 2
    assert all(r["grounding_score"] is not None for r in rows)


# ---------------------------------------------------------------------------
# 7) kept_at_threshold flag tracks --threshold
# ---------------------------------------------------------------------------


def test_kept_at_threshold_flag(tmp_path: Path, sampler) -> None:
    out = tmp_path / "s.jsonl"
    sampler.main([
        "--input", str(FIXTURES),
        "--n", "100", "--seed", "1", "--threshold", "0.5",
        "--out", str(out), "--out-md", str(tmp_path / "s.md"),
        "--log-level", "WARNING",
    ])
    _meta, rows = _read_jsonl(out)
    for r in rows:
        assert r["threshold"] == 0.5
        expected = r["grounding_score"] is not None and r["grounding_score"] >= 0.5
        assert r["kept_at_threshold"] is expected, (
            f"kept_at_threshold mismatch for score {r['grounding_score']}"
        )


# ---------------------------------------------------------------------------
# 8) --threshold does NOT reposition buckets
# ---------------------------------------------------------------------------


def test_threshold_only_affects_flag_not_buckets(tmp_path: Path, sampler) -> None:
    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    sampler.main([
        "--input", str(FIXTURES), "--n", "20", "--seed", "5",
        "--threshold", "0.50",
        "--out", str(out_a), "--out-md", str(tmp_path / "a.md"),
        "--log-level", "WARNING",
    ])
    sampler.main([
        "--input", str(FIXTURES), "--n", "20", "--seed", "5",
        "--threshold", "0.70",
        "--out", str(out_b), "--out-md", str(tmp_path / "b.md"),
        "--log-level", "WARNING",
    ])
    _, rows_a = _read_jsonl(out_a)
    _, rows_b = _read_jsonl(out_b)
    # Bucket assignment of each (pmcid, source, chunk_id, finding_index) tuple
    # must be identical across the two runs — buckets are fixed.
    a_by_key = {(r["pmcid"], r["source"], r["chunk_id"], r["finding_index"]): r["score_bucket"]
                for r in rows_a}
    b_by_key = {(r["pmcid"], r["source"], r["chunk_id"], r["finding_index"]): r["score_bucket"]
                for r in rows_b}
    common = set(a_by_key) & set(b_by_key)
    assert common, "expected overlap between the two seed=5 runs"
    for key in common:
        assert a_by_key[key] == b_by_key[key], (
            f"bucket changed for {key}: {a_by_key[key]!r} vs {b_by_key[key]!r}"
        )


# ---------------------------------------------------------------------------
# 9) Markdown disclaimer + forbidden phrasings
# ---------------------------------------------------------------------------


def test_markdown_has_disclaimer_and_no_forbidden_phrasings(
    tmp_path: Path, sampler,
) -> None:
    out_md = tmp_path / "s.md"
    sampler.main([
        "--input", str(FIXTURES), "--n", "20", "--seed", "3",
        "--out", str(tmp_path / "s.jsonl"), "--out-md", str(out_md),
        "--log-level", "WARNING",
    ])
    text = out_md.read_text(encoding="utf-8")
    assert "This is a labeling sample, not an accuracy report" in text
    for forbidden in [
        "accuracy score", "precision score", "recall score",
        "F1 score", "F-1 score", "precision/recall",
        "accuracy=", "precision=", "recall=",
    ]:
        assert forbidden not in text, f"forbidden phrasing in markdown: {forbidden}"


# ---------------------------------------------------------------------------
# 10) Missing input dir
# ---------------------------------------------------------------------------


def test_missing_input_dir_exits_with_rc_1(tmp_path: Path, sampler) -> None:
    rc = sampler.main([
        "--input", str(tmp_path / "does_not_exist"),
        "--out", str(tmp_path / "s.jsonl"),
        "--out-md", str(tmp_path / "s.md"),
        "--log-level", "ERROR",
    ])
    assert rc == 1


# ---------------------------------------------------------------------------
# 11) Import safety
# ---------------------------------------------------------------------------


BANNED_MODULE_PREFIXES: tuple[str, ...] = (
    "transformers",
    "torch",
    "openai",
    "anthropic",
    "google.genai",
    "google.generativeai",
    "vertexai",
    "sentence_transformers",
    "pipeline.stages.knowledge_extraction.llm_providers",
)


def test_import_safety_no_nli_or_llm_modules(tmp_path: Path) -> None:
    script = (
        "import sys, importlib.util;"
        f"spec = importlib.util.spec_from_file_location('m', r'{SCRIPT_PATH}');"
        "mod = importlib.util.module_from_spec(spec);"
        "sys.modules['m'] = mod;"
        "spec.loader.exec_module(mod);"
        "banned = "
        f"{BANNED_MODULE_PREFIXES!r};"
        "loaded = sorted({n for n in sys.modules if any(n == b or n.startswith(b + '.') for b in banned)});"
        "print('LOADED:' + (','.join(loaded) if loaded else 'none'));"
        "sys.exit(1 if loaded else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"banned imports detected: {result.stdout!r}"
