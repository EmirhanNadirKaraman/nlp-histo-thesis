"""Regression tests for `scripts/estimate_pipeline_cost_percentiles.py`.

Pins `est_chunks` semantics (mirrors `MapStage._make_chunks` count) and
`pick_percentile` nearest-rank indexing + the empty-corpus guard.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.paths import SCRIPTS

# scripts/ is not a package, so it must be on sys.path to import the script by
# name. The repository root is already there (pytest inserts the rootdir).
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from estimate_pipeline_cost_percentiles import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PROFILES,
    STRIDE,
    PaperMeta,
    _REPORTED_PROFILES,
    cumulative_cost_by_cut,
    load_observed_escalation_rates,
    pick_percentile,
)
from pipeline.stages.knowledge_extraction.config import MapConfig  # noqa: E402


# ── CHUNK_SIZE / CHUNK_OVERLAP sourced from MapConfig ────────────────────────

def test_reported_profiles_are_cheap_and_real_only():
    """`default` profile retired 2026-05-16. Allowlist guards against regressions
    if a future voter_configs edit re-adds it under another name."""
    assert _REPORTED_PROFILES == ("cheap", "real")
    assert set(PROFILES) <= {"cheap", "real"}
    assert "default" not in PROFILES


def test_chunk_constants_track_map_config_defaults():
    """If MapConfig defaults change, the estimator picks them up automatically."""
    defaults = MapConfig()
    assert CHUNK_SIZE == defaults.chunk_size
    assert CHUNK_OVERLAP == defaults.chunk_overlap
    assert STRIDE == defaults.chunk_size - defaults.chunk_overlap


# ── est_chunks mirrors MapStage._make_chunks counting ───────────────────────

def _make_paper(n_words: int) -> PaperMeta:
    return PaperMeta(
        pmcid="PMC_TEST", title=None,
        n_te=1, n_words=n_words, n_chars=0,
        n_figures=0, n_tables=0, n_distinct_paths=0,
    )


@pytest.mark.parametrize(
    "n_sentences, expected_chunks",
    [
        (1, 1),     # tiny paper → 1 chunk
        (8, 1),     # exactly one stride
        (9, 2),     # spills into a second chunk start
        (100, 13),  # matches _make_chunks worked example
        (200, 25),  # ceil(200/8) = 25
    ],
)
def test_est_chunks_matches_make_chunks_count(n_sentences, expected_chunks, monkeypatch):
    """est_chunks = ceil(n_sentences / stride) === len(range(0, n_sentences, stride))."""
    # Inject n_sentences directly by monkey-patching the property's input —
    # est_sentences is derived from n_words, but the test target is est_chunks.
    paper = _make_paper(n_words=0)
    monkeypatch.setattr(type(paper), "est_sentences", property(lambda self: n_sentences))
    assert paper.est_chunks == expected_chunks


# ── pick_percentile ─────────────────────────────────────────────────────────

def _papers(n: int) -> list[PaperMeta]:
    return [
        PaperMeta(pmcid=f"PMC{i:04d}", title=None, n_te=i, n_words=i,
                  n_chars=0, n_figures=0, n_tables=0, n_distinct_paths=0)
        for i in range(1, n + 1)
    ]


@pytest.mark.parametrize(
    "n, p, expected_idx",
    [
        (10, 0.50, 4),    # ceil(5) - 1 = 4 → PMC0005
        (10, 0.80, 7),    # ceil(8) - 1 = 7 → PMC0008
        (10, 0.90, 8),    # ceil(9) - 1 = 8 → PMC0009
        (10, 1.00, 9),    # last paper
        (100, 0.90, 89),  # PMC0090
    ],
)
def test_pick_percentile_nearest_rank(n, p, expected_idx):
    papers = _papers(n)
    idx, paper = pick_percentile(papers, p)
    assert idx == expected_idx
    assert paper.pmcid == f"PMC{expected_idx + 1:04d}"


def test_pick_percentile_empty_corpus_raises():
    with pytest.raises(ValueError, match="empty paper list"):
        pick_percentile([], 0.9)


@pytest.mark.parametrize("bad_p", [0.0, -0.1, 1.5])
def test_pick_percentile_out_of_range_raises(bad_p):
    with pytest.raises(ValueError, match="p must be in"):
        pick_percentile(_papers(5), bad_p)


# ── cumulative_cost_by_cut ──────────────────────────────────────────────────

class _FakeBook:
    """Stand-in PriceBook: flat $1 input + $0 output per MTok for every model.

    Lets the test pin chunk counts and call counts without coupling to the
    live price book. Returned costs collapse to `(in_tok / 1_000_000) × 1.0`,
    so a chunk at the script's `OBS_INPUT_TOK_PER_CHUNK=4733` costs
    4.733e-3 per L1 voter call. Doubles for batch via `_FakeBook.cost`.
    """

    @staticmethod
    def get(model):
        class _P:
            input_per_1m = 1.0
            output_per_1m = 0.0
        return _P()

    @staticmethod
    def cost(model, in_tok, out_tok, batch=False):
        normal = in_tok / 1_000_000 * 1.0
        return normal * (0.5 if batch else 1.0)


_FAKE_PROFILE: dict[str, list[str] | str] = {
    "l1": ["fake-l1"],
    "l2": ["fake-l2"],
    "l3": "fake-l3",
}


def test_cumulative_cost_by_cut_running_sum_is_monotonic():
    """Cumulative cost must be non-decreasing as the cut grows."""
    papers = _papers(20)
    cuts = [0.10, 0.50, 0.90, 1.00]
    series = cumulative_cost_by_cut(
        papers, _FakeBook(), _FAKE_PROFILE,
        l2_rate=0.2, l3_rate=0.1, cuts=cuts,
    )
    assert [pt.cut for pt in series] == cuts
    assert [pt.n_papers for pt in series] == [2, 10, 18, 20]
    # Strictly non-decreasing across cuts (all per-paper costs >= 0).
    normal = [pt.normal_cost for pt in series]
    batch = [pt.batch_cost for pt in series]
    assert normal == sorted(normal)
    assert batch == sorted(batch)
    # Batch is exactly half normal under the fake book.
    for pt in series:
        assert pt.batch_cost == pytest.approx(pt.normal_cost / 2)


def test_cumulative_cost_by_cut_full_corpus_matches_naive_sum():
    """At cut=1.0, cumulative cost = sum over every paper."""
    papers = _papers(7)
    series = cumulative_cost_by_cut(
        papers, _FakeBook(), _FAKE_PROFILE,
        l2_rate=0.0, l3_rate=0.0, cuts=[1.0],
    )
    [final] = series
    assert final.n_papers == 7
    assert final.total_chunks == sum(p.est_chunks for p in papers)


def test_cumulative_cost_by_cut_empty_corpus_returns_empty():
    assert cumulative_cost_by_cut([], _FakeBook(), _FAKE_PROFILE,
                                  l2_rate=0.2, l3_rate=0.1,
                                  cuts=[0.5, 1.0]) == []


@pytest.mark.parametrize("bad_cut", [0.0, -0.1, 1.5])
def test_cumulative_cost_by_cut_rejects_out_of_range(bad_cut):
    with pytest.raises(ValueError, match="cut must be in"):
        cumulative_cost_by_cut(
            _papers(5), _FakeBook(), _FAKE_PROFILE,
            l2_rate=0.2, l3_rate=0.1, cuts=[bad_cut],
        )


# ── load_observed_escalation_rates ──────────────────────────────────────────

def _write_report(dir_: Path, name: str, *, n: int, l2: int, l3: int,
                  use_totals: bool = True) -> Path:
    body = {
        "papers": [
            {"pmcid": "PMC_TEST", "total_chunks": n,
             "l2_escalated": l2, "l3_escalated": l3},
        ],
    }
    if use_totals:
        body["totals"] = {"total_chunks": n, "l2_escalated": l2, "l3_escalated": l3}
    path = dir_ / name
    path.write_text(__import__("json").dumps(body))
    return path


def test_load_observed_rates_returns_none_when_dir_missing(tmp_path):
    assert load_observed_escalation_rates(tmp_path / "nonexistent") is None


def test_load_observed_rates_returns_none_when_dir_empty(tmp_path):
    assert load_observed_escalation_rates(tmp_path) is None


def test_load_observed_rates_single_report(tmp_path):
    _write_report(tmp_path, "escalation_report_20260516T000001.json",
                  n=30, l2=6, l3=0)
    observed = load_observed_escalation_rates(tmp_path)
    assert observed is not None
    assert observed.total_chunks == 30
    assert observed.total_l2_escalated == 6
    assert observed.total_l3_escalated == 0
    assert observed.l2_rate == pytest.approx(0.2)
    assert observed.l3_rate == pytest.approx(0.0)
    assert observed.n_reports == 1


def test_load_observed_rates_chunk_weighted_across_reports(tmp_path):
    """Bigger paper dominates — verifies chunk-weighting, not per-report mean."""
    _write_report(tmp_path, "escalation_report_a.json", n=10, l2=10, l3=10)
    _write_report(tmp_path, "escalation_report_b.json", n=100, l2=10, l3=0)
    observed = load_observed_escalation_rates(tmp_path)
    assert observed is not None
    assert observed.total_chunks == 110
    # Per-report mean would give l2 ~ 55%; chunk-weighted is 20/110 ≈ 18.2%.
    assert observed.l2_rate == pytest.approx(20 / 110)
    assert observed.l3_rate == pytest.approx(10 / 110)


def test_load_observed_rates_falls_back_to_papers_when_no_totals(tmp_path):
    _write_report(tmp_path, "escalation_report_old.json",
                  n=8, l2=2, l3=1, use_totals=False)
    observed = load_observed_escalation_rates(tmp_path)
    assert observed is not None
    assert observed.total_chunks == 8
    assert observed.total_l2_escalated == 2
    assert observed.total_l3_escalated == 1


def test_load_observed_rates_skips_malformed_and_empty(tmp_path):
    (tmp_path / "escalation_report_broken.json").write_text("{ not json")
    (tmp_path / "escalation_report_empty.json").write_text(
        '{"totals": {"total_chunks": 0, "l2_escalated": 0, "l3_escalated": 0}}'
    )
    _write_report(tmp_path, "escalation_report_good.json", n=20, l2=5, l3=2)
    observed = load_observed_escalation_rates(tmp_path)
    assert observed is not None
    assert observed.n_reports == 1
    assert observed.total_chunks == 20
    assert observed.l2_rate == pytest.approx(0.25)
