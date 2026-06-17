"""Tests for the leave-one-voter-out diagnostic (eval/silver/experiments/E12_voter_loo/voter_loo.py).

The load-bearing guards: dropping a voter nulls EXACTLY that raw index and mutates
nothing else (so baseline + 6 variants off the same cache stay independent), and
the label self-guard fails loud when the cache's slot count diverges from the
current voter set. No API, no engine.
"""
from __future__ import annotations

import copy

import pytest

from eval.silver.experiments.E12_voter_loo.voter_loo import (
    _assert_cache_matches_voters,
    _drop_voter,
    _max_slots,
    _voter_labels,
)


def _summary(tag):
    return {"chunk_id": "c1", "findings": [{"claim": tag}], "summary_text": tag, "audit_metadata": {}}


def _cache(n_l1=3, n_l2=3):
    return {
        "PMC1|1": {
            "case_id": "PMC1|1", "pmcid": "PMC1", "te_id": 1,
            "chunk_map": {"c1": []}, "source_texts": {"c1": ""},
            "l1": {"c1": [_summary(f"l1v{i}") for i in range(n_l1)]},
            "l2": {"c1": [_summary(f"l2v{i}") for i in range(n_l2)]},
            "l3": {"c1": _summary("l3")},
        }
    }


# ── _drop_voter: nulls exactly index i, shares the rest, never mutates source ──

def test_drop_voter_nulls_only_target_index():
    cache = _cache()
    dropped = _drop_voter(cache, "l1", 1)
    l1 = dropped["PMC1|1"]["l1"]["c1"]
    assert l1[1] is None                                   # target nulled
    assert l1[0]["summary_text"] == "l1v0"                 # siblings intact
    assert l1[2]["summary_text"] == "l1v2"
    # the OTHER level and l3 are the same objects (shared, not copied)
    assert dropped["PMC1|1"]["l2"] is cache["PMC1|1"]["l2"]
    assert dropped["PMC1|1"]["l3"] is cache["PMC1|1"]["l3"]


def test_drop_voter_does_not_mutate_source():
    cache = _cache()
    snapshot = copy.deepcopy(cache)
    _ = _drop_voter(cache, "l1", 0)
    _ = _drop_voter(cache, "l2", 2)
    assert cache == snapshot                               # source untouched by either drop


def test_drop_voter_l2_independent_of_l1():
    cache = _cache()
    d = _drop_voter(cache, "l2", 0)
    assert d["PMC1|1"]["l2"]["c1"][0] is None
    assert d["PMC1|1"]["l1"] is cache["PMC1|1"]["l1"]      # L1 untouched when dropping L2


# ── label self-guard ─────────────────────────────────────────────────────────

def test_voter_labels_shape_matches_make_voters():
    l1, l2 = _voter_labels()
    assert len(l1) == 3 and len(l2) == 3
    assert all(isinstance(p, str) and isinstance(m, str) for p, m in l1 + l2)


def test_max_slots_counts_widest_list():
    assert _max_slots(_cache(n_l1=3), "l1") == 3
    assert _max_slots(_cache(n_l2=2), "l2") == 2


def test_assert_cache_matches_voters_passes_on_match():
    _assert_cache_matches_voters(_cache(3, 3), 3, 3)        # no raise


def test_assert_cache_matches_voters_fails_on_mismatch():
    # cache primed with 4 L1 voters but current _make_voters has 3 → must fail loud
    with pytest.raises(SystemExit):
        _assert_cache_matches_voters(_cache(n_l1=4), 3, 3)
    with pytest.raises(SystemExit):
        _assert_cache_matches_voters(_cache(n_l2=2), 3, 3)
