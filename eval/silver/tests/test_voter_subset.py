"""Tests for the voter-subset / leave-one-voter-out calibration stages.

Covers: cache filtering doesn't mutate the original; drop_lN_i removes the intended
voter; voter_subset_screen uses each config's FIXED theta; voter_subset_refine crosses
pairs with the FULL theta grid; checkpoint keys include voter_subset (with the
backward-compat 6-tuple for "all"); map_theta/map_gates use BEST_VOTER_SUBSET.

No API, no engine.
"""
from __future__ import annotations

import copy

import pytest

import eval.silver.run_new_summarization_sweeps as m
from eval.silver.map_theta_sweep import REJECT_THETA_GRID, THETA_GRID


@pytest.fixture(autouse=True)
def _clear_filter_memo():
    m._FILTER_CACHE.clear()          # id()-keyed memo must not leak across tests
    yield
    m._FILTER_CACHE.clear()


def _summary(tag):
    return {"chunk_id": "c1", "findings": [{"claim": tag}], "summary_text": tag, "audit_metadata": {}}


def _cache(n_l1=3, n_l2=3):
    return {"PMC1|1": {
        "case_id": "PMC1|1", "pmcid": "PMC1", "te_id": 1,
        "chunk_map": {"c1": []}, "source_texts": {"c1": ""},
        "l1": {"c1": [_summary(f"l1v{i}") for i in range(n_l1)]},
        "l2": {"c1": [_summary(f"l2v{i}") for i in range(n_l2)]},
        "l3": {"c1": _summary("l3")},
    }}


def _cfg(**over):
    base = dict(label="c", embedder="gemini", scorer_kind="embedding",
                alignment_strategy="soft_max", tau=0.15, count_alpha=0.25,
                reuse_weight=0.15, contradiction_weight=0.20)
    base.update(over)
    return base


# ── cache filtering ──────────────────────────────────────────────────────────

def test_filtered_cache_does_not_mutate_original():
    cache = _cache()
    snap = copy.deepcopy(cache)
    out = m._filtered_voter_cache(cache, "drop_l1_1")
    assert out["PMC1|1"]["l1"]["c1"][1] is None             # dropped
    assert out["PMC1|1"]["l1"]["c1"][0]["summary_text"] == "l1v0"
    assert out["PMC1|1"]["l2"] is cache["PMC1|1"]["l2"]      # other level shared
    assert cache == snap                                    # ORIGINAL untouched


def test_drop_l1_and_l2_remove_intended_voter():
    cache = _cache()
    d1 = m._filtered_voter_cache(cache, "drop_l1_0")
    assert d1["PMC1|1"]["l1"]["c1"][0] is None
    assert all(d1["PMC1|1"]["l1"]["c1"][i] is not None for i in (1, 2))
    assert d1["PMC1|1"]["l2"]["c1"][0] is not None           # L2 untouched
    d2 = m._filtered_voter_cache(cache, "drop_l2_2")
    assert d2["PMC1|1"]["l2"]["c1"][2] is None
    assert d2["PMC1|1"]["l1"] is cache["PMC1|1"]["l1"]       # L1 untouched


def test_filtered_cache_all_is_identity():
    cache = _cache()
    assert m._filtered_voter_cache(cache, "all") is cache


def test_filtered_cache_safe_on_short_list():
    cache = _cache(n_l1=2)                                   # only 2 L1 voters
    out = m._filtered_voter_cache(cache, "drop_l1_2")        # idx 2 absent → no-op, no IndexError
    assert [d is not None for d in out["PMC1|1"]["l1"]["c1"]] == [True, True]


def test_subset_meta_and_counts():
    level, idx, model = m._subset_meta("drop_l1_1")
    assert level == "l1" and idx == 1 and "/" in model
    assert m._subset_meta("all") == ("", "", "")
    assert m._subset_voter_counts("drop_l1_0") == (2, 3)
    assert m._subset_voter_counts("drop_l2_1") == (3, 2)
    assert m._subset_voter_counts("all") == (3, 3)


# ── backward-compat checkpoint key (live-run safety) ─────────────────────────

def test_all_voter_subset_key_is_6tuple_backward_compat():
    k_all = m._canonical_key("gemini", "v", 0.9, 0.2, "keep", True, "all")
    k_drop = m._canonical_key("gemini", "v", 0.9, 0.2, "keep", True, "drop_l1_2")
    assert len(k_all) == 6                                   # unchanged for existing stages
    assert len(k_drop) == 7 and k_drop[6] == "drop_l1_2"
    assert k_drop[:6] == k_all                               # prefix-identical


def test_cell_and_row_key_agree_with_voter_subset():
    spec = m._spec_from_config(_cfg())
    cell = {"embedder": "gemini", "spec": spec, "theta": 0.7, "reject": 0.1,
            "policy": "keep", "polarity": True, "voter_subset": "drop_l2_0"}
    row = {"embedder": "gemini", "variant": spec.name, "theta": 0.7, "reject_theta": 0.1,
           "legacy_single_voter_policy": "keep", "force_escalate_on_polarity_conflict": "true",
           "voter_subset": "drop_l2_0"}
    assert m._cell_key_from_cell(cell) == m._cell_key_from_row(row)
    row_old = {k: v for k, v in row.items() if k != "voter_subset"}     # old checkpoint row
    assert len(m._cell_key_from_row(row_old)) == 6                       # → "all" → 6-tuple


# ── stage cell shapes ────────────────────────────────────────────────────────

def test_screen_cells_use_each_configs_fixed_theta(monkeypatch):
    monkeypatch.setattr(m, "VOTER_SUBSET_SCREEN_CONFIGS",
                        [_cfg(label="q", theta=0.90, reject_theta=0.20)])
    cells = list(m._iter_cells("voter_subset_screen"))
    assert len(cells) == len(m.VOTER_SUBSET_GRID)            # 7 subsets × 1 config
    assert all(c["theta"] == 0.90 and c["reject"] == 0.20 for c in cells)   # NO theta sweep
    assert {c["voter_subset"] for c in cells} == set(m.VOTER_SUBSET_GRID)


def test_refine_cells_cross_full_theta_grid(monkeypatch):
    monkeypatch.setattr(m, "VOTER_SUBSET_REFINE_CONFIGS",
                        [_cfg(label="e", voter_subset="drop_l1_2")])
    cells = list(m._iter_cells("voter_subset_refine"))
    valid = sum(1 for t in THETA_GRID for rj in REJECT_THETA_GRID if rj < t)
    assert len(cells) == valid                              # full theta×reject grid
    assert all(c["voter_subset"] == "drop_l1_2" for c in cells)
    assert {c["theta"] for c in cells} == {t for t in THETA_GRID
                                           if any(rj < t for rj in REJECT_THETA_GRID)}


def test_map_theta_and_gates_use_best_voter_subset(monkeypatch):
    monkeypatch.setattr(m, "BEST_VOTER_SUBSET", "drop_l1_1")
    for stage in ("map_theta", "map_gates"):
        cells = list(m._iter_cells(stage))
        assert cells and all(c["voter_subset"] == "drop_l1_1" for c in cells)


def test_grid_stages_stay_all(monkeypatch):
    # structure_screen / family_refine must keep voter_subset="all" (signature safety)
    monkeypatch.setattr(m, "BEST_VOTER_SUBSET", "drop_l1_1")   # must NOT leak to these
    assert all(c["voter_subset"] == "all" for c in m._iter_cells("structure_screen"))
