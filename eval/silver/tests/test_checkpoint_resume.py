"""Tests for per-cell checkpoint/resume in the screen→refine harness.

The load-bearing property is the *round-trip*: a cell key built from a cell dict
and from a reloaded CSV row must be identical (so resume skips exactly the done
cells and never re-appends a duplicate). Also covered: the signature reacts to
resolved weight VALUES (not just spec names), and a mismatch rotates the stale
checkpoint aside. No API, no cache — `run_sweep` / `_load_map_context` are stubbed.
"""
from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest

import eval.silver.analysis.run_new_summarization_sweeps as m
from eval.silver.analysis.map_theta_sweep import ScorerSpec
from nlp_histo.pipeline.stages.knowledge_extraction.config import AgreementConfig


def _args(**over):
    base = dict(split="all", seed=42, dev_fraction=0.8, sim_threshold=0.55,
                embed_cache=None, fresh=False)
    base.update(over)
    return SimpleNamespace(**base)


def _row_for_cell(c: dict) -> dict:
    """A minimal stamped row for cell ``c`` (what _run_stage would persist)."""
    return {
        "embedder": c["embedder"], "scorer_kind": c["scorer_kind"], "variant": c["spec"].name,
        "alignment_strategy": c["alignment"],
        "theta": round(c["theta"], 2), "reject_theta": round(c["reject"], 2),
        "legacy_single_voter_policy": c["policy"],
        "force_escalate_on_polarity_conflict": str(bool(c["polarity"])).lower(),
        # production stamps voter_subset into every persisted row (map_theta pins
        # BEST_VOTER_SUBSET, not "all") — omitting it makes the row key drift 6-tuple
        # vs the cell's 7-tuple and breaks resume matching.
        "voter_subset": c.get("voter_subset", "all"),
        "strict_f1_optimal": 0.5, "f1_optimal": 0.5, "escalate_rate": 0.0,
    }


# ── cell-key round-trip (the drift guard) ────────────────────────────────────

def test_cell_and_row_keys_agree_incl_csv_strings():
    c = next(iter(m._iter_cells("map_theta")))
    typed = {"embedder": c["embedder"], "variant": c["spec"].name,
             "theta": round(c["theta"], 2), "reject_theta": round(c["reject"], 2),
             "legacy_single_voter_policy": c["policy"],
             "force_escalate_on_polarity_conflict": str(bool(c["polarity"])).lower(),
             # map_theta pins BEST_VOTER_SUBSET (not "all"); production stamps it into the row
             "voter_subset": c.get("voter_subset", "all")}
    # CSV stores everything as strings (theta as "0.30" etc.)
    csv_row = {k: (f"{v:.2f}" if k in ("theta", "reject_theta") else str(v))
               for k, v in typed.items()}
    assert m._cell_key_from_cell(c) == m._cell_key_from_row(typed)
    assert m._cell_key_from_cell(c) == m._cell_key_from_row(csv_row)


# ── signature reacts to weight VALUES, not just spec names ────────────────────

def test_plan_signature_tracks_weights_not_just_names():
    args = _args()

    def cells_with_tau(tau):
        spec = ScorerSpec("embedding", "embedding",
                          AgreementConfig(scorer_kind="embedding", tau=tau))
        return [{"embedder": "gemini", "spec": spec, "scorer_kind": "embedding",
                 "alignment": "soft_max", "theta": 0.8, "reject": 0.2,
                 "policy": "keep", "polarity": True}]

    s1 = m._plan_signature("x", cells_with_tau(0.15), args)
    s1b = m._plan_signature("x", cells_with_tau(0.15), args)
    s2 = m._plan_signature("x", cells_with_tau(0.30), args)   # same spec NAME, different tau
    assert s1 == s1b
    assert s1 != s2


def test_plan_signature_tracks_run_params():
    cells = list(m._iter_cells("map_theta"))
    assert m._plan_signature("map_theta", cells, _args()) != \
        m._plan_signature("map_theta", cells, _args(seed=7))


# ── resume integration (stubbed engine) ──────────────────────────────────────

@pytest.fixture
def patched(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "REPORTS_DIR", tmp_path)
    calls: list[tuple] = []

    def fake_run_sweep(*, scorer_specs, thetas, reject_thetas, single_voter_policies,
                       force_escalate_on_polarity_conflict_grid, **kw):
        spec = scorer_specs[0]
        calls.append((spec.name, round(thetas[0], 2), round(reject_thetas[0], 2),
                      single_voter_policies[0],
                      str(bool(force_escalate_on_polarity_conflict_grid[0])).lower()))
        h = spec.weights.hybrid
        return [{
            "scorer": spec.name, "tau": spec.weights.tau, "count_alpha": spec.weights.count_alpha,
            "reuse_weight": spec.weights.reuse_weight,
            "contradiction_weight": spec.weights.contradiction_weight,
            "w_category": h.w_category, "w_embedding": h.w_embedding,
            "w_entity": h.w_entity, "w_evidence": h.w_evidence,
            "theta": round(thetas[0], 2), "reject_theta": round(reject_thetas[0], 2),
            "legacy_single_voter_policy": single_voter_policies[0],
            "force_escalate_on_polarity_conflict":
                str(bool(force_escalate_on_polarity_conflict_grid[0])).lower(),
            "strict_f1_optimal": 0.5, "f1_optimal": 0.5, "escalate_rate": 0.0,
        }]

    monkeypatch.setattr(m, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(m, "_load_map_context",
                        lambda emb, **k: SimpleNamespace(
                            voter_cache={}, silver_by_case={}, embedder=None,
                            embed_cache=None, agreement_embed_fn=None))
    return SimpleNamespace(calls=calls, tmp=tmp_path)


def test_fresh_run_computes_all_cells_no_dupes(patched):
    rows = m._run_stage("map_gates", _args())          # 4 cells (policy×polarity)
    assert len(patched.calls) == 4
    keys = {m._cell_key_from_row(r) for r in rows}
    assert len(rows) == 4 and len(keys) == 4
    assert len(m._read_checkpoint_rows(m._ckpt_csv("map_gates"))) == 4


def test_resume_skips_done_and_no_duplicates(patched):
    stage, args = "map_gates", _args()
    cells = list(m._iter_cells(stage))
    sig = m._plan_signature(stage, cells, args)
    # pre-seed: matching meta + 2 of the 4 cells already done
    m._ckpt_meta(stage).parent.mkdir(parents=True, exist_ok=True)
    m._ckpt_meta(stage).write_text(json.dumps({"signature": sig, "stage": stage}))
    with m._ckpt_csv(stage).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=m._CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for c in cells[:2]:
            w.writerow(_row_for_cell(c))

    rows = m._run_stage(stage, args)
    assert len(patched.calls) == 2                       # only the 2 missing cells ran
    keys = {m._cell_key_from_row(r) for r in rows}
    assert keys == {m._cell_key_from_cell(c) for c in cells}   # full coverage
    assert len(rows) == 4 and len(keys) == 4                    # no duplicates


def test_full_resume_runs_nothing(patched):
    m._run_stage("map_gates", _args())
    patched.calls.clear()
    rows = m._run_stage("map_gates", _args())
    assert patched.calls == []                           # all done → no recompute
    assert len({m._cell_key_from_row(r) for r in rows}) == 4


def test_signature_mismatch_rotates_and_starts_fresh(patched):
    stage = "map_gates"
    m._ckpt_meta(stage).parent.mkdir(parents=True, exist_ok=True)
    m._ckpt_meta(stage).write_text(json.dumps({"signature": "STALE", "stage": stage}))
    m._ckpt_csv(stage).write_text("embedder,variant\n")
    done, resume = m._checkpoint_state(stage, "CURRENT", fresh=False)
    assert done == [] and resume is False
    assert list(patched.tmp.glob("**/*.stale-*"))        # stale files moved aside (now under <stage>/)


def test_deviation_tolerates_csv_empty_strings():
    # reloaded one-to-one row: inert weights are "" (CSV None), tau/blend are strings
    row = {"embedder": "gemini", "scorer_kind": "embedding", "alignment_strategy": "hungarian",
           "tau": "0.15", "count_alpha": "", "reuse_weight": "", "contradiction_weight": "",
           "w_category": "0.25", "w_embedding": "0.40", "w_entity": "0.25", "w_evidence": "0.10"}
    assert m._deviation(row) == 1                         # only hungarian deviates
