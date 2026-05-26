"""Tests for ``scripts/eval/run_summarization_experiments.py`` — orchestration
logic only.

No real sweeps run; no caches touched; no API calls. Each test exercises
either pure resolution logic (phase → experiments, topological ordering,
``_per_scorer_best``) or hard-fail paths that short-circuit *before* any
``run_sweep`` invocation. The orchestrator's expensive paths (calling into
``eval.silver.map_theta_sweep.run_sweep``) are reached only via ``--run``
on executable EXPs, and none of these tests do that.

Loaded with ``importlib.util`` because ``scripts/`` is not a package
(matches the pattern used by `scripts/run_paper.py` and run_all_sweeps.py).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ORCH_PATH = _REPO_ROOT / "scripts" / "eval" / "run_summarization_experiments.py"


@pytest.fixture(scope="module")
def orch():
    """Load the orchestrator module from its file path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "run_summarization_experiments", _ORCH_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ids(experiments) -> list[str]:
    return [e.exp_id for e in experiments]


# ── 1-3, 5, 6. Phase → experiment resolution ────────────────────────────────


def test_phase_gemini_branch_selects_only_exp_1_2_3(orch):
    exps = orch._resolve_experiments(
        phase="gemini_branch", exp=None, only=None, include_deps=False,
    )
    assert _ids(exps) == ["EXP_1", "EXP_2", "EXP_3"]


def test_phase_openai_branch_selects_only_exp_4_5_6(orch):
    exps = orch._resolve_experiments(
        phase="openai_branch", exp=None, only=None, include_deps=False,
    )
    assert _ids(exps) == ["EXP_4", "EXP_5", "EXP_6"]


def test_phase_branches_includes_exp_1_through_6_only(orch):
    """Phase 1 (`branches`) runs both branches but NO shared EXPs leak in."""
    exps = orch._resolve_experiments(
        phase="branches", exp=None, only=None, include_deps=False,
    )
    ids = _ids(exps)
    assert set(ids) == {"EXP_1", "EXP_2", "EXP_3", "EXP_4", "EXP_5", "EXP_6"}
    for shared in ("EXP_7", "EXP_F", "EXP_A", "EXP_B.1", "EXP_B.2",
                   "EXP_C", "EXP_D", "EXP_E"):
        assert shared not in ids, f"{shared} leaked into --phase branches"


def test_phase_compare_does_not_auto_include_branches_without_include_deps(orch):
    """`--phase compare` resolves EXACTLY EXP_7 — deps NOT auto-pulled."""
    exps = orch._resolve_experiments(
        phase="compare", exp=None, only=None, include_deps=False,
    )
    assert _ids(exps) == ["EXP_7"]


def test_phase_compare_with_include_deps_pulls_predecessors(orch):
    """Sanity for the `--include-deps` opt-in: deps appear, topo-ordered."""
    exps = orch._resolve_experiments(
        phase="compare", exp=None, only=None, include_deps=True,
    )
    ids = _ids(exps)
    # EXP_7 depends on EXP_3 + EXP_6, which transitively need EXP_1, EXP_2, EXP_4, EXP_5
    assert {"EXP_1", "EXP_2", "EXP_3", "EXP_4", "EXP_5", "EXP_6", "EXP_7"} <= set(ids)
    # And EXP_7 comes last (topological order)
    assert ids[-1] == "EXP_7"


def test_phase_confirm_resolves_only_exp_f(orch):
    exps = orch._resolve_experiments(
        phase="confirm", exp=None, only=None, include_deps=False,
    )
    assert _ids(exps) == ["EXP_F"]


def test_phase_all_orders_branches_before_compare_before_confirm(orch):
    """Topological order: branches → compare → confirm → validation."""
    exps = orch._resolve_experiments(
        phase="all", exp=None, only=None, include_deps=False,
    )
    ids = _ids(exps)
    # Each branch's EXPs come before the comparison.
    for branch_exp in ("EXP_1", "EXP_2", "EXP_3", "EXP_4", "EXP_5", "EXP_6"):
        assert ids.index(branch_exp) < ids.index("EXP_7"), (
            f"{branch_exp} should come before EXP_7"
        )
    # EXP_7 (compare) before EXP_F (confirm)
    assert ids.index("EXP_7") < ids.index("EXP_F")
    # And all validation EXPs come after their dependency EXP_7
    for val_exp in ("EXP_A", "EXP_B.1", "EXP_B.2", "EXP_C", "EXP_E"):
        assert ids.index("EXP_7") < ids.index(val_exp), (
            f"{val_exp} should come after EXP_7 (its dependency)"
        )


# ── 4, 5. Hard-fail when required state is missing ──────────────────────────


def test_missing_compare_state_hard_fails(orch, tmp_path):
    """EXP 7 raises SystemExit when both FINAL_*_MAP_CONFIG keys are missing."""
    ctx = orch.ExperimentContext(output_dir=tmp_path, state={})
    with pytest.raises(SystemExit, match="FINAL_GEMINI_MAP_CONFIG"):
        orch._run_exp_7(ctx)


def test_missing_compare_state_partial_also_hard_fails(orch, tmp_path):
    """Only one branch present → still hard-fails, names the missing one."""
    seeded = {
        "FINAL_GEMINI_MAP_CONFIG": {
            "embedder": "gemini", "scorer": "embedding_default",
            "theta": 0.8, "reject_theta": 0.2, "polarity_flag": True,
            "strict_f1": 0.85, "hybrid_weights": None, "agreement_weights": None,
        },
    }
    ctx = orch.ExperimentContext(output_dir=tmp_path, state=seeded)
    with pytest.raises(SystemExit, match="FINAL_OPENAI_MAP_CONFIG"):
        orch._run_exp_7(ctx)


def test_missing_confirm_state_hard_fails(orch, tmp_path):
    """EXP F raises SystemExit when PROVISIONAL_FINAL_MAP_CONFIG is missing."""
    ctx = orch.ExperimentContext(output_dir=tmp_path, state={})
    with pytest.raises(SystemExit, match="PROVISIONAL_FINAL_MAP_CONFIG"):
        orch._run_exp_f(ctx)


# ── 7. EXP 7 writes PROVISIONAL_FINAL_* (not FINAL_*) ──────────────────────


def test_exp_7_writes_provisional_not_final(orch, tmp_path):
    """Selection by EXP 7 is PROVISIONAL until EXP A bootstrap CI promotes it.

    Asserts ``state_updates`` carries ``PROVISIONAL_FINAL_MAP_CONFIG`` and
    does NOT carry ``FINAL_MAP_CONFIG`` — that key is reserved for the
    post-bootstrap final pick.
    """
    g = {"embedder": "gemini", "scorer": "embedding_default", "theta": 0.8,
         "reject_theta": 0.2, "polarity_flag": True, "strict_f1": 0.85,
         "hybrid_weights": None, "agreement_weights": None}
    o = {"embedder": "openai", "scorer": "embedding_default", "theta": 0.7,
         "reject_theta": 0.1, "polarity_flag": True, "strict_f1": 0.82,
         "hybrid_weights": None, "agreement_weights": None}
    ctx = orch.ExperimentContext(
        output_dir=tmp_path,
        state={"FINAL_GEMINI_MAP_CONFIG": g, "FINAL_OPENAI_MAP_CONFIG": o},
    )
    result = orch._run_exp_7(ctx)
    assert "PROVISIONAL_FINAL_MAP_CONFIG" in result.state_updates
    assert "PROVISIONAL_FINAL_EMBEDDER" in result.state_updates
    assert "FINAL_MAP_CONFIG" not in result.state_updates, (
        "EXP 7 must not promote to FINAL_MAP_CONFIG — that's EXP A's job after "
        "bootstrap CIs / tie logic land."
    )
    # Winner = higher strict_f1 = Gemini (0.85 > 0.82)
    assert result.state_updates["PROVISIONAL_FINAL_EMBEDDER"] == "gemini"
    assert result.state_updates["PROVISIONAL_FINAL_MAP_CONFIG"]["embedder"] == "gemini"


# ── 8. --split test rejected for EXP 1-7 without --allow-test-tuning ────────


def test_split_test_rejected_for_exp_1_without_allow_test_tuning(orch, capsys):
    """``--split test`` on a calibration EXP without ``--allow-test-tuning``
    must exit 2 and print a clear rejection message naming the override flag.
    """
    rc = orch.main([
        "--exp", "exp_1_gemini_scorer",
        "--split", "test",
        "--run",
    ])
    captured = capsys.readouterr()
    assert rc == 2
    assert "--split test rejected" in captured.err
    assert "--allow-test-tuning" in captured.err


def test_split_test_allowed_with_allow_test_tuning(orch, tmp_path):
    """Explicit override accepted. Stays in dry-run (no --run) so no sweep."""
    rc = orch.main([
        "--exp", "exp_1_gemini_scorer",
        "--split", "test",
        "--allow-test-tuning",
        "--state-path", str(tmp_path / "state.json"),
        "--output-dir", str(tmp_path),
    ])
    assert rc == 0


def test_split_test_allowed_for_exp_f_without_flag(orch, tmp_path):
    """EXP F is exempt from the --split test protection (it always uses test
    internally regardless of CLI; rejecting CLI test for it would be a UX bug).
    """
    rc = orch.main([
        "--exp", "exp_f_test_confirmation",
        "--split", "test",
        "--state-path", str(tmp_path / "state.json"),
        "--output-dir", str(tmp_path),
    ])
    assert rc == 0


# ── 9. _per_scorer_best picks correctly ─────────────────────────────────────


def test_per_scorer_best_picks_max_strict_f1_per_scorer(orch):
    """Group by `scorer`, keep the row with the highest strict_f1 per group."""
    rows = [
        {"scorer": "embedding_default", "strict_f1": 0.50, "escalate_rate": 0.30},
        {"scorer": "embedding_default", "strict_f1": 0.55, "escalate_rate": 0.40},
        {"scorer": "hybrid_default",    "strict_f1": 0.60, "escalate_rate": 0.50},
        {"scorer": "hybrid_default",    "strict_f1": 0.58, "escalate_rate": 0.20},
        {"scorer": "hybrid_balanced",   "strict_f1": 0.62, "escalate_rate": 0.45},
    ]
    best = orch._per_scorer_best(rows, metric="strict_f1")
    by_scorer = {r["scorer"]: r for r in best}
    assert len(best) == 3
    assert by_scorer["embedding_default"]["strict_f1"] == 0.55
    assert by_scorer["hybrid_default"]["strict_f1"] == 0.60
    assert by_scorer["hybrid_balanced"]["strict_f1"] == 0.62


def test_per_scorer_best_tiebreaks_lower_escalate_rate(orch):
    """When two rows share the metric, prefer the one with lower escalate_rate."""
    rows = [
        {"scorer": "embedding_default", "strict_f1": 0.50, "escalate_rate": 0.50},
        {"scorer": "embedding_default", "strict_f1": 0.50, "escalate_rate": 0.10},
        {"scorer": "embedding_default", "strict_f1": 0.50, "escalate_rate": 0.30},
    ]
    best = orch._per_scorer_best(rows, metric="strict_f1")
    assert len(best) == 1
    assert best[0]["escalate_rate"] == 0.10


def test_per_scorer_best_empty_input(orch):
    """Defensive: empty rows in → empty best-of out."""
    assert orch._per_scorer_best([], metric="strict_f1") == []
