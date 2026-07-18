"""B-108: the default sweep run must not write into the tracked results tree.

`eval/results/grounding_sweep.md` is a frozen thesis snapshot (5 papers, run
grounding_compare_calv1_runB_20260516T163007). The sweep has no frozen input
pin -- it reads whatever is in out/summaries -- so a default run reflects the
current corpus, not that snapshot. Writing there rewrote a published table with
numbers corresponding to no published result, and dirtied the worktree for
anyone following HOW_TO_RUN section 12.

The module is loaded by file path and ``_lib`` is popped afterwards, matching the
fixtures in test_grounding.py and test_compute_proxy_metrics.py. That is not
ceremony: ``eval/sweeps/_lib.py`` and ``scripts/eval/_lib.py`` are different
modules that both get imported as bare ``_lib`` after their own directory is put
on ``sys.path[0]``, so whichever loads first claims ``sys.modules["_lib"]`` for
the whole session. A plain module-scope import here makes an unrelated
proxy-metrics test fail under some pytest-randomly orderings.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from tests.paths import REPO_ROOT

MODULE_PATH = REPO_ROOT / "eval" / "sweeps" / "grounding.py"
TRACKED_REPORT = REPO_ROOT / "eval" / "results" / "grounding_sweep.md"


@pytest.fixture
def grounding():
    """Load eval/sweeps/grounding.py in isolation, then release ``_lib``."""
    spec = importlib.util.spec_from_file_location("grounding_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)
    sys.modules.pop("_lib", None)


def _tracked_files() -> set[str]:
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-tree", "-r", "--name-only", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout
    return {line for line in out.split("\n") if line}


@pytest.mark.parametrize("attr", ["DEFAULT_OUT_MD", "DEFAULT_OUT_CSV"])
def test_default_outputs_are_not_tracked_paths(grounding, attr):
    """A default run must never target a file git tracks."""
    default = getattr(grounding, attr)
    assert str(default) not in _tracked_files(), (
        f"default output {default} is a tracked artifact; a plain run would dirty it"
    )


@pytest.mark.parametrize("attr", ["DEFAULT_OUT_MD", "DEFAULT_OUT_CSV"])
def test_default_outputs_land_in_the_generated_tree(grounding, attr):
    assert getattr(grounding, attr).parts[0] == "out"


def test_tracked_snapshot_is_still_tracked():
    """The frozen report stays in the repository -- this fix redirects the
    writer, it does not untrack the thesis artifact."""
    assert "eval/results/grounding_sweep.md" in _tracked_files()
    assert TRACKED_REPORT.is_file()


def test_defaults_do_not_collide_with_the_snapshot(grounding):
    assert grounding.DEFAULT_OUT_MD.resolve() != TRACKED_REPORT.resolve()


def test_explicit_override_still_reaches_the_tracked_path(grounding):
    """Regenerating the snapshot on purpose must remain possible."""
    args = grounding.parse_args([
        "--out-md", "eval/results/grounding_sweep.md",
        "--out-csv", "eval/results/grounding_sweep.csv",
    ])
    assert args.out_md == Path("eval/results/grounding_sweep.md")
    assert args.out_csv == Path("eval/results/grounding_sweep.csv")


def test_parse_args_defaults_match_module_constants(grounding):
    args = grounding.parse_args([])
    assert args.out_md == grounding.DEFAULT_OUT_MD
    assert args.out_csv == grounding.DEFAULT_OUT_CSV
    assert args.input == grounding.DEFAULT_INPUT
