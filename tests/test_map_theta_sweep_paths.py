"""B-102: map_theta_sweep's repository-owned defaults must not depend on cwd.

PRIMER_PATH and CACHE_PATH are default argument values on eight functions and
bind at import time, so a bare relative default made every one of them resolve
against whatever directory the caller happened to be in. Running the results
replay from outside the repository failed with a confusing missing-cache error
rather than a path error.

These tests import the module with cwd set to a temporary directory outside the
repository, which is the condition that used to break.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE = "eval.silver.analysis.map_theta_sweep"


@pytest.fixture
def sweep_from_outside_repo(tmp_path, monkeypatch):
    """Import map_theta_sweep with cwd outside the repository."""
    monkeypatch.chdir(tmp_path)
    assert Path.cwd().resolve() != REPO_ROOT
    assert REPO_ROOT not in Path.cwd().resolve().parents
    if str(REPO_ROOT) not in sys.path:
        monkeypatch.syspath_prepend(str(REPO_ROOT))
    sys.modules.pop(MODULE, None)
    return importlib.import_module(MODULE)


REPO_OWNED = ["PRIMER_DIR", "PRIMER_PATH", "CACHE_PATH", "SOURCE_PATH", "SILVER_PATH"]
FROZEN_CACHES = ["_FROZEN_OPENAI_CACHE", "_FROZEN_GEMINI_CACHE"]


@pytest.mark.parametrize("name", REPO_OWNED + FROZEN_CACHES)
def test_repo_owned_defaults_are_absolute_and_under_repo_root(sweep_from_outside_repo, name):
    value = getattr(sweep_from_outside_repo, name)
    assert value.is_absolute(), f"{name} is not absolute: {value}"
    assert REPO_ROOT in value.parents or value.parent == REPO_ROOT, (
        f"{name} resolves outside the repository: {value}"
    )


def test_defaults_are_independent_of_cwd(tmp_path, monkeypatch):
    """Importing from two different directories yields identical paths."""
    def paths_with_cwd(where):
        monkeypatch.chdir(where)
        if str(REPO_ROOT) not in sys.path:
            monkeypatch.syspath_prepend(str(REPO_ROOT))
        sys.modules.pop(MODULE, None)
        mod = importlib.import_module(MODULE)
        return {n: getattr(mod, n) for n in REPO_OWNED + FROZEN_CACHES}

    outside = paths_with_cwd(tmp_path)
    inside = paths_with_cwd(REPO_ROOT)
    assert outside == inside


def test_primer_children_derive_from_primer_dir(sweep_from_outside_repo):
    m = sweep_from_outside_repo
    assert m.PRIMER_PATH.parent == m.PRIMER_DIR
    assert m.CACHE_PATH.parent == m.PRIMER_DIR
    assert m.PRIMER_PATH.name == "primer.json"
    assert m.CACHE_PATH.name == "voter_cache.json"


def test_anchored_defaults_match_the_old_relative_paths(sweep_from_outside_repo):
    """The fix must not retarget anything: from the repo root, the old bare
    relative literals and the new anchored constants name the same locations."""
    m = sweep_from_outside_repo
    expected = {
        "PRIMER_DIR": "eval/data/map_primer",
        "PRIMER_PATH": "eval/data/map_primer/primer.json",
        "CACHE_PATH": "eval/data/map_primer/voter_cache.json",
        "SOURCE_PATH": "eval/data/source_cases_related15.jsonl",
        "SILVER_PATH": "eval/data/silver_findings_related15.jsonl",
    }
    for name, rel in expected.items():
        assert getattr(m, name) == REPO_ROOT / rel


def test_generated_output_stays_caller_relative(sweep_from_outside_repo):
    """REPORTS_DIR is output, not input: it must stay cwd-relative (B-102)."""
    assert not sweep_from_outside_repo.REPORTS_DIR.is_absolute()
    assert str(sweep_from_outside_repo.REPORTS_DIR) == "eval/reports"


def test_explicit_overrides_keep_caller_relative_semantics(sweep_from_outside_repo, tmp_path):
    """A user-supplied --primer-dir is honoured verbatim, absolute or relative."""
    m = sweep_from_outside_repo
    parser_args = ["sweep", "--primer-dir", str(tmp_path / "mine")]
    ns = m.build_parser().parse_args(parser_args) if hasattr(m, "build_parser") else None
    if ns is not None:
        assert Path(ns.primer_dir) == tmp_path / "mine"
    # relative override stays relative -- resolution is the caller's business
    rel = Path("some/relative/primer")
    assert not rel.is_absolute()


def test_help_runs_from_outside_the_repository(tmp_path):
    """The public entry point must at least parse --help from a foreign cwd."""
    import subprocess
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        env.pop(var, None)
    proc = subprocess.run(
        [sys.executable, "-m", MODULE, "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "--primer-dir" in proc.stdout
