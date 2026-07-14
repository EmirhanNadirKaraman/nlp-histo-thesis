"""Stable repository-root anchor for the test suite.

Resource-path resolution only — this module never touches ``sys.path``, the
working directory, or the environment. Tests that need a path relative to the
repository root (e.g. to load a module under ``scripts/``, which is not a
package and so cannot be imported by name) import ``REPO_ROOT`` / ``SCRIPTS``
from here instead of recomputing ``Path(__file__).resolve().parents[n]``
locally, so the depth is stated in exactly one place and does not change when a
test module moves between packages.

Mirrors ``eval/paths.py``, which does the same for the evaluation harness.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
