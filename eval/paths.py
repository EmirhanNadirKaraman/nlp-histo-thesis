"""Stable repository-root anchor for the evaluation harness.

Resource-path resolution only — this module never touches ``sys.path``, the
working directory, or the environment. Modules under ``eval/`` that need to
build a path relative to the repository root import ``REPO_ROOT`` from here
instead of recomputing ``Path(__file__).resolve().parents[n]`` locally, so the
depth is stated in exactly one place and does not change when a module moves.

``eval`` is not part of the installed distribution; ``eval.*`` commands run from
the repository root with ``python -m``.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
