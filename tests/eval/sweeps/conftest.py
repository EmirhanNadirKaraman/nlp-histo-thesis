"""Shared hermeticity guard for the Layer-A grounding-sweep tests.

These tests exercise CLIs (``eval/sweeps/grounding.py`` /
``eval/sweeps/grounding_plot.py``) whose production defaults read and write
the tracked report ``eval/results/grounding_sweep.md``. A test that forgets to
redirect ``--sweep-md`` / ``--out-*`` to ``tmp_path`` would silently mutate that
committed file (embedding machine-specific temp paths) and leave the working
tree dirty. The autouse guard below fails loudly if any test in this directory
changes the tracked report, keeping the suite hermetic and order-independent.

Production defaults are unchanged — this only guards the tests.
"""

from __future__ import annotations


import pytest

from tests.paths import REPO_ROOT

_TRACKED_REPORT = REPO_ROOT / "eval" / "results" / "grounding_sweep.md"


@pytest.fixture(autouse=True)
def _tracked_report_unchanged():
    """Assert no grounding-sweep test mutates the tracked ``grounding_sweep.md``."""
    before = _TRACKED_REPORT.read_bytes() if _TRACKED_REPORT.exists() else None
    yield
    after = _TRACKED_REPORT.read_bytes() if _TRACKED_REPORT.exists() else None
    assert after == before, (
        f"a grounding-sweep test mutated the tracked report {_TRACKED_REPORT}. "
        "Point the CLI's --sweep-md / --out-* at tmp_path instead of the default "
        "so generated output stays in the test's temporary directory."
    )
