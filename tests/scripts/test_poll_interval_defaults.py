"""Regression test for the `--poll-interval` duplicate-default bug.

Tier 1 config audit (2026-05-15) consolidated three diverging defaults
(CLI=60, `_run_all_batch`=20, `_run_batch`=60) onto a single module-level
constant `DEFAULT_POLL_INTERVAL_SEC`. This test pins that consolidation so
the defaults cannot silently drift apart again.

Implementation note: we deliberately use `inspect.signature(...).parameters
[name].default` rather than `func.__defaults__` tuple indexing — the latter
breaks the moment a new parameter is added before `poll_interval`.
"""
from __future__ import annotations

import inspect




def _load_run_paper():
    # The implementation is an installed module now — no path loading needed.
    from nlp_histo.workflows import knowledge

    return knowledge


def test_poll_interval_defaults_agree():
    run_paper = _load_run_paper()
    cli_default = run_paper.DEFAULT_POLL_INTERVAL_SEC
    all_batch_default = (
        inspect.signature(run_paper._run_all_batch).parameters["poll_interval"].default
    )
    single_batch_default = (
        inspect.signature(run_paper._run_batch).parameters["poll_interval"].default
    )

    assert cli_default == 60, (
        f"Tier 1 audit pinned the canonical poll-interval default to 60s; "
        f"got {cli_default}. Update the constant deliberately if you mean to."
    )
    assert all_batch_default == cli_default, (
        f"_run_all_batch poll_interval default ({all_batch_default}) drifted "
        f"from DEFAULT_POLL_INTERVAL_SEC ({cli_default})."
    )
    assert single_batch_default == cli_default, (
        f"_run_batch poll_interval default ({single_batch_default}) drifted "
        f"from DEFAULT_POLL_INTERVAL_SEC ({cli_default})."
    )
