"""B-081: GeminiBatchProvider.retrieve() must map results by metadata key, not
list position. Regression test for cross-paper contamination caused by
positional custom_id recovery against reordered/dropped Gemini responses.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pipeline.stages.summarization.batch.gemini_batch import GeminiBatchProvider, _ID_SEP
from pipeline.stages.summarization.batch.models import ProviderJob


def _provider(inline_responses):
    p = GeminiBatchProvider.__new__(GeminiBatchProvider)  # skip __init__ (no API key)
    client = MagicMock()
    client.batches.get.return_value = SimpleNamespace(
        dest=SimpleNamespace(inlined_responses=inline_responses)
    )
    p._client = client
    return p


def _resp(custom_id, text='{"ok": 1}'):
    md = {"custom_id": custom_id} if custom_id is not None else None
    response = SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=2),
    )
    return SimpleNamespace(metadata=md, error=None, response=response)


def _job(custom_ids):
    return ProviderJob(
        provider="gemini", job_id="batches/abc", status="completed", model="m",
        request_count=len(custom_ids),
        output_location=f"batches/abc{_ID_SEP}{','.join(custom_ids)}",
    )


def test_matches_by_metadata_when_responses_reordered():
    # Submit order A,B,C; provider returns them as C,A,B. Positional recovery
    # would mislabel every one; metadata matching must pin each to its own id.
    submit_ids = ["PMCa__C1__l1__0", "PMCb__C1__l1__0", "PMCc__C1__l1__0"]
    reordered = [_resp("PMCc__C1__l1__0", '{"c":1}'),
                 _resp("PMCa__C1__l1__0", '{"a":1}'),
                 _resp("PMCb__C1__l1__0", '{"b":1}')]
    results = _provider(reordered).retrieve(_job(submit_ids))
    got = {r.custom_id: r.content for r in results}
    assert got["PMCa__C1__l1__0"] == '{"a":1}'
    assert got["PMCb__C1__l1__0"] == '{"b":1}'
    assert got["PMCc__C1__l1__0"] == '{"c":1}'


def test_length_mismatch_raises():
    # One response dropped → positional recovery is unsafe → fail loud.
    submit_ids = ["PMCa__C1__l1__0", "PMCb__C1__l1__0", "PMCc__C1__l1__0"]
    dropped = [_resp("PMCa__C1__l1__0"), _resp("PMCc__C1__l1__0")]
    with pytest.raises(RuntimeError, match="count mismatch"):
        _provider(dropped).retrieve(_job(submit_ids))


def test_positional_fallback_when_metadata_absent(caplog):
    # Older API / no metadata round-trip → fall back to position but WARN.
    submit_ids = ["PMCa__C1__l1__0", "PMCb__C1__l1__0"]
    no_md = [_resp(None, '{"a":1}'), _resp(None, '{"b":1}')]
    with caplog.at_level("WARNING"):
        results = _provider(no_md).retrieve(_job(submit_ids))
    got = {r.custom_id: r.content for r in results}
    assert got["PMCa__C1__l1__0"] == '{"a":1}'  # position 0
    assert got["PMCb__C1__l1__0"] == '{"b":1}'  # position 1
    assert any("matched by POSITION" in rec.message for rec in caplog.records)
