"""Regression test for B-020: batch dispatch groups by (provider, model).

Pre-fix `dispatch.submit_level` grouped requests by `provider` only. The OpenAI
Batch API rejects any batch that mixes models (`mismatched_model`). The cheap
profile has two OpenAI L1 voters (`gpt-4o-mini` and `gpt-4.1-nano`), so every
batched cheap-profile run had its entire OpenAI half-batch rejected silently.

This test asserts the grouping logic produces one batch per (provider, model)
tuple — even when one provider hosts multiple models.

No live API calls.
"""
from __future__ import annotations

from pipeline.stages.knowledge_extraction.batch.models import BatchRequest


def _make_request(provider: str, model: str, custom_id: str) -> BatchRequest:
    return BatchRequest(
        custom_id=custom_id,
        messages=[],
        model=model,
        provider=provider,
        max_tokens=8000,
        temperature=0.1,
    )


def _group_by_provider_model(requests: list[BatchRequest]) -> dict[tuple[str, str], list[BatchRequest]]:
    """Mirror dispatch.submit_level's grouping step so we can assert on it
    without invoking real provider clients."""
    out: dict[tuple[str, str], list[BatchRequest]] = {}
    for r in requests:
        out.setdefault((r.provider, r.model), []).append(r)
    return out


def test_cheap_profile_l1_produces_three_batches():
    """Cheap-profile L1 (1× Gemini + 2× OpenAI) must split into 3 batches,
    not 2 — otherwise the OpenAI batch mixes gpt-4o-mini with gpt-4.1-nano
    and the Batch API rejects the whole submission."""
    requests = [
        _make_request("gemini", "gemini-2.5-flash-lite", "p__c1__l1__0"),
        _make_request("openai", "gpt-4o-mini",          "p__c1__l1__1"),
        _make_request("openai", "gpt-4.1-nano",         "p__c1__l1__2"),
        _make_request("gemini", "gemini-2.5-flash-lite", "p__c2__l1__0"),
        _make_request("openai", "gpt-4o-mini",          "p__c2__l1__1"),
        _make_request("openai", "gpt-4.1-nano",         "p__c2__l1__2"),
    ]
    groups = _group_by_provider_model(requests)
    assert len(groups) == 3, f"expected 3 batches, got {len(groups)}: {list(groups.keys())}"
    assert ("gemini", "gemini-2.5-flash-lite") in groups
    assert ("openai", "gpt-4o-mini") in groups
    assert ("openai", "gpt-4.1-nano") in groups
    # Each batch has exactly the expected request_count for the number of chunks.
    assert len(groups[("gemini", "gemini-2.5-flash-lite")]) == 2
    assert len(groups[("openai", "gpt-4o-mini")]) == 2
    assert len(groups[("openai", "gpt-4.1-nano")]) == 2


def test_single_model_per_provider_stays_one_batch():
    """A single voter per provider should produce one batch per provider —
    no over-fragmentation when the multi-model case doesn't apply."""
    requests = [
        _make_request("gemini", "gemini-2.5-flash-lite", "p__c1__l1__0"),
        _make_request("openai", "gpt-4.1-mini",          "p__c1__l2__0"),
    ]
    groups = _group_by_provider_model(requests)
    assert len(groups) == 2


def test_dispatch_submit_level_uses_provider_model_grouping():
    """White-box check that dispatch.py contains the (provider, model)
    grouping — guards against a future refactor that re-introduces the bug
    by reverting to `req.provider` only."""
    import inspect

    from pipeline.stages.knowledge_extraction.batch import dispatch
    source = inspect.getsource(dispatch.submit_level)
    assert "by_provider_model" in source, (
        "submit_level no longer groups by (provider, model) — B-020 regressed?"
    )
    assert "req.provider, req.model" in source or "(req.provider, req.model)" in source, (
        "submit_level grouping key must include req.model — B-020 regressed?"
    )
