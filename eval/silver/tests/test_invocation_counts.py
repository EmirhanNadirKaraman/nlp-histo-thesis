"""Per-tier INVOCATION counting in ``map_theta_sweep._replay`` (cost columns).

The cascade row reports ``early_accept_rate`` (L1+L2 *accepts* summed) and
``escalate_rate`` (L3 accepts). Cost needs the L1↔L2 split, which those two can't
recover — hence the ``invoked`` counts. A tier is INVOKED when it actually runs:
L1 on every chunk with L1 data; L2 only when L1 escalated; L3 only when L2
escalated (or had no L2 data). Invariant: ``invoked["l3"] == accept["l3"]``.

Engine-free: ``AgreementChecker`` / ``AuditableSummary`` are stubbed so the test
drives only the routing+counting logic (the decision logic is tested elsewhere).
"""
from __future__ import annotations

import types

import pytest

import nlp_histo.pipeline.stages.knowledge_extraction.agreement as agreement_mod
import nlp_histo.pipeline.stages.knowledge_extraction.models as models_mod
import eval.silver.analysis.map_theta_sweep as mts
from nlp_histo.pipeline.stages.knowledge_extraction.interfaces.scoring import ChunkDecision

_DEC = {"KEEP": ChunkDecision.KEEP, "REJECT": ChunkDecision.REJECT,
        "ESCALATE": ChunkDecision.ESCALATE}


class _StubSummary:
    """Stand-in for AuditableSummary: carries summary_text "<level>:<DECISION>"."""
    def __init__(self, d):
        self._d = d
        self.summary_text = d.get("summary_text", "")
    @classmethod
    def model_validate(cls, d): return cls(d)
    def model_dump(self): return self._d


class _StubChecker:
    """Returns the decision scripted in each summary's summary_text suffix."""
    def __init__(self, scorer=None, **kw): pass
    def compute(self, outs, source_text=""):
        decision = _DEC[outs[0].summary_text.split(":", 1)[1]]
        return types.SimpleNamespace(decision=decision, score_details={})
    def best(self, outs, bundle): return outs[0]


def _s(level: str, decision: str) -> dict:
    # findings=[] → _replay never calls _finding_to_pipeline (counting is finding-agnostic)
    return {"summary_text": f"{level}:{decision}", "findings": [], "audit_metadata": {}}


@pytest.fixture
def _stub(monkeypatch):
    monkeypatch.setattr(agreement_mod, "AgreementChecker", _StubChecker)
    monkeypatch.setattr(models_mod, "AuditableSummary", _StubSummary)
    yield


def _scripted_cache() -> dict:
    # cA: L1 KEEP            → invoke L1 only;        resolve l1
    # cB: L1 ESC → L2 KEEP   → invoke L1,L2;          resolve l2
    # cC: L1 ESC → L2 ESC    → invoke L1,L2,L3;       resolve l3
    # cD: no L1 → L2 KEEP    → invoke L2 only (no L1);resolve l2
    # cE: L1 ESC → no L2     → invoke L1,L3 (no L2);  resolve l3
    chunks = ["cA", "cB", "cC", "cD", "cE"]
    return {"PMC1|1": {
        "case_id": "PMC1|1", "pmcid": "PMC1", "te_id": 1,
        "chunk_map": {c: [] for c in chunks},
        "source_texts": {c: "" for c in chunks},
        "l1": {"cA": [_s("l1", "KEEP")], "cB": [_s("l1", "ESCALATE")],
               "cC": [_s("l1", "ESCALATE")], "cD": [], "cE": [_s("l1", "ESCALATE")]},
        "l2": {"cA": [], "cB": [_s("l2", "KEEP")], "cC": [_s("l2", "ESCALATE")],
               "cD": [_s("l2", "KEEP")], "cE": []},
        "l3": {"cC": _s("l3", "KEEP"), "cE": _s("l3", "KEEP")},
    }}


def test_invocation_counts_match_scripted_routing(_stub):
    _outs, accept, _early, _npol, invoked = mts._replay(
        _scripted_cache(), scorer=None, theta=0.6, reject_theta=0.1)
    assert invoked == {"l1": 4, "l2": 3, "l3": 2}            # L1: cA,cB,cC,cE; L2: cB,cC,cD; L3: cC,cE
    assert accept["l1"] == 1 and accept["l2"] == 2 and accept["l3"] == 2
    assert accept["reject"] == 0


def test_l3_invoked_equals_l3_accept_invariant(_stub):
    _o, accept, _e, _n, invoked = mts._replay(
        _scripted_cache(), scorer=None, theta=0.6, reject_theta=0.1)
    # L3 is the escalation terminus: every chunk that reaches it resolves there.
    assert invoked["l3"] == accept["l3"]


def test_l2_invocation_not_recoverable_from_accept_rates(_stub):
    """The motivating gap: early_accept (l1+l2 accepts) + escalate (l3) can't yield
    L2 invocations — but invoked["l2"] can. Here l2 *accepts*=2 yet L2 *ran* 3×."""
    _o, accept, _e, _n, invoked = mts._replay(
        _scripted_cache(), scorer=None, theta=0.6, reject_theta=0.1)
    assert accept["l2"] == 2 and invoked["l2"] == 3          # cC invoked L2 but resolved at L3
