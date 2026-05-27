"""
Canonical voter model names, factory functions, and cascade profiles.

Single source of truth for all LLM model IDs used in the MAP cascade.
Both run_paper.py (batch + sync runners) and map_theta_sweep.py import
from here so the two are always in sync.

Profiles
--------
Pick one with ``get_profile(name)`` (or set ``NLP_HISTO_PROFILE``). There is
no implicit default — callers must select a profile explicitly so accidental
runs cannot silently burn an unintended cascade.

  cheap      — 2-tier smoke cascade across the OpenAI + Gemini providers.
               L1: Gemini-Flash-Lite, GPT-4o-mini, GPT-4.1-nano (no Claude
               voter). L2 and L3 are both GPT-4.1-mini: L3 is set equal to L2
               so the structural 3-tier ``CascadeProfile`` dataclass still
               works, but the cascade is effectively 2 distinct tiers.
  real       — 3-tier production cascade. L1: Gemini-Flash-Lite, GPT-4o-mini,
               GPT-4.1-nano (cheapest tier; no Claude voter at L1). L2:
               Gemini-Flash, GPT-4.1-mini, Claude-Haiku. L3: Claude-Sonnet
               (temperature=0.0). Use for real evaluation runs.
  haiku_only — Single-voter Claude-Haiku at L1 (N=1 → KEEP under
               ``legacy_single_voter_policy="keep"``; L2/L3 never fire). Use
               for cost-effective production runs; ~\\$0.0036/chunk, accepts
               a ~4.2pp strict_f1 drop vs Sonnet-only per EXP B.2 dev.

The previous ``default`` profile (Claude-Haiku at L1 for 3-provider
diversity, mirroring the legacy hardcoded sync config) was removed —
``real`` is the production cascade and ``cheap`` is the smoke cascade,
both retained. Reintroduce only if the L1-provider-diversity hypothesis is
empirically required.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# ── Model IDs ─────────────────────────────────────────────────────────────────

# L1 — cheapest tier
GEMINI_L1 = "gemini-2.5-flash-lite"
OPENAI_L1 = "gpt-4o-mini"
OPENAI_L1_B = "gpt-4.1-nano"
CLAUDE_L1 = "claude-haiku-4-5-20251001"

# L2 — mid tier
GEMINI_L2 = "gemini-2.5-flash"
OPENAI_L2 = "gpt-4.1-mini"
CLAUDE_L2 = "claude-haiku-4-5-20251001"   # no cheap step between Haiku and Sonnet

# L3 — escalation
# Sonnet 4.6 has no datestamped alias; only the bare ID resolves (verified
# 2026-05-14 — datestamped form returns 404 from Anthropic). Haiku 4.5 accepts
# both, kept datestamped for pin-stability.
CLAUDE_L3 = "claude-sonnet-4-6"

# Aliases kept for back-compat with tests / scripts that import these directly.
CLAUDE_HAIKU  = "claude-haiku-4-5-20251001"
CLAUDE_SONNET = "claude-sonnet-4-6"


# ── Voter list factories (back-compat for run_paper.py) ───────────────────────

def make_l1_voters():
    """L1 voter list: Gemini-Flash-Lite, GPT-4o-mini, GPT-4.1-nano.

    Cheapest tier; no Claude voter at L1 — Claude enters at L2 as Haiku.
    """
    from .models import VoterBatchConfig
    return [
        VoterBatchConfig(GEMINI_L1, provider="gemini", temperature=0.1),
        VoterBatchConfig(OPENAI_L1, provider="openai", temperature=0.1),
        VoterBatchConfig(OPENAI_L1_B, provider="openai", temperature=0.1),
    ]


def make_l2_voters():
    """L2 voter list (for `real` profile): Gemini-Flash, GPT-4.1-mini, Claude-Haiku."""
    from .models import VoterBatchConfig
    return [
        VoterBatchConfig(GEMINI_L2, provider="gemini", temperature=0.1),
        VoterBatchConfig(OPENAI_L2, provider="openai", temperature=0.1),
        VoterBatchConfig(CLAUDE_L2, provider="claude", temperature=0.1),
    ]


def make_l3_voter():
    """L3 escalation voter: Claude Sonnet."""
    from .models import VoterBatchConfig
    return VoterBatchConfig(CLAUDE_L3, provider="claude")


# ── Profiles ──────────────────────────────────────────────────────────────────

@dataclass
class CascadeProfile:
    """Concrete cascade selection for the batch MAP runner."""
    name: str
    l1_voters: list  # list[VoterBatchConfig]
    l2_voters: list  # list[VoterBatchConfig]
    l3_voter:  object  # VoterBatchConfig


def _make_cheap_profile() -> CascadeProfile:
    """2-tier smoke cascade across OpenAI + Gemini.

    L1: Gemini-Flash-Lite (gemini), GPT-4o-mini (openai), GPT-4.1-nano (openai).
    L2: GPT-4.1-mini (openai).
    L3 is set to the same model/provider as L2 (gpt-4.1-mini, openai) because
    CascadeProfile structurally requires an L3 voter, but semantically this is
    a 2-tier cascade — any L2 disagreement re-runs at the same tier rather
    than escalating to a more expensive model.
    All voters use temperature=0.1.
    """
    from .models import VoterBatchConfig
    l1 = [
        VoterBatchConfig(GEMINI_L1, provider="gemini", temperature=0.1),
        VoterBatchConfig(OPENAI_L1, provider="openai", temperature=0.1),
        VoterBatchConfig(OPENAI_L1_B, provider="openai", temperature=0.1),
    ]
    l2 = [VoterBatchConfig(OPENAI_L2, provider="openai", temperature=0.1)]
    l3 = VoterBatchConfig(OPENAI_L2, provider="openai", temperature=0.1)
    return CascadeProfile(name="cheap", l1_voters=l1, l2_voters=l2, l3_voter=l3)


def _make_real_profile() -> CascadeProfile:
    """3-tier production cascade across 3 providers.

    L1: Gemini-Flash-Lite, GPT-4o-mini, GPT-4.1-nano — cheapest tier (no
    Claude voter at L1; Claude enters at L2 as Haiku). temperature=0.1.
    L2: mid-tier — Gemini-Flash, GPT-4.1-mini, Claude-Haiku. temperature=0.1.
    L3: Claude-Sonnet. temperature=0.0 (deterministic final escalation).
    Use for real evaluation runs.
    """
    from .models import VoterBatchConfig
    l1 = [
        VoterBatchConfig(GEMINI_L1, provider="gemini", temperature=0.1),
        VoterBatchConfig(OPENAI_L1, provider="openai", temperature=0.1),
        VoterBatchConfig(OPENAI_L1_B, provider="openai", temperature=0.1),
    ]
    l2 = [
        VoterBatchConfig(GEMINI_L2, provider="gemini", temperature=0.1),
        VoterBatchConfig(OPENAI_L2, provider="openai", temperature=0.1),
        VoterBatchConfig(CLAUDE_L2, provider="claude", temperature=0.1),
    ]
    l3 = VoterBatchConfig(CLAUDE_L3, provider="claude", temperature=0.0)
    return CascadeProfile(name="real", l1_voters=l1, l2_voters=l2, l3_voter=l3)


def _make_haiku_only_profile() -> CascadeProfile:
    """Single-voter Haiku at L1 — no escalation fires.

    L1 has a single Claude Haiku voter. With ``legacy_single_voter_policy``
    set to ``"keep"`` (the production default), ``AgreementChecker.compute``
    returns ``ChunkDecision.KEEP`` with ``confidence=1.0`` at N=1 — every
    chunk is accepted at L1 and L2 / L3 are never invoked. L2 and L3 slots
    are populated structurally (the dataclass requires them) but are dead
    code at runtime.

    Temperature is ``0.1`` at every tier to match the EXP B.2 measurement:
    the cited Haiku-only strict_f1 = 0.696 was generated at L2 in the
    ``real`` profile, where Haiku runs at ``t=0.1``. Shipping at any other
    temperature would invalidate the empirical basis for the cited
    accuracy (cf. ``docs/EXP_B2_RESULTS.md``). It also keeps the cached
    L2-Haiku voter outputs in ``eval/data/map_primer/voter_cache.json``
    reusable should we want to replay any silver-set chunk under this
    profile.

    Cost: ~\\$0.0036 / chunk (token assumption: 1500 in + 600 out). Use for
    cost-effective production runs that accept the EXP B.2 dev-split result
    of strict_f1 ≈ 0.696 — 73 % cheaper than ``claude-sonnet-4-6`` (Sonnet
    alone) and 82 % cheaper than the calibrated cascade, at a 4.2pp
    strict_f1 drop vs Sonnet. The family-bias diagnostic confirmed the
    accuracy ranking is not a matcher artifact.
    """
    from .models import VoterBatchConfig
    haiku = VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.1)
    return CascadeProfile(
        name="haiku_only",
        l1_voters=[haiku],
        l2_voters=[haiku],
        l3_voter=haiku,
    )


_PROFILE_BUILDERS = {
    "cheap":      _make_cheap_profile,
    "real":       _make_real_profile,
    "haiku_only": _make_haiku_only_profile,
}


def list_profiles() -> list[str]:
    return list(_PROFILE_BUILDERS.keys())


def get_profile(name: str | None = None) -> CascadeProfile:
    """Resolve a cascade profile by name.

    Resolution order: explicit ``name`` arg → ``$NLP_HISTO_PROFILE``. There is
    no implicit fallback — if neither is set, ``ValueError`` is raised so
    accidental runs cannot silently burn an unintended cascade. Unknown names
    also raise so typos surface immediately.
    """
    resolved = name or os.environ.get("NLP_HISTO_PROFILE")
    if not resolved:
        raise ValueError(
            "No cascade profile selected. Pass an explicit name to "
            "get_profile() or set $NLP_HISTO_PROFILE. "
            f"Available: {sorted(_PROFILE_BUILDERS)}"
        )
    if resolved not in _PROFILE_BUILDERS:
        raise ValueError(
            f"Unknown cascade profile {resolved!r}. "
            f"Available: {sorted(_PROFILE_BUILDERS)}"
        )
    return _PROFILE_BUILDERS[resolved]()
