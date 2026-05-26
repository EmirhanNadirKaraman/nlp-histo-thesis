"""Tests for the unified YAML config loader."""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from pipeline.config_loader import load_config
from pipeline.stages.pdf_text_extraction.config import (
    LogLevel,
    TableDetectorType,
)


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "run.yaml"
    p.write_text(dedent(body))
    return p


def test_empty_yaml_uses_defaults(tmp_path: Path):
    p = _write(tmp_path, "")
    pdf, sumcfg = load_config(p)
    assert pdf.runtime.num_workers == 1
    assert pdf.runtime.log_level == LogLevel.INFO
    assert pdf.table_detector == TableDetectorType.HYBRID
    assert sumcfg.map.theta == 0.8


def test_partial_override_keeps_other_defaults(tmp_path: Path):
    p = _write(tmp_path, """
        pdf_extraction:
          runtime:
            num_workers: 8
            log_level: DEBUG
    """)
    pdf, _ = load_config(p)
    assert pdf.runtime.num_workers == 8
    assert pdf.runtime.log_level == LogLevel.DEBUG
    # untouched defaults survive
    assert pdf.runtime.seed == 42
    assert pdf.docling.do_ocr is False


def test_enum_coerced_from_string(tmp_path: Path):
    p = _write(tmp_path, """
        pdf_extraction:
          table_detector: tatr
          runtime:
            log_level: DEBUG
    """)
    pdf, _ = load_config(p)
    assert pdf.table_detector == TableDetectorType.TATR
    assert pdf.runtime.log_level == LogLevel.DEBUG


def test_path_field_wrapped(tmp_path: Path):
    p = _write(tmp_path, """
        pdf_extraction:
          paths:
            output_root: /tmp/custom-out
    """)
    pdf, _ = load_config(p)
    assert isinstance(pdf.paths.output_root, Path)
    assert str(pdf.paths.output_root) == "/tmp/custom-out"


def test_unknown_field_raises(tmp_path: Path):
    p = _write(tmp_path, """
        pdf_extraction:
          runtime:
            nonexistent: 1
    """)
    with pytest.raises(ValueError, match="Unknown config field"):
        load_config(p)


def test_summarization_section_independent(tmp_path: Path):
    p = _write(tmp_path, """
        summarization:
          map:
            theta: 0.65
          resolve:
            grounding_weight: 0.7
    """)
    _, sumcfg = load_config(p)
    assert sumcfg.map.theta == 0.65
    assert sumcfg.resolve.grounding_weight == 0.7
    # untouched defaults survive
    assert sumcfg.map.reject_theta == 0.2


def test_routing_pins_loaded(tmp_path: Path):
    """B-062 + config-layout-v2 (2026-05-26): cascade-path + single-voter
    policy pins live on ``RoutingConfig`` (moved from ``MapConfig`` in
    layout v2). Both runner builders read ``sum_cfg.routing.*``. Strict
    loader rejects unknown keys, so this also guards against the field
    being renamed/dropped or accidentally re-introduced under ``map``."""
    # Defaults when unspecified → legacy cascade.
    _, default_cfg = load_config(_write(tmp_path, ""))
    assert default_cfg.routing.enable_router is False
    assert default_cfg.routing.router_single_voter_policy == "escalate"
    assert default_cfg.routing.legacy_single_voter_policy == "keep"
    # Explicit override round-trips (bool coercion + str pass-through).
    p = _write(tmp_path, """
        summarization:
          routing:
            enable_router: true
            router_single_voter_policy: keep
            legacy_single_voter_policy: escalate
    """)
    _, sumcfg = load_config(p)
    assert sumcfg.routing.enable_router is True
    assert sumcfg.routing.router_single_voter_policy == "keep"
    assert sumcfg.routing.legacy_single_voter_policy == "escalate"


def test_legacy_map_routing_paths_rejected(tmp_path: Path):
    """Config-layout-v2 (2026-05-26): the three routing-policy fields moved
    from ``MapConfig`` to ``RoutingConfig``. Stale YAML using the v1 path
    (``summarization.map.enable_router`` etc.) must fail loudly through the
    strict loader rather than silently route to a non-existent field.

    This is a deliberate hard break — no compatibility alias was added —
    because (a) this repo has a single canonical YAML, (b) ``configs/run.yaml``
    was migrated in the same diff, and (c) a clear ``Unknown config field``
    error is a stronger forcing function than a deprecation warning that can
    be ignored."""
    import pytest

    for stale_key in (
        "enable_router: true",
        "router_single_voter_policy: keep",
        "legacy_single_voter_policy: escalate",
    ):
        p = _write(tmp_path, f"""
            summarization:
              map:
                {stale_key}
        """)
        with pytest.raises(ValueError, match="Unknown config field: MapConfig"):
            load_config(p)


# ── AgreementConfig — soft-alignment weights (H-EMB-01) ──────────────────────
# Centralisation check (2026-05-26): the four weights flow from
# config.py → run.yaml → SummarizationRunner → EmbeddingSimilarityStrategy
# without hardcoded overrides anywhere on the path. These tests pin every link.


def test_agreement_config_defaults():
    """`AgreementConfig()` carries the production defaults verbatim.

    Hardcoded against the values used by ``_align`` historically — any drift
    here would silently change MAP agreement-gate behaviour without a YAML
    edit. Cite this test if anyone proposes changing a default outside a
    documented calibration sweep."""
    from pipeline.stages.summarization.config import AgreementConfig
    cfg = AgreementConfig()
    assert cfg.tau == 0.15
    assert cfg.count_alpha == 0.25
    assert cfg.reuse_weight == 0.15
    assert cfg.contradiction_weight == 0.20


def test_agreement_config_loaded_from_run_yaml():
    """The shipped ``configs/run.yaml`` block matches the dataclass defaults.

    Catches the case where someone edits ``run.yaml`` in isolation: if the
    YAML drifts from the dataclass default, this test surfaces it — production
    runs follow the YAML, calibration sweeps (``map_theta_sweep``) build
    ``AgreementConfig()`` with dataclass defaults, so a silent split between
    the two would make calibration numbers non-faithful to production."""
    _, sumcfg = load_config(Path("configs/run.yaml"))
    assert sumcfg.agreement.tau == 0.15
    assert sumcfg.agreement.count_alpha == 0.25
    assert sumcfg.agreement.reuse_weight == 0.15
    assert sumcfg.agreement.contradiction_weight == 0.20


def test_agreement_config_yaml_override_propagates(tmp_path: Path):
    """An explicit YAML override changes every loaded value and only those.

    Round-trips a 2× override on `tau` and `contradiction_weight`; asserts
    the two named fields move and the other two stay at default. Guards
    against accidental coupling between fields in the loader (e.g. a typo
    that propagates one value to multiple slots)."""
    p = _write(tmp_path, """
        summarization:
          agreement:
            tau: 0.30
            contradiction_weight: 0.40
    """)
    _, sumcfg = load_config(p)
    assert sumcfg.agreement.tau == 0.30
    assert sumcfg.agreement.contradiction_weight == 0.40
    # untouched defaults survive
    assert sumcfg.agreement.count_alpha == 0.25
    assert sumcfg.agreement.reuse_weight == 0.15


def test_agreement_strategy_consumes_all_four_fields():
    """`EmbeddingSimilarityStrategy.from_config(cfg.agreement)` propagates all
    four fields into the constructed strategy instance.

    This is the final link in the wiring chain:
        run.yaml → AgreementConfig → from_config → strategy._tau etc.

    Production paths (``SummarizationRunner`` at ``runner.py:214`` and
    ``BatchSummarizationRunner`` at ``batch/runner.py:158``, ``:1148``) all
    funnel through this exact call. If this test passes, the runners forward
    the values by construction; no separate runner-build smoke test needed."""
    from pipeline.stages.summarization.agreement.embedding_similarity import (
        EmbeddingSimilarityStrategy,
    )
    from pipeline.stages.summarization.config import AgreementConfig
    custom = AgreementConfig(
        tau=0.42, count_alpha=0.13, reuse_weight=0.07, contradiction_weight=0.99,
    )
    strategy = EmbeddingSimilarityStrategy.from_config(custom, embed_fn=None)
    assert strategy._tau == 0.42
    assert strategy._count_alpha == 0.13
    assert strategy._reuse_weight == 0.07
    assert strategy._contradiction_weight == 0.99


# ── AgreementConfig.scorer_kind — similarity-strategy selector ───────────────
# Promoted 2026-05-26. The previous default (hardcoded
# `EmbeddingSimilarityStrategy` in `runner.py:213`) is preserved as the
# config default; this knob just lets Stage 1's winner be pinned in YAML.


def test_agreement_scorer_kind_default_is_embedding():
    """Default ``scorer_kind="embedding"`` preserves the historical hardcoded choice."""
    from pipeline.stages.summarization.config import AgreementConfig
    assert AgreementConfig().scorer_kind == "embedding"


def test_agreement_scorer_kind_loaded_from_run_yaml():
    """Shipped ``configs/run.yaml`` pins ``scorer_kind: embedding`` (current
    production default). Drift detector for accidental YAML edits."""
    _, sumcfg = load_config(Path("configs/run.yaml"))
    assert sumcfg.agreement.scorer_kind == "embedding"


def test_agreement_scorer_kind_yaml_override_to_hybrid(tmp_path: Path):
    """Explicit YAML override flips the loaded value; untouched fields keep
    their defaults."""
    p = _write(tmp_path, """
        summarization:
          agreement:
            scorer_kind: hybrid
    """)
    _, sumcfg = load_config(p)
    assert sumcfg.agreement.scorer_kind == "hybrid"
    # Other agreement fields untouched.
    assert sumcfg.agreement.tau == 0.15
    assert sumcfg.agreement.contradiction_weight == 0.20


def test_agreement_scorer_kind_invalid_value_raises_at_scorer_build():
    """An invalid ``scorer_kind`` raises ``ValueError`` at scorer construction.

    The dataclass loader is permissive (it accepts any string and stores it
    verbatim) — validation lives in
    ``SemanticAgreementScorer.from_agreement_config``. That's the entry point
    both runners use, so the bad value is caught before any cascade work
    starts. The error message names the bad value and the expected set."""
    import pytest
    from pipeline.stages.summarization.agreement import SemanticAgreementScorer
    from pipeline.stages.summarization.config import AgreementConfig
    bad = AgreementConfig(scorer_kind="bogus")  # loader-allowed; runtime-invalid
    with pytest.raises(ValueError, match=r"scorer_kind=.*bogus.*expected.*embedding.*hybrid"):
        SemanticAgreementScorer.from_agreement_config(bad, embed_fn=None)


def test_agreement_scorer_kind_dispatches_to_correct_strategy():
    """``from_agreement_config`` picks the right strategy class per ``scorer_kind``.

    Verifies the dispatch with both production branches:
      * ``embedding`` → ``EmbeddingSimilarityStrategy``
      * ``hybrid``    → ``HybridStructuredSimilarity``

    Soft-alignment weights flow through ``AgreementConfig`` to either branch
    identically; the hybrid blend weights (``w_category`` etc.) stay at the
    ``HybridStructuredSimilarity`` ctor defaults until a follow-up promotion."""
    from pipeline.stages.summarization.agreement import (
        EmbeddingSimilarityStrategy,
        HybridStructuredSimilarity,
        SemanticAgreementScorer,
    )
    from pipeline.stages.summarization.config import AgreementConfig

    emb = SemanticAgreementScorer.from_agreement_config(
        AgreementConfig(scorer_kind="embedding"), embed_fn=None,
    )
    assert isinstance(emb._strategy, EmbeddingSimilarityStrategy)
    # Soft-alignment weights propagate identically into either branch.
    assert emb._strategy._tau == 0.15

    hyb = SemanticAgreementScorer.from_agreement_config(
        AgreementConfig(scorer_kind="hybrid"), embed_fn=None,
    )
    assert isinstance(hyb._strategy, HybridStructuredSimilarity)
    assert hyb._strategy._tau == 0.15
    # Hybrid blend weights stay at constructor defaults (not yet promoted).
    assert hyb._strategy._w_embedding == 0.40


# ── AgreementConfig.hybrid — HybridStructuredSimilarity blend weights ────────
# Promoted 2026-05-26 as a nested ``HybridConfig`` sub-config. Only active
# when scorer_kind="hybrid"; the embedding scorer ignores the block entirely.
# Convention: weights should sum to 1.0 (not enforced in code).


def test_hybrid_config_defaults():
    """``HybridConfig()`` defaults match the historical
    ``HybridStructuredSimilarity`` ctor: 0.25/0.40/0.25/0.10 — and they sum to 1.0."""
    from pipeline.stages.summarization.config import AgreementConfig, HybridConfig
    h = AgreementConfig().hybrid
    assert isinstance(h, HybridConfig)
    assert h.w_category == 0.25
    assert h.w_embedding == 0.40
    assert h.w_entity == 0.25
    assert h.w_evidence == 0.10
    assert (h.w_category + h.w_embedding + h.w_entity + h.w_evidence) == pytest.approx(1.0)


def test_hybrid_config_loaded_from_run_yaml():
    """Shipped ``configs/run.yaml`` pins the production-default hybrid weights.
    Drift detector for accidental YAML edits that would silently shift the
    Stage 1b baseline."""
    _, sumcfg = load_config(Path("configs/run.yaml"))
    h = sumcfg.agreement.hybrid
    assert h.w_category == 0.25
    assert h.w_embedding == 0.40
    assert h.w_entity == 0.25
    assert h.w_evidence == 0.10


def test_hybrid_config_yaml_override(tmp_path: Path):
    """Explicit YAML override flips the loaded values; untouched weights
    keep their defaults. Round-trips an ``embedding_heavy``-style override."""
    p = _write(tmp_path, """
        summarization:
          agreement:
            hybrid:
              w_embedding: 0.65
              w_evidence: 0.05
    """)
    _, sumcfg = load_config(p)
    h = sumcfg.agreement.hybrid
    assert h.w_embedding == 0.65
    assert h.w_evidence == 0.05
    # untouched defaults survive
    assert h.w_category == 0.25
    assert h.w_entity == 0.25


def test_hybrid_strategy_consumes_hybrid_config():
    """``HybridStructuredSimilarity.from_config`` reads ``agreement_cfg.hybrid``
    and propagates each weight into the constructed strategy. Plus a back-compat
    spot-check: an explicit ``w_category=…`` kwarg overrides only that field
    and leaves the other three at the config values (callers that built the
    strategy programmatically before this promotion shouldn't break)."""
    from pipeline.stages.summarization.agreement.hybrid_structured import (
        HybridStructuredSimilarity,
    )
    from pipeline.stages.summarization.config import AgreementConfig, HybridConfig

    cfg = AgreementConfig(
        hybrid=HybridConfig(
            w_category=0.50, w_embedding=0.30, w_entity=0.15, w_evidence=0.05,
        ),
    )
    strat = HybridStructuredSimilarity.from_config(cfg, embed_fn=None)
    assert strat._w_category == 0.50
    assert strat._w_embedding == 0.30
    assert strat._w_entity == 0.15
    assert strat._w_evidence == 0.05
    # Soft-alignment weights still flow through unchanged.
    assert strat._tau == 0.15

    # Per-field override via explicit kwarg (back-compat).
    strat_override = HybridStructuredSimilarity.from_config(
        cfg, embed_fn=None, w_category=0.99,
    )
    assert strat_override._w_category == 0.99
    # The other three keep their config values (not the ctor defaults).
    assert strat_override._w_embedding == 0.30
    assert strat_override._w_entity == 0.15
    assert strat_override._w_evidence == 0.05


def test_embedding_strategy_ignores_hybrid_config():
    """``EmbeddingSimilarityStrategy.from_config`` does not consume the hybrid
    block — switching ``hybrid.w_*`` to extreme values must not perturb the
    embedding strategy's state. Asserts the strategy's soft-alignment weights
    are identical whether or not the hybrid block is at default."""
    from pipeline.stages.summarization.agreement.embedding_similarity import (
        EmbeddingSimilarityStrategy,
    )
    from pipeline.stages.summarization.config import AgreementConfig, HybridConfig

    cfg_default = AgreementConfig()
    cfg_extreme_hybrid = AgreementConfig(
        hybrid=HybridConfig(
            w_category=0.99, w_embedding=0.0, w_entity=0.0, w_evidence=0.01,
        ),
    )

    s_default = EmbeddingSimilarityStrategy.from_config(cfg_default, embed_fn=None)
    s_extreme = EmbeddingSimilarityStrategy.from_config(cfg_extreme_hybrid, embed_fn=None)

    # The embedding strategy's state depends only on the soft-alignment quartet,
    # which is identical between the two configs.
    assert s_default._tau == s_extreme._tau
    assert s_default._count_alpha == s_extreme._count_alpha
    assert s_default._reuse_weight == s_extreme._reuse_weight
    assert s_default._contradiction_weight == s_extreme._contradiction_weight
    # And there's no hybrid-related private state on the embedding strategy.
    assert not hasattr(s_default, "_w_category")


# ── AgreementConfig.force_escalate_on_polarity_conflict ─────────────────────
# Promoted 2026-05-26. B-051 hard-fail policy. Default True preserves the
# safety guard verbatim; False is the ablation setting (still records the
# marker, just doesn't override the scorer decision). Behavioural tests for
# both branches live in test_b051_hard_fail_polarity.py.


def test_polarity_flag_default_is_true():
    """Default ``force_escalate_on_polarity_conflict=True`` preserves the B-051 safety guard."""
    from pipeline.stages.summarization.config import AgreementConfig
    assert AgreementConfig().force_escalate_on_polarity_conflict is True


def test_polarity_flag_loaded_from_run_yaml():
    """Shipped ``configs/run.yaml`` pins the flag to ``true`` (current production
    safety default). Drift detector for accidental YAML edits."""
    _, sumcfg = load_config(Path("configs/run.yaml"))
    assert sumcfg.agreement.force_escalate_on_polarity_conflict is True


def test_polarity_flag_yaml_override_to_false(tmp_path: Path):
    """Explicit YAML override flips the loaded value; untouched fields keep
    their defaults. Round-trip uses YAML ``false`` (lowercase) — yaml.safe_load
    coerces this to Python ``False`` cleanly."""
    p = _write(tmp_path, """
        summarization:
          agreement:
            force_escalate_on_polarity_conflict: false
    """)
    _, sumcfg = load_config(p)
    assert sumcfg.agreement.force_escalate_on_polarity_conflict is False
    # Other agreement fields untouched.
    assert sumcfg.agreement.tau == 0.15
    assert sumcfg.agreement.scorer_kind == "embedding"


def test_normalize_extra_synonyms_loaded_as_mapping(tmp_path: Path):
    """B-037: NormalizeConfig.extra_synonyms is a `dict[str, str]` and must
    pass through the loader without being mistaken for a nested dataclass."""
    p = _write(tmp_path, """
        summarization:
          normalize:
            extra_synonyms:
              acme corp: ACME
              foo bar: FOO
    """)
    _, sumcfg = load_config(p)
    assert sumcfg.normalize.extra_synonyms == {"acme corp": "ACME", "foo bar": "FOO"}


def test_tatr_render_dpi_overridable(tmp_path: Path):
    """B-034: TATRConfig.render_dpi is now a real config knob (was hardcoded
    `_RENDER_DPI = 150` at module level)."""
    p = _write(tmp_path, """
        pdf_extraction:
          tatr:
            render_dpi: 300
    """)
    pdf, _ = load_config(p)
    assert pdf.tatr.render_dpi == 300


# ── B-028: deleted DatabaseConfig sub-fields fail loudly ──────────────────────
# `schema`, `create_tables_if_missing`, `batch_size`, `connect_timeout_sec`
# never had consumers (`PostgresDatabaseIngester` only accepts `db_url`). They
# were removed in the 2026-05-15 Tier 1 config audit rather than wired. Any
# YAML still referencing them must fail at load time, not silently no-op.
@pytest.mark.parametrize(
    "dead_field, value",
    [
        ("schema", '"custom"'),
        ("create_tables_if_missing", "true"),
        ("batch_size", "500"),
        ("connect_timeout_sec", "30"),
    ],
)
def test_deleted_database_keys_rejected(tmp_path: Path, dead_field: str, value: str):
    p = _write(tmp_path, f"""
        pdf_extraction:
          database:
            {dead_field}: {value}
    """)
    with pytest.raises(ValueError, match=f"DatabaseConfig\\.{dead_field}"):
        load_config(p)


# ── RuntimeConfig kill-switch fields still load (sanity) ──────────────────────
# Phase-1 audit incorrectly flagged these as dead. They are kill-switch
# features consumed at runner.py:375-378 (blacklist_if_rows_exceed) and
# runner.py:603+ (multi_source_crops). Defaults are None/False, so the
# default code path is unchanged — but the config knob must still load.
def test_runtime_kill_switch_fields_still_load(tmp_path: Path):
    p = _write(tmp_path, """
        pdf_extraction:
          runtime:
            blacklist_if_rows_exceed: 5000
            multi_source_crops: true
    """)
    pdf, _ = load_config(p)
    assert pdf.runtime.blacklist_if_rows_exceed == 5000
    assert pdf.runtime.multi_source_crops is True


# ── Seed wiring: configured seed reaches PipelineRunner._seed_pipeline ────────
def test_seed_reaches_seed_pipeline(tmp_path: Path, monkeypatch):
    """Mock-based wiring test — does NOT touch global RNG state.

    Asserts the seed value loaded from YAML is the value the runner passes
    to its internal seeding helper. Implementation detail of how the helper
    seeds (`random.seed`, `numpy.random.seed`, `torch.manual_seed`) is out
    of scope for this test.
    """
    from pipeline.stages.pdf_text_extraction import runner as runner_mod

    captured = {}

    def fake_seed_pipeline(self):
        captured["seed"] = self._cfg.runtime.seed

    monkeypatch.setattr(runner_mod.PipelineRunner, "_seed_pipeline", fake_seed_pipeline)

    p = _write(tmp_path, """
        pdf_extraction:
          runtime:
            seed: 99
    """)
    pdf, _ = load_config(p)
    runner_mod.PipelineRunner(pdf)
    assert captured["seed"] == 99
