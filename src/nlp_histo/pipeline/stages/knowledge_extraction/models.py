"""
Pydantic schemas for all pipeline stages:
  MAP → NORMALIZE → GROUP → CANONICALIZE → RELATE → RESOLVE
"""
from __future__ import annotations

from enum import Enum
from typing import List, Literal

import logging

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from .enum_logging import log_bad_finding, log_enum_observation

_log = logging.getLogger(__name__)


# Schema / prompt versioning
# Bump these whenever the MAP output schema or prompt changes in a
# behaviour-affecting way. They are included in cache keys / run metadata so
# stale outputs don't collide with new ones.
MAP_SCHEMA_VERSION: str = "map_v10_nli_and_voter_config_in_key"
MAP_PROMPT_VERSION: str = "map_prompt_v5_expression_absent_vs_negative"
MAP_STAGE_NAME:     str = "map"

# CANONICALIZE behavioural-version stamp (B-049). Bumped when canonicalization
# semantics change in a way that invalidates cached summaries. Fed into
# `compute_pipeline_config_hash` via the `thresholds` dict on both runners so
# cached `out/summaries/summaries/*.json` re-runs cleanly on the next
# invocation. Not user-tunable — change the value, not the field.
CANONICALIZE_DIRECTION_POLICY_VERSION: str = "per_direction_no_folding_v2"

# MAP agreement-gate behavioural-version stamp (B-051). Bumped when the MAP
# cascade KEEP/ESCALATE policy changes in a way that invalidates cached
# per-paper results. Fed into `compute_pipeline_config_hash` on both runners.
# Companion to the `MAP_SCHEMA_VERSION` bump that invalidates the chunk-level
# `PipelineCache` — both are required to cover both cache layers.
MAP_AGREEMENT_POLICY_VERSION: str = "polarity_hard_fail_v1"


# Cascade signature helper
def compute_finding_id(
    pmcid: str,
    chunk_id: str,
    position_in_chunk: int,
    claim: str,
) -> str:
    """Stable 12-char hash identifying a MAP Finding.

    Deterministic over (pmcid, chunk_id, position_in_chunk, normalized claim).
    Whitespace in the claim is collapsed and case-folded so cosmetic
    re-serialisation does not change the id.
    """
    import hashlib
    normalized_claim = " ".join((claim or "").lower().split())
    payload = f"{pmcid}|{chunk_id}|{position_in_chunk}|{normalized_claim}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def compute_cascade_signature(voter_specs: list[tuple[str, str]]) -> str:
    """Stable short hash of an ordered (provider, model) sequence.

    Used in MAP cache keys and run-artifact metadata so the same chunk under a
    different cascade configuration is considered a cache miss. This signature
    captures identity only — provider + model. Sampler settings
    (temperature, top_p) are captured separately by
    ``compute_voter_config_hash`` so a temperature tweak under the same
    (provider, model) still invalidates the cache.
    """
    import hashlib
    import json
    payload = json.dumps([list(t) for t in voter_specs], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def compute_voter_config_hash(
    voter_specs_with_sampling: list[tuple[str, str, float, float | None]],
) -> str:
    """Stable short hash of an ordered (provider, model, temperature, top_p) sequence.

    Companion to ``compute_cascade_signature`` — captures sampler settings so
    editing ``voter_configs.py`` (e.g. flipping L1 temperature 0.0 → 0.2) under
    the same model set produces a different MAP cache key. ``top_p`` may be
    ``None`` for providers that don't expose it; the hash encodes that
    explicitly via ``json.dumps`` so ``None`` and ``1.0`` are distinguishable.

    Temperatures are rounded to 3 dp before hashing to avoid float-identity
    misses in the same way as the in-run voter cache (see
    ``map_stage._voter_cache_key``).
    """
    import hashlib
    import json
    payload = json.dumps(
        [
            [provider, model, round(float(temperature), 3), top_p]
            for provider, model, temperature, top_p in voter_specs_with_sampling
        ],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class MapRunMetadata(BaseModel):
    """Provenance attached to every MAP cache entry and run artifact.

    The fields are deliberately strings so they can round-trip through JSON
    without any custom (de)serialisers. ``cascade_profile`` is the human-
    readable profile name (``default`` / ``smoke_haiku`` / ``dev_sonnet`` /
    ``custom``); ``cascade_signature`` is the deterministic content hash that
    actually drives cache invalidation.

    ``provider`` and ``model`` identify the producer of the stored result —
    in the cascade this is the voter or escalation model whose output was
    selected, not the cascade as a whole.

    ``nli_model_id`` is the active grounding/relation NLI checkpoint
    (``$NLP_HISTO_NLI_MODEL`` resolved at process start). Stored on each MAP
    entry because grounding scores are computed during MAP and persisted with
    the result — swapping NLI models without invalidating those scores would
    serve stale entailment judgements as if fresh.

    ``voter_config_hash`` is the deterministic hash of
    (provider, model, temperature, top_p) tuples across all cascade levels.
    Defaults to empty string for backward-compat with on-disk entries written
    before the field existed; the cache lookup checks both fields and treats
    a mismatch as a miss.
    """
    schema_version:     str = MAP_SCHEMA_VERSION
    prompt_version:     str = MAP_PROMPT_VERSION
    stage_name:         str = MAP_STAGE_NAME
    provider:           str
    model:              str
    cascade_profile:    str = "custom"
    cascade_signature:  str
    nli_model_id:       str = ""
    voter_config_hash:  str = ""


# Phase 1: new enums and scope model

class DirectionEnum(str, Enum):
    positive      = "positive"
    negative      = "negative"
    absent        = "absent"
    partial       = "partial"
    unclear       = "unclear"
    no_direction  = "no_direction"


# Shared direction-class constants. Used by canonicalize_stage.py (group-level
# is_conflicted computation), relate_stage.py (non-polarity skip), and
# corpus_relate.py (cross-paper non-polarity skip). One source of truth so the
# polarity-class definition can't drift between sites.
#
# `partial` is included in POLARITY_BEARING_DIRS for the group-level
# is_conflicted flag — meaning `positive + partial → is_conflicted=True`. The
# question of whether `partial` should really count as a contradicting
# polarity (vs. a qualifier on `positive`) is owned by B-025 and is not a
# final decision in B-049.
POLARITY_BEARING_DIRS: frozenset[str] = frozenset({"positive", "negative", "absent", "partial"})
NON_POLARITY_DIRS:     frozenset[str] = frozenset({"unclear", "no_direction"})


def direction_value(d: object) -> str:
    """Normalize a direction to its string value.

    Handles three input shapes that all flow into the same gates:
    * `DirectionEnum` member — live Pydantic-parsed CanonicalRule.direction
    * Raw string — metadata dicts read from cached JSON in corpus_relate.py
    * `None` — legacy/cached rows; treated as `"unclear"` by the project
      convention used at canonicalize_stage.py:60 and :110.

    A naive `direction in NON_POLARITY_DIRS` check would silently fail on
    `None` (None not in any set) and would work only by accident on raw
    strings (because `DirectionEnum(str, Enum)` is a str subclass). This
    helper makes the normalization explicit at every gate site.
    """
    if d is None:
        return "unclear"
    if hasattr(d, "value"):  # DirectionEnum or any other Enum
        return d.value  # type: ignore[attr-defined]
    return str(d)


class RelationTypeEnum(str, Enum):
    has_feature        = "has_feature"        # entity exhibits / shows / has a histological feature
    expression         = "expression"         # biomarker/protein expression level
    prognostic         = "prognostic"         # entity predicts or associates with a clinical outcome
    comparative        = "comparative"        # relative frequency or likelihood vs another group
    demographic        = "demographic"        # age, sex, prevalence in a population
    treatment_response = "treatment_response" # response to a treatment regimen
    unclear            = "unclear"            # relation type genuinely not inferable



_SCOPE_FIELDS_DEFAULTS: dict[str, object] = {
    "disease_subtype":   None,
    "cohort_n":          None,
    "assay_method":      None,
    "biomarker_cutoff":  None,
    "tissue_site":       None,
    "treatment_context": None,
    "endpoint":          None,
    "study_design":      None,
    "scope_parsed":      False,
}


class FindingScope(BaseModel):
    # No JSON-schema defaults — every sub-field is required when scope is
    # non-null so the OpenAI strict schema is satisfied. The model is
    # instructed to emit null for sub-fields it cannot determine.
    #
    # Internal Python callers may still construct an empty scope via
    # ``FindingScope.empty()``; legacy cached payloads with missing keys are
    # filled in by ``_fill_legacy_defaults`` below.
    disease_subtype:   str | None
    cohort_n:          int | None
    assay_method:      str | None
    biomarker_cutoff:  str | None
    tissue_site:       str | None
    treatment_context: str | None
    endpoint:          str | None
    study_design:      str | None
    scope_parsed:      bool

    @classmethod
    def empty(cls) -> "FindingScope":
        return cls(**_SCOPE_FIELDS_DEFAULTS)

    @model_validator(mode="before")
    @classmethod
    def _fill_legacy_defaults(cls, data: object) -> object:
        """Tolerate cached payloads / internal callers that omit sub-fields."""
        if isinstance(data, dict):
            for key, default in _SCOPE_FIELDS_DEFAULTS.items():
                if key not in data:
                    data[key] = default
        return data

    @field_validator(
        "disease_subtype", "assay_method", "biomarker_cutoff",
        "tissue_site", "treatment_context", "endpoint", "study_design",
        mode="before",
    )
    @classmethod
    def _empty_string_to_none(cls, v: object) -> object:
        if v == "null" or v == "":
            return None
        return v

    @model_validator(mode="after")
    def _compute_scope_parsed(self) -> "FindingScope":
        # scope_parsed is trivially derivable: true iff any sub-field is non-null.
        # Overriding whatever the LLM emitted removes one more thing it can get
        # wrong; the field stays in the schema only because OpenAI strict mode
        # requires every property to be present.
        self.scope_parsed = any(
            getattr(self, k) is not None
            for k in _SCOPE_FIELDS_DEFAULTS
            if k != "scope_parsed"
        )
        return self


# MAP output

_CATEGORY_VALID: tuple[str, ...] = (
    "morphology", "IHC", "molecular_genetics", "staging",
    "treatment", "prognosis", "demographic",
)
# Case-insensitive lookup for category repair (e.g. "Demographic" → "demographic",
# "ihc" → "IHC"). Only `IHC` differs from its lowercase form, but LLMs emit
# title-case and uppercase variants for every label.
_CATEGORY_CANONICAL_BY_LOWER: dict[str, str] = {c.lower(): c for c in _CATEGORY_VALID}

_CATEGORY_ALIASES: dict[str, str] = {
    "demographics": "demographic",  # legacy LLM alias from pre-alignment prompts
}

# `category` and `relation_type` are orthogonal axes, but LLMs frequently
# bleed a category-name into relation_type ("prognosis", "molecular_genetics",
# "IHC", "morphology"). Alias-repair recovers the finding rather than coercing
# to "unclear" (which would drop it at GROUP).
#
# Keys are lowercase; case-folded raw values match via the case-folded branch
# of `_coerce_invalid_relation_type`.
#
# "staging" is intentionally not aliased: descriptive (has_feature) vs
# outcome-driven (prognostic) split needs claim context. Falls through to
# "unclear" with reason="cross_field_bleed" so loss stays observable.
_RELATION_TYPE_ALIASES: dict[str, str] = {
    "prognosis":          "prognostic",
    "treatment":          "treatment_response",
    "morphology":         "has_feature",
    "ihc":                "expression",
    "molecular_genetics": "expression",
}

# Lowercase set of category names used to detect cross-field bleed during
# relation_type coercion. Drives the `reason="cross_field_bleed"` log so we
# can count exactly how many findings are lost (or recovered) due to this bug.
_CATEGORY_NAMES_LOWER: frozenset[str] = frozenset(c.lower() for c in _CATEGORY_VALID)

# Case-insensitive lookup for valid confidence literals.
_CONFIDENCE_VALID: tuple[str, ...] = ("high", "medium", "low")
_CONFIDENCE_LOWER_SET: frozenset[str] = frozenset(_CONFIDENCE_VALID)

# LLMs reach for synonyms outside the {high, medium, low} vocabulary,
# especially "moderate" (English-ese for medium) and "very high". Without an
# alias map these silently drop the whole Finding via Literal validation.
# Keys are lowercased so the case-repair branch matches them after case-fold.
_CONFIDENCE_ALIASES: dict[str, str] = {
    "moderate":       "medium",
    "uncertain":      "low",
    "weak":           "low",
    "strong":         "high",
    "very_high":      "high",
    "very high":      "high",
    "v_high":         "high",
    "vhigh":          "high",
    "very_low":       "low",
    "very low":       "low",
    "v_low":          "low",
    "vlow":           "low",
}

# LLMs occasionally reach for hedging words outside the DirectionEnum vocabulary
# ("maybe", "possibly", "perhaps"). Without an alias map these silently coerce
# to `unclear` via the unknown-value branch and lose the hedging signal in the
# raw_direction column (still preserved on `_raw_direction` per B-015).
# Mapping them all to `unclear` is the conservative read — they signal
# uncertainty, not polarity. Keys are lowercased so the case-fold branch matches.
_DIRECTION_ALIASES: dict[str, str] = {
    "maybe":    "unclear",
    "possibly": "unclear",
    "perhaps":  "unclear",
    "likely":   "unclear",
    "unknown":  "unclear",
    "none":     "no_direction",  # LLMs occasionally emit the literal word "none"
    "n/a":      "no_direction",
    "na":       "no_direction",
}


class Finding(BaseModel):
    # `category` and `relation_type` both use the singular "demographic".
    # Legacy LLM output saying "demographics" is repaired by _repair_category_alias.
    #
    # Strict-schema invariant: no field has a Pydantic default. Optional fields
    # are typed as `X | None` so Pydantic emits anyOf:[X,null] and the OpenAI
    # strict schema can list every property in `required`.
    category: Literal["morphology", "IHC", "molecular_genetics", "staging", "treatment", "prognosis", "demographic"]
    claim: str = Field(description="Atomic, telegraphic medical fact")
    evidence: List[str] = Field(description="Citation IDs, e.g. ['S1|PMC123|456']")
    confidence: Literal["high", "medium", "low"]
    verbatim_support: str = Field(description="Exact quote from source text")
    subject_entity:    str | None
    outcome_entity:    str | None
    relation_type:     RelationTypeEnum
    direction:         DirectionEnum
    scope:             FindingScope | None
    grounding_score:   float | None

    # Pipeline-assigned stable identifier. Kept as a PrivateAttr so it never
    # appears in the OpenAI strict schema generated from this model, and so
    # cached LLM outputs that omit it parse cleanly. The runner fills it via
    # `set_finding_id` after MAP completes; persistence reads it via the
    # `finding_id` property.
    _finding_id: str | None = PrivateAttr(default=None)

    # Raw LLM-emitted values, captured before any enum coercion or alias
    # repair. PrivateAttrs so they never appear in the OpenAI strict schema.
    # Populated by the wrap-validator below; persistence reads them via the
    # raw_* properties and writes them to sum_map_findings.raw_*.
    _raw_relation_type: str | None = PrivateAttr(default=None)
    _raw_direction:     str | None = PrivateAttr(default=None)
    _raw_category:      str | None = PrivateAttr(default=None)

    @property
    def finding_id(self) -> str | None:
        return self._finding_id

    def set_finding_id(self, value: str) -> None:
        self._finding_id = value

    @property
    def raw_relation_type(self) -> str | None:
        return self._raw_relation_type

    @property
    def raw_direction(self) -> str | None:
        return self._raw_direction

    @property
    def raw_category(self) -> str | None:
        return self._raw_category

    @model_validator(mode="wrap")
    @classmethod
    def _capture_raw_then_validate(cls, data, handler):
        """Stash raw LLM-emitted relation_type/direction/category before any
        coercion or alias repair runs, so the original strings survive on the
        instance even when the field validators coerce them to enum members."""
        raw_relation = None
        raw_direction = None
        raw_category = None
        if isinstance(data, dict):
            raw_relation = data.get("relation_type")
            raw_direction = data.get("direction")
            raw_category = data.get("category")
        model = handler(data)
        if raw_relation is not None:
            model._raw_relation_type = str(raw_relation)
        if raw_direction is not None:
            model._raw_direction = str(raw_direction)
        if raw_category is not None:
            model._raw_category = str(raw_category)
        return model

    @model_validator(mode="before")
    @classmethod
    def _fill_legacy_defaults(cls, data: object) -> object:
        """Tolerate cached payloads that omit Phase-1 optional keys."""
        if isinstance(data, dict):
            data.setdefault("subject_entity", None)
            data.setdefault("outcome_entity", None)
            data.setdefault("scope", None)
            data.setdefault("grounding_score", None)
            data.setdefault("relation_type", "unclear")
            data.setdefault("direction", "no_direction")
        return data

    @field_validator("category", mode="before")
    @classmethod
    def _repair_category_alias(cls, v: object) -> object:
        if not isinstance(v, str):
            return v
        # Exact valid value — pass through.
        if v in _CATEGORY_VALID:
            return v
        # Alias map (e.g. "demographics" → "demographic").
        if v in _CATEGORY_ALIASES:
            repaired = _CATEGORY_ALIASES[v]
            _log.warning("Repaired category alias %r → %r", v, repaired)
            log_enum_observation(
                field_name="category", raw_value=v,
                valid_values=list(_CATEGORY_VALID),
                coerced_value=repaired, reason="alias_repair",
            )
            return repaired
        # Case-insensitive match against valid values (e.g. "Demographic" →
        # "demographic", "ihc" → "IHC").
        lowered = v.lower()
        if lowered in _CATEGORY_CANONICAL_BY_LOWER:
            repaired = _CATEGORY_CANONICAL_BY_LOWER[lowered]
            _log.warning("Repaired category casing %r → %r", v, repaired)
            log_enum_observation(
                field_name="category", raw_value=v,
                valid_values=list(_CATEGORY_VALID),
                coerced_value=repaired, reason="case_repair",
            )
            return repaired
        # Lowercase form may match an alias (e.g. "Demographics" → "demographics" → "demographic").
        if lowered in _CATEGORY_ALIASES:
            repaired = _CATEGORY_ALIASES[lowered]
            _log.warning("Repaired category alias (case-folded) %r → %r", v, repaired)
            log_enum_observation(
                field_name="category", raw_value=v,
                valid_values=list(_CATEGORY_VALID),
                coerced_value=repaired, reason="alias_repair",
            )
            return repaired
        return v

    @field_validator("relation_type", mode="before")
    @classmethod
    def _coerce_invalid_relation_type(cls, v: object) -> object:
        valid = [e.value for e in RelationTypeEnum]
        valid_set = set(valid)
        if v is None or v == "null" or v == "":
            _log.warning("Missing relation_type — coercing to 'unclear'")
            log_enum_observation(
                field_name="relation_type", raw_value=v, valid_values=valid,
                coerced_value=RelationTypeEnum.unclear.value, reason="missing",
            )
            return RelationTypeEnum.unclear
        if not isinstance(v, str):
            return v
        # Exact valid value — pass through.
        if v in valid_set:
            return v
        # Cross-field bleed: raw value is actually a category name. Tag the log
        # with `cross_field_bleed` regardless of whether we recover via alias —
        # `coerced_value` distinguishes recovered (valid enum) vs lost (unclear).
        lowered = v.lower()
        is_bleed = lowered in _CATEGORY_NAMES_LOWER
        # Alias map (e.g. "prognosis" → "prognostic").
        if v in _RELATION_TYPE_ALIASES:
            repaired = _RELATION_TYPE_ALIASES[v]
            _log.warning("Repaired relation_type alias %r → %r", v, repaired)
            log_enum_observation(
                field_name="relation_type", raw_value=v, valid_values=valid,
                coerced_value=repaired,
                reason="cross_field_bleed" if is_bleed else "alias_repair",
            )
            return repaired
        # Case-insensitive match (e.g. "Has_Feature" → "has_feature").
        if lowered in valid_set:
            _log.warning("Repaired relation_type casing %r → %r", v, lowered)
            log_enum_observation(
                field_name="relation_type", raw_value=v, valid_values=valid,
                coerced_value=lowered, reason="case_repair",
            )
            return lowered
        # Lowercase form may match an alias.
        if lowered in _RELATION_TYPE_ALIASES:
            repaired = _RELATION_TYPE_ALIASES[lowered]
            _log.warning("Repaired relation_type alias (case-folded) %r → %r", v, repaired)
            log_enum_observation(
                field_name="relation_type", raw_value=v, valid_values=valid,
                coerced_value=repaired,
                reason="cross_field_bleed" if is_bleed else "alias_repair",
            )
            return repaired
        _log.warning("Unknown relation_type %r — coercing to 'unclear'", v)
        log_enum_observation(
            field_name="relation_type", raw_value=v, valid_values=valid,
            coerced_value=RelationTypeEnum.unclear.value,
            reason="cross_field_bleed" if is_bleed else "unknown_value",
        )
        return RelationTypeEnum.unclear

    @field_validator("direction", mode="before")
    @classmethod
    def _coerce_invalid_direction(cls, v: object) -> object:
        valid = [e.value for e in DirectionEnum]
        valid_set = set(valid)
        # Old caches and lenient producers may emit null/""/"null" — those mean
        # "direction does not apply": coerce to no_direction.
        if v is None or v == "null" or v == "":
            log_enum_observation(
                field_name="direction", raw_value=v, valid_values=valid,
                coerced_value=DirectionEnum.no_direction.value,
                reason="null_to_no_direction",
            )
            return DirectionEnum.no_direction
        if not isinstance(v, str):
            return v
        if v in valid_set:
            return v
        # Case-insensitive match (e.g. "Positive" → "positive", "NEGATIVE" → "negative").
        lowered = v.lower()
        if lowered in valid_set:
            _log.warning("Repaired direction casing %r → %r", v, lowered)
            log_enum_observation(
                field_name="direction", raw_value=v, valid_values=valid,
                coerced_value=lowered, reason="case_repair",
            )
            return lowered
        # Hedging-word alias repair ("maybe"/"possibly"/… → "unclear",
        # "none"/"n/a" → "no_direction"). Runs after case repair so the alias
        # table is case-insensitive without each entry needing both casings.
        if lowered in _DIRECTION_ALIASES:
            repaired = _DIRECTION_ALIASES[lowered]
            _log.warning("Repaired direction alias %r → %r", v, repaired)
            log_enum_observation(
                field_name="direction", raw_value=v, valid_values=valid,
                coerced_value=repaired, reason="alias_repair",
            )
            return repaired
        _log.warning("Unknown direction %r — coercing to 'unclear'", v)
        log_enum_observation(
            field_name="direction", raw_value=v, valid_values=valid,
            coerced_value=DirectionEnum.unclear.value, reason="unknown_value",
        )
        return DirectionEnum.unclear

    @field_validator("confidence", mode="before")
    @classmethod
    def _repair_confidence_casing(cls, v: object) -> object:
        if not isinstance(v, str):
            return v
        if v in _CONFIDENCE_LOWER_SET:
            return v
        lowered = v.lower()
        if lowered in _CONFIDENCE_LOWER_SET:
            _log.warning("Repaired confidence casing %r → %r", v, lowered)
            log_enum_observation(
                field_name="confidence", raw_value=v,
                valid_values=list(_CONFIDENCE_VALID),
                coerced_value=lowered, reason="case_repair",
            )
            return lowered
        if lowered in _CONFIDENCE_ALIASES:
            repaired = _CONFIDENCE_ALIASES[lowered]
            _log.warning("Repaired confidence alias %r → %r", v, repaired)
            log_enum_observation(
                field_name="confidence", raw_value=v,
                valid_values=list(_CONFIDENCE_VALID),
                coerced_value=repaired, reason="alias_repair",
            )
            return repaired
        return v

    @field_validator("scope", "subject_entity", "outcome_entity",
                     "grounding_score", mode="before")
    @classmethod
    def _null_string_to_none(cls, v: object) -> object:
        return None if v == "null" else v


class AuditMetadata(BaseModel):
    sentences_analyzed: int
    sentences_cited: List[str]
    pmcids_referenced: List[str]
    uncited_sentences: List[str]


class AuditableSummary(BaseModel):
    """Output of the MAP stage for a single chunk."""
    chunk_id: str
    findings: List[Finding]
    summary_text: str
    audit_metadata: AuditMetadata

    @field_validator("findings", mode="before")
    @classmethod
    def _drop_invalid_findings(cls, v: object) -> object:
        if not isinstance(v, list):
            return v
        valid_categories = [
            "morphology", "IHC", "molecular_genetics", "staging",
            "treatment", "prognosis", "demographic",
        ]
        clean = []
        for item in v:
            try:
                Finding.model_validate(item)
                clean.append(item)
            except Exception as exc:
                _log.warning("Dropping malformed finding: %s", exc)
                log_bad_finding(raw=item, error=str(exc))
                # Surface invalid category values in structured form so they
                # show up in enum_observations.jsonl aggregations alongside
                # relation_type / direction coercions. category="expression"
                # and similar relation_type-bleed values would otherwise only
                # appear in bad_findings.jsonl as free-form Pydantic errors.
                if isinstance(item, dict):
                    raw_category = item.get("category")
                    if (
                        isinstance(raw_category, str)
                        and raw_category not in valid_categories
                        and raw_category not in _CATEGORY_ALIASES
                    ):
                        log_enum_observation(
                            field_name="category",
                            raw_value=raw_category,
                            valid_values=valid_categories,
                            coerced_value=None,
                            reason="invalid_literal_dropped",
                        )
        return clean


# Phase 2: NORMALIZE output

class SourceSpan(BaseModel):
    """Provenance pointer to a single sentence in the source document."""
    sentence_id:     str        # e.g. "S3"
    pmcid:           str
    text_element_id: int
    verbatim:        str


class NormalFinding(BaseModel):
    """
    Output of the NORMALIZE stage.

    One NormalFinding represents a single distinct claim (subject, outcome,
    direction) after deduplication across voters and entity normalization.
    Multiple source spans are merged when they share the same claim and source
    sentence; different claims from the same text element remain separate.
    """
    normal_id:            str               # "NF_{pmcid}_{sha8(subject+outcome+relation_type+te_ids)}"
    subject_entity:       str | None        # normalized; None if MAP did not extract
    outcome_entity:       str | None        # normalized; None if MAP did not extract
    subject_cui:          str | None = None  # UMLS CUI resolved during NORMALIZE
    outcome_cui:          str | None = None  # UMLS CUI resolved during NORMALIZE
    relation_type:        RelationTypeEnum  # groupability key; unclear = non-groupable
    direction:            DirectionEnum | None
    category:             Literal["morphology", "IHC", "molecular_genetics", "staging", "treatment", "prognosis", "demographic"]
    predicate_text:       str               # representative claim text
    scope:                FindingScope      # from the best-grounded source finding
    source_finding_ids:   List[str]         # Finding.finding_id values merged here (reserved for Phase 3+)
    evidence:             List[SourceSpan]  # deduplicated source spans
    pmcids:               List[str]         # unique PMCIDs
    mean_grounding_score: float | None      # mean of grounding_score across source findings


# Phase 3: GROUP output

class FindingGroup(BaseModel):
    """
    Output of the GROUP stage.

    Contains only groupable NormalFindings — those where subject_entity,
    outcome_entity are non-None and relation_type is not unclear.
    Non-groupable findings are excluded before GROUP runs and remain in
    the normalized-findings cache.

    All NormalFindings about the same (subject, outcome, relation_type) are
    collected here regardless of direction.  Mixed directions within a group
    are intentional — contradictions surface as relations in Phase 5 RELATE,
    not as separate groups.
    """
    group_id:             str                    # "GRP_{sha8(pmcid)}_{sha8(subject)}_{sha8(outcome)}_{relation_type}_{sha8(category)}"
    subject_entity:       str                    # always non-None (groupability invariant)
    outcome_entity:       str                    # always non-None (groupability invariant)
    relation_type:        RelationTypeEnum       # grouping key
    category:             Literal["morphology", "IHC", "molecular_genetics", "staging", "treatment", "prognosis", "demographic"]
    member_ids:           List[str]              # NormalFinding.normal_id values
    direction_counts:     dict[str, int]         # direction.value → count (None keys stored as "unclear")
    scope_heterogeneity:  float                  # 0.0 = all scope fields agree; 1.0 = maximum variation


# Phase 2 forward-looking model (not wired in Phase 1)

class AtomicFinding(BaseModel):
    """
    Future Phase 2 runtime object. Defined here for schema stability.
    Not imported or used anywhere in Phase 1 runner/filter code.
    """
    finding_id:       str
    category:         Literal["morphology", "IHC", "molecular_genetics", "staging", "treatment", "prognosis", "demographic"]
    claim:            str
    subject_entity:   str | None
    outcome_entity:   str | None
    direction:        DirectionEnum | None
    scope:            FindingScope        # required (not Optional) — Phase 2 invariant
    confidence:       Literal["high", "medium", "low"]
    evidence:         List[str]
    verbatim_support: str
    grounding_score:  float | None


# Phase 4: CANONICALIZE output

class CanonicalRule(BaseModel):
    """
    Output of the CANONICALIZE stage for one direction-bin of a FindingGroup.

    One CanonicalRule represents the best-grounded, direction-consistent
    predicate text that can be generalised across member NormalFindings.
    """
    canonical_id:        str                   # "CR_{sha8(group_id)}_{direction}"
    group_id:            str                   # source FindingGroup.group_id
    subject_entity:      str
    outcome_entity:      str
    relation_type:       RelationTypeEnum
    direction:           DirectionEnum | None  # the single direction this rule represents
    predicate_text:      str                   # highest mean_grounding_score predicate in the bin
    # Group-level signal (B-049): True iff the parent canonicalization group
    # produced ≥2 polarity-bearing direction bins. Every per-direction rule
    # from a conflicted group carries the same flag. Does not mean this
    # individual rule contains contradictory members — by construction each
    # rule contains exactly one direction's worth of members. `partial`
    # counts as polarity-bearing here; whether that's the right call is the
    # open semantic question owned by B-025.
    is_conflicted:       bool
    study_coverage:      Literal["single_study", "multi_study", "unknown"]
    category:            Literal["morphology", "IHC", "molecular_genetics", "staging", "treatment", "prognosis", "demographic"]
    supporting_pmcids:   List[str]             # unique PMCIDs across member NFs
    member_normal_ids:   List[str]             # NormalFinding.normal_id values in this bin
    mean_grounding_score: float | None
    finding_count:       int                   # number of NormalFindings in this bin
    subject_cui:         str | None = None     # UMLS CUI for subject_entity (set post-canonicalize)
    outcome_cui:         str | None = None     # UMLS CUI for outcome_entity (set post-canonicalize)
    # Representative scope from the best-grounded NormalFinding in this rule's
    # direction-bin. Optional + default ``None`` for backward compatibility
    # with older artifacts that pre-date the field — RELATE falls back to
    # predicate-only NLI input when this is None.  See
    # ``RelateConfig.scope_aware_nli``.
    scope:               FindingScope | None = None
    # Representative verbatim source sentence (from the best-grounded
    # NormalFinding's first SourceSpan). Lets the RELATE stage feed the NLI
    # with the exact source text instead of (or alongside) the abstracted
    # predicate_text.  See ``RelateConfig.use_verbatim_for_nli``.  Optional
    # for backward compatibility.
    representative_verbatim:         str | None = None
    # Pointer to the paragraph containing the verbatim sentence above.  Use
    # ``provenance.paragraph_lookup.get_paragraph_for_rule`` (DB-backed) to
    # fetch the full ``text_elements.text_content`` — kept out of JSON to
    # avoid ~10× output blow-up.  Optional for backward compatibility.
    representative_text_element_id:  int | None = None


# Phase 5: RELATE output

class RelationTypeLabel(str, Enum):
    SUPPORT        = "SUPPORT"
    CONTRADICT     = "CONTRADICT"
    UNRELATED      = "UNRELATED"


class Relation(BaseModel):
    """
    Output of the RELATE stage: a directional NLI-derived relation between
    two CanonicalRules. ``rule_id_a`` and ``rule_id_b`` are symmetric for
    SUPPORT.
    """
    rule_id_a:       str
    rule_id_b:       str
    relation_type:   RelationTypeLabel
    # Label-conditional NLI projection: entailment score when relation_type ==
    # SUPPORT, contradiction score when relation_type == CONTRADICT. For
    # unambiguous per-direction entailment/contradiction values, see RawNLIPair.
    nli_score_a_to_b: float
    nli_score_b_to_a: float


class RawNLIPair(BaseModel):
    """
    Raw NLI scores for one eligible pair from the RELATE stage, including
    pairs classified as UNRELATED.  Saved alongside Relation[] so that
    entailment/contradiction thresholds can be swept offline without
    re-running the NLI model.
    """
    rule_id_a:   str
    rule_id_b:   str
    ent_a_to_b:  float
    ent_b_to_a:  float
    con_a_to_b:  float
    con_b_to_a:  float
    classified_label: RelationTypeLabel   # UNRELATED when neither threshold fired
    pmcid_a:     str = ""                 # source paper for rule_id_a (corpus-level only)
    pmcid_b:     str = ""                 # source paper for rule_id_b (corpus-level only)


# Phase 6: RESOLVE output

class FinalRule(BaseModel):
    """
    Output of the RESOLVE stage: a scored, ranked canonical rule.

    Inherits all fields from CanonicalRule and adds scoring metadata.
    """
    final_id:              str                   # "FR_{canonical_id}"
    canonical_id:          str
    group_id:              str
    subject_entity:        str
    outcome_entity:        str
    relation_type:         RelationTypeEnum
    direction:             DirectionEnum | None
    predicate_text:        str
    is_conflicted:         bool
    study_coverage:        Literal["single_study", "multi_study", "unknown"]
    category:              Literal["morphology", "IHC", "molecular_genetics", "staging", "treatment", "prognosis", "demographic"]
    supporting_pmcids:     List[str]
    member_normal_ids:     List[str]
    mean_grounding_score:  float | None
    finding_count:         int
    # Scoring
    final_score:           float                 # 0–1, higher is better
    support_count:         int                   # SUPPORT relations touching this rule
    contradict_count:      int                   # CONTRADICT relations touching this rule
    # Always 0 since the SCOPE_QUALIFY branch in _classify_pair was removed
    # (B-006). Field + DB column kept so existing readers (templates, DB
    # downstream consumers) don't break; revive only if an asymmetric-
    # entailment branch is reintroduced.
    scope_qualify_count:   int = 0
    is_contradicted:       bool
    contradicted_by:       List[str]             # canonical_ids that contradict this rule
    # Observability — which RESOLVE scoring branch produced this rule's
    # final_score. ``relations_present`` uses the votes+grounding+relations
    # formula; ``relations_absent`` uses the grounding-dominant fallback.
    # See ``stages/resolve_stage.py`` for the exact branch logic.
    # Default ``relations_absent`` keeps backward compatibility with cached
    # entries from before the field existed.
    score_mode:            Literal["relations_present", "relations_absent"] = "relations_absent"


# Corpus-level RELATE output

class CorpusRelation(Relation):
    """
    Output of CorpusRelateStage: a pairwise relation from the full pooled corpus.

    Extends Relation with provenance (pmcid_a/b, comparison_scope), human-readable
    claim text, and grounding metadata.

    comparison_scope distinguishes intra-paper edges (both rules from the same
    paper's processing run) from cross-paper edges (rules from different papers).
    NLI methodology is identical for both; the label is purely informational.

    Per-paper JSON `relations` remain the authoritative input for ResolveStage
    scoring.  This model is for the analytical corpus graph only.
    """
    relation_id:              str
    comparison_scope:         Literal["intra_paper", "cross_paper"]
    same_paper:               bool
    pmcid_a:                  str
    pmcid_b:                  str
    subject_entity:           str
    outcome_entity:           str
    category:                 str
    relation_type_structural: str                # has_feature / expression / etc.
    direction_a:              str | None
    direction_b:              str | None
    predicate_a:              str
    predicate_b:              str
    mean_grounding_a:         float | None
    mean_grounding_b:         float | None
    finding_count_a:          int
    finding_count_b:          int
    supporting_pmcids_a:      List[str]
    supporting_pmcids_b:      List[str]
    is_conflicted_a:          bool
    study_coverage_a:         Literal["single_study", "multi_study", "unknown"]
    is_conflicted_b:          bool
    study_coverage_b:         Literal["single_study", "multi_study", "unknown"]
    # Scope qualifier check — "scope_unknown" until FindingScope is carried on
    # CanonicalRule (v2 work).  Values: scope_unknown | scope_compatible |
    # scope_mismatch | scope_qualified.
    scope_check_result:       str = "scope_unknown"
    scope_note:               str = ""


# Rejection summary

class RejectedFinding(BaseModel):
    """A finding dropped or excluded at a pipeline stage, with reason metadata."""
    stage: Literal["grounding_map", "group_non_groupable"]
    reason: str                       # e.g. "grounding_score=0.23 < threshold=0.50"
    claim: str
    category: str
    chunk_id: str | None = None       # set for grounding_map stage
    grounding_score: float | None = None
    subject_entity: str | None = None
    outcome_entity: str | None = None
    relation_type: str | None = None


class RejectionSummary(BaseModel):
    """
    Structured report of all findings rejected or excluded during one paper's run.

    Attached to the pipeline result under the ``rejection_summary`` key.
    Use this to audit filter calibration: high grounding_rejection_rate suggests
    the threshold is too strict; high non_groupable_rate suggests the MAP prompt
    is failing to extract entities.
    """
    pmcid: str
    grounding_threshold: float | None
    # Counts
    map_findings_total: int
    map_grounding_rejected: int
    normal_findings_total: int
    non_groupable_total: int
    # Non-groupable breakdown (can overlap — a finding may fail multiple conditions)
    non_groupable_no_subject: int
    non_groupable_no_outcome: int
    non_groupable_unclear_relation: int
    # Rates
    grounding_rejection_rate: float   # map_grounding_rejected / map_findings_total
    non_groupable_rate: float         # non_groupable_total / normal_findings_total
    # Per-category breakdown for grounding rejections
    grounding_rejected_by_category: dict[str, int]
    # Full item list
    rejected: List[RejectedFinding]
