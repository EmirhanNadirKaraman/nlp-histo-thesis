"""Single source of truth for the NLI model used by GROUNDING and RELATE.

Both stages read the active model spec from here so swaps stay one-place.
Selection precedence:

    1. ``$NLP_HISTO_NLI_MODEL`` — registry key OR raw HF id (for ad-hoc swaps).
    2. ``default`` key in ``configs/nli_models.yaml``.

Batch size override: ``$NLP_HISTO_NLI_BATCH_SIZE``.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Repository-level resource, anchored to this file rather than the working directory:
# grounding -> knowledge_extraction -> stages -> pipeline -> <repo root>/configs/…
# The NLI model registry is an immutable application default and ships inside the
# package. It is read through importlib.resources: an installed wheel has no
# repository above it, so a parents[n] walk would resolve to a stranger's directory
# (empirically: from a wheel it landed on /private/tmp/configs/nli_models.yaml and
# silently fell back). NLP_HISTO_NLI_MODELS overrides with an external file.
def _config_path():
    import os
    from importlib import resources

    override = os.getenv("NLP_HISTO_NLI_MODELS")
    if override:
        return Path(override).expanduser()
    return resources.files("nlp_histo.resources").joinpath("nli_models.yaml")


_FALLBACK_HF_ID = "pritamdeka/PubMedBERT-MNLI-MedNLI"
_FALLBACK_BATCH_SIZE = 16


@dataclass(frozen=True)
class NLIModelSpec:
    key: str
    hf_id: str
    batch_size: int
    notes: str = ""


@lru_cache(maxsize=1)
def _load_registry() -> tuple[dict[str, NLIModelSpec], str]:
    if not _config_path().is_file():
        logger.warning(
            "NLI registry %s missing; using built-in fallback (%s).",
            _config_path(), _FALLBACK_HF_ID,
        )
        fallback = NLIModelSpec(
            key="fallback", hf_id=_FALLBACK_HF_ID,
            batch_size=_FALLBACK_BATCH_SIZE,
        )
        return {"fallback": fallback}, "fallback"

    data = yaml.safe_load(_config_path().read_text(encoding="utf-8"))
    raw_models = data.get("models") or {}
    models = {
        key: NLIModelSpec(
            key=key,
            hf_id=spec["hf_id"],
            batch_size=int(spec.get("batch_size", _FALLBACK_BATCH_SIZE)),
            notes=str(spec.get("notes", "")).strip(),
        )
        for key, spec in raw_models.items()
    }
    default_key = data.get("default")
    if default_key not in models:
        raise ValueError(
            f"{_config_path()}: default key {default_key!r} not found in models."
        )
    return models, default_key


def get_active_spec() -> NLIModelSpec:
    """Return the currently selected NLI model spec."""
    models, default_key = _load_registry()
    sel = os.environ.get("NLP_HISTO_NLI_MODEL") or default_key

    if sel in models:
        spec = models[sel]
    else:
        spec = NLIModelSpec(
            key=sel, hf_id=sel, batch_size=_FALLBACK_BATCH_SIZE,
            notes="ad-hoc HF id (not in registry)",
        )

    bs_override = os.environ.get("NLP_HISTO_NLI_BATCH_SIZE")
    if bs_override:
        spec = NLIModelSpec(
            key=spec.key, hf_id=spec.hf_id,
            batch_size=int(bs_override), notes=spec.notes,
        )
    return spec


def list_models() -> list[NLIModelSpec]:
    """Return all registered candidate specs (for future eval harness)."""
    models, _ = _load_registry()
    return list(models.values())
