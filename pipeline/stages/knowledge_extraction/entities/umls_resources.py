"""
Process-wide singleton for the scispaCy + UMLS entity linker.

Before this module existed, ``normalize_stage`` and ``entity_linker``
each loaded their own copy of ``en_core_sci_lg`` + ``scispacy_linker``.  The
UMLS KB resident set is several GB; loading it twice in one process is the
quickest path to an OOM kill mid-pipeline.

Every component that needs the linker must obtain it through ``get_nlp()`` so
the model is loaded at most once per process.

The loader is silent-failing: if scispaCy / the model is unavailable,
``get_nlp()`` returns ``None`` and callers must degrade gracefully.
"""
from __future__ import annotations

import logging
import os
import threading

from pipeline.utils.memory_logging import MemoryLogger, get_default_memory_logger

from .umls_utils import UMLS_THRESHOLD

logger = logging.getLogger(__name__)

# Module-level MemoryLogger so scispaCy / UMLS load events share the unified
# MEMORY line format used by the per-paper pipeline runner.
_mem: MemoryLogger = get_default_memory_logger()

# ── Process-wide state ────────────────────────────────────────────────────────

_NLP = None
_LINKER = None
_AVAILABLE: bool | None = None       # None = not probed; True/False after
_LOAD_LOCK = threading.Lock()

# Small-model singleton — separate cache from the linker-attached `_NLP`.
# Used for sentence segmentation and biomedical-entity presence detection
# (no UMLS linking). Loading scispaCy `en_core_sci_sm` is cheap (~40MB) but
# repeating it once per paper still costs O(N) seconds in batch mode, and
# allowing direct `spacy.load(...)` calls fragments the cache across
# call sites. See B-029 (PDF extraction) and B-038 (summarisation loader).
_SMALL_NLP_BY_NAME: dict[str, object] = {}
_SMALL_NLP_LOCK = threading.Lock()

# Disable scispaCy/UMLS entirely without uninstalling it. Useful for low-RAM
# machines and for the new --skip-umls-* CLI flags, which set this env var
# before calling into the pipeline.
_DISABLE_ENV = "NLP_HISTO_DISABLE_UMLS"


def umls_disabled() -> bool:
    """True when the kill-switch env var is set to a truthy value."""
    return os.environ.get(_DISABLE_ENV, "").lower() in {"1", "true", "yes"}


def _quiet_nmslib() -> None:
    """nmslib emits INFO lines on every index load — drop to WARNING."""
    for name in ("nmslib", "scispacy", "scispacy.candidate_generation"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _quiet_sklearn_pickle_warnings() -> None:
    """Silence the InconsistentVersionWarning emitted when scispaCy unpickles
    its TfidfTransformer + TfidfVectorizer artifacts (pickled with sklearn
    1.1.x; we run 1.6.x).  The TF-IDF data structures are stable across these
    versions; the warning is cosmetic.  Two pickles are unpickled per linker
    load, so without this filter every UMLS load emits two warnings."""
    import warnings  # noqa: PLC0415
    try:
        from sklearn.exceptions import InconsistentVersionWarning  # noqa: PLC0415
    except ImportError:
        return
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


def _log_memory(label: str) -> None:
    """Emit a MEMORY checkpoint with stage=UMLS for the given label.

    ``label`` strings such as "before scispaCy load" / "after scispaCy load" /
    "before UMLS enrichment" map to the ``event`` field of the MEMORY line.
    """
    event = label.replace(" ", "_")
    _mem.checkpoint("UMLS", event)


def get_nlp(
    *,
    model_candidates: tuple[str, ...] = ("en_core_sci_lg", "en_core_sci_sm"),
    threshold: float = UMLS_THRESHOLD,
):
    """Return the shared scispaCy nlp object, loading it on first call.

    Returns ``None`` when scispaCy/UMLS is unavailable or has been disabled.
    Thread-safe: concurrent first-callers block on the load lock.
    """
    global _NLP, _LINKER, _AVAILABLE

    if _AVAILABLE is not None:
        if _AVAILABLE and _NLP is not None:
            logger.debug("UMLS: reusing existing scispaCy model/linker")
        return _NLP

    if umls_disabled():
        logger.info(
            "UMLS disabled via $%s — scispaCy linker will not be loaded",
            _DISABLE_ENV,
        )
        _AVAILABLE = False
        return None

    with _LOAD_LOCK:
        if _AVAILABLE is not None:        # raced another loader
            return _NLP
        _quiet_nmslib()
        _quiet_sklearn_pickle_warnings()
        _log_memory("before scispaCy load")
        try:
            import spacy                                  # type: ignore
            import scispacy                               # noqa: F401  # type: ignore
            from scispacy.linking import EntityLinker     # noqa: F401  # type: ignore

            nlp = None
            chosen: str | None = None
            for name in model_candidates:
                try:
                    logger.info("UMLS: loading %s (one-time, all stages reuse)", name)
                    nlp = spacy.load(
                        name,
                        disable=["parser", "attribute_ruler", "lemmatizer"],
                    )
                    chosen = name
                    break
                except OSError:
                    logger.info("UMLS: %s not installed, trying next", name)
                    continue
            if nlp is None:
                raise RuntimeError(
                    f"No scispaCy model found among {model_candidates!r}"
                )

            nlp.add_pipe("scispacy_linker", config={
                "resolve_abbreviations": True,
                "linker_name": "umls",
                "threshold": threshold,
            })
            _NLP = nlp
            _LINKER = nlp.get_pipe("scispacy_linker")
            _AVAILABLE = True
            logger.info("UMLS: scispaCy + linker ready (%s)", chosen)
            _log_memory("after scispaCy load")
            return _NLP

        except Exception as exc:
            logger.warning(
                "UMLS: linker unavailable — downstream stages will skip CUI work: %s",
                exc,
            )
            _AVAILABLE = False
            return None


def get_linker():
    """Return the loaded scispacy_linker pipe, or None when unavailable."""
    if _AVAILABLE is None:
        get_nlp()
    return _LINKER


def is_available() -> bool:
    return bool(_AVAILABLE)


def get_small_nlp(model_name: str = "en_core_sci_sm"):
    """Return the cached small scispaCy model (no UMLS linker attached).

    Use this for sentence segmentation, entity-presence checks, and other
    lightweight NLP work that doesn't need the UMLS knowledge base.
    The full linker-attached `get_nlp()` loads `en_core_sci_lg` plus the
    multi-GB UMLS KB; routing every site through that would waste memory.

    One singleton per model name. Thread-safe; concurrent first-callers
    block on the load lock. Returns ``None`` and logs a warning when the
    model is not installed (callers should treat as opt-out — see
    `PipelineRunner._get_nlp` for the falsy-sentinel pattern).

    Disable kill-switch: `NLP_HISTO_DISABLE_UMLS=1` also disables this
    loader so memory-constrained environments stay consistent across all
    scispaCy entry-points.
    """
    if umls_disabled():
        logger.info(
            "Small scispaCy loader disabled via $%s — returning None",
            _DISABLE_ENV,
        )
        return None

    cached = _SMALL_NLP_BY_NAME.get(model_name)
    if cached is not None:
        return cached

    with _SMALL_NLP_LOCK:
        cached = _SMALL_NLP_BY_NAME.get(model_name)
        if cached is not None:                # raced another loader
            return cached
        _log_memory(f"before small scispaCy load ({model_name})")
        try:
            import spacy  # type: ignore
            # No linker pipe attached — `get_small_nlp` is the cheap variant.
            # Disable parser/attribute_ruler/lemmatizer to match the heavy
            # `get_nlp()` config so sentence segmentation behaviour stays
            # consistent across the two singletons.
            nlp = spacy.load(
                model_name,
                disable=["parser", "attribute_ruler", "lemmatizer"],
            )
            # Re-enable the sentencizer explicitly — without the parser
            # spaCy needs a senter component to produce `.sents`.
            if "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            _SMALL_NLP_BY_NAME[model_name] = nlp
            logger.info("Small scispaCy loaded: %s (one-time, all sites reuse)", model_name)
            _log_memory(f"after small scispaCy load ({model_name})")
            return nlp
        except OSError as exc:
            logger.warning(
                "Small scispaCy model %r not installed — callers will fall back: %s",
                model_name, exc,
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Small scispaCy load failed for %r — callers will fall back: %s",
                model_name, exc,
            )
            return None


# Test / reset hook (do not call from production code).

def _reset_for_tests() -> None:
    global _NLP, _LINKER, _AVAILABLE
    _NLP = None
    _LINKER = None
    _AVAILABLE = None
