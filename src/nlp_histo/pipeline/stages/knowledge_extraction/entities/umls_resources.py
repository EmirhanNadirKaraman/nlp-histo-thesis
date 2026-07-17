"""
Process-wide singleton for the scispaCy + UMLS entity linker.

Before this module existed, ``normalize_stage`` and ``entity_linker``
each loaded their own copy of ``en_core_sci_lg`` + ``scispacy_linker``.  The
UMLS KB resident set is several GB; loading it twice in one process is the
quickest path to an OOM kill mid-pipeline.

Every component that needs the linker must obtain it through ``get_nlp()`` so
the model is loaded at most once per process.

The loader is silent-failing: if scispaCy / the model is unavailable,
``get_nlp()`` returns ``None`` and callers must degrade gracefully. That is
deliberate for the live per-paper pipeline, which is allowed to skip CUI work
and carry on.

It is **not** acceptable for workflows whose published numbers depend on CUI
enrichment: silently skipping the linker there produces plausible-but-wrong
output that reports success (B-107). Those callers must use ``require_umls()``,
which probes the loader and raises ``UmlsUnavailableError`` instead of degrading.
``get_nlp()``'s return-``None`` contract is unchanged.
"""
from __future__ import annotations

import logging
import os
import threading

from nlp_histo.pipeline.utils.memory_logging import MemoryLogger, get_default_memory_logger

from .umls_utils import UMLS_THRESHOLD

logger = logging.getLogger(__name__)

# Module-level MemoryLogger so scispaCy / UMLS load events share the unified
# MEMORY line format used by the per-paper pipeline runner.
_mem: MemoryLogger = get_default_memory_logger()

# ── Process-wide state ────────────────────────────────────────────────────────

_NLP = None
_LINKER = None
_AVAILABLE: bool | None = None       # None = not probed; True/False after
# Why the last load attempt failed. ``_AVAILABLE=False`` alone cannot distinguish
# "deliberately disabled via the kill-switch" from "tried to load and broke", and
# require_umls() must report those differently.
_FAILURE_REASON: str | None = None
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
    global _NLP, _LINKER, _AVAILABLE, _FAILURE_REASON

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
            # Record the cause verbatim. A no-network machine surfaces this as a
            # requests.ConnectionError wrapping urllib3's NameResolutionError over a
            # socket.gaierror — require_umls() replays it to the user, since "why"
            # is the whole diagnostic (B-107).
            _FAILURE_REASON = f"{type(exc).__name__}: {exc}"
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


def failure_reason() -> str | None:
    """Why the last load attempt failed.

    ``None`` when the loader has not been probed, succeeded, or was disabled via the
    kill-switch (which is not a failure). Use ``umls_disabled()`` to tell those apart.
    """
    return _FAILURE_REASON


class UmlsUnavailableError(RuntimeError):
    """A caller that REQUIRES the UMLS linker could not have it.

    Raised only by ``require_umls()``. ``get_nlp()`` keeps returning ``None`` so the
    live pipeline's deliberate skip-CUI-and-continue path is unaffected.
    """


# Having the artifacts on disk is not sufficient, which is the single most
# counter-intuitive part of a UMLS failure — so require_umls() always says it.
_CACHE_IS_NOT_ENOUGH = (
    "Note: a warm cache does not make this work offline. scispaCy resolves its cache\n"
    "filename from the REMOTE ETag — get_from_cache() issues an unconditional\n"
    "requests.head(url) and builds the name as sha256(url).sha256(etag)\n"
    "(scispacy/file_cache.py). With no network the filename cannot be computed, so a\n"
    "byte-complete ~/.scispacy/datasets (≈2.1 GB) is unusable. Pre-downloading the\n"
    "model does NOT fix this.\n"
    "\n"
    "The network access needed here is a FREE model/metadata fetch from\n"
    "s3-us-west-2.amazonaws.com. It is not a paid model or API call, and this command\n"
    "still makes no paid requests."
)


def require_umls(*, context: str, affected_outputs: tuple[str, ...] = ()) -> None:
    """Probe the linker; raise ``UmlsUnavailableError`` unless it is usable.

    For workflows whose published numbers depend on CUI enrichment. Call it *before*
    writing anything, so a run that cannot be correct cannot leave output behind.

    ``context`` names the workflow in the error; ``affected_outputs`` names the
    artifacts that would have been wrong.
    """
    get_nlp()  # idempotent: loads on first call, replays the cached verdict after.
    if is_available():
        return

    if umls_disabled():
        cause = (
            f"UMLS is disabled via ${_DISABLE_ENV}.\n"
            f"\n"
            f"Unset it to run {context}. The kill-switch exists for low-RAM machines and "
            f"for workflows that tolerate missing CUIs — this one does not."
        )
        remedy = ""
    else:
        cause = f"The scispaCy/UMLS linker could not be initialised:\n\n  {failure_reason()}"
        remedy = f"\n\n{_CACHE_IS_NOT_ENOUGH}"

    affected = ""
    if affected_outputs:
        affected = "\n\nAffected outputs — these depend on CUI enrichment and would be wrong:\n" + "\n".join(
            f"  - {name}" for name in affected_outputs
        )

    raise UmlsUnavailableError(
        f"{context} requires the UMLS entity linker, which is unavailable.\n"
        f"\n"
        f"{cause}"
        f"{affected}"
        f"\n\n"
        f"Refusing to run: without the linker, CUI enrichment is skipped and the numbers\n"
        f"below would still be produced, still exit 0, and still look correct (B-107).\n"
        f"Nothing has been written."
        f"{remedy}"
    )


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
    global _NLP, _LINKER, _AVAILABLE, _FAILURE_REASON
    _NLP = None
    _LINKER = None
    _AVAILABLE = None
    _FAILURE_REASON = None
