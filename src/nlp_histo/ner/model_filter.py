"""Guard against a `--model` filter that matches nothing while data exists (B-115).

`ner merge` / `ner export` filter `entities.model_name`. When that filter names a model
the database has never seen, every query returns empty and the command previously printed
"No entities found" and exited 0 — indistinguishable from "the corpus really is empty".

That is exactly how B-115 hid: the consumers defaulted to the package name
`en_core_sci_lg` while `ner.py` stores spaCy's ``meta["name"]`` (`core_sci_lg`), so the
documented commands produced nothing against a corpus of 1.79M entities, silently, for
the corpus's entire lifetime.

An empty result is only trustworthy when the requested model is one the database actually
holds. Otherwise it is a mismatch and must be loud.
"""
from __future__ import annotations

from nlp_histo.database import Entity


class NoMatchingEntitiesError(RuntimeError):
    """The requested model filter matches no stored entity, but other models do.

    Distinct from "there are no entities at all", which is a legitimate empty result
    (a fresh database) and is not raised.
    """


def stored_model_names(session) -> list[str]:
    """Every distinct non-null ``entities.model_name`` present, sorted."""
    return sorted(
        name for (name,) in session.query(Entity.model_name).distinct() if name
    )


def check_model_filter(session, model_name: str | None) -> None:
    """Raise ``NoMatchingEntitiesError`` if ``model_name`` matches nothing that exists.

    No-ops when: no filter was requested; the table holds no entities at all (a genuine
    empty corpus); or the requested model is present. Only the mismatch case raises —
    the one where "no results" means "you asked the wrong question".
    """
    if not model_name:
        return

    present = stored_model_names(session)
    if not present:
        return  # genuinely empty corpus — an empty result is the honest answer
    if model_name in present:
        return  # the filter is valid; an empty result is a real result

    available = ", ".join(repr(m) for m in present)
    raise NoMatchingEntitiesError(
        f"No entities are stored under model {model_name!r}, but the database does hold "
        f"entities under: {available}.\n"
        f"\n"
        f"Refusing to report an empty result as success — the filter does not match the "
        f"data, so 'no entities found' would be misleading rather than informative.\n"
        f"\n"
        f"Note the stored identifier is spaCy's nlp.meta['name'], which omits the "
        f"language prefix: loading the 'en_core_sci_lg' package stores 'core_sci_lg'. "
        f"Pass one of the names above via --model, or omit --model to use the default "
        f"(B-115)."
    )
