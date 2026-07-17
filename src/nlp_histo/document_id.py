"""Document identity — one place, shared by acquisition and the pipeline.

Two names, deliberately distinct, because conflating them is what caused B-119:

* **PMC accession** — the bare NLM value, ``PMC8395919``.
* **document ID** — this project's established composite identifier,
  ``PMC8395919_dermatopathology-08-00036``: the accession plus the publisher's filename
  stem, produced by ``acquire organize`` and used as ``documents.pmcid``.

The document ID is what every artifact keys on — 977 rows in the corpus, the frozen
replay artifacts (``out/summaries/summaries/PMC10100421_HIS-82-393.json``), the silver
labels (``case_id: "PMC11649514_HIS-86-204|3562"``) — so it is load-bearing history, not
a style choice. Normalising it to bare accessions would orphan all of it and is filed as
post-thesis migration debt, not a fix.

What this module guards is narrower: an NLM **version suffix** must never reach the
database. AWS names its objects ``PMC8395919.1.pdf`` (``.1`` = article version), and a
naive ``Path.stem`` would mint ``PMC8395919.1`` as a *new* document ID — a duplicate of a
paper the corpus already has under its FTP-derived ID.
"""
from __future__ import annotations

import re

# Strips a numeric version ONLY where NLM puts one: immediately after PMC<digits>, and
# only when the whole token ends there or a publisher component follows.
#
#   PMC8395919.1                          → PMC8395919
#   PMC8395919.1_dermatopathology-08-…    → PMC8395919_dermatopathology-08-…
#   PMC8395919_dermatopathology-08-…      → unchanged (no version present)
#   PMC8395919.1.2                        → unchanged (not a shape NLM produces; refuse
#                                            to guess rather than truncate blindly)
#   anything not starting PMC<digits>.<digits> → unchanged
#
# The lookahead is the safety: without it, `.` anywhere after the accession could be eaten,
# and publisher stems do contain dots.
_VERSION_SUFFIX = re.compile(r"^(PMC\d+)\.\d+(?=$|_)")


def canonical_document_id(raw: str) -> str:
    """Return *raw* with an NLM version suffix removed; otherwise unchanged.

    Never truncates a publisher component, an unrelated period, or a non-PMC name — the
    only edit it will make is dropping ``.<digits>`` directly after ``PMC<digits>``.
    """
    return _VERSION_SUFFIX.sub(r"\1", raw)


def pmc_accession(document_id: str) -> str | None:
    """The bare accession inside a document ID, or ``None`` if there isn't one.

    ``PMC8395919_dermatopathology-08-00036`` → ``PMC8395919``. For identity *comparison*
    across sources — never for storage: the composite ID is what the corpus uses.
    """
    m = re.match(r"^(PMC\d+)", document_id)
    return m.group(1) if m else None
