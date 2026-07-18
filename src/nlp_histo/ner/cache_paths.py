"""Where the entity-linking cache lives.

Historically the cache was written next to the module::

    CACHE_FILE = Path(__file__).parent / "entity_linking_cache.json"

That is a packaging bug: once the project is installed, ``__file__`` points into
``site-packages``, so the process would try to write a ~30 MB file into the
installed package (read-only on many systems, and lost on every reinstall).

Resolution order, most explicit first:

1. an explicit path passed by the caller (function argument / CLI flag);
2. the ``NLP_HISTO_ENTITY_CACHE`` environment variable (a file path);
3. a user-cache default: ``$XDG_CACHE_HOME/nlp-histo/entity_linking_cache.json``,
   falling back to ``~/.cache/nlp-histo/entity_linking_cache.json``.

Resolving a path never touches the filesystem: no directory is created merely by
importing this module or asking where the cache would go. The parent directory is
created only by :func:`ensure_parent`, which the writer calls immediately before
writing.

The cache format and its keys are unchanged — only its location is now explicit.
An absent cache behaves exactly as before (empty mapping, recomputed on demand).
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "NLP_HISTO_ENTITY_CACHE"
CACHE_FILENAME = "entity_linking_cache.json"


def default_entity_cache_path() -> Path:
    """The cache location when the caller gives none. Does not touch the disk."""
    configured = os.environ.get(ENV_VAR)
    if configured:
        return Path(configured).expanduser()

    cache_root = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    return cache_root / "nlp-histo" / CACHE_FILENAME


def resolve_entity_cache_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Return the cache path: explicit argument > env var > user-cache default."""
    if explicit is not None:
        return Path(explicit).expanduser()
    return default_entity_cache_path()


def ensure_parent(path: Path) -> Path:
    """Create the cache's parent directory. Called only when a write is imminent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
