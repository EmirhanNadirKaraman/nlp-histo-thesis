"""nlp-histo — auditable knowledge extraction from histopathology literature.

The installed library namespace. Repository-only code (experiment drivers,
developer tools, thesis reproduction scripts) lives outside this package and
imports from it; nothing here reaches back into the repository tree.

The distribution version is declared once, in ``pyproject.toml``. Read it at
runtime via ``importlib.metadata.version("nlp-histo")`` rather than duplicating
it here.
"""
