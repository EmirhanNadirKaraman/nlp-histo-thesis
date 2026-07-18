"""Immutable runtime resources shipped inside the installed package.

These are application defaults the library needs in order to work at all — the
model price table and the NLI model registry. They are small, version-controlled,
and read through ``importlib.resources``, never located by walking up from
``__file__``: an installed wheel has no repository above it.

They are *defaults*, not user configuration. A caller can always override:
``PriceBook.load(path)`` / ``NLP_HISTO_MODEL_PRICES`` and ``NLP_HISTO_NLI_MODELS``.
Run-specific configuration (``configs/run.yaml``) is deliberately not here — that is
the user's file and stays in the repository.
"""
