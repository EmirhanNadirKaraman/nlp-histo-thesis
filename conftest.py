"""Repo-root pytest configuration.

Guards against a pytest-randomly ↔ thinc incompatibility.

``pytest-randomly`` reseeds every registered ``pytest_randomly.random_seeder``
entry point once per test with ``base_seed + crc32(nodeid)`` — a value up to
~2**33. It clamps its OWN numpy reseed (``seed % 2**32``) but passes the RAW seed
to third-party entry-point reseeders. The only one installed here is thinc's
``fix_random_seed`` (pulled in transitively by scispaCy), which forwards the seed
straight to ``numpy.random.seed`` — and numpy rejects any seed > 2**32-1. With a
large random base seed (the default when you run ``python -m pytest`` without an
explicit ``--randomly-seed``) most tests overflow, raising
``ValueError: Seed must be between 0 and 2**32 - 1`` at setup/teardown → a
full-suite error cascade. (Small explicit seeds like ``--randomly-seed=1`` almost
never overflow, which is why it hid during development.)

Fix: pre-populate ``pytest_randomly.entrypoint_reseeds`` with clamped wrappers so
each entry-point reseeder receives ``seed % 2**32`` — mirroring pytest-randomly's
own numpy clamp. This runs in ``pytest_configure``, before pytest-randomly's first
reseed (at collection), so the module global is already set and never lazily
rebuilt.
"""
from __future__ import annotations


def pytest_configure(config):
    try:
        import pytest_randomly
    except ImportError:
        return
    # Idempotent: rebuild the reseeder list from the entry points (the source of
    # truth) rather than trusting whatever pytest-randomly may have already lazily
    # built — so we win regardless of pytest_configure hook ordering, and never
    # double-wrap. The per-test reseeds (setup/call/teardown) that actually overflow
    # all run after pytest_configure, so this is installed in time.
    if getattr(pytest_randomly, "_nlp_histo_seed_clamped", False):
        return

    from importlib.metadata import entry_points

    def _clamped(reseed):
        # numpy (and thus thinc's fix_random_seed) requires 0 <= seed <= 2**32-1.
        return lambda seed: reseed(seed % 2**32)

    eps = entry_points(group="pytest_randomly.random_seeder")
    pytest_randomly.entrypoint_reseeds = [_clamped(e.load()) for e in eps]
    pytest_randomly._nlp_histo_seed_clamped = True
