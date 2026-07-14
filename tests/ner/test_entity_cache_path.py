"""Where the entity-linking cache lives, and when it is created.

The cache used to be written next to the module (``Path(__file__).parent /
"entity_linking_cache.json"``). Once installed, that path points into
``site-packages`` — so a ~30 MB file would be written into the installed package.
These tests pin the replacement contract:

    explicit argument  >  $NLP_HISTO_ENTITY_CACHE  >  user cache dir

No test here touches the real user cache or the repository's local 30 MB cache:
every path is a ``tmp_path``, and the environment variable is monkeypatched.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlp_histo.ner import batch_ner
from nlp_histo.ner.cache_paths import (
    CACHE_FILENAME,
    ENV_VAR,
    default_entity_cache_path,
    resolve_entity_cache_path,
)


@pytest.fixture(autouse=True)
def _no_ambient_cache_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a developer's real NLP_HISTO_ENTITY_CACHE leak into these tests."""
    monkeypatch.delenv(ENV_VAR, raising=False)


# ── precedence ────────────────────────────────────────────────────────────────

def test_explicit_path_wins_over_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "from_env.json"))
    explicit = tmp_path / "from_arg.json"
    assert resolve_entity_cache_path(explicit) == explicit


def test_env_var_is_honoured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "cache" / "entities.json"
    monkeypatch.setenv(ENV_VAR, str(target))
    assert resolve_entity_cache_path() == target
    assert default_entity_cache_path() == target


def test_default_lives_in_the_user_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_entity_cache_path() == tmp_path / "nlp-histo" / CACHE_FILENAME


def test_default_is_outside_the_installed_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression this whole module exists for: never write into site-packages."""
    import nlp_histo.ner as ner_pkg

    package_dir = Path(ner_pkg.__file__).resolve().parent
    default = default_entity_cache_path().resolve()
    assert package_dir not in default.parents, (
        f"the default cache path {default} is inside the installed package "
        f"{package_dir} — an installed wheel would be written to"
    )


def test_user_paths_are_expanded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "~/somewhere/cache.json")
    assert "~" not in str(resolve_entity_cache_path())


# ── no filesystem side effects from importing or resolving ────────────────────

def test_resolving_creates_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    before = set(tmp_path.rglob("*"))
    resolve_entity_cache_path()
    default_entity_cache_path()
    assert set(tmp_path.rglob("*")) == before, "merely resolving the path touched the disk"


def test_reading_an_absent_cache_creates_nothing_and_returns_empty(tmp_path: Path) -> None:
    missing = tmp_path / "nested" / "absent.json"
    assert batch_ner.load_entity_cache(missing) == {}
    assert not missing.parent.exists(), "a read must not create the cache directory"


# ── writes ────────────────────────────────────────────────────────────────────

def test_write_creates_the_parent_directory_only_when_needed(tmp_path: Path) -> None:
    target = tmp_path / "made" / "on" / "demand" / CACHE_FILENAME
    assert not target.parent.exists()
    batch_ner.save_entity_cache({"CEAN": "C0007114"}, target)
    assert target.is_file()


def test_round_trip_preserves_the_json_format(tmp_path: Path) -> None:
    """Cache format and keys are unchanged — a plain JSON object of text → CUI."""
    target = tmp_path / CACHE_FILENAME
    payload = {"epithelioid haemangioma": "C0018916", "CEAN": "C0007114"}

    batch_ner.save_entity_cache(payload, target)

    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk == payload
    assert batch_ner.load_entity_cache(target) == payload


def test_writer_and_reader_resolve_the_same_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "shared.json"))
    batch_ner.save_entity_cache({"a": "C1"})
    assert batch_ner.load_entity_cache() == {"a": "C1"}


def test_module_adjacent_cache_path_is_gone() -> None:
    """The old `Path(__file__).parent / cache` constant must not come back."""
    assert not hasattr(batch_ner, "CACHE_FILE"), (
        "batch_ner.CACHE_FILE resolved next to the module — that writes into "
        "site-packages once installed. Use resolve_entity_cache_path()."
    )
