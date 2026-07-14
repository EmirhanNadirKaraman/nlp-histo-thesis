"""Runtime paths must survive being installed — no repository beside the package.

Each test here corresponds to a way the package used to reach for the repository tree
and would have failed (usually *silently*) once installed:

* ``.env`` was located relative to ``__file__``. After the src-layout move that
  resolved to ``src/nlp_histo/.env``, which does not exist — so DB_CONFIG silently
  fell back to ``postgres/postgres@localhost``.
* the price table and NLI registry were found by walking six parents up. From a wheel
  that landed outside the package: the price book came back empty (all costs "n/a")
  and the NLI registry fell back to a built-in default.
* the embedding cache defaulted to ``eval/data/…sqlite`` relative to the cwd. Away
  from the repository root it misses — and a miss means PAID embedding calls.
* the manifest writer asked git about a directory six parents up from ``__file__``.
"""
from __future__ import annotations

import ast
from importlib import resources
from pathlib import Path

import pytest


# ── A. immutable packaged resources ───────────────────────────────────────────

def test_env_file_is_not_searched_inside_the_package() -> None:
    """`.env` must be found from the working directory, never relative to __file__."""
    from nlp_histo.database import db_connection

    src = Path(db_connection.__file__).read_text(encoding="utf-8")
    assert "find_dotenv" in src
    assert "Path(__file__).parent.parent / '.env'" not in src, (
        "this resolved to src/nlp_histo/.env after the src-layout move, so .env was "
        "silently ignored and DB_CONFIG fell back to postgres/postgres@localhost"
    )

def test_env_file_override(tmp_path, monkeypatch) -> None:
    from nlp_histo.database.init_db import _env_path

    explicit = tmp_path / "custom.env"
    monkeypatch.setenv("NLP_HISTO_ENV_FILE", str(explicit))
    assert _env_path() == explicit

@pytest.mark.parametrize("name", ["model_prices.json", "nli_models.yaml"])
def test_packaged_resource_is_reachable(name: str) -> None:
    res = resources.files("nlp_histo.resources").joinpath(name)
    assert res.is_file(), f"{name} is not package data"
    assert res.read_text(encoding="utf-8").strip()

def test_price_book_loads_from_the_packaged_resource() -> None:
    from nlp_histo.pipeline.stages.knowledge_extraction.costing.pricing import PriceBook

    book = PriceBook.load()
    assert book.known_models(), (
        "the packaged price table produced an empty book — costs would render as n/a"
    )

def test_nli_registry_loads_from_the_packaged_resource() -> None:
    from nlp_histo.pipeline.stages.knowledge_extraction.grounding import nli_config

    assert nli_config._config_path().is_file()

def test_resource_lookups_do_not_walk_parents() -> None:
    """The regression guard: no `.parents[n]` climb may come back for these."""
    from nlp_histo.pipeline.stages.knowledge_extraction.costing import pricing
    from nlp_histo.pipeline.stages.knowledge_extraction.grounding import nli_config

    for mod in (pricing, nli_config):
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        walks = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Attribute) and n.attr == "parents"]
        assert not walks, f"{mod.__name__} still locates its resource by filesystem depth"

def test_resource_override_env_vars(tmp_path, monkeypatch) -> None:
    from nlp_histo.pipeline.stages.knowledge_extraction.costing.pricing import (
        default_price_path,
    )
    from nlp_histo.pipeline.stages.knowledge_extraction.grounding.nli_config import (
        _config_path,
    )

    prices = tmp_path / "p.json"
    monkeypatch.setenv("NLP_HISTO_MODEL_PRICES", str(prices))
    assert Path(str(default_price_path())) == prices

    registry = tmp_path / "n.yaml"
    monkeypatch.setenv("NLP_HISTO_NLI_MODELS", str(registry))
    assert Path(str(_config_path())) == registry
