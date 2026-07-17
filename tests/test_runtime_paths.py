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

def test_embedding_cache_default_is_outside_the_package(monkeypatch, tmp_path) -> None:
    from nlp_histo.evaluation.matching import matcher

    monkeypatch.delenv(matcher.OPENAI_CACHE_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    default = matcher.default_embedding_cache_path("openai")
    package_dir = Path(matcher.__file__).resolve().parent
    assert package_dir not in default.resolve().parents
    assert default == tmp_path / "nlp-histo" / "embedding_cache_openai.sqlite"

def test_embedding_cache_env_overrides(monkeypatch, tmp_path) -> None:
    from nlp_histo.evaluation.matching import matcher

    monkeypatch.setenv(matcher.GEMINI_CACHE_ENV, str(tmp_path / "g.sqlite"))
    assert matcher.default_embedding_cache_path("gemini") == tmp_path / "g.sqlite"

def test_resolving_the_embedding_cache_creates_nothing(monkeypatch, tmp_path) -> None:
    from nlp_histo.evaluation.matching import matcher

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    matcher.default_embedding_cache_path("openai")
    assert not list(tmp_path.iterdir())

def test_matcher_no_longer_hardcodes_the_repository_cache() -> None:
    from nlp_histo.evaluation.matching import matcher

    src = Path(matcher.__file__).read_text(encoding="utf-8")
    assert 'Path("eval/data/embedding_cache_openai.sqlite")' not in src, (
        "a cwd-relative cache default misses away from the repository root — and a "
        "miss means PAID embedding calls"
    )

def test_replay_requires_the_frozen_embedding_cache(tmp_path, capsys) -> None:
    """An offline command must refuse to start rather than silently make paid calls."""
    from nlp_histo.workflows import replay

    (tmp_path / "out" / "summaries" / "summaries").mkdir(parents=True)
    primer = tmp_path / "eval" / "data" / "map_primer"
    primer.mkdir(parents=True)
    (primer / "voter_cache.json").write_text("{}")
    # ...but no embedding_cache_openai.sqlite

    rc = replay.main(["--artifact-root", str(tmp_path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "embedding_cache_openai.sqlite" in err

def test_replay_resolves_its_cache_under_the_artifact_root(tmp_path) -> None:
    from nlp_histo.workflows import replay

    replay._REPO_ROOT = tmp_path
    assert replay.frozen_embedding_cache() == (
        tmp_path / "eval" / "data" / "embedding_cache_openai.sqlite"
    )

def test_manifest_git_info_outside_a_checkout(tmp_path, monkeypatch) -> None:
    """Outside a git repository the manifest records empty fields — it must not crash."""
    from nlp_histo.pipeline.stages.pdf_text_extraction.outputs.manifest_writer import (
        _git_info,
    )

    monkeypatch.chdir(tmp_path)
    info = _git_info()
    assert info.get("sha") in (None, "")

def test_manifest_git_info_does_not_walk_parents() -> None:
    from nlp_histo.pipeline.stages.pdf_text_extraction.outputs import manifest_writer

    tree = ast.parse(Path(manifest_writer.__file__).read_text(encoding="utf-8"))
    walks = [n for n in ast.walk(tree)
             if isinstance(n, ast.Attribute) and n.attr == "parents"]
    assert not walks, "an installed wheel is not N parents below a git checkout"

def test_output_defaults_stay_relative_to_the_caller(tmp_path, monkeypatch) -> None:
    """`out/…` means "under the directory you ran from" — deliberately, not a bug.

    They must not be reinterpreted as relative to the installed package.
    """
    from nlp_histo.pipeline.stages.pdf_text_extraction.config import PathConfig

    monkeypatch.chdir(tmp_path)
    cfg = PathConfig()
    assert not cfg.output_root.is_absolute()
    assert cfg.output_root == Path("out")
    assert (tmp_path / cfg.output_root).parent == tmp_path
