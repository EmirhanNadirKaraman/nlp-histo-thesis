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

from pathlib import Path



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
