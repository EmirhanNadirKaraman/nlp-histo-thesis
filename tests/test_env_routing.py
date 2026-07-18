"""B-113 — an explicit env file must not be silently overridden.

The near-miss: `NLP_HISTO_ENV_FILE=/tmp/test.env` (DB_NAME=new_local_db) resolved to
`nlp_histo`, the production corpus, because an earlier `source .env` had exported
DB_NAME and python-dotenv loads with override=False. A one-PDF ingest was seconds from
writing into 977 papers; only a hand-written assert stopped it.

The precedence itself is deliberate and documented (ENV_LOADING.md: environment beats
file beats default), so these tests pin the *narrow* contract:

* an EXPLICIT env file that disagrees about routing → error, before connecting;
* ordinary automatic .env discovery → untouched, env still wins;
* secrets → never a conflict; injecting DB_PASSWORD from the environment is legitimate.
"""
from __future__ import annotations

import pytest

from nlp_histo.database.env_routing import (
    EnvRoutingConflict,
    detect_routing_conflict,
    format_target,
    format_target_config,
    raise_on_routing_conflict,
)


def _write_env(tmp_path, **values):
    p = tmp_path / "test.env"
    p.write_text("\n".join(f"{k}={v}" for k, v in values.items()) + "\n", encoding="utf-8")
    return p


# disagreement

def test_disagreement_on_db_name_is_a_conflict(tmp_path) -> None:
    """The exact near-miss: file says scratch, environment says production."""
    env_file = _write_env(tmp_path, DB_NAME="new_local_db", DB_HOST="localhost")
    conflicts = detect_routing_conflict(env_file, {"DB_NAME": "nlp_histo"})
    assert conflicts == ["DB_NAME"]


def test_raise_names_the_variable_and_how_to_resolve(tmp_path) -> None:
    env_file = _write_env(tmp_path, DB_NAME="new_local_db")
    with pytest.raises(EnvRoutingConflict) as exc:
        raise_on_routing_conflict(env_file, {"DB_NAME": "nlp_histo"})
    msg = str(exc.value)
    assert "DB_NAME" in msg
    assert "-u DB_NAME" in msg                    # how to use the file
    assert "unset NLP_HISTO_ENV_FILE" in msg      # how to use the environment
    # names only — the values are the user's business and may be sensitive
    assert "new_local_db" not in msg
    assert "nlp_histo" not in msg.replace("NLP_HISTO_ENV_FILE", "").replace(
        "nlp_histo (B-113)", ""
    ).replace("database/ENV_LOADING.md", "")


@pytest.mark.parametrize("var,file_val,env_val", [
    ("DB_HOST", "localhost", "prod.internal"),
    ("DB_PORT", "5432", "6543"),
    ("DB_NAME", "scratch", "production"),
    ("DB_USER", "test_user", "app_user"),
    ("DB_SCHEMA", "scratch_schema", "public"),
])
def test_every_routing_field_is_covered(tmp_path, var, file_val, env_val) -> None:
    env_file = _write_env(tmp_path, **{var: file_val})
    assert detect_routing_conflict(env_file, {var: env_val}) == [var]


def test_multiple_conflicts_are_all_reported(tmp_path) -> None:
    env_file = _write_env(tmp_path, DB_NAME="a", DB_HOST="h1", DB_PORT="5432")
    conflicts = detect_routing_conflict(
        env_file, {"DB_NAME": "b", "DB_HOST": "h2", "DB_PORT": "5432"}
    )
    assert conflicts == ["DB_HOST", "DB_NAME"]  # DB_PORT agrees, so not flagged


# agreement

def test_agreement_is_not_a_conflict(tmp_path) -> None:
    """Same value from both sources — nothing is being overridden."""
    env_file = _write_env(tmp_path, DB_NAME="same_db", DB_HOST="localhost")
    assert detect_routing_conflict(
        env_file, {"DB_NAME": "same_db", "DB_HOST": "localhost"}
    ) == []
    raise_on_routing_conflict(env_file, {"DB_NAME": "same_db"})  # must not raise


# no explicit env file: documented env-wins behaviour is untouched

def test_automatic_discovery_is_never_a_conflict() -> None:
    """Ordinary `.env` discovery keeps the documented precedence (ENV_LOADING.md).

    Only an EXPLICIT selection carries the intent this check protects.
    """
    assert detect_routing_conflict(None, {"DB_NAME": "anything"}) == []
    raise_on_routing_conflict(None, {"DB_NAME": "anything"})


def test_missing_env_file_is_not_a_conflict(tmp_path) -> None:
    assert detect_routing_conflict(tmp_path / "nope.env", {"DB_NAME": "x"}) == []


# partial files

def test_variables_absent_from_the_file_are_not_conflicts(tmp_path) -> None:
    """A file that declares only DB_NAME says nothing about DB_HOST — inheriting the
    rest of the connection from the environment is the normal, intended pattern."""
    env_file = _write_env(tmp_path, DB_NAME="same_db")
    assert detect_routing_conflict(
        env_file, {"DB_NAME": "same_db", "DB_HOST": "whatever", "DB_USER": "anyone"}
    ) == []


def test_variables_absent_from_the_environment_are_not_conflicts(tmp_path) -> None:
    """Nothing to override — the file simply applies."""
    env_file = _write_env(tmp_path, DB_NAME="scratch", DB_HOST="localhost")
    assert detect_routing_conflict(env_file, {}) == []


def test_empty_file_is_not_a_conflict(tmp_path) -> None:
    p = tmp_path / "empty.env"
    p.write_text("", encoding="utf-8")
    assert detect_routing_conflict(p, {"DB_NAME": "x"}) == []


# secrets are not routing

def test_password_override_is_legitimate_not_a_conflict(tmp_path) -> None:
    """Injecting a secret from the environment while routing comes from the file is the
    normal CI/container pattern. Flagging it would punish good practice — and a wrong
    password fails loudly by itself."""
    env_file = _write_env(tmp_path, DB_NAME="same_db", DB_PASSWORD="from-file")
    assert detect_routing_conflict(
        env_file, {"DB_NAME": "same_db", "DB_PASSWORD": "from-environment"}
    ) == []
    raise_on_routing_conflict(env_file, {"DB_PASSWORD": "from-environment"})


# no connection is attempted after a failure

def test_conflict_is_raised_before_any_connection(tmp_path, monkeypatch) -> None:
    """The check must fire before an engine is ever created, not after connecting."""
    import sqlalchemy

    monkeypatch.setattr(
        sqlalchemy, "create_engine",
        lambda *a, **k: pytest.fail("a connection was attempted despite the conflict"),
    )
    env_file = _write_env(tmp_path, DB_NAME="scratch")
    with pytest.raises(EnvRoutingConflict):
        raise_on_routing_conflict(env_file, {"DB_NAME": "production"})


# target rendering: never leak the password

class _FakeURL:
    username, host, port, database = "u", "h", 5432, "d"
    password = "hunter2"


def test_format_target_never_includes_the_password(monkeypatch) -> None:
    monkeypatch.delenv("DB_SCHEMA", raising=False)
    out = format_target(_FakeURL())
    assert out == "u@h:5432/d"
    assert "hunter2" not in out


def test_format_target_config_matches_the_url_rendering(monkeypatch) -> None:
    """One formatter, one format — a command cannot invent a leaky variant."""
    monkeypatch.delenv("DB_SCHEMA", raising=False)
    cfg = {"user": "u", "host": "h", "port": 5432, "database": "d", "password": "hunter2"}
    assert format_target_config(cfg) == format_target(_FakeURL())
    assert "hunter2" not in format_target_config(cfg)


def test_schema_is_shown_when_set(monkeypatch) -> None:
    monkeypatch.setenv("DB_SCHEMA", "my_schema")
    assert format_target(_FakeURL()).endswith(" schema=my_schema")
