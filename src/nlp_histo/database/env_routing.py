"""Explicit env-file routing conflicts, and one place to render a connection target.

Two small pieces, both from B-113:

**Conflict detection.** ``NLP_HISTO_ENV_FILE`` chooses *which* file is read; it does not
make that file's values win. python-dotenv loads with ``override=False``, and that
precedence — environment beats file beats default — is deliberate and documented
(``ENV_LOADING.md``): it is how you override one value for one command. The failure mode
is not the ordering but the *silence*: setting ``NLP_HISTO_ENV_FILE`` reads as "use this
configuration", and when an inherited ``DB_NAME`` quietly wins, a command aimed at a
scratch database writes to production instead. That happened during the 2026-07-16 ingest
verification and was caught only by a hand-written assertion.

So: when — and only when — an env file was named **explicitly**, a disagreement about
*where the connection points* is an error rather than a silent substitution. Ordinary
automatic ``.env`` discovery is untouched, and so is the documented env-wins behaviour.

**Target rendering.** ``format_target()`` is the single place that formats a resolved
connection for humans. Never the password, never a URL carrying credentials.
"""
from __future__ import annotations

import os
from pathlib import Path

# Fields that decide WHICH database you talk to. A disagreement here silently redirects
# writes, which is the whole point of the check.
ROUTING_VARS: tuple[str, ...] = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_SCHEMA")

# Deliberately NOT routing: injecting a secret from the environment while reading the rest
# from a file is a legitimate, common pattern (CI, containers). Treating it as a conflict
# would punish good practice — and a wrong password fails loudly on its own anyway.
SECRET_VARS: tuple[str, ...] = ("DB_PASSWORD",)


class EnvRoutingConflict(RuntimeError):
    """An explicit env file disagrees with the environment about connection routing."""


def _parse_env_file(path: Path) -> dict[str, str]:
    """Read ``KEY=VALUE`` pairs without mutating the environment.

    Must run *before* ``load_dotenv``: afterwards the file's values are indistinguishable
    from inherited ones, and the conflict is exactly what we are trying to see. Uses
    python-dotenv's parser when available so quoting/escaping match what will actually be
    loaded; falls back to a minimal reader otherwise.
    """
    try:
        from dotenv import dotenv_values

        return {k: v for k, v in dotenv_values(str(path)).items() if v is not None}
    except ImportError:
        values: dict[str, str] = {}
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
        return values


def detect_routing_conflict(
    env_file: str | os.PathLike[str] | None,
    environ: dict[str, str] | None = None,
) -> list[str]:
    """Return the routing variables where an explicit ``env_file`` loses to ``environ``.

    Empty when: no explicit file was given (automatic discovery — documented env-wins
    applies untouched); the file is unreadable or absent; or every routing value agrees.
    Secrets are never compared. Returns names only — values may be sensitive and are the
    caller's to know, not ours to print.
    """
    if not env_file:
        return []
    path = Path(env_file)
    if not path.is_file():
        return []

    environ = os.environ if environ is None else environ
    declared = _parse_env_file(path)

    conflicts = []
    for var in ROUTING_VARS:
        if var in declared and var in environ and declared[var] != environ[var]:
            conflicts.append(var)
    return conflicts


def raise_on_routing_conflict(
    env_file: str | os.PathLike[str] | None,
    environ: dict[str, str] | None = None,
) -> None:
    """Fail before connecting when an explicit env file is being silently overridden."""
    conflicts = detect_routing_conflict(env_file, environ)
    if not conflicts:
        return

    names = ", ".join(conflicts)
    unset = " ".join(f"-u {c}" for c in conflicts)
    raise EnvRoutingConflict(
        f"NLP_HISTO_ENV_FILE points at {env_file}, but these routing variables are already "
        f"set in the environment and would silently win: {names}.\n"
        f"\n"
        f"Refusing to connect: the environment beats the file (deliberately — see "
        f"database/ENV_LOADING.md), so this command would target whichever database your "
        f"shell already names, not the one you asked for. That is how a test run reaches "
        f"production (B-113).\n"
        f"\n"
        f"Resolve it by picking one source of truth:\n"
        f"  • use the file:        env {unset} <command>\n"
        f"  • use the environment: unset NLP_HISTO_ENV_FILE, and let the variables apply\n"
        f"\n"
        f"(Values are not shown. DB_PASSWORD and other secrets are not treated as "
        f"conflicts — injecting a secret from the environment is legitimate.)"
    )


def _format_target(user, host, port, database, schema: str | None = None) -> str:
    """The one rendering of a connection target. Never the password, never a URL.

    ``str(engine.url)`` embeds credentials, so it must not be printed; everything that
    shows a target goes through here so no command can invent its own format — or leak.
    """
    suffix = f" schema={schema}" if schema else ""
    return f"{user}@{host}:{port}/{database}{suffix}"


def format_target(url) -> str:
    """Render a SQLAlchemy engine URL as a secret-free target."""
    return _format_target(
        url.username, url.host, url.port, url.database, os.getenv("DB_SCHEMA")
    )


def format_target_config(cfg) -> str:
    """Render a ``DB_CONFIG``-shaped mapping as a secret-free target."""
    return _format_target(
        cfg["user"], cfg["host"], cfg["port"], cfg["database"],
        cfg.get("schema") or os.getenv("DB_SCHEMA"),
    )


def print_target(db, *, label: str = "Target") -> None:
    """Echo the resolved target of ``db`` once, before it is written to.

    `db init` / `db check` already announced their target; `ingest` and the NER commands
    did not — so a redirected write gave no sign of where it was going (B-113).
    """
    print(f"{label}: {format_target(db.engine.url)}", flush=True)
