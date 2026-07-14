"""The public CLI: help is cheap and safe, dispatch reaches the right workflow.

The load-bearing property here is *what --help does not do*. Printing help must not
open a PostgreSQL connection, load scispaCy, initialise UMLS, construct a provider
SDK client, or create a cache or output directory — otherwise `nlp-histo --help` on a
fresh machine fails or, worse, quietly writes somewhere. The handlers import their
workflows lazily to guarantee that, and these tests pin it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nlp_histo.cli.main import COST_WARNING, main

HELP_PATHS = [
    [],
    ["db"],
    ["db", "init"],
    ["db", "check"],
    ["acquire"],
    ["acquire", "download"],
    ["acquire", "unpack"],
    ["acquire", "organize"],
    ["ingest"],
    ["ner"],
    ["ner", "extract"],
    ["ner", "merge"],
    ["ner", "export"],
    ["knowledge"],
    ["replay"],
    ["replay", "chapter9"],
]


@pytest.mark.parametrize("path", HELP_PATHS, ids=lambda p: " ".join(p) or "root")
def test_help_exits_zero_and_prints_usage(path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main([*path, "--help"])
    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_unknown_command_is_non_zero(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["not-a-command"])
    assert exc.value.code != 0


def test_bare_invocation_prints_help_and_returns_non_zero(capsys) -> None:
    assert main([]) == 1
    assert "usage:" in capsys.readouterr().out


def test_command_group_without_subcommand_shows_group_help(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["db"])  # group help path exits via argparse


# ── help must not touch the world ─────────────────────────────────────────────

HEAVY = (
    "sqlalchemy.engine.base",  # a live Engine implies a DB connect attempt
    "spacy",
    "scispacy",
    "torch",
    "transformers",
    "openai",
    "anthropic",
)


def test_help_loads_no_heavy_module_and_touches_no_disk(tmp_path, monkeypatch) -> None:
    """--help must not import models/SDKs, connect to a DB, or create directories."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))  # any cache write lands here
    monkeypatch.setattr("nlp_histo.cli.main.__doc__", "")  # no-op, keeps monkeypatch imported

    before = {m for m in sys.modules}
    with pytest.raises(SystemExit):
        main(["--help"])
    newly = {m for m in sys.modules} - before

    leaked = sorted(m for m in newly if m.split(".")[0] in {h.split(".")[0] for h in HEAVY})
    assert not leaked, f"--help imported heavy modules: {leaked}"
    assert not list(tmp_path.iterdir()), "--help created files/directories"


def test_missing_required_path_argument_errors_clearly(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["acquire", "download"])  # --pmcid-file / --output-dir are required
    assert exc.value.code != 0
    assert "required" in capsys.readouterr().err.lower()


def test_knowledge_help_states_that_it_costs_money(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["knowledge", "--help"])
    out = capsys.readouterr().out
    assert "PAID" in out or "PAID" in COST_WARNING
    assert "paid" in out.lower()


# ── dispatch reaches the intended workflow ────────────────────────────────────

def test_db_init_dispatches_to_init_db(monkeypatch) -> None:
    calls: list[list[str]] = []
    import nlp_histo.database.init_db as init_db

    monkeypatch.setattr(init_db, "main", lambda argv=None: calls.append(list(argv or [])) or 0)
    assert main(["db", "init"]) == 0
    assert calls == [[]]


def test_db_check_forwards_check_only(monkeypatch) -> None:
    calls: list[list[str]] = []
    import nlp_histo.database.init_db as init_db

    monkeypatch.setattr(init_db, "main", lambda argv=None: calls.append(list(argv or [])) or 0)
    assert main(["db", "check"]) == 0
    assert calls == [["--check-only"]]


def test_acquire_download_dispatches_with_explicit_paths(monkeypatch, tmp_path) -> None:
    seen: dict = {}
    import nlp_histo.acquisition.downloader as downloader

    monkeypatch.setattr(
        downloader, "download_papers",
        lambda pmcid_file, output_dir, *, overwrite=False: seen.update(
            file=Path(pmcid_file), out=Path(output_dir), overwrite=overwrite
        ),
    )
    ids = tmp_path / "ids.txt"
    ids.write_text("PMC1\n")
    assert main(["acquire", "download", "--pmcid-file", str(ids),
                 "--output-dir", str(tmp_path / "tars")]) == 0
    assert seen["file"] == ids
    assert seen["out"] == tmp_path / "tars"
    assert seen["overwrite"] is False


def test_ner_extract_forwards_entity_cache_flag(monkeypatch, tmp_path) -> None:
    seen: list[list[str]] = []
    from nlp_histo.ner import batch_ner

    monkeypatch.setattr(batch_ner, "main", lambda argv=None: seen.append(list(argv or [])) or 0)
    cache = tmp_path / "c.json"
    assert main(["ner", "extract", "--entity-cache", str(cache)]) == 0
    assert seen == [["--entity-cache", str(cache)]]


def test_replay_chapter9_dispatches_to_replay_workflow(monkeypatch) -> None:
    seen: list[list[str]] = []
    from nlp_histo.workflows import replay

    monkeypatch.setattr(replay, "main", lambda argv=None: seen.append(list(argv or [])) or 0)
    assert main(["replay", "chapter9", "--artifact-root", "/somewhere"]) == 0
    assert seen == [["--artifact-root", "/somewhere"]]


def test_knowledge_dispatches_to_knowledge_workflow(monkeypatch) -> None:
    seen: list[list[str]] = []
    from nlp_histo.workflows import knowledge

    monkeypatch.setattr(knowledge, "main", lambda argv=None: seen.append(list(argv or [])) or 0)
    assert main(["knowledge", "--profile", "cheap"]) == 0
    assert seen == [["--profile", "cheap"]]


# ── acquisition import safety ─────────────────────────────────────────────────

def test_acquisition_modules_are_import_safe(tmp_path, monkeypatch) -> None:
    """Importing must perform no network access and create no directories.

    The downloader used to run its whole download loop at module scope.
    """
    monkeypatch.chdir(tmp_path)
    import importlib

    for mod in ("downloader", "tarballs", "organizer"):
        importlib.import_module(f"nlp_histo.acquisition.{mod}")
    assert not list(tmp_path.iterdir()), "importing acquisition created files"


def test_replay_import_creates_no_output_dirs(tmp_path, monkeypatch) -> None:
    """The replay used to mkdir its output tree at import."""
    monkeypatch.chdir(tmp_path)
    import importlib

    importlib.import_module("nlp_histo.workflows.replay")
    assert not list(tmp_path.iterdir())


def test_replay_missing_artifacts_is_a_clear_error(tmp_path, capsys) -> None:
    from nlp_histo.workflows import replay

    rc = replay.main(["--artifact-root", str(tmp_path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "missing required inputs" in err
    assert "voter_cache.json" in err
