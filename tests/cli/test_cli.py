"""The public CLI: help is cheap and safe, dispatch reaches the right workflow.

The load-bearing property here is what --help does not do. Printing help must not
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


# help must not touch the world

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


def test_explicit_passthrough_reaches_the_workflow_help(monkeypatch) -> None:
    """`nlp-histo ingest -- --help` forwards `--help` to the runner, dropping the `--`.

    The `--` must not be forwarded: argparse reads it as the positional separator and
    then rejects `--help` as an unrecognized positional (B-111). The CLI's own ingest
    help advertises this invocation, so it has to work.
    """
    forwarded: list[list[str]] = []
    from nlp_histo.pipeline.stages.pdf_text_extraction import runner

    monkeypatch.setattr(runner, "main", lambda argv=None: forwarded.append(list(argv or [])))
    main(["ingest", "--", "--help"])
    assert forwarded == [["--help"]]


def test_plain_help_after_command_is_the_clis_own(monkeypatch) -> None:
    """Without an explicit `--`, `--help` belongs to this CLI, not the workflow —
    that is what keeps the knowledge cost warning reachable."""
    from nlp_histo.pipeline.stages.pdf_text_extraction import runner

    monkeypatch.setattr(runner, "main", lambda argv=None: pytest.fail("must not dispatch"))
    with pytest.raises(SystemExit):
        main(["ingest", "--help"])


# dispatch reaches the intended workflow

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

    from nlp_histo.acquisition.downloader import DownloadReport

    def _fake_download(pmcid_file, output_dir, *, overwrite=False, source="aws"):
        seen.update(
            file=Path(pmcid_file), out=Path(output_dir),
            overwrite=overwrite, source=source,
        )
        # Mirror the real signature: a DownloadReport, not a bare count. Any `failed`
        # would make the CLI exit non-zero (B-117).
        return DownloadReport(requested=1, succeeded=1, failed=0, skipped=0)

    monkeypatch.setattr(downloader, "download_papers", _fake_download)
    ids = tmp_path / "ids.txt"
    ids.write_text("PMC1\n")
    assert main(["acquire", "download", "--pmcid-file", str(ids),
                 "--output-dir", str(tmp_path / "tars")]) == 0
    assert seen["file"] == ids
    assert seen["out"] == tmp_path / "tars"
    assert seen["overwrite"] is False
    assert seen["source"] == "aws", "the durable backend is the default (B-118)"


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


# acquisition import safety

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
    assert "artifact validation failed" in err
    assert "voter_cache.json" in err
    # every missing input is reported in one pass, each with why it is needed
    assert "embedding_cache_openai.sqlite" in err
    assert "Required for:" in err


# B-117's exit-code contract (total/partial failure, skips, empty input, zero-byte and
# non-archive responses) lives in tests/test_acquire_download_outcomes.py — the dispatch
# test below only checks that the CLI forwards its arguments.


# B-107: a UMLS failure must stop the replay, not shade its numbers

def test_replay_refuses_and_exits_nonzero_when_umls_is_unavailable(tmp_path, capsys, monkeypatch) -> None:
    """Previously: one WARNING, exit 0, and wrong 06/12 tables on disk."""
    from nlp_histo.workflows import replay
    from nlp_histo.pipeline.stages.knowledge_extraction.entities.umls_resources import (
        UmlsUnavailableError,
    )

    # Artifact validation must pass so we reach the UMLS gate, not stop short of it.
    monkeypatch.setattr(replay, "validate_artifacts", lambda root: [])

    def _unavailable() -> None:
        raise UmlsUnavailableError(
            "The chapter-9 replay requires the UMLS entity linker, which is unavailable.\n"
            "  - 06_exp_f_test_split.csv / .json\n"
            "Nothing has been written."
        )

    monkeypatch.setattr(replay, "_require_umls_or_refuse", _unavailable)

    out_dir = tmp_path / "replay-out"
    rc = replay.main([
        "--artifact-root", str(tmp_path), "--output-dir", str(out_dir),
    ])

    assert rc == 3, "a UMLS failure must exit non-zero"
    err = capsys.readouterr().err
    assert "UMLS" in err and "06_exp_f_test_split" in err
    assert not out_dir.exists(), "refused run must not leave an output directory behind"


def test_replay_umls_gate_runs_before_any_output_exists(tmp_path, monkeypatch) -> None:
    """Ordering is the guarantee: the gate must fire before the output tree is made,
    so a refused run cannot leave partial CSVs that read like results."""
    from nlp_histo.workflows import replay

    monkeypatch.setattr(replay, "validate_artifacts", lambda root: [])
    # The embedding-cache gate (B-112) also runs in configure(); stub it so this test
    # isolates the UMLS ordering guarantee. Its own ordering is covered in
    # test_replay_embedding_cache_preflight.py.
    monkeypatch.setattr(replay, "validate_embedding_cache_entries", lambda root: None)
    out_dir = tmp_path / "replay-out"
    seen: dict[str, bool] = {}

    def _probe() -> None:
        seen["out_dir_existed_at_probe_time"] = out_dir.exists()

    monkeypatch.setattr(replay, "_require_umls_or_refuse", _probe)
    monkeypatch.setattr(replay, "_run_replay", lambda: None)

    assert replay.main(["--artifact-root", str(tmp_path), "--output-dir", str(out_dir)]) == 0
    assert seen["out_dir_existed_at_probe_time"] is False


def test_replay_succeeds_normally_when_umls_is_available(tmp_path, monkeypatch) -> None:
    """The working path is unchanged: gate passes, replay runs, exit 0."""
    from nlp_histo.workflows import replay

    monkeypatch.setattr(replay, "validate_artifacts", lambda root: [])
    monkeypatch.setattr(replay, "_require_umls_or_refuse", lambda: None)
    monkeypatch.setattr(replay, "validate_embedding_cache_entries", lambda root: None)
    ran: list[bool] = []
    monkeypatch.setattr(replay, "_run_replay", lambda: ran.append(True))

    rc = replay.main([
        "--artifact-root", str(tmp_path), "--output-dir", str(tmp_path / "o"),
    ])
    assert rc == 0
    assert ran == [True]
