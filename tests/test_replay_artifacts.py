"""The chapter-9 replay must validate its whole artifact set before it runs.

The failure this guards against is not a crash — it is a *partial result*. With an
incomplete artifact root the replay used to run anyway and emit 4 CSVs instead of 9.
Four CSVs look like an answer. They are not: the analyses whose inputs were missing
simply never ran, and nothing said so at the top.

So validation is one pass over the complete required set, before any directory is
created and before any analysis executes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nlp_histo.workflows import replay

SQLITE_HEADER = b"SQLite format 3\x00"


def _complete_artifact_root(root: Path) -> Path:
    """A minimal tree containing every required artifact, one of each category."""
    (root / "out" / "summaries" / "summaries").mkdir(parents=True)
    (root / "out" / "summaries" / "summaries" / "PMC1.json").write_text("{}")

    (root / "out" / "summaries" / "cascade_decisions").mkdir(parents=True)
    (root / "out" / "summaries" / "cascade_decisions" / "PMC1.jsonl").write_text("{}\n")

    primer = root / "eval" / "data" / "map_primer"
    primer.mkdir(parents=True)
    (primer / "voter_cache.json").write_text("{}")
    (root / "eval" / "data" / "silver_findings_related15.jsonl").write_text('{"a": 1}\n')
    # Both embedders are exercised by the replay — 05 loads the map context with
    # "openai", 06/10/12 with "gemini" — so both caches are required (B-112).
    (root / "eval" / "data" / "embedding_cache_openai.sqlite").write_bytes(
        SQLITE_HEADER + b"\x00" * 16
    )
    (root / "eval" / "data" / "embedding_cache_gemini.sqlite").write_bytes(
        SQLITE_HEADER + b"\x00" * 16
    )

    (root / "scripts" / "eval").mkdir(parents=True)
    (root / "scripts" / "eval" / "run_summarization_experiments.py").write_text("x = 1\n")

    (root / "reports").mkdir(parents=True)
    (root / "reports" / "stage6_PR.md").write_text("# stage 6\n")

    for mode in ("predicate", "verbatim", "scope_predicate", "scope_verbatim"):
        (root / "out" / "summaries" / f"corpus_relations_{mode}.json").write_text("{}")
    return root


# ── the contract itself ───────────────────────────────────────────────────────

def test_complete_fixture_passes_validation(tmp_path: Path) -> None:
    assert replay.validate_artifacts(_complete_artifact_root(tmp_path)) == []


def test_every_required_category_is_covered_by_the_fixture(tmp_path: Path) -> None:
    """If a new required artifact is added, this fixture must grow with it."""
    root = _complete_artifact_root(tmp_path)
    for artifact in replay.REQUIRED_ARTIFACTS:
        assert artifact.validate(root) is None, (
            f"fixture does not satisfy {artifact.describe()}"
        )
    kinds = {a.kind for a in replay.REQUIRED_ARTIFACTS}
    assert kinds == {"file", "dir", "sqlite", "glob"}, kinds


def test_the_orchestrator_and_rubric_report_are_required(tmp_path: Path) -> None:
    """These two were silently unchecked — they are what produced the 4/9 partial run."""
    required = {a.relative_path for a in replay.REQUIRED_ARTIFACTS}
    assert Path("scripts/eval/run_summarization_experiments.py") in required
    assert Path("reports/stage6_PR.md") in required
    assert Path("out/summaries/cascade_decisions") in required
    assert Path("eval/data/silver_findings_related15.jsonl") in required


# ── failure modes ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "victim",
    [
        "out/summaries/summaries",
        "out/summaries/cascade_decisions",
        "eval/data/map_primer/voter_cache.json",
        "eval/data/silver_findings_related15.jsonl",
        "eval/data/embedding_cache_openai.sqlite",
        "eval/data/embedding_cache_gemini.sqlite",
        "scripts/eval/run_summarization_experiments.py",
        "reports/stage6_PR.md",
    ],
)
def test_any_single_missing_artifact_fails(tmp_path: Path, victim: str) -> None:
    root = _complete_artifact_root(tmp_path)
    target = root / victim
    if target.is_dir():
        for child in target.iterdir():
            child.unlink()
        target.rmdir()
    else:
        target.unlink()

    problems = replay.validate_artifacts(root)
    assert len(problems) == 1
    assert victim in problems[0]


def test_multiple_missing_artifacts_are_reported_together(tmp_path: Path, capsys) -> None:
    root = _complete_artifact_root(tmp_path)
    (root / "reports" / "stage6_PR.md").unlink()
    (root / "eval" / "data" / "silver_findings_related15.jsonl").unlink()

    rc = replay.main(["--artifact-root", str(root)])
    assert rc == 2

    err = capsys.readouterr().err
    assert "stage6_PR.md" in err
    assert "silver_findings_related15.jsonl" in err
    assert "Required for:" in err


def test_wrong_file_type_is_rejected(tmp_path: Path) -> None:
    root = _complete_artifact_root(tmp_path)
    cache = root / "eval" / "data" / "map_primer" / "voter_cache.json"
    cache.unlink()
    cache.mkdir()  # a directory where a file is required

    problems = replay.validate_artifacts(root)
    assert any("not a regular file" in p for p in problems)


def test_empty_sqlite_cache_is_rejected(tmp_path: Path) -> None:
    root = _complete_artifact_root(tmp_path)
    (root / "eval" / "data" / "embedding_cache_openai.sqlite").write_bytes(b"")

    problems = replay.validate_artifacts(root)
    assert any("empty" in p for p in problems), problems


def test_non_sqlite_cache_is_rejected(tmp_path: Path) -> None:
    """A truncated or wrong-format cache must not be treated as usable — a cache miss
    means PAID embedding calls in a workflow documented as free."""
    root = _complete_artifact_root(tmp_path)
    (root / "eval" / "data" / "embedding_cache_openai.sqlite").write_bytes(b"not sqlite")

    problems = replay.validate_artifacts(root)
    assert any("not a SQLite database" in p for p in problems)


def test_missing_corpus_relation_variants_are_rejected(tmp_path: Path) -> None:
    """Analysis 07 needs at least one variant; with none it silently emitted no CSV."""
    root = _complete_artifact_root(tmp_path)
    for f in (root / "out" / "summaries").glob("corpus_relations*.json"):
        f.unlink()

    problems = replay.validate_artifacts(root)
    assert any("corpus_relations*.json" in p for p in problems), problems


def test_a_single_corpus_relation_variant_is_enough(tmp_path: Path) -> None:
    """The frozen tree carries only two of the four variants — requiring all would be wrong."""
    root = _complete_artifact_root(tmp_path)
    for f in (root / "out" / "summaries").glob("corpus_relations*.json"):
        f.unlink()
    (root / "out" / "summaries" / "corpus_relations_predicate.json").write_text("{}")

    assert replay.validate_artifacts(root) == []


def test_empty_required_directory_is_rejected(tmp_path: Path) -> None:
    root = _complete_artifact_root(tmp_path)
    for child in (root / "out" / "summaries" / "cascade_decisions").iterdir():
        child.unlink()

    problems = replay.validate_artifacts(root)
    assert any("empty" in p for p in problems)


# ── nothing may be produced when validation fails ─────────────────────────────

def test_failed_validation_creates_no_output_and_runs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "incomplete"
    root.mkdir()
    out = tmp_path / "out"

    entered = []
    monkeypatch.setattr(replay, "_run_replay", lambda: entered.append(True))

    rc = replay.main(["--artifact-root", str(root), "--output-dir", str(out)])

    assert rc == 2
    assert not entered, "the replay executed despite failing validation"
    assert not out.exists(), "an output directory was created before validation passed"
    assert not list(tmp_path.rglob("*.csv")), "a CSV was written despite failing validation"


def test_incomplete_root_cannot_produce_a_partial_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 4-of-9 regression, pinned: a root missing only the orchestrator and the
    rubric report used to pass the old three-item preflight and then half-run."""
    root = _complete_artifact_root(tmp_path)
    (root / "scripts" / "eval" / "run_summarization_experiments.py").unlink()
    (root / "reports" / "stage6_PR.md").unlink()

    entered = []
    monkeypatch.setattr(replay, "_run_replay", lambda: entered.append(True))

    assert replay.main(["--artifact-root", str(root)]) == 2
    assert not entered


# ── paths come from --artifact-root, not the cwd ──────────────────────────────

def test_inputs_resolve_from_the_artifact_root_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _complete_artifact_root(tmp_path / "root")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # a cwd with no artifacts at all

    assert replay.validate_artifacts(root) == []
    replay._REPO_ROOT = root
    assert replay.frozen_embedding_cache() == (
        root / "eval" / "data" / "embedding_cache_openai.sqlite"
    )


def test_help_is_side_effect_free(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        replay.main(["--help"])
    assert exc.value.code == 0
    assert not list(tmp_path.iterdir()), "--help touched the filesystem"
    assert "artifact-root" in capsys.readouterr().out
