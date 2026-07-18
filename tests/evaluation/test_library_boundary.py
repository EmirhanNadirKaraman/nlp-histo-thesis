"""The installed evaluation library must not reach back into the repository.

``nlp_histo.evaluation`` ships in the wheel; the thesis experiment drivers (E01–E14,
sweeps, calibration, report generation) do not. So the library may never, at import
time, read ``eval/data`` or ``eval/reports``, walk up to a repository root, or load a
frozen artifact — none of which exist beside an installed wheel.

These tests pin that boundary. They are cheap AST/import checks, not behaviour tests:
the matcher's own behaviour stays covered by the existing matcher test-suite.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import nlp_histo.evaluation

LIBRARY_MODULES = [
    "nlp_histo.evaluation.schemas",
    "nlp_histo.evaluation.jsonl_utils",
    "nlp_histo.evaluation.split",
    "nlp_histo.evaluation.matching",
    "nlp_histo.evaluation.matching.embedders",
    "nlp_histo.evaluation.matching.matcher",
]


def _source_files() -> list[Path]:
    root = Path(nlp_histo.evaluation.__file__).parent
    return sorted(root.rglob("*.py"))


@pytest.mark.parametrize("module", LIBRARY_MODULES)
def test_library_modules_import(module: str) -> None:
    """Importing must work with no repository, no database, no API key, no artifacts."""
    assert importlib.import_module(module) is not None


@pytest.mark.parametrize("module", LIBRARY_MODULES)
def test_no_repository_root_walk(module: str) -> None:
    """No `Path(__file__).parents[n]` — an installed wheel has no repository tree."""
    src = Path(importlib.import_module(module).__file__).read_text(encoding="utf-8")
    walks = [
        node
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Attribute) and node.attr == "parents"
    ]
    assert not walks, f"{module} walks up to a repository root ({len(walks)} `.parents` access)"


def test_no_module_executes_work_at_import() -> None:
    """No top-level statements beyond imports, constants, defs and classes."""
    offenders: list[str] = []
    for path in _source_files():
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(
                node,
                (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
                 ast.ClassDef, ast.Assign, ast.AnnAssign, ast.If, ast.Try),
            ):
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue  # docstring
            offenders.append(f"{path.name}:{node.lineno} {type(node).__name__}")
    assert not offenders, f"evaluation library runs work at import: {offenders}"


def test_library_reads_no_artifact_at_import() -> None:
    """No open()/read_text()/read_bytes()/load() call at module scope."""
    readers = {"open", "read_text", "read_bytes", "load", "safe_load", "loads"}
    offenders: list[str] = []
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # only module-level statements — calls inside functions are fine
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                    if name in readers:
                        offenders.append(f"{path.name}:{sub.lineno} {name}()")
    assert not offenders, f"evaluation library reads artifacts at import: {offenders}"


def test_matcher_cache_default_is_not_cwd_relative(monkeypatch, tmp_path) -> None:
    """The embedding cache must not default to a repository-relative path.

    It used to be ``Path("eval/data/embedding_cache_openai.sqlite")`` — pinned in this
    file while the value was still frozen. The path pass replaced it with an explicit
    resolver (argument > env var > user cache dir), because a cwd-relative default
    misses away from the repository root, and a miss means paid embedding calls.
    The repository experiment drivers now pass their frozen sqlite paths explicitly,
    so their behaviour — and their avoidance of paid calls — is unchanged.
    """
    from nlp_histo.evaluation.matching import matcher

    assert not hasattr(matcher, "DEFAULT_CACHE_PATH")
    assert not hasattr(matcher, "DEFAULT_GEMINI_CACHE_PATH")

    monkeypatch.delenv(matcher.OPENAI_CACHE_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    default = matcher.default_embedding_cache_path("openai")
    assert default.is_absolute()
    assert default == tmp_path / "nlp-histo" / "embedding_cache_openai.sqlite"


def test_repository_experiment_drivers_import_the_installed_library() -> None:
    """The dependency points repo → library, never library → repo."""
    from eval.silver.data import sample  # a repository-only driver

    src = Path(sample.__file__).read_text(encoding="utf-8")
    assert "nlp_histo.evaluation" in src
    assert "eval.silver.data.schemas" not in src, "driver still imports the pre-move path"
