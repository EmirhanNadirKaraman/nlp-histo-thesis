"""End-to-end tests for ``eval/sweeps/grounding.py``."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "eval" / "sweeps" / "grounding.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Sentinel distinguishing "``_lib`` was absent" from "``_lib`` was None".
_LIB_UNSET = object()


def _restore_sys_module(name: str, prev: object) -> None:
    """Restore ``sys.modules[name]`` to ``prev`` (delete it if it was absent).

    Used so the ``_lib``-manipulating fixtures below neither pollute nor erase
    the bare ``_lib`` entry for tests that run after them: they save the prior
    value on setup and put it back verbatim on teardown.
    """
    if prev is _LIB_UNSET:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = prev  # type: ignore[assignment]


@pytest.fixture()
def grounding_module():
    spec = importlib.util.spec_from_file_location(
        "eval_sweeps_grounding_under_test", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # grounding.py prepends its own dir to sys.path and does a bare ``import
    # _lib``. If a prior test (e.g. scripts/eval/* loaders) left a *different*
    # module cached under the bare name ``_lib``, that stale module would win
    # and the sweep would fail with an AttributeError — an order-dependent bug.
    # Save the current ``_lib`` and clear it so the fresh import binds
    # eval/sweeps/_lib.py; restore the exact prior state on teardown so we
    # neither pollute nor erase ``_lib`` for tests that run after us.
    prev_lib = sys.modules.get("_lib", _LIB_UNSET)
    sys.modules.pop("_lib", None)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)
    _restore_sys_module("_lib", prev_lib)


def test_cli_happy_path_writes_csv_and_md(tmp_path: Path, grounding_module) -> None:
    out_csv = tmp_path / "g.csv"
    out_md = tmp_path / "g.md"
    rc = grounding_module.main([
        "--input", str(FIXTURES / "summaries"),
        "--thresholds", "0.30,0.50,0.70",
        "--out-csv", str(out_csv),
        "--out-md", str(out_md),
        "--log-level", "WARNING",
    ])
    assert rc == 0
    assert out_csv.exists() and out_csv.stat().st_size > 0
    assert out_md.exists() and out_md.stat().st_size > 0


def test_cli_default_thresholds_parse(grounding_module) -> None:
    parsed = grounding_module.parse_thresholds(grounding_module.DEFAULT_THRESHOLDS)
    assert parsed == [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def test_cli_bad_thresholds_raises(grounding_module) -> None:
    with pytest.raises(Exception):  # argparse.ArgumentTypeError
        grounding_module.parse_thresholds("0.5,abc,0.7")


def test_multi_config_warning_emitted_by_default(tmp_path: Path, grounding_module) -> None:
    out_md = tmp_path / "g.md"
    rc = grounding_module.main([
        "--input", str(FIXTURES / "summaries_multi_config"),
        "--thresholds", "0.5",
        "--out-csv", str(tmp_path / "g.csv"),
        "--out-md", str(out_md),
        "--log-level", "WARNING",
    ])
    assert rc == 0
    text = out_md.read_text(encoding="utf-8")
    assert "Multiple pipeline_config_hashes detected" in text
    assert "hashA" in text and "hashB" in text


def test_strict_single_config_exits_on_multi_config(tmp_path: Path, grounding_module) -> None:
    rc = grounding_module.main([
        "--input", str(FIXTURES / "summaries_multi_config"),
        "--thresholds", "0.5",
        "--out-csv", str(tmp_path / "g.csv"),
        "--out-md", str(tmp_path / "g.md"),
        "--strict-single-config",
        "--log-level", "ERROR",
    ])
    assert rc == 2


def test_strict_single_config_passes_on_clean_input(tmp_path: Path, grounding_module) -> None:
    rc = grounding_module.main([
        "--input", str(FIXTURES / "summaries"),
        "--thresholds", "0.5",
        "--out-csv", str(tmp_path / "g.csv"),
        "--out-md", str(tmp_path / "g.md"),
        "--strict-single-config",
        "--log-level", "WARNING",
    ])
    assert rc == 0


def test_markdown_has_disclaimer_and_no_forbidden_phrasings(
    tmp_path: Path, grounding_module
) -> None:
    out_md = tmp_path / "g.md"
    grounding_module.main([
        "--input", str(FIXTURES / "summaries"),
        "--thresholds", "0.30,0.50,0.70",
        "--out-csv", str(tmp_path / "g.csv"),
        "--out-md", str(out_md),
        "--log-level", "WARNING",
    ])
    text = out_md.read_text(encoding="utf-8")
    assert "does NOT measure accuracy, precision, or recall" in text
    for forbidden in [
        "accuracy score", "precision score", "recall score",
        "F1 score", "F-1 score", "precision/recall",
        "accuracy=", "precision=", "recall=",
    ]:
        assert forbidden not in text, f"forbidden phrasing in report: {forbidden}"


@pytest.fixture()
def _stale_lib_pollution():
    """Simulate another test (e.g. a scripts/eval/* loader) having cached a
    *different* module under the bare name ``_lib`` and never cleaned it up.
    This is the exact trigger for the historical order-dependent failure."""
    other = REPO_ROOT / "scripts" / "eval" / "_lib.py"
    spec = importlib.util.spec_from_file_location("_lib", other)
    stale = importlib.util.module_from_spec(spec)
    prev_lib = sys.modules.get("_lib", _LIB_UNSET)
    sys.modules["_lib"] = stale
    spec.loader.exec_module(stale)
    # Sanity: this really is the *wrong* _lib (no sweep writer), as in the bug.
    assert not hasattr(stale, "write_sweep_markdown")
    yield
    _restore_sys_module("_lib", prev_lib)


def test_disclaimer_is_order_independent_under_stale_lib(
    tmp_path: Path, _stale_lib_pollution, grounding_module
) -> None:
    """Regression for the ``_lib`` sys.modules collision: ``_stale_lib_pollution``
    runs *before* ``grounding_module`` (fixture request order), so a stale
    ``_lib`` is cached when grounding.py is imported. The fixture's setup-time
    ``sys.modules.pop('_lib')`` rebinds the correct eval/sweeps/_lib.py, so the
    sweep runs and the disclaimer is present. Without that fix this raised
    ``AttributeError`` at grounding.py:94, making the disclaimer test order-dependent."""
    out_md = tmp_path / "g.md"
    rc = grounding_module.main([
        "--input", str(FIXTURES / "summaries"),
        "--thresholds", "0.5",
        "--out-csv", str(tmp_path / "g.csv"),
        "--out-md", str(out_md),
        "--log-level", "WARNING",
    ])
    assert rc == 0
    # Generated output landed in the test's tmp dir, never the tracked report.
    assert out_md.exists() and out_md.parent == tmp_path
    assert "does NOT measure accuracy, precision, or recall" in out_md.read_text(
        encoding="utf-8"
    )


@pytest.fixture()
def _known_lib_sentinel():
    """Cache a recognizable sentinel under ``_lib`` and, on teardown — which
    runs *after* grounding_module's teardown (fixtures finalize in reverse
    setup order) — assert the sentinel was restored. This proves the
    grounding_module fixture puts the prior ``_lib`` back rather than deleting
    it (i.e. it neither pollutes nor erases state for later tests)."""
    import types

    sentinel = types.ModuleType("_lib")
    sentinel.__nlp_histo_sentinel__ = True  # type: ignore[attr-defined]
    prev = sys.modules.get("_lib", _LIB_UNSET)
    sys.modules["_lib"] = sentinel
    yield sentinel
    assert sys.modules.get("_lib") is sentinel, (
        "grounding_module fixture did not restore the prior sys.modules['_lib'] "
        "on teardown (it must save/restore, not delete)"
    )
    _restore_sys_module("_lib", prev)


def test_grounding_module_restores_prior_lib_on_teardown(
    tmp_path: Path, _known_lib_sentinel, grounding_module
) -> None:
    """``_known_lib_sentinel`` sets ``_lib`` to a sentinel *before*
    grounding_module runs. During the test the correct eval/sweeps/_lib.py is
    bound (sentinel cleared), so the sweep works; the restore is asserted by
    ``_known_lib_sentinel``'s teardown, which runs after grounding_module tears
    down."""
    assert sys.modules.get("_lib") is not _known_lib_sentinel  # correct _lib active
    rc = grounding_module.main([
        "--input", str(FIXTURES / "summaries"),
        "--thresholds", "0.5",
        "--out-csv", str(tmp_path / "g.csv"),
        "--out-md", str(tmp_path / "g.md"),
        "--log-level", "WARNING",
    ])
    assert rc == 0


def test_csv_columns_include_missing_score_pct(tmp_path: Path, grounding_module) -> None:
    out_csv = tmp_path / "g.csv"
    grounding_module.main([
        "--input", str(FIXTURES / "summaries"),
        "--thresholds", "0.5",
        "--out-csv", str(out_csv),
        "--out-md", str(tmp_path / "g.md"),
        "--log-level", "WARNING",
    ])
    lines = [line for line in out_csv.read_text(encoding="utf-8").splitlines()
             if not line.startswith("#") and line.strip()]
    header = lines[0]
    assert "missing_score_count" in header
    assert "missing_score_pct" in header
    assert "kept_pct_of_scored" in header
    assert "rejected_pct_of_scored" in header
    # And the "naked" kept_pct that would obscure the denominator is NOT here.
    assert ",kept_pct," not in "," + header + ","


def test_missing_input_dir_exits_with_rc_1(tmp_path: Path, grounding_module) -> None:
    rc = grounding_module.main([
        "--input", str(tmp_path / "does_not_exist"),
        "--out-csv", str(tmp_path / "g.csv"),
        "--out-md", str(tmp_path / "g.md"),
        "--log-level", "ERROR",
    ])
    assert rc == 1


BANNED_MODULE_PREFIXES: tuple[str, ...] = (
    "transformers",
    "torch",
    "openai",
    "anthropic",
    "google.genai",
    "google.generativeai",
    "vertexai",
    "sentence_transformers",
    "pipeline.stages.knowledge_extraction.llm.llm_providers",
)


def test_import_safety_no_nli_or_llm_modules(tmp_path: Path) -> None:
    """Importing the script must not pull in NLI / LLM / embedding modules."""
    script = (
        "import sys, importlib.util;"
        f"spec = importlib.util.spec_from_file_location('m', r'{SCRIPT_PATH}');"
        "mod = importlib.util.module_from_spec(spec);"
        "sys.modules['m'] = mod;"
        "spec.loader.exec_module(mod);"
        "banned = "
        f"{BANNED_MODULE_PREFIXES!r};"
        "loaded = sorted({n for n in sys.modules if any(n == b or n.startswith(b + '.') for b in banned)});"
        "print('LOADED:' + (','.join(loaded) if loaded else 'none'));"
        "sys.exit(1 if loaded else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"banned imports detected: {result.stdout!r}"
