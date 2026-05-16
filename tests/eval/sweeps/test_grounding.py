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


@pytest.fixture()
def grounding_module():
    spec = importlib.util.spec_from_file_location(
        "eval_sweeps_grounding_under_test", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)
    sys.modules.pop("_lib", None)


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


def test_csv_columns_include_missing_score_pct(tmp_path: Path, grounding_module) -> None:
    out_csv = tmp_path / "g.csv"
    grounding_module.main([
        "--input", str(FIXTURES / "summaries"),
        "--thresholds", "0.5",
        "--out-csv", str(out_csv),
        "--out-md", str(tmp_path / "g.md"),
        "--log-level", "WARNING",
    ])
    lines = [l for l in out_csv.read_text(encoding="utf-8").splitlines()
             if not l.startswith("#") and l.strip()]
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
    "pipeline.stages.summarization.llm_providers",
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
