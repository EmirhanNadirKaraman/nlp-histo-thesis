"""Tests for ``eval/sweeps/grounding_plot.py``.

Layer A invariant: matplotlib's ``Agg`` backend is acceptable (it doesn't
talk to APIs or run pipeline code); the banned-imports list is unchanged.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "eval" / "sweeps" / "grounding_plot.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture()
def plot_module():
    spec = importlib.util.spec_from_file_location(
        "eval_sweeps_grounding_plot_under_test", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


def _write_sweep_csv(path: Path, *, rows: list[dict]) -> None:
    """Write a minimal sweep CSV with the same shape grounding.py emits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "threshold", "total_findings", "scored_findings",
        "missing_score_count", "missing_score_pct",
        "kept_findings", "rejected_findings",
        "kept_pct_of_scored", "rejected_pct_of_scored",
        "papers_covered",
        "mean_grounding_score_kept", "mean_grounding_score_rejected",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        # Mirror grounding.py's leading ``#``-comment metadata block.
        f.write("# sweep_name: \"grounding_threshold\"\n")
        f.write("# pipeline_config_hashes: [\"fixture_hash\"]\n")
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_sweep_rows() -> list[dict]:
    return [
        {
            "threshold": 0.30, "total_findings": 100, "scored_findings": 100,
            "missing_score_count": 0, "missing_score_pct": 0.0,
            "kept_findings": 90, "rejected_findings": 10,
            "kept_pct_of_scored": 90.0, "rejected_pct_of_scored": 10.0,
            "papers_covered": 2,
            "mean_grounding_score_kept": 0.96,
            "mean_grounding_score_rejected": 0.10,
        },
        {
            "threshold": 0.50, "total_findings": 100, "scored_findings": 100,
            "missing_score_count": 0, "missing_score_pct": 0.0,
            "kept_findings": 85, "rejected_findings": 15,
            "kept_pct_of_scored": 85.0, "rejected_pct_of_scored": 15.0,
            "papers_covered": 2,
            "mean_grounding_score_kept": 0.97,
            "mean_grounding_score_rejected": 0.18,
        },
        {
            "threshold": 0.70, "total_findings": 100, "scored_findings": 100,
            "missing_score_count": 0, "missing_score_pct": 0.0,
            "kept_findings": 80, "rejected_findings": 20,
            "kept_pct_of_scored": 80.0, "rejected_pct_of_scored": 20.0,
            "papers_covered": 2,
            "mean_grounding_score_kept": 0.98,
            "mean_grounding_score_rejected": 0.28,
        },
    ]


def test_read_sweep_csv_skips_comment_lines(tmp_path: Path, plot_module) -> None:
    csv_path = tmp_path / "sweep.csv"
    _write_sweep_csv(csv_path, rows=_make_sweep_rows())
    rows = plot_module.read_sweep_csv(csv_path)
    assert len(rows) == 3
    assert rows[0]["threshold"] == 0.30
    assert rows[0]["kept_pct_of_scored"] == 90.0


def test_main_writes_two_pngs_against_fixture(tmp_path: Path, plot_module) -> None:
    csv_path = tmp_path / "sweep.csv"
    _write_sweep_csv(csv_path, rows=_make_sweep_rows())
    out_ret = tmp_path / "retention.png"
    out_dist = tmp_path / "distribution.png"
    rc = plot_module.main([
        "--input", str(FIXTURES / "summaries"),
        "--sweep-csv", str(csv_path),
        "--out-retention", str(out_ret),
        "--out-distribution", str(out_dist),
        "--threshold", "0.5",
        "--log-level", "WARNING",
    ])
    assert rc == 0
    assert out_ret.exists() and out_ret.stat().st_size > 1000
    assert out_dist.exists() and out_dist.stat().st_size > 1000
    assert out_ret.read_bytes()[:8] == PNG_MAGIC
    assert out_dist.read_bytes()[:8] == PNG_MAGIC


def test_missing_sweep_csv_exits_with_rc_1(tmp_path: Path, plot_module) -> None:
    rc = plot_module.main([
        "--input", str(FIXTURES / "summaries"),
        "--sweep-csv", str(tmp_path / "absent.csv"),
        "--out-retention", str(tmp_path / "r.png"),
        "--out-distribution", str(tmp_path / "d.png"),
        "--log-level", "ERROR",
    ])
    assert rc == 1


def test_missing_input_dir_exits_with_rc_1(tmp_path: Path, plot_module) -> None:
    csv_path = tmp_path / "sweep.csv"
    _write_sweep_csv(csv_path, rows=_make_sweep_rows())
    rc = plot_module.main([
        "--input", str(tmp_path / "nope"),
        "--sweep-csv", str(csv_path),
        "--out-retention", str(tmp_path / "r.png"),
        "--out-distribution", str(tmp_path / "d.png"),
        "--log-level", "ERROR",
    ])
    assert rc == 1


def test_matplotlib_backend_is_agg(plot_module) -> None:
    import matplotlib
    assert matplotlib.get_backend().lower() == "agg", (
        "grounding_plot.py must force the Agg backend so it works headless"
    )


def test_plot_retention_handles_missing_rejected_means(tmp_path: Path, plot_module) -> None:
    """First row at low threshold may have no rejected findings → mean is None.
    The plotter should not crash."""
    rows = _make_sweep_rows()
    rows[0]["mean_grounding_score_rejected"] = None
    out = tmp_path / "r.png"
    plot_module.plot_retention(rows, out, producer_threshold=0.5)
    assert out.exists() and out.read_bytes()[:8] == PNG_MAGIC


# ---------------------------------------------------------------------------
# Markdown injection (idempotent)
# ---------------------------------------------------------------------------


def test_inject_plot_refs_into_markdown_inserts_block(
    tmp_path: Path, plot_module,
) -> None:
    md = tmp_path / "sweep.md"
    md.write_text(
        "# Sweep\n\n## Retention by threshold\n\n| t | k |\n|---|---|\n| 0.5 | 90 |\n\n"
        "## Notes\n\n- existing notes\n",
        encoding="utf-8",
    )
    ret = tmp_path / "retention.png"
    dist = tmp_path / "distribution.png"
    ret.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    dist.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    changed = plot_module.inject_plot_refs_into_markdown(
        md, retention_png=ret, distribution_png=dist,
    )
    assert changed is True
    text = md.read_text(encoding="utf-8")
    assert plot_module.MD_PLOT_BEGIN in text
    assert plot_module.MD_PLOT_END in text
    assert "![Retention curve at varying thresholds](retention.png)" in text
    assert "![Distribution of grounding scores](distribution.png)" in text
    # Block lands before "## Notes".
    assert text.index(plot_module.MD_PLOT_BEGIN) < text.index("## Notes")


def test_inject_plot_refs_is_idempotent(tmp_path: Path, plot_module) -> None:
    md = tmp_path / "sweep.md"
    md.write_text("# Sweep\n\n## Notes\n", encoding="utf-8")
    ret = tmp_path / "retention.png"
    dist = tmp_path / "distribution.png"
    ret.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    dist.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    plot_module.inject_plot_refs_into_markdown(
        md, retention_png=ret, distribution_png=dist,
    )
    first = md.read_text(encoding="utf-8")
    # Second call must produce identical content (no duplicated block).
    plot_module.inject_plot_refs_into_markdown(
        md, retention_png=ret, distribution_png=dist,
    )
    second = md.read_text(encoding="utf-8")
    assert first == second
    # Exactly one BEGIN / END pair.
    assert second.count(plot_module.MD_PLOT_BEGIN) == 1
    assert second.count(plot_module.MD_PLOT_END) == 1


def test_inject_plot_refs_appends_when_no_notes_anchor(
    tmp_path: Path, plot_module,
) -> None:
    md = tmp_path / "sweep.md"
    md.write_text("# Sweep\n\nNo notes anchor here.\n", encoding="utf-8")
    ret = tmp_path / "retention.png"
    dist = tmp_path / "distribution.png"
    ret.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    dist.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    plot_module.inject_plot_refs_into_markdown(
        md, retention_png=ret, distribution_png=dist,
    )
    text = md.read_text(encoding="utf-8")
    assert plot_module.MD_PLOT_BEGIN in text
    assert text.endswith(plot_module.MD_PLOT_END + "\n")


def test_inject_plot_refs_missing_markdown_returns_false(
    tmp_path: Path, plot_module,
) -> None:
    absent = tmp_path / "no.md"
    ret = tmp_path / "retention.png"
    dist = tmp_path / "distribution.png"
    ret.write_bytes(b"x")
    dist.write_bytes(b"x")
    changed = plot_module.inject_plot_refs_into_markdown(
        absent, retention_png=ret, distribution_png=dist,
    )
    assert changed is False


def test_main_embeds_into_markdown(tmp_path: Path, plot_module) -> None:
    csv_path = tmp_path / "sweep.csv"
    _write_sweep_csv(csv_path, rows=_make_sweep_rows())
    md_path = tmp_path / "sweep.md"
    md_path.write_text(
        "# Sweep\n\n## Retention by threshold\n\n| t | k |\n|---|---|\n\n"
        "## Notes\n",
        encoding="utf-8",
    )
    rc = plot_module.main([
        "--input", str(FIXTURES / "summaries"),
        "--sweep-csv", str(csv_path),
        "--sweep-md", str(md_path),
        "--out-retention", str(tmp_path / "retention.png"),
        "--out-distribution", str(tmp_path / "distribution.png"),
        "--threshold", "0.5",
        "--log-level", "WARNING",
    ])
    assert rc == 0
    text = md_path.read_text(encoding="utf-8")
    assert plot_module.MD_PLOT_BEGIN in text
    assert "retention.png" in text
    assert "distribution.png" in text


# Sanity: the banned-imports list still excludes API/NLI/embedding libs even
# though matplotlib is now in the dependency graph. Run as a subprocess so
# this test's own pytest run doesn't pollute sys.modules.
import subprocess

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


def test_import_safety_no_nli_or_llm_modules() -> None:
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
