#!/usr/bin/env python3
"""
run_all_sweeps.py — run every PDF-extraction sweep variant in a single
Python process so Docling + TATR models load once and stay cached.

Equivalent to ``/tmp/rerun_sweeps.sh`` but ~5 min faster on a full
cycle: each shell ``run`` invocation in that script spawns a fresh
Python process which reloads Docling (~20s) and TATR (~10s).  Here we
amortize both: TATR uses its module-level ``_SHARED_PROCESSOR/_MODEL``
already, and Docling now caches via ``_DOCLING_CONVERTER_CACHE`` keyed
by the tuple of options that influence converter state.

Variants are defined inline below; each is a ``SweepSpec`` with a name
and a ``configure(cfg)`` callable that mutates the canonical
``PipelineConfig`` to express the variant.

Outputs land under ``out/sweeps/<name>/`` exactly like the shell script.
After all sweeps finish, share-map rebuild and per-variant seeding run
automatically (same as the old shell flow's tail).

Usage::

    python scripts/eval/run_all_sweeps.py
    python scripts/eval/run_all_sweeps.py --only baseline tatr_090
    python scripts/eval/run_all_sweeps.py --pdf-dir eval/pdfs --workers 2
    python scripts/eval/run_all_sweeps.py --skip-share-map
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from tqdm import tqdm

# Make ``pipeline.*`` importable when this script is run directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.stages.pdf_text_extraction import PipelineConfig, PipelineRunner  # noqa: F401  (kept for backwards-compat imports)
from pipeline.stages.pdf_text_extraction.batch import ParallelBatchRunner
from pipeline.stages.pdf_text_extraction.config import TableDetectorType
from pipeline.stages.pdf_text_extraction.runner import _retarget_paths


_DEFAULT_PDF_DIR = _REPO_ROOT / "eval" / "pdfs"
_DEFAULT_SWEEPS_ROOT = _REPO_ROOT / "out" / "sweeps"


@dataclass
class SweepSpec:
    name: str
    configure: Callable[[PipelineConfig], None]
    workers: int = 2


def _apply_stage1_baseline(cfg: PipelineConfig) -> None:
    """Canonical Stage 1 baseline: helpers OFF, two-pass ON.

    Every field is set explicitly so a future change to PipelineConfig
    defaults cannot silently shift the experiment baseline.  In particular
    MaskingConfig.drop_tables_inside_figures defaults to True since
    2026-05-18, so we must force it False here.
    """
    cfg.two_pass.enabled                       = True
    cfg.two_pass.render_dpi                    = 150
    cfg.tatr.render_dpi                        = 150
    cfg.docling.reconstruct_tables_from_lists  = False
    cfg.cropping.merge_tables_by_caption       = False
    cfg.cropping.merge_figures_by_caption      = False
    cfg.cropping.expand_tables_with_footnotes  = False
    cfg.masking.drop_tables_inside_figures     = False


def _stage1(detector: TableDetectorType, threshold: Optional[float]):
    """Build a Stage-1 configure() callable.

    threshold=None for docling-only (no TATR threshold relevant).
    """
    def configure(cfg: PipelineConfig) -> None:
        _apply_stage1_baseline(cfg)
        cfg.table_detector = detector
        if threshold is not None:
            cfg.tatr.threshold = threshold
    return configure


# Stage 1 — detector / threshold selection.
# Helper flags (reconstruct/merge/expand/drop) are explicitly OFF in
# _apply_stage1_baseline so this comparison isolates the detector.
# ``workers=1`` for the docling-only variant avoids previously-observed OOM.
ALL_SWEEPS: List[SweepSpec] = [
    SweepSpec("01_docling",    _stage1(TableDetectorType.DOCLING, None), workers=1),
    SweepSpec("02_tatr_090",   _stage1(TableDetectorType.TATR,    0.90), workers=2),
    SweepSpec("03_tatr_095",   _stage1(TableDetectorType.TATR,    0.95), workers=2),
    SweepSpec("04_tatr_099",   _stage1(TableDetectorType.TATR,    0.99), workers=2),
    SweepSpec("05_hybrid_090", _stage1(TableDetectorType.HYBRID,  0.90), workers=2),
    SweepSpec("06_hybrid_095", _stage1(TableDetectorType.HYBRID,  0.95), workers=2),
    SweepSpec("07_hybrid_099", _stage1(TableDetectorType.HYBRID,  0.99), workers=2),
]


# Stage 5 — table_in_figure suppression strategies on BEST_BASE.
# BEST_BASE is hardcoded to "07_hybrid_099" below; edit after Stage 1
# F1 scoring to point at whichever Stage 1 variant won.
BEST_BASE = "07_hybrid_099"

_STAGE1_BASES = {
    "01_docling":    (TableDetectorType.DOCLING, None),
    "02_tatr_090":   (TableDetectorType.TATR,    0.90),
    "03_tatr_095":   (TableDetectorType.TATR,    0.95),
    "04_tatr_099":   (TableDetectorType.TATR,    0.99),
    "05_hybrid_090": (TableDetectorType.HYBRID,  0.90),
    "06_hybrid_095": (TableDetectorType.HYBRID,  0.95),
    "07_hybrid_099": (TableDetectorType.HYBRID,  0.99),
}


def _apply_best_base(cfg: PipelineConfig) -> None:
    """Apply Stage 1 baseline + the chosen detector/threshold from BEST_BASE."""
    _apply_stage1_baseline(cfg)
    if BEST_BASE not in _STAGE1_BASES:
        raise ValueError(
            f"BEST_BASE={BEST_BASE!r} must be one of {sorted(_STAGE1_BASES)}"
        )
    detector, threshold = _STAGE1_BASES[BEST_BASE]
    cfg.table_detector = detector
    if threshold is not None:
        cfg.tatr.threshold = threshold


def _stage5(drop: bool, premask: bool):
    """Build a Stage-5 configure() callable on the BEST_BASE config.

    drop:    masking.drop_tables_inside_figures
    premask: masking.mask_figures_before_table_detection
    """
    def configure(cfg: PipelineConfig) -> None:
        _apply_best_base(cfg)
        cfg.masking.drop_tables_inside_figures          = drop
        cfg.masking.mask_figures_before_table_detection = premask
    return configure


ALL_SWEEPS.extend([
    SweepSpec("20_best_drop_tables_inside_figures_on",                 _stage5(drop=True,  premask=False), workers=2),
    SweepSpec("21_best_premask_figures_for_table_detection",           _stage5(drop=False, premask=True),  workers=2),
    SweepSpec("22_best_drop_plus_premask_figures_for_table_detection", _stage5(drop=True,  premask=True),  workers=2),
])


def _build_config(spec: SweepSpec, *, pdf_dir: Path, sweeps_root: Path,
                  workers_override: Optional[int],
                  ensure_dirs: bool = True) -> PipelineConfig:
    cfg = PipelineConfig()
    # Sweep defaults: no DB writes, no raw text dumps (saves disk + time).
    cfg.database.enabled = False
    cfg.text.write_raw_text = False
    # Resume semantics: skip PDFs whose media JSON already exists, and
    # enable the per-stage cache so a PDF interrupted mid-pipeline can
    # resume from the last completed stage on the next run.
    cfg.runtime.skip_existing_media_json = True
    cfg.runtime.skip_existing_outputs    = True
    # Apply variant mutations
    spec.configure(cfg)
    cfg.runtime.num_workers = workers_override if workers_override is not None else spec.workers
    # Retarget every per-variant path under out/sweeps/<name>/
    _retarget_paths(cfg, sweeps_root / spec.name)
    if ensure_dirs:
        cfg.prepare()
    else:
        # Dry-run path: validate flag values without creating directories.
        cfg.validate()
    return cfg


_DONE_MARKER = "_DONE.json"


def _count_pdfs(pdf_dir: Path) -> int:
    return sum(1 for _ in pdf_dir.glob("*.pdf"))


def _variant_already_done(out_dir: Path, expected_pdf_count: int) -> bool:
    """Return True iff the variant has a _DONE marker stamped for the
    same number of PDFs that currently live in the source directory.

    A PDF-count mismatch (e.g. the user added more PDFs since last run)
    invalidates the marker so the new PDFs get processed on resume.
    Already-processed PDFs are still skipped via
    `cfg.runtime.skip_existing_media_json`.
    """
    marker = out_dir / _DONE_MARKER
    if not marker.exists():
        return False
    try:
        data = json.loads(marker.read_text())
    except Exception:
        return False
    return data.get("pdf_count") == expected_pdf_count


def _mark_variant_done(out_dir: Path, pdf_count: int) -> None:
    marker = out_dir / _DONE_MARKER
    marker.write_text(json.dumps({
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pdf_count": pdf_count,
    }, indent=2))


def _clean_partial_pdf_outputs(out_dir: Path, pdf_dir: Path) -> int:
    """Delete crop artifacts for any PDF that started but did not finish.

    A PDF is considered "completed" iff ``json/<pmcid>_media.json`` exists.
    For each PDF in ``pdf_dir`` whose media JSON is missing, this scans
    ``figures/`` and ``tables/`` for filenames starting with ``<pmcid>_``
    and deletes them.

    Necessary because ``PyMuPDFMediaCropper`` appends ``_1``, ``_2`` suffixes
    on filename collisions (``components/media_cropper.py:334``) — without
    this cleanup, a Ctrl-C mid-crop on the first attempt would silently
    leave behind half a crop set, and the next attempt's crops would be
    written alongside (doubling the share-map entries) instead of
    overwriting.

    Idempotent and cheap: only touches PDFs whose ``_media.json`` is
    missing, so completed work is preserved. Returns the number of
    artifacts deleted (informational).
    """
    json_dir    = out_dir / "json"
    figures_dir = out_dir / "figures"
    tables_dir  = out_dir / "tables"
    if not figures_dir.exists() and not tables_dir.exists():
        return 0

    deleted = 0
    for pdf in pdf_dir.glob("*.pdf"):
        pmcid = pdf.stem
        if (json_dir / f"{pmcid}_media.json").exists():
            continue  # finished PDF — keep its crops
        for d in (figures_dir, tables_dir):
            if not d.exists():
                continue
            for stale in d.glob(f"{pmcid}_*"):
                try:
                    stale.unlink()
                    deleted += 1
                except OSError:
                    pass
    return deleted


def _run_one(spec: SweepSpec, *, pdf_dir: Path, sweeps_root: Path,
             workers_override: Optional[int],
             force_restart: bool = False,
             ) -> tuple[str, float, bool, str]:
    """Run a single sweep variant.

    Returns (name, wall_seconds, ok, status_label) where status_label is
    one of {"ran", "resumed", "skipped"} for the post-run summary.

    Resume semantics:
      * `force_restart=True` removes the output directory before running.
      * Otherwise, if a `_DONE.json` marker is present for the current
        PDF count, the variant is skipped entirely.
      * Otherwise the variant runs; `cfg.runtime.skip_existing_media_json`
        causes already-processed PDFs to be skipped, and the per-stage
        cache lets partially-processed PDFs resume from the last
        completed stage.
    """
    out_dir = sweeps_root / spec.name
    if force_restart and out_dir.exists():
        logging.info("--restart: removing %s", out_dir)
        shutil.rmtree(out_dir)

    pdf_count = _count_pdfs(pdf_dir)

    if _variant_already_done(out_dir, pdf_count):
        logging.info("[%s] already done (%d PDFs) — skipping", spec.name, pdf_count)
        return spec.name, 0.0, True, "skipped"

    status = "resumed" if out_dir.exists() else "ran"
    if status == "resumed":
        n_deleted = _clean_partial_pdf_outputs(out_dir, pdf_dir)
        if n_deleted:
            logging.info(
                "[%s] cleaned %d stale crop file(s) from interrupted runs",
                spec.name, n_deleted,
            )
    t0 = time.perf_counter()
    cfg = _build_config(spec, pdf_dir=pdf_dir, sweeps_root=sweeps_root,
                         workers_override=workers_override)
    # Use ParallelBatchRunner so num_workers, the per-PDF tqdm bar, and the
    # `[N/total]` log line all activate. PipelineRunner.run_batch is a
    # sequential for-loop that bypasses all three.
    try:
        ParallelBatchRunner(cfg, max_workers=cfg.runtime.num_workers).run(
            pdf_dir=pdf_dir,
        )
        ok = True
    except Exception as exc:
        logging.exception("sweep %s failed: %s", spec.name, exc)
        ok = False
    elapsed = time.perf_counter() - t0
    if ok:
        _mark_variant_done(out_dir, pdf_count)
    return spec.name, elapsed, ok, status


def _print_variants_table(
    sweeps: List[SweepSpec],
    *,
    pdf_dir: Path,
    sweeps_root: Path,
    workers_override: Optional[int],
) -> None:
    """Print one row per variant with its resolved config.

    Builds the PipelineConfig for each variant (without running extraction)
    so the printed flags match exactly what a real run would see.
    """
    header = ("variant", "detector", "tatr", "2pass",
              "recon", "m_tbl", "m_fig", "exp", "drop", "premask", "out_dir")
    rows: List[tuple] = []
    for spec in sweeps:
        cfg = _build_config(spec, pdf_dir=pdf_dir, sweeps_root=sweeps_root,
                             workers_override=workers_override,
                             ensure_dirs=False)
        detector = cfg.table_detector.value
        tatr_thr = (f"{cfg.tatr.threshold:.2f}"
                    if detector in ("tatr", "hybrid") else "-")
        rows.append((
            spec.name,
            detector,
            tatr_thr,
            "ON" if cfg.two_pass.enabled else "OFF",
            "ON" if cfg.docling.reconstruct_tables_from_lists else "OFF",
            "ON" if cfg.cropping.merge_tables_by_caption else "OFF",
            "ON" if cfg.cropping.merge_figures_by_caption else "OFF",
            "ON" if cfg.cropping.expand_tables_with_footnotes else "OFF",
            "ON" if cfg.masking.drop_tables_inside_figures else "OFF",
            "ON" if cfg.masking.mask_figures_before_table_detection else "OFF",
            str(cfg.paths.output_root),
        ))

    widths = [max(len(str(c)) for c in col)
              for col in zip(header, *rows)]

    def _fmt(row: tuple) -> str:
        return "  ".join(str(c).ljust(w) for c, w in zip(row, widths))

    print(_fmt(header))
    print(_fmt(tuple("-" * w for w in widths)))
    for row in rows:
        print(_fmt(row))


def _post_run_share_map() -> None:
    """Rebuild share-map and re-seed per-variant labels (the shell tail)."""
    import subprocess
    PY = sys.executable
    here = Path(__file__).parent
    logging.info("rebuilding share_map...")
    r1 = subprocess.run([PY, str(here / "build_share_map.py")],
                         capture_output=True, text=True)
    print(r1.stdout)
    if r1.returncode != 0:
        print(r1.stderr, file=sys.stderr)
    logging.info("re-seeding per-variant labels...")
    r2 = subprocess.run([PY, str(here / "seed_variant_labels.py")],
                         capture_output=True, text=True)
    print(r2.stdout)
    if r2.returncode != 0:
        print(r2.stderr, file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdf-dir", type=Path, default=_DEFAULT_PDF_DIR,
                   help="Source PDF directory.  Default: eval/pdfs/")
    p.add_argument("--sweeps-root", type=Path, default=_DEFAULT_SWEEPS_ROOT,
                   help="Where to write each variant's outputs.")
    p.add_argument("--only", nargs="+", default=None,
                   help="Run only these sweep names (e.g. --only baseline tatr_090).")
    p.add_argument("--workers", type=int, default=None,
                   help="Override per-variant worker count.")
    p.add_argument("--skip-share-map", action="store_true",
                   help="Skip the share-map rebuild + label seed at the end.")
    p.add_argument("--list-variants", "--dry-run", dest="list_variants",
                   action="store_true",
                   help="Print the resolved config for each variant and exit "
                        "without running extraction.")
    p.add_argument("--restart", action="store_true",
                   help="Wipe each selected variant's output directory before "
                        "running (force fresh re-run). Default: resume — "
                        "variants stamped with _DONE.json are skipped and "
                        "partially-completed variants pick up from where they "
                        "left off via per-PDF + per-stage caching.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.list_variants and not args.pdf_dir.exists():
        print(f"error: pdf-dir not found: {args.pdf_dir}", file=sys.stderr)
        return 2

    sweeps = ALL_SWEEPS
    if args.only:
        wanted = set(args.only)
        sweeps = [s for s in ALL_SWEEPS if s.name in wanted]
        missing = wanted - {s.name for s in ALL_SWEEPS}
        if missing:
            print(f"error: unknown sweep name(s): {sorted(missing)}", file=sys.stderr)
            print(f"  available: {sorted(s.name for s in ALL_SWEEPS)}", file=sys.stderr)
            return 2

    if args.list_variants:
        _print_variants_table(
            sweeps,
            pdf_dir=args.pdf_dir,
            sweeps_root=args.sweeps_root,
            workers_override=args.workers,
        )
        return 0

    args.sweeps_root.mkdir(parents=True, exist_ok=True)

    print(f"=== running {len(sweeps)} sweep(s) in one process ===")
    print(f"  pdf-dir:     {args.pdf_dir}")
    print(f"  sweeps-root: {args.sweeps_root}")
    print(f"  variants:    {[s.name for s in sweeps]}")
    print()

    results: List[tuple[str, float, bool, str]] = []
    total_t0 = time.perf_counter()
    with tqdm(
        total=len(sweeps),
        unit="variant",
        desc="Sweep progress",
        position=0,
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} variants  [{elapsed}<{remaining}]  {postfix}",
    ) as outer:
        for i, spec in enumerate(sweeps, 1):
            outer.set_postfix_str(f"running {spec.name}")
            tqdm.write(
                f"[{time.strftime('%H:%M:%S')}] === [{i}/{len(sweeps)}] {spec.name} starting ==="
            )
            name, elapsed, ok, status = _run_one(
                spec,
                pdf_dir=args.pdf_dir,
                sweeps_root=args.sweeps_root,
                workers_override=args.workers,
                force_restart=args.restart,
            )
            results.append((name, elapsed, ok, status))
            if status == "skipped":
                flag = "SKIPPED (already done)"
            elif ok:
                flag = f"OK ({status})"
            else:
                flag = "FAILED"
            tqdm.write(
                f"[{time.strftime('%H:%M:%S')}] === [{i}/{len(sweeps)}] {name} {flag}  ({elapsed:.1f}s) ==="
            )
            outer.update(1)
        outer.set_postfix_str("done")

    total_elapsed = time.perf_counter() - total_t0
    print()
    print("=== summary ===")
    for name, elapsed, ok, status in results:
        if status == "skipped":
            flag = "SKIP  "
        elif ok:
            flag = "OK    "
        else:
            flag = "FAILED"
        print(f"  {flag}  {elapsed:>7.1f}s  {name}  ({status})")
    print(f"  total:  {total_elapsed:>7.1f}s  ({len(results)} sweeps)")

    if not args.skip_share_map:
        print()
        _post_run_share_map()

    return 0 if all(ok for _, _, ok, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
