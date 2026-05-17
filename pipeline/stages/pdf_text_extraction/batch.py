"""
ParallelBatchRunner

Processes multiple PDFs concurrently using a thread pool.

Why a separate file?
  PipelineRunner is intentionally single-threaded so it stays simple.
  Parallelism concerns (thread-local runners, progress tracking, KeyboardInterrupt
  handling) live here instead, keeping runner.py clean.

Thread safety model (same as scripts/latest_ingest.py):
  - Each worker thread gets its *own* PipelineRunner via threading.local().
    This gives every thread an independent Docling converter, since Docling is
    not safe to share across threads.
  - The BlacklistManager is already thread-safe (internal lock).
  - Stats are collected under a lock and summarised at the end.

Usage::

    from pipeline.stages.pdf_text_extraction.batch import ParallelBatchRunner
    from pipeline.stages.pdf_text_extraction import PipelineConfig

    cfg = PipelineConfig()
    cfg.prepare()

    runner = ParallelBatchRunner(cfg, max_workers=4)
    stats = runner.run(pdf_dir=Path("files/organized_pdfs"), max_docs=10)
    print(stats)
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pipeline.stages.pdf_text_extraction.blacklist import BlacklistManager
from pipeline.stages.pdf_text_extraction.config import PipelineConfig
from pipeline.stages.pdf_text_extraction.outputs.manifest_writer import (
    RunManifestWriter,
    make_run_id,
)

logger = logging.getLogger(__name__)


class ParallelBatchRunner:
    """
    Parallel batch processor for PDFs.

    Parameters
    ----------
    config:
        PipelineConfig shared across all workers (read-only after prepare()).
    max_workers:
        Number of worker threads.  Defaults to ``cpu_count // 2`` (min 1),
        matching the heuristic in scripts/latest_ingest.py.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        self._cfg = config or PipelineConfig()
        cpu = os.cpu_count() or 1
        self._max_workers = max_workers or max(1, cpu // 2)

        # Shared across all threads — already thread-safe
        self._blacklist  = BlacklistManager(self._cfg.paths.blacklist_file)
        self._completed  = BlacklistManager(self._cfg.paths.completed_file) if self._cfg.paths.completed_file else None

        # Per-thread PipelineRunner instances (Docling is not thread-safe to share)
        self._local = threading.local()

        # Stats — mutated only under _stats_lock
        self._stats_lock = threading.Lock()
        self._stats: Dict[str, int] = {
            "processed": 0,
            "failed":    0,
            "skipped":   0,
        }

        # Per-batch run-id; populated in run_paths.  Threaded through to
        # PipelineRunner.run_document so per-doc stats files reference the
        # same run.
        self._run_id: Optional[str] = None

    # ── Per-thread runner factory ─────────────────────────────────────────────

    def _get_runner(self):
        """Return (or create) the PipelineRunner for the current thread."""
        if not hasattr(self._local, "runner"):
            from pipeline.stages.pdf_text_extraction.runner import PipelineRunner
            # Each thread builds its own runner so Docling converters are isolated.
            runner = PipelineRunner(self._cfg)
            # Share the same blacklist instance across all runners so one thread's
            # failure immediately blocks the same pmcid in other threads.
            runner._blacklist = self._blacklist
            runner._completed = self._completed
            self._local.runner = runner
        return self._local.runner

    # ── Worker ────────────────────────────────────────────────────────────────

    def _process_one(self, pdf_path: Path, pmcid: str) -> str:
        """
        Process a single PDF.  Returns 'processed' | 'skipped' | 'failed'.
        Called from a worker thread.
        """
        if self._completed and self._completed.contains(pmcid):
            logger.info("⚡ %s — skipped (already completed)", pmcid)
            return "skipped"

        if self._cfg.runtime.skip_blacklisted and self._blacklist.contains(pmcid):
            logger.info("⚡ %s — skipped (blacklisted)", pmcid)
            return "skipped"

        try:
            runner = self._get_runner()
            ok = runner.run_document(pdf_path, pmcid, run_id=self._run_id)
            return "processed" if ok else "failed"
        except Exception as exc:  # noqa: BLE001
            logger.error("❌ %s — worker error: %s", pmcid, exc)
            logger.debug(traceback.format_exc())
            if self._cfg.runtime.update_blacklist_on_failure:
                self._blacklist.add(pmcid, reason=str(exc))
            return "failed"

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        pdf_dir: Path,
        glob: str = "*.pdf",
        max_docs: Optional[int] = None,
        pmcid_fn: Optional[Callable[[Path], str]] = None,
    ) -> Dict[str, int]:
        """
        Process PDFs in ``pdf_dir`` using a thread pool.

        Args:
            pdf_dir:   Directory containing PDF files.
            glob:      Filename glob pattern (default ``*.pdf``).
            max_docs:  Cap on number of PDFs to process.
            pmcid_fn:  Optional callable mapping a Path to a PMCID string.
                       Defaults to using the file stem.

        Returns:
            Dict with keys ``processed``, ``failed``, ``skipped``.
        """
        self._cfg.prepare()
        pdfs: List[Path] = sorted(pdf_dir.glob(glob))
        if max_docs is not None:
            pdfs = pdfs[:max_docs]
        return self.run_paths(
            pdfs,
            pmcid_fn=pmcid_fn,
            _input_meta={
                "pdf_dir":  str(pdf_dir),
                "glob":     glob,
                "max_docs": max_docs,
            },
        )

    def run_paths(
        self,
        paths: List[Path],
        pmcid_fn: Optional[Callable[[Path], str]] = None,
        *,
        _input_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """
        Process an explicit list of PDF paths using a thread pool.

        Useful when the caller has already filtered / sampled the file list
        (e.g. random subset for evaluation).

        Args:
            paths:    Pre-built list of PDF paths to process.
            pmcid_fn: Optional callable mapping a Path to a PMCID string.
                      Defaults to using the file stem.

        Returns:
            Dict with keys ``processed``, ``failed``, ``skipped``.
        """
        self._stats = {"processed": 0, "failed": 0, "skipped": 0}

        logger.info(
            "ParallelBatchRunner: %d PDFs, %d workers",
            len(paths), self._max_workers,
        )

        work = [
            (pdf, pmcid_fn(pdf) if pmcid_fn else pdf.stem)
            for pdf in paths
        ]

        # Observability manifest — best-effort; failure must not affect extraction.
        manifest: Optional[RunManifestWriter] = None
        try:
            self._run_id = make_run_id()
            manifest = RunManifestWriter(
                cfg=self._cfg,
                run_metadata_dir=self._cfg.paths.run_metadata_dir,
                run_id=self._run_id,
            )
            manifest.set_input(
                runner="ParallelBatchRunner",
                max_workers=self._max_workers,
                n_paths=len(paths),
                **(_input_meta or {}),
            )
            manifest.set_attempted([pmcid for _, pmcid in work])
            logger.info("ParallelBatchRunner: run_id=%s", self._run_id)
        except Exception:
            logger.exception("RunManifestWriter init failed — continuing without manifest")
            manifest = None

        try:
            with ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="PDFWorker",
            ) as executor:
                future_to_pmcid = {
                    executor.submit(self._process_one, pdf, pmcid): pmcid
                    for pdf, pmcid in work
                }

                with tqdm(total=len(future_to_pmcid), unit="pdf", desc="PDF extraction") as pbar:
                    for future in as_completed(future_to_pmcid):
                        pmcid = future_to_pmcid[future]
                        try:
                            outcome = future.result()
                        except Exception as exc:  # noqa: BLE001
                            logger.error("Unhandled worker exception for %s: %s", pmcid, exc)
                            outcome = "failed"

                        with self._stats_lock:
                            self._stats[outcome] += 1

                        pbar.update(1)
                        pbar.set_postfix(
                            ok=self._stats["processed"],
                            fail=self._stats["failed"],
                            skip=self._stats["skipped"],
                        )

        except KeyboardInterrupt:
            logger.warning("Interrupted — waiting for active threads to finish…")
            logger.warning("Press Ctrl+C again to force quit (may corrupt data)")

        logger.info(
            "Done: %d processed, %d failed, %d skipped",
            self._stats["processed"],
            self._stats["failed"],
            self._stats["skipped"],
        )

        if manifest is not None:
            try:
                manifest.write(batch_stats=dict(self._stats))
            except Exception:
                logger.exception("RunManifestWriter.write failed")
        self._run_id = None

        return dict(self._stats)
