"""
Per-stage output cache for PipelineRunner (B-027).

Disk layout::

    out/stage_cache/<stage_name>/<pmcid>.json   # serialised result
    out/stage_cache/<stage_name>/<pmcid>.hash   # one-line config hash

A cache hit requires (a) `enabled=True`, (b) both files exist, (c) the
sidecar hash equals the current pipeline-config hash, and (d) the loader
parses the artifact cleanly. Sidecar-read failures and loader failures
inside the narrow `_LOADER_EXPECTED_ERRORS` set log a WARNING and fall
through to recompute. Anything outside that set propagates so real bugs
are not silently masked.

Write failures propagate too — a silent skip on a broken cache disk
would make debug behaviour confusing.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from nlp_histo.pipeline.stages.pdf_text_extraction.models.dto import (
    BoundingBox,
    DetectedRegion,
    HierarchicalRow,
    LayoutElement,
    TableDetectionResult,
)

logger = logging.getLogger(__name__)


# ── Stage names ───────────────────────────────────────────────────────────────

class StageName:
    """Disk paths depend on these. Renaming a value here invalidates the
    on-disk cache layout (see `docs/readmes/HOW_TO_RUN.md`)."""

    TABLE_DETECTION = "table_detection"
    FILTERING = "filtering"
    TEXT_ASSEMBLY = "text_assembly"


# ── Cache version ─────────────────────────────────────────────────────────────
#
# !!! BUMP THIS WHEN ANY OF THE FOLLOWING CHANGES FOR A CACHED STAGE !!!
#   * the on-disk JSON format (loader/dumper schema)
#   * the loader/dumper assumptions (e.g. type coercion, default values)
#   * the cached stage's behaviour (algorithm tweak)
#   * a module-level constant the stage reads (e.g. `_NON_BIO_ENT_LABELS`,
#     `_SIDEBAR_METADATA`, `MIN_ANCHOR_H` in `parsers/layout_utils.py`)
#   * an external resource not represented in the config hash (model
#     weights upgrade, regex tweak, synonym list change)
#
# Failing to bump means previous-format / previous-behaviour artifacts
# silently survive and downstream stages get stale data. The constant
# lives here next to the cache that consumes it. Each cached stage's
# implementation carries a one-line reminder pointing back here so
# contributors arriving from the stage side don't miss the bump.
STAGE_CACHE_VERSION: dict[str, int] = {
    StageName.TABLE_DETECTION: 1,
    StageName.FILTERING: 1,
    StageName.TEXT_ASSEMBLY: 1,
}


# Narrow exception set for sidecar-read AND loader-parse paths. Anything
# outside this propagates (so bugs in the loader aren't silently masked).
# `UnicodeDecodeError` covers a sidecar that happens to contain non-UTF-8
# bytes; the rest cover JSON / structural validation failures.
_LOADER_EXPECTED_ERRORS = (
    OSError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    KeyError,
    ValueError,
    TypeError,
)


# ── Inline atomic write helpers ───────────────────────────────────────────────
# Inlined deliberately rather than extracted from
# `knowledge_extraction.persistence.write_json` — see B-027 plan "implementation
# contingency". The summarisation helper has many branches (Pydantic,
# datetime, Path, …) tied to telemetry concerns; keeping a local 15-line
# copy here lets B-027 land without destabilising summarisation output
# bytes.

def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write `payload` to `path` atomically (temp-fd + rename).

    Creates parent directories if needed. Uses `indent=2`, `ensure_ascii=False`,
    `default=str` — same defaults as `knowledge_extraction.persistence.write_json`
    so cached files are diffable / human-inspectable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
            fh.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _atomic_write_text(path: Path, content: str) -> None:
    """Write `content` to `path` atomically. UTF-8, no trailing newline.

    `_StageCache` reads sidecars with `read_text(encoding="utf-8").strip()`,
    so a trailing newline would be tolerated — but omitting it keeps the
    file boring and predictable to inspect.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── Per-stage (loader, dumper, summarise) triples ─────────────────────────────

def dump_table_detection(result: TableDetectionResult) -> dict:
    return {
        "regions": [
            {
                "bbox": {
                    "x1": r.bbox.x1, "y1": r.bbox.y1,
                    "x2": r.bbox.x2, "y2": r.bbox.y2,
                    "page": r.bbox.page,
                },
                "score": r.score,
                "source": r.source,
                "label": r.label,
            }
            for r in result.regions
        ],
        "source": result.source,
        "pdf_path": str(result.pdf_path),
        "page_dims": {str(k): v for k, v in result.page_dims.items()},
    }


def load_table_detection(path: Path) -> TableDetectionResult:
    raw = json.loads(path.read_text(encoding="utf-8"))
    regions = [
        DetectedRegion(
            bbox=BoundingBox(
                x1=r["bbox"]["x1"], y1=r["bbox"]["y1"],
                x2=r["bbox"]["x2"], y2=r["bbox"]["y2"],
                page=r["bbox"]["page"],
            ),
            score=r["score"],
            source=r["source"],
            label=r["label"],
        )
        for r in raw["regions"]
    ]
    page_dims = {int(k): v for k, v in raw["page_dims"].items()}
    return TableDetectionResult(
        regions=regions,
        source=raw["source"],
        pdf_path=Path(raw["pdf_path"]),  # restore Path runtime type
        page_dims=page_dims,
    )


def dump_layout_elements(result: list[LayoutElement]) -> list[dict]:
    return [el.to_dict() for el in result]


def load_layout_elements(path: Path) -> list[LayoutElement]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [LayoutElement.from_dict(d) for d in raw]


def dump_hierarchical_rows(result: list[HierarchicalRow]) -> list[dict]:
    return [
        {
            "path_string": row.path_string,
            "path_list": list(row.path_list),
            "depth": row.depth,
            "text": row.text,
            "source_chunks": list(row.source_chunks),
        }
        for row in result
    ]


def load_hierarchical_rows(path: Path) -> list[HierarchicalRow]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        HierarchicalRow(
            path_string=d["path_string"],
            path_list=list(d["path_list"]),
            depth=d["depth"],
            text=d["text"],
            source_chunks=list(d.get("source_chunks", [])),
        )
        for d in raw
    ]


# ── _StageCache ───────────────────────────────────────────────────────────────

class _StageCache:
    """Per-(stage, pmcid) JSON cache with config-hash invalidation.

    Use:
        cache = _StageCache(root=Path("out/stage_cache"), enabled=True)
        result = cache.get_or_compute(
            stage_name=StageName.TABLE_DETECTION, pmcid="PMC123",
            config_hash=hash_str,
            compute_fn=lambda: detector.detect(pdf_path),
            loader_fn=load_table_detection,
            dumper_fn=dump_table_detection,
            summarise_fn=lambda r: f"{len(r.regions)} regions",
        )

    `enabled=False` forces every call to recompute (the documented
    `runtime.skip_existing_outputs=False` behaviour).
    """

    def __init__(self, root: Path, enabled: bool) -> None:
        self._root = Path(root)
        self._enabled = enabled

    def get_or_compute(
        self,
        *,
        stage_name: str,
        pmcid: str,
        config_hash: str,
        compute_fn: Callable[[], Any],
        loader_fn: Callable[[Path], Any],
        dumper_fn: Callable[[Any], Any],
        summarise_fn: Callable[[Any], str],
    ) -> Any:
        artifact = self._root / stage_name / f"{pmcid}.json"
        sidecar = artifact.with_suffix(".hash")

        if self._enabled and artifact.exists() and sidecar.exists():
            try:
                stored = sidecar.read_text(encoding="utf-8").strip()
            except _LOADER_EXPECTED_ERRORS as exc:
                logger.warning(
                    "CACHE INVALID [%s/%s] sidecar read failed with %s: %s; recomputing",
                    stage_name, pmcid, type(exc).__name__, exc,
                )
                stored = None

            if stored is not None and stored == config_hash:
                try:
                    result = loader_fn(artifact)
                    logger.info(
                        "CACHE HIT  [%s/%s] %s — %s",
                        stage_name, pmcid, artifact, summarise_fn(result),
                    )
                    return result
                except _LOADER_EXPECTED_ERRORS as exc:
                    logger.warning(
                        "CACHE INVALID [%s/%s] %s — loader failed with %s: %s; recomputing",
                        stage_name, pmcid, artifact, type(exc).__name__, exc,
                    )
            elif stored is not None:
                logger.info(
                    "CACHE STALE [%s/%s] hash changed (cached=%s, current=%s); recomputing",
                    stage_name, pmcid, stored, config_hash,
                )
        elif self._enabled and artifact.exists() and not sidecar.exists():
            logger.info(
                "CACHE MISS [%s/%s] artifact present but sidecar missing; recomputing",
                stage_name, pmcid,
            )
        elif self._enabled:
            logger.info(
                "CACHE MISS [%s/%s] no cached artifact; running stage",
                stage_name, pmcid,
            )
        else:
            logger.info(
                "CACHE OFF  [%s/%s] runtime.skip_existing_outputs=False; running stage",
                stage_name, pmcid,
            )

        result = compute_fn()
        # Both writes atomic; the helpers create parent directories.
        # Write failures propagate.
        _atomic_write_json(artifact, dumper_fn(result))
        _atomic_write_text(sidecar, config_hash)
        return result
