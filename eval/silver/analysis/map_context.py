"""Shared MAP calibration/replay context loader.

Loads the primed voter cache + silver labels + the agreement embedder and
pre-warms the agreement embedding cache — with NO LLM calls. This is the
load-bearing helper for every offline replay/calibration tool:
``run_new_summarization_sweeps`` (E05–E08 cascade calibration), ``voter_loo``,
``grounding_sweep_related15`` (E03), the chapter-9 offline replay, and the
experiments harness.

Extracted from the now-legacy ``old_files/run_summarization_sweeps.py`` so the
live loader is not stranded in a deprecated module. ``CACHE_PATH`` / ``SILVER_PATH``
continue to live in ``map_theta_sweep`` (the related15 prime + silver paths) and
are re-exported here for convenience.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from eval.silver.data.jsonl_utils import read_jsonl
from eval.silver.analysis.map_theta_sweep import (
    CACHE_PATH,
    SILVER_PATH,
    _make_cached_embed_fn,
    _prewarm_agreement_cache,
)
from eval.silver.matching.matcher import (
    DEFAULT_GEMINI_CACHE_PATH,
    GEMINI_EMBEDDING_MODEL,
    make_embedding_cache,
)
from eval.silver.data.schemas import SilverCaseResult

__all__ = ["_MapContext", "_load_map_context", "CACHE_PATH", "SILVER_PATH"]


@dataclass
class _MapContext:
    voter_cache: dict
    silver_by_case: dict
    embedder: object
    embed_cache: object
    agreement_embed_fn: object


def _load_map_context(
    embedder_kind: str,
    *,
    embed_cache_path: Optional[str],
    cache_path: Path = CACHE_PATH,
    silver_path: Path = SILVER_PATH,
) -> _MapContext:
    """Load voter cache + silver + the gemini/openai embedder and pre-warm the
    agreement cache. Mirrors map_theta_sweep's sweep-mode setup (no LLM calls).

    ``cache_path`` / ``silver_path`` default to the related15 primer/silver but can
    point at another split (e.g. heldout15) for held-out evaluation."""
    if not cache_path.exists():
        raise SystemExit(
            f"voter cache not found: {cache_path}\n"
            "Run `python -m eval.silver.analysis.map_theta_sweep prime` (with --source/--primer-dir) "
            "then `collect`.")
    if not silver_path.exists():
        raise SystemExit(
            f"silver labels not found: {silver_path}\n"
            "Run `python -m eval.silver.generation.generate --batch`.")

    voter_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    silver_by_case = {rec.case_id: rec for rec in read_jsonl(silver_path, SilverCaseResult)}

    if embedder_kind == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("GOOGLE_API_KEY not set")
        from eval.silver.matching.embedders import GeminiEmbedder
        from nlp_histo.pipeline.stages.knowledge_extraction.agreement.providers import (
            GeminiEmbedder as AgreementGeminiEmbedder,
        )
        embedder = GeminiEmbedder(api_key)
        path = Path(embed_cache_path) if embed_cache_path else DEFAULT_GEMINI_CACHE_PATH
        embed_cache = make_embedding_cache(path, GEMINI_EMBEDDING_MODEL)
        raw_agreement_fn = AgreementGeminiEmbedder()
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY not set")
        from eval.silver.matching.embedders import OpenAIEmbedder
        from eval.silver.matching.matcher import DEFAULT_CACHE_PATH, EMBEDDING_MODEL
        from nlp_histo.pipeline.stages.knowledge_extraction.agreement.providers import (
            OpenAIEmbedder as AgreementOpenAIEmbedder,
        )
        embedder = OpenAIEmbedder(api_key)
        path = Path(embed_cache_path) if embed_cache_path else DEFAULT_CACHE_PATH
        embed_cache = make_embedding_cache(path, EMBEDDING_MODEL)
        raw_agreement_fn = AgreementOpenAIEmbedder()

    _prewarm_agreement_cache(voter_cache, raw_agreement_fn, embed_cache)
    agreement_embed_fn = _make_cached_embed_fn(embed_cache, raw_agreement_fn)
    return _MapContext(voter_cache, silver_by_case, embedder, embed_cache, agreement_embed_fn)
