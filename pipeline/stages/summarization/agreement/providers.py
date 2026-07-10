"""Embedding provider implementations.

Keeps provider-specific API calls out of scorer logic so scorers stay
testable and provider-agnostic.
"""
from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

# Type alias for any callable that maps strings to embedding vectors.
EmbedFn = Callable[[list[str]], list[list[float]]]


OPENAI_BATCH_SIZE = 512  # conservative limit to avoid token-length errors


class OpenAIEmbedder:
    """
    Callable wrapper around the OpenAI Embeddings API.

    Implements EmbedFn so it can be injected into any scorer that accepts an
    embedding function.  The openai package is imported lazily so the rest of
    the pipeline does not require it at import time.

    Automatically chunks lists longer than ``batch_size`` into multiple API
    calls and concatenates the results, so callers never need to batch manually.

    Parameters
    ----------
    model:
        OpenAI embedding model name.  Defaults to text-embedding-3-small.
    batch_size:
        Maximum texts per API call.  Default 512.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = OPENAI_BATCH_SIZE,
    ) -> None:
        self._model = model
        self._batch_size = batch_size
        self._call_count = 0
        self._total_texts = 0

    def __call__(self, texts: list[str]) -> list[list[float]]:
        import openai  # lazy import — optional dependency

        if not texts:
            return []

        client = openai.OpenAI()
        results: list[list[float]] = []
        n_batches = (len(texts) + self._batch_size - 1) // self._batch_size
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            self._call_count += 1
            self._total_texts += len(batch)
            batch_num = i // self._batch_size + 1
            logger.info(
                "Embedding batch %d/%d (%d texts, %d total so far)",
                batch_num, n_batches, len(batch), self._total_texts,
            )
            results.extend(self._embed_batch(client, batch))
        return results

    def _embed_batch(self, client, texts: list[str]) -> list[list[float]]:
        import time
        delay = 5.0
        for attempt in range(6):
            try:
                response = client.embeddings.create(model=self._model, input=texts)
                return [item.embedding for item in response.data]
            except Exception as exc:
                if attempt == 5:
                    raise
                if "429" in str(exc) or "quota" in str(exc).lower() or "rate" in str(exc).lower():
                    logger.warning(
                        "Rate limit hit (attempt %d/6) — retrying in %.0fs", attempt + 1, delay
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 120.0)
                else:
                    raise

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self._model!r})"


GEMINI_BATCH_SIZE = 100  # Gemini embed_content limit per request


class GeminiEmbedder:
    """
    Callable wrapper around the Google Gemini Embeddings API.

    Uses the google-genai SDK (google.genai).  Drop-in replacement for
    OpenAIEmbedder — implements the same EmbedFn interface.

    Automatically chunks lists longer than ``batch_size`` into multiple API
    calls and concatenates the results, so callers never need to batch manually.

    Parameters
    ----------
    model:
        Gemini embedding model name.  Defaults to gemini-embedding-001.
    task_type:
        Gemini task type hint.  "SEMANTIC_SIMILARITY" works well for
        agreement scoring between extracted claims.
    batch_size:
        Maximum texts per API call.  Gemini enforces a hard limit of 100.
    """

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        task_type: str = "SEMANTIC_SIMILARITY",
        batch_size: int = GEMINI_BATCH_SIZE,
    ) -> None:
        import os
        import google.genai as genai  # lazy import — optional dependency
        self._client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self._model = model
        self._task_type = task_type
        self._batch_size = batch_size
        self._call_count = 0
        self._total_texts = 0

    def __call__(self, texts: list[str]) -> list[list[float]]:

        if not texts:
            return []

        results: list[list[float]] = []
        n_batches = (len(texts) + self._batch_size - 1) // self._batch_size
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            self._call_count += 1
            self._total_texts += len(batch)
            batch_num = i // self._batch_size + 1
            logger.info(
                "Embedding batch %d/%d (%d texts, %d total so far)",
                batch_num, n_batches, len(batch), self._total_texts,
            )
            results.extend(self._embed_batch(batch))
        return results

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        import time
        delay = 5.0
        for attempt in range(6):
            try:
                result = self._client.models.embed_content(
                    model=self._model,
                    contents=texts,
                    config={"task_type": self._task_type},
                )
                return [e.values for e in result.embeddings]
            except Exception as exc:
                if attempt == 5:
                    raise
                if "429" in str(exc) or "quota" in str(exc).lower() or "rate" in str(exc).lower():
                    logger.warning(
                        "Rate limit hit (attempt %d/6) — retrying in %.0fs", attempt + 1, delay
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 120.0)
                else:
                    raise

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self._model!r})"
