"""SimilarityStrategy — protocol for pairwise voter similarity scoring."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from pipeline.stages.knowledge_extraction.models import AuditableSummary


@runtime_checkable
class SimilarityStrategy(Protocol):
    """
    Computes pairwise similarity between two AuditableSummary voter outputs.

    V1 design principle
    -------------------
    In the production v1 claim-centered agreement path, implementations
    compare structured findings — primarily ``Finding.claim``,
    ``Finding.category``, and ``Finding.evidence``.
    ``AuditableSummary.summary_text`` is not used in this path.

    Both fields are available in ``AuditableSummary``, so future strategies
    that operate at the summary-prose level (e.g. sentence-level embedding of
    ``summary_text`` as an auxiliary signal) can be added without any Protocol
    change.  Such strategies should be clearly documented as prose-level and
    used only as optional auxiliary or evaluation components, not as the
    primary agreement signal.

    Implementations must be stateless across calls (safe for concurrent use).

    Required method
    ---------------
    similarity(a, b) -> float
        Score in [0, 1].  1.0 = identical; 0.0 = completely dissimilar.

    Optional method
    ---------------
    compute_matrix(outputs) -> np.ndarray
        Build the full N×N similarity matrix in one batched call.
        SemanticAgreementScorer detects and calls this when available to
        minimise API round trips (e.g. embedding strategies).
    """

    def similarity(self, a: AuditableSummary, b: AuditableSummary) -> float: ...
