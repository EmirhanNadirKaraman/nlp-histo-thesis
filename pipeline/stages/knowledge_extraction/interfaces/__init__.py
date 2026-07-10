"""Protocol interfaces for the knowledge_extraction pipeline."""
from .agreement import MapOutputScorer
from .contradiction import ContradictionChecker
from .grounding import GroundingChecker
from .scoring import AgreementContext, ChunkDecision, ScoreBundle, VoterContext
from .similarity import SimilarityStrategy

__all__ = [
    "MapOutputScorer",
    "GroundingChecker",
    "ContradictionChecker",
    "ScoreBundle",
    "ChunkDecision",
    "AgreementContext",
    "VoterContext",
    "SimilarityStrategy",
]
