"""Agreement scorers and checker for the ABC MAP cascade."""
from .category_jaccard import CategoryJaccardScorer
from .checker import AgreementChecker
from .composite import CascadedCompositeScorer
from .decision import (
    CascadeDecisionLog,
    CascadeDecisionRecord,
    ChunkOutcome,
    evaluate_chunk,
    make_decision_record,
    producer_from_outcome,
)
from .embedding import EmbeddingScorer
from .embedding_similarity import EmbeddingSimilarityStrategy
from .hybrid_structured import HybridStructuredSimilarity
from .lexical_similarity import LexicalSimilarityStrategy
from .llm_judge import LLMJudgeScorer
from .providers import EmbedFn, GeminiEmbedder, OpenAIEmbedder
from .semantic_scorer import SemanticAgreementScorer
from pipeline.stages.knowledge_extraction.interfaces.agreement import MapOutputScorer
from pipeline.stages.knowledge_extraction.interfaces.scoring import ChunkDecision, ScoreBundle

__all__ = [
    "MapOutputScorer",
    "AgreementChecker",
    "EmbeddingScorer",
    "CategoryJaccardScorer",
    "LLMJudgeScorer",
    "CascadedCompositeScorer",
    "SemanticAgreementScorer",
    "LexicalSimilarityStrategy",
    "EmbeddingSimilarityStrategy",
    "HybridStructuredSimilarity",
    "ScoreBundle",
    "ChunkDecision",
    "EmbedFn",
    "GeminiEmbedder",
    "OpenAIEmbedder",
    # Shared cascade decision surface (sync + batch use the same path)
    "evaluate_chunk",
    "ChunkOutcome",
    "CascadeDecisionLog",
    "CascadeDecisionRecord",
    "make_decision_record",
    "producer_from_outcome",
]
