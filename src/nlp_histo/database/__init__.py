"""
Database package for NLP Histopathology project.
Provides models and connection utilities for PostgreSQL.
"""

from .models import (
    Document,
    TextElement,
    Figure,
    Table,
    TextElementFigureReference,
    TextElementTableReference,
    Entity,
    PipelineRun,
    SumMapFinding,
    SumMapVoterOutput,
    SumNormalFinding,
    SumNormalFindingSpan,
    SumFindingGroup,
    SumGroupMember,
    SumCanonicalRule,
    SumRelation,
    SumFinalRule,
    SumCorpusRelation,
    SumRejectionSummary,
    SumRejectedFinding,
    LlmJudgeCache,
    Base,
)
from .db_connection import (
    DatabaseConnection,
    get_db_connection,
    close_db_connection,
    get_database_url,
    DB_CONFIG
)

__all__ = [
    'Document',
    'TextElement',
    'Figure',
    'Table',
    'TextElementFigureReference',
    'TextElementTableReference',
    'Entity',
    'PipelineRun',
    'SumMapFinding',
    'SumMapVoterOutput',
    'SumNormalFinding',
    'SumNormalFindingSpan',
    'SumFindingGroup',
    'SumGroupMember',
    'SumCanonicalRule',
    'SumRelation',
    'SumFinalRule',
    'SumCorpusRelation',
    'SumRejectionSummary',
    'SumRejectedFinding',
    'LlmJudgeCache',
    'Base',
    'DatabaseConnection',
    'get_db_connection',
    'close_db_connection',
    'get_database_url',
    'DB_CONFIG',
]
