"""split canonical_scope into is_conflicted + study_coverage

Replaces the single canonical_scope column (single_study | multi_study |
conflicted | unknown) with two separate columns that can independently express
both conflict and study-coverage simultaneously.

Affected tables:
  sum_canonical_rules   — canonical_scope → is_conflicted + study_coverage
  sum_corpus_relations  — canonical_scope_a/b → is_conflicted_a/b + study_coverage_a/b

Data migration: derive is_conflicted from old value == "conflicted",
study_coverage from old value (conflicted → "multi_study" as best approximation
since the old schema could not store both).

Revision ID: 0009
Revises: 0008
Create Date: 2026-04-25
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── sum_canonical_rules ───────────────────────────────────────────────────
    op.add_column("sum_canonical_rules",
        sa.Column("is_conflicted", sa.Boolean(), nullable=True))
    op.add_column("sum_canonical_rules",
        sa.Column("study_coverage", sa.String(20), nullable=True))

    op.execute("""
        UPDATE sum_canonical_rules
        SET
            is_conflicted  = (canonical_scope = 'conflicted'),
            study_coverage = CASE
                WHEN canonical_scope IN ('multi_study', 'conflicted') THEN 'multi_study'
                WHEN canonical_scope = 'single_study'                 THEN 'single_study'
                ELSE                                                       'unknown'
            END
    """)

    op.alter_column("sum_canonical_rules", "is_conflicted", nullable=False)
    op.alter_column("sum_canonical_rules", "study_coverage", nullable=False)
    op.drop_column("sum_canonical_rules", "canonical_scope")

    # ── sum_corpus_relations ──────────────────────────────────────────────────
    op.add_column("sum_corpus_relations",
        sa.Column("is_conflicted_a", sa.Boolean(), nullable=True))
    op.add_column("sum_corpus_relations",
        sa.Column("study_coverage_a", sa.String(20), nullable=True))
    op.add_column("sum_corpus_relations",
        sa.Column("is_conflicted_b", sa.Boolean(), nullable=True))
    op.add_column("sum_corpus_relations",
        sa.Column("study_coverage_b", sa.String(20), nullable=True))

    op.execute("""
        UPDATE sum_corpus_relations
        SET
            is_conflicted_a  = (canonical_scope_a = 'conflicted'),
            study_coverage_a = CASE
                WHEN canonical_scope_a IN ('multi_study', 'conflicted') THEN 'multi_study'
                WHEN canonical_scope_a = 'single_study'                 THEN 'single_study'
                ELSE                                                         'unknown'
            END,
            is_conflicted_b  = (canonical_scope_b = 'conflicted'),
            study_coverage_b = CASE
                WHEN canonical_scope_b IN ('multi_study', 'conflicted') THEN 'multi_study'
                WHEN canonical_scope_b = 'single_study'                 THEN 'single_study'
                ELSE                                                         'unknown'
            END
    """)

    op.alter_column("sum_corpus_relations", "is_conflicted_a", nullable=False)
    op.alter_column("sum_corpus_relations", "study_coverage_a", nullable=False)
    op.alter_column("sum_corpus_relations", "is_conflicted_b", nullable=False)
    op.alter_column("sum_corpus_relations", "study_coverage_b", nullable=False)
    op.drop_column("sum_corpus_relations", "canonical_scope_a")
    op.drop_column("sum_corpus_relations", "canonical_scope_b")


def downgrade() -> None:
    # ── sum_canonical_rules ───────────────────────────────────────────────────
    op.add_column("sum_canonical_rules",
        sa.Column("canonical_scope", sa.String(20), nullable=True))

    op.execute("""
        UPDATE sum_canonical_rules
        SET canonical_scope = CASE
            WHEN is_conflicted                       THEN 'conflicted'
            WHEN study_coverage = 'multi_study'      THEN 'multi_study'
            WHEN study_coverage = 'single_study'     THEN 'single_study'
            ELSE                                          'unknown'
        END
    """)

    op.alter_column("sum_canonical_rules", "canonical_scope", nullable=False)
    op.drop_column("sum_canonical_rules", "is_conflicted")
    op.drop_column("sum_canonical_rules", "study_coverage")

    # ── sum_corpus_relations ──────────────────────────────────────────────────
    op.add_column("sum_corpus_relations",
        sa.Column("canonical_scope_a", sa.String(30), nullable=True))
    op.add_column("sum_corpus_relations",
        sa.Column("canonical_scope_b", sa.String(30), nullable=True))

    op.execute("""
        UPDATE sum_corpus_relations
        SET
            canonical_scope_a = CASE
                WHEN is_conflicted_a                      THEN 'conflicted'
                WHEN study_coverage_a = 'multi_study'     THEN 'multi_study'
                WHEN study_coverage_a = 'single_study'    THEN 'single_study'
                ELSE                                           'unknown'
            END,
            canonical_scope_b = CASE
                WHEN is_conflicted_b                      THEN 'conflicted'
                WHEN study_coverage_b = 'multi_study'     THEN 'multi_study'
                WHEN study_coverage_b = 'single_study'    THEN 'single_study'
                ELSE                                           'unknown'
            END
    """)

    op.alter_column("sum_corpus_relations", "canonical_scope_a", nullable=False)
    op.alter_column("sum_corpus_relations", "canonical_scope_b", nullable=False)
    op.drop_column("sum_corpus_relations", "is_conflicted_a")
    op.drop_column("sum_corpus_relations", "study_coverage_a")
    op.drop_column("sum_corpus_relations", "is_conflicted_b")
    op.drop_column("sum_corpus_relations", "study_coverage_b")
