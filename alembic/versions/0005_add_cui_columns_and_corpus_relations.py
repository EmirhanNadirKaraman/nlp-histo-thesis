"""add subject_cui/outcome_cui to sum_canonical_rules and create sum_corpus_relations

Revision ID: 0005
Revises: 0004
Create Date: 2026-04-13
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # sum_canonical_rules: add UMLS CUI columns
    op.add_column("sum_canonical_rules", sa.Column("subject_cui", sa.String(20), nullable=True))
    op.add_column("sum_canonical_rules", sa.Column("outcome_cui", sa.String(20), nullable=True))
    op.create_index("ix_scr_subject_cui", "sum_canonical_rules", ["subject_cui"])
    op.create_index("ix_scr_outcome_cui", "sum_canonical_rules", ["outcome_cui"])

    # sum_corpus_relations: corpus-level cross-paper NLI relations
    op.create_table(
        "sum_corpus_relations",
        sa.Column("id",                       sa.Integer,     primary_key=True, autoincrement=True),
        sa.Column("corpus_run_id",            sa.String(20),  nullable=False),
        sa.Column("relation_id",              sa.String(20),  nullable=False),
        sa.Column("comparison_scope",         sa.String(20),  nullable=False),
        sa.Column("same_paper",               sa.Boolean,     nullable=False),
        sa.Column("rule_id_a",                sa.String(100), nullable=False),
        sa.Column("rule_id_b",                sa.String(100), nullable=False),
        sa.Column("pmcid_a",                  sa.String(50),  nullable=False),
        sa.Column("pmcid_b",                  sa.String(50),  nullable=False),
        sa.Column("relation_type",            sa.String(20),  nullable=False),
        sa.Column("nli_score_a_to_b",         sa.Float,       nullable=False),
        sa.Column("nli_score_b_to_a",         sa.Float,       nullable=False),
        sa.Column("subject_entity",           sa.Text,        nullable=False),
        sa.Column("outcome_entity",           sa.Text,        nullable=False),
        sa.Column("category",                 sa.String(50),  nullable=False),
        sa.Column("relation_type_structural", sa.String(50),  nullable=False),
        sa.Column("direction_a",              sa.String(20),  nullable=True),
        sa.Column("direction_b",              sa.String(20),  nullable=True),
        sa.Column("predicate_a",              sa.Text,        nullable=False),
        sa.Column("predicate_b",              sa.Text,        nullable=False),
        sa.Column("mean_grounding_a",         sa.Float,       nullable=True),
        sa.Column("mean_grounding_b",         sa.Float,       nullable=True),
        sa.Column("finding_count_a",          sa.Integer,     nullable=False),
        sa.Column("finding_count_b",          sa.Integer,     nullable=False),
        sa.Column("supporting_pmcids_a",      postgresql.ARRAY(sa.Text), nullable=True),
        sa.Column("supporting_pmcids_b",      postgresql.ARRAY(sa.Text), nullable=True),
        sa.Column("canonical_scope_a",        sa.String(20),  nullable=False),
        sa.Column("canonical_scope_b",        sa.String(20),  nullable=False),
        sa.Column("scope_check_result",       sa.String(30),  nullable=False, server_default="scope_unknown"),
        sa.Column("scope_note",               sa.Text,        nullable=False, server_default=""),
        sa.Column("created_at",               sa.TIMESTAMP,   server_default=sa.text("now()")),
        sa.UniqueConstraint("corpus_run_id", "relation_id", name="uq_corpus_relation"),
    )
    op.create_index("ix_cor_corpus_run",  "sum_corpus_relations", ["corpus_run_id"])
    op.create_index("ix_cor_scope",       "sum_corpus_relations", ["comparison_scope"])
    op.create_index("ix_cor_rule_a",      "sum_corpus_relations", ["rule_id_a"])
    op.create_index("ix_cor_rule_b",      "sum_corpus_relations", ["rule_id_b"])
    op.create_index("ix_cor_pmcid_a",     "sum_corpus_relations", ["pmcid_a"])
    op.create_index("ix_cor_pmcid_b",     "sum_corpus_relations", ["pmcid_b"])


def downgrade() -> None:
    op.drop_table("sum_corpus_relations")
    op.drop_index("ix_scr_outcome_cui", "sum_canonical_rules")
    op.drop_index("ix_scr_subject_cui", "sum_canonical_rules")
    op.drop_column("sum_canonical_rules", "outcome_cui")
    op.drop_column("sum_canonical_rules", "subject_cui")
