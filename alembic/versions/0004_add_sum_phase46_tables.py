"""add sum_canonical_rules, sum_relations, sum_final_rules

Phase 4-6 persistence: store CANONICALIZE, RELATE, and RESOLVE stages outputs.

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-08
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # sum_canonical_rules
    op.create_table(
        "sum_canonical_rules",
        sa.Column("id",                   sa.Integer,     primary_key=True, autoincrement=True),
        sa.Column("pipeline_run_id",      sa.Integer,     sa.ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("pmcid",                sa.String(50),  nullable=False),
        sa.Column("canonical_id",         sa.String(100), nullable=False),
        sa.Column("finding_group_id",     sa.Integer,     sa.ForeignKey("sum_finding_groups.id", ondelete="SET NULL"), nullable=True),
        sa.Column("group_id",             sa.String(100), nullable=False),
        sa.Column("subject_entity",       sa.Text,        nullable=False),
        sa.Column("outcome_entity",       sa.Text,        nullable=False),
        sa.Column("relation_type",        sa.String(50),  nullable=False),
        sa.Column("direction",            sa.String(20),  nullable=True),
        sa.Column("predicate_text",       sa.Text,        nullable=False),
        sa.Column("canonical_scope",      sa.String(20),  nullable=False),
        sa.Column("category",             sa.String(50),  nullable=False),
        sa.Column("pmcids",               postgresql.ARRAY(sa.Text), nullable=True),
        sa.Column("member_normal_ids",    postgresql.ARRAY(sa.Text), nullable=True),
        sa.Column("mean_grounding_score", sa.Float,       nullable=True),
        sa.Column("finding_count",        sa.Integer,     nullable=False),
        sa.Column("created_at",           sa.TIMESTAMP,   server_default=sa.text("now()")),
    )
    op.create_index("ix_scr_run",     "sum_canonical_rules", ["pipeline_run_id"])
    op.create_index("ix_scr_pmcid",   "sum_canonical_rules", ["pmcid"])
    op.create_index("ix_scr_subject", "sum_canonical_rules", ["subject_entity"])
    op.create_index("ix_scr_outcome", "sum_canonical_rules", ["outcome_entity"])
    op.create_unique_constraint(
        "uq_sum_canonical_rule",
        "sum_canonical_rules",
        ["pipeline_run_id", "canonical_id"],
    )

    # sum_relations
    op.create_table(
        "sum_relations",
        sa.Column("id",                  sa.Integer,     primary_key=True, autoincrement=True),
        sa.Column("pipeline_run_id",     sa.Integer,     sa.ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("pmcid",               sa.String(50),  nullable=False),
        sa.Column("rule_id_a",           sa.String(100), nullable=False),
        sa.Column("rule_id_b",           sa.String(100), nullable=False),
        sa.Column("canonical_rule_a_id", sa.Integer,     sa.ForeignKey("sum_canonical_rules.id", ondelete="CASCADE"), nullable=True),
        sa.Column("canonical_rule_b_id", sa.Integer,     sa.ForeignKey("sum_canonical_rules.id", ondelete="CASCADE"), nullable=True),
        sa.Column("relation_type",       sa.String(20),  nullable=False),
        sa.Column("nli_score_a_to_b",    sa.Float,       nullable=False),
        sa.Column("nli_score_b_to_a",    sa.Float,       nullable=False),
        sa.Column("created_at",          sa.TIMESTAMP,   server_default=sa.text("now()")),
    )
    op.create_index("ix_sr_run",    "sum_relations", ["pipeline_run_id"])
    op.create_index("ix_sr_pmcid",  "sum_relations", ["pmcid"])
    op.create_index("ix_sr_rule_a", "sum_relations", ["rule_id_a"])
    op.create_index("ix_sr_rule_b", "sum_relations", ["rule_id_b"])

    # sum_final_rules
    op.create_table(
        "sum_final_rules",
        sa.Column("id",                  sa.Integer,     primary_key=True, autoincrement=True),
        sa.Column("pipeline_run_id",     sa.Integer,     sa.ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("pmcid",               sa.String(50),  nullable=False),
        sa.Column("final_id",            sa.String(100), nullable=False),
        sa.Column("canonical_rule_id",   sa.Integer,     sa.ForeignKey("sum_canonical_rules.id", ondelete="SET NULL"), nullable=True),
        sa.Column("canonical_id",        sa.String(100), nullable=False),
        sa.Column("subject_entity",      sa.Text,        nullable=False),
        sa.Column("outcome_entity",      sa.Text,        nullable=False),
        sa.Column("relation_type",       sa.String(50),  nullable=False),
        sa.Column("direction",           sa.String(20),  nullable=True),
        sa.Column("predicate_text",      sa.Text,        nullable=False),
        sa.Column("category",            sa.String(50),  nullable=False),
        sa.Column("final_score",         sa.Float,       nullable=False),
        sa.Column("support_count",       sa.Integer,     nullable=False),
        sa.Column("contradict_count",    sa.Integer,     nullable=False),
        sa.Column("scope_qualify_count", sa.Integer,     nullable=False),
        sa.Column("is_contradicted",     sa.Boolean,     nullable=False),
        sa.Column("contradicted_by",     postgresql.ARRAY(sa.Text), nullable=True),
        sa.Column("created_at",          sa.TIMESTAMP,   server_default=sa.text("now()")),
    )
    op.create_index("ix_sfr_run",     "sum_final_rules", ["pipeline_run_id"])
    op.create_index("ix_sfr_pmcid",   "sum_final_rules", ["pmcid"])
    op.create_index("ix_sfr_subject", "sum_final_rules", ["subject_entity"])
    op.create_index("ix_sfr_outcome", "sum_final_rules", ["outcome_entity"])
    op.create_unique_constraint(
        "uq_sum_final_rule",
        "sum_final_rules",
        ["pipeline_run_id", "final_id"],
    )


def downgrade() -> None:
    op.drop_table("sum_final_rules")
    op.drop_table("sum_relations")
    op.drop_table("sum_canonical_rules")
