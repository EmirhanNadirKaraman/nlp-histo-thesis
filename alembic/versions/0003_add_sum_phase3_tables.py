"""add sum_finding_groups, sum_group_members

Phase 3 persistence: store GROUP stage output — one row per FindingGroup
and one row per (group, NormalFinding) membership.

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-08
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── sum_finding_groups ────────────────────────────────────────────────────
    op.create_table(
        "sum_finding_groups",
        sa.Column("id",                  sa.Integer,     primary_key=True, autoincrement=True),
        sa.Column("pipeline_run_id",      sa.Integer,     sa.ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("pmcid",               sa.String(50),  nullable=False),
        sa.Column("group_id",            sa.String(100), nullable=False),
        sa.Column("subject_entity",      sa.Text,        nullable=False),
        sa.Column("outcome_entity",      sa.Text,        nullable=False),
        sa.Column("relation_type",       sa.String(50),  nullable=False),
        sa.Column("category",            sa.String(50),  nullable=False),
        sa.Column("scope_heterogeneity", sa.Float,       nullable=False),
        sa.Column("direction_counts",    sa.JSON,        nullable=False),
        sa.Column("created_at",          sa.TIMESTAMP,   server_default=sa.text("now()")),
    )
    op.create_index("ix_sfg_run",     "sum_finding_groups", ["pipeline_run_id"])
    op.create_index("ix_sfg_pmcid",   "sum_finding_groups", ["pmcid"])
    op.create_index("ix_sfg_subject", "sum_finding_groups", ["subject_entity"])
    op.create_index("ix_sfg_outcome", "sum_finding_groups", ["outcome_entity"])
    op.create_unique_constraint(
        "uq_sum_finding_group",
        "sum_finding_groups",
        ["pipeline_run_id", "group_id"],
    )

    # ── sum_group_members ─────────────────────────────────────────────────────
    op.create_table(
        "sum_group_members",
        sa.Column("id",               sa.Integer,     primary_key=True, autoincrement=True),
        sa.Column("finding_group_id",  sa.Integer,     sa.ForeignKey("sum_finding_groups.id", ondelete="CASCADE"), nullable=False),
        sa.Column("normal_finding_id", sa.Integer,     sa.ForeignKey("sum_normal_findings.id", ondelete="SET NULL"), nullable=True),
        sa.Column("normal_id",         sa.String(100), nullable=False),
        sa.Column("created_at",        sa.TIMESTAMP,   server_default=sa.text("now()")),
    )
    op.create_index("ix_sgm_group", "sum_group_members", ["finding_group_id"])
    op.create_index("ix_sgm_nf",    "sum_group_members", ["normal_finding_id"])
    op.create_unique_constraint(
        "uq_sum_group_member",
        "sum_group_members",
        ["finding_group_id", "normal_id"],
    )


def downgrade() -> None:
    op.drop_table("sum_group_members")
    op.drop_table("sum_finding_groups")
