"""add sum_rejection_summaries and sum_rejected_findings tables

Revision ID: 0008
Revises: 0007
Create Date: 2026-04-16
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSON

revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "sum_rejection_summaries",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("pipeline_run_id", sa.Integer(),
                  sa.ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
                  nullable=False, unique=True),
        sa.Column("pmcid", sa.String(50), nullable=False),
        sa.Column("grounding_threshold", sa.Float(), nullable=True),
        sa.Column("map_findings_total", sa.Integer(), nullable=False),
        sa.Column("map_grounding_rejected", sa.Integer(), nullable=False),
        sa.Column("normal_findings_total", sa.Integer(), nullable=False),
        sa.Column("non_groupable_total", sa.Integer(), nullable=False),
        sa.Column("non_groupable_no_subject", sa.Integer(), nullable=False),
        sa.Column("non_groupable_no_outcome", sa.Integer(), nullable=False),
        sa.Column("non_groupable_unclear_relation", sa.Integer(), nullable=False),
        sa.Column("grounding_rejection_rate", sa.Float(), nullable=False),
        sa.Column("non_groupable_rate", sa.Float(), nullable=False),
        sa.Column("grounding_rejected_by_category", JSON(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("now()")),
    )
    op.create_index("ix_srs_run",   "sum_rejection_summaries", ["pipeline_run_id"])
    op.create_index("ix_srs_pmcid", "sum_rejection_summaries", ["pmcid"])

    op.create_table(
        "sum_rejected_findings",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("pipeline_run_id", sa.Integer(),
                  sa.ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("rejection_summary_id", sa.Integer(),
                  sa.ForeignKey("sum_rejection_summaries.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("pmcid", sa.String(50), nullable=False),
        sa.Column("stage", sa.String(30), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("claim", sa.Text(), nullable=False),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("chunk_id", sa.String(20), nullable=True),
        sa.Column("grounding_score", sa.Float(), nullable=True),
        sa.Column("subject_entity", sa.Text(), nullable=True),
        sa.Column("outcome_entity", sa.Text(), nullable=True),
        sa.Column("relation_type", sa.String(50), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("now()")),
    )
    op.create_index("ix_srf_run",      "sum_rejected_findings", ["pipeline_run_id"])
    op.create_index("ix_srf_summary",  "sum_rejected_findings", ["rejection_summary_id"])
    op.create_index("ix_srf_pmcid",    "sum_rejected_findings", ["pmcid"])
    op.create_index("ix_srf_stage",    "sum_rejected_findings", ["stage"])
    op.create_index("ix_srf_category", "sum_rejected_findings", ["category"])


def downgrade() -> None:
    op.drop_index("ix_srf_category", "sum_rejected_findings")
    op.drop_index("ix_srf_stage",    "sum_rejected_findings")
    op.drop_index("ix_srf_pmcid",    "sum_rejected_findings")
    op.drop_index("ix_srf_summary",  "sum_rejected_findings")
    op.drop_index("ix_srf_run",      "sum_rejected_findings")
    op.drop_table("sum_rejected_findings")

    op.drop_index("ix_srs_pmcid", "sum_rejection_summaries")
    op.drop_index("ix_srs_run",   "sum_rejection_summaries")
    op.drop_table("sum_rejection_summaries")
