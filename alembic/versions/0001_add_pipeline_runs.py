"""add pipeline_runs table

Revision ID: 0001
Revises: (none — first migration)
Create Date: 2026-04-08

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "pipeline_runs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(100), nullable=False),
        sa.Column("pmcid", sa.String(50), nullable=False),
        sa.Column(
            "document_id",
            sa.Integer,
            sa.ForeignKey("documents.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("status", sa.String(20), nullable=False, server_default="running"),
        sa.Column("config_snapshot", sa.JSON, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column(
            "started_at",
            sa.TIMESTAMP,
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("finished_at", sa.TIMESTAMP, nullable=True),
    )
    op.create_index("ix_pipeline_runs_run_id", "pipeline_runs", ["run_id"], unique=True)
    op.create_index("ix_pipeline_runs_pmcid", "pipeline_runs", ["pmcid"])
    op.create_index("ix_pipeline_runs_document_id", "pipeline_runs", ["document_id"])
    op.create_index(
        "ix_pipeline_runs_pmcid_started_at",
        "pipeline_runs",
        ["pmcid", "started_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_pipeline_runs_pmcid_started_at", table_name="pipeline_runs")
    op.drop_index("ix_pipeline_runs_document_id", table_name="pipeline_runs")
    op.drop_index("ix_pipeline_runs_pmcid", table_name="pipeline_runs")
    op.drop_index("ix_pipeline_runs_run_id", table_name="pipeline_runs")
    op.drop_table("pipeline_runs")
