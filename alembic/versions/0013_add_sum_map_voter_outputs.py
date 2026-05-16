"""add sum_map_voter_outputs table

Captures each voter's AuditableSummary per (chunk, level, voter_index)
including voters whose output was discarded by agreement at that level.
Lets thesis analysis answer "what did each model say for chunk C5?"
rather than only "which voter won".

Revision ID: 0013
Revises: 0012
Create Date: 2026-05-16
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0013"
down_revision: Union[str, None] = "0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "sum_map_voter_outputs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "pipeline_run_id", sa.Integer(),
            sa.ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("pmcid",         sa.String(50),  nullable=False),
        sa.Column("chunk_id",      sa.String(20),  nullable=False),
        sa.Column("level",         sa.String(4),   nullable=False),
        sa.Column("voter_index",   sa.Integer(),   nullable=False),
        sa.Column("provider",      sa.String(20),  nullable=False),
        sa.Column("model",         sa.String(100), nullable=False),
        sa.Column("is_selected",   sa.Boolean(),   nullable=False),
        sa.Column("failed",        sa.Boolean(),   nullable=False),
        sa.Column("error_message", sa.Text(),      nullable=True),
        sa.Column("finding_count", sa.Integer(),   nullable=False),
        sa.Column("latency_ms",    sa.Float(),     nullable=True),
        sa.Column("raw_output",    JSONB(),        nullable=True),
        sa.Column("created_at",    sa.TIMESTAMP(), server_default=sa.text("now()")),
        sa.UniqueConstraint(
            "pipeline_run_id", "pmcid", "chunk_id", "level", "voter_index",
            name="uq_sum_map_voter_output",
        ),
    )
    op.create_index("ix_smvo_run",   "sum_map_voter_outputs", ["pipeline_run_id"])
    op.create_index("ix_smvo_pmcid", "sum_map_voter_outputs", ["pmcid"])
    op.create_index(
        "ix_smvo_chunk", "sum_map_voter_outputs",
        ["pipeline_run_id", "pmcid", "chunk_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_smvo_chunk", table_name="sum_map_voter_outputs")
    op.drop_index("ix_smvo_pmcid", table_name="sum_map_voter_outputs")
    op.drop_index("ix_smvo_run",   table_name="sum_map_voter_outputs")
    op.drop_table("sum_map_voter_outputs")
