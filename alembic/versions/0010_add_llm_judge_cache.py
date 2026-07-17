"""add llm_judge_cache table

Postgres-backed cache for LLM judge evaluation results (eval/llm_judge/).
Replaces the previous SQLite file-based cache.

Revision ID: 0010
Revises: 0009
Create Date: 2026-04-27
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0010"
down_revision: Union[str, None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "llm_judge_cache",
        sa.Column("cache_key",       sa.String(64),  primary_key=True),
        sa.Column("task",            sa.String(50),  nullable=False),
        sa.Column("judge_model",     sa.String(100), nullable=False),
        sa.Column("prompt_version",  sa.String(20),  nullable=False),
        sa.Column("schema_version",  sa.String(20),  nullable=False),
        sa.Column("label_source",    sa.String(50),  nullable=False),
        sa.Column("request_payload", JSONB,          nullable=False),
        sa.Column("result_payload",  JSONB,          nullable=False),
        sa.Column("pmcid",           sa.String(50),  nullable=True),
        sa.Column("pipeline_run_id", sa.Integer(),   nullable=True),
        sa.Column("source_id",       sa.Integer(),   nullable=True),
        sa.Column("created_at",      sa.TIMESTAMP(), server_default=sa.text("now()")),
        sa.Column("updated_at",      sa.TIMESTAMP(), server_default=sa.text("now()")),
    )
    op.create_index("ix_ljc_task",      "llm_judge_cache", ["task"])
    op.create_index("ix_ljc_versions",  "llm_judge_cache", ["judge_model", "prompt_version", "schema_version"])
    op.create_index("ix_ljc_pmcid_run", "llm_judge_cache", ["pmcid", "pipeline_run_id"])


def downgrade() -> None:
    op.drop_index("ix_ljc_pmcid_run", table_name="llm_judge_cache")
    op.drop_index("ix_ljc_versions",  table_name="llm_judge_cache")
    op.drop_index("ix_ljc_task",      table_name="llm_judge_cache")
    op.drop_table("llm_judge_cache")
