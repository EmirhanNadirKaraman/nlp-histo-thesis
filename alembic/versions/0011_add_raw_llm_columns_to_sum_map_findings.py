"""add raw LLM columns to sum_map_findings

Captures the pre-coercion / pre-alias-repair LLM-emitted strings for
relation_type, direction, and category. Lets us audit what models
*actually said* versus the enum value we coerced to.

Revision ID: 0011
Revises: 0010
Create Date: 2026-05-14
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: Union[str, None] = "0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "sum_map_findings",
        sa.Column("raw_relation_type", sa.Text(), nullable=True),
    )
    op.add_column(
        "sum_map_findings",
        sa.Column("raw_direction", sa.Text(), nullable=True),
    )
    op.add_column(
        "sum_map_findings",
        sa.Column("raw_category", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("sum_map_findings", "raw_category")
    op.drop_column("sum_map_findings", "raw_direction")
    op.drop_column("sum_map_findings", "raw_relation_type")
