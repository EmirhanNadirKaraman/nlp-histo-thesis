"""add semantic_types column to entities table

Absorbs the one-off database/migrations/add_semantic_types.py script into the
Alembic chain.  Uses IF NOT EXISTS so databases that already ran the old script
are unaffected.

Revision ID: 0006
Revises: 0005
Create Date: 2026-04-13
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE entities ADD COLUMN IF NOT EXISTS semantic_types VARCHAR;"
    )


def downgrade() -> None:
    op.drop_column("entities", "semantic_types")
