"""add score_mode column to sum_final_rules

Captures which RESOLVE scoring branch produced final_score:
  - "relations_present" (votes + grounding + relations formula)
  - "relations_absent"  (grounding-dominant fallback)

Nullable so historical rows written before the column existed are valid.
See pipeline/stages/summarization/current_stages/resolve_stage.py for the
exact branch logic.

Revision ID: 0012
Revises: 0011
Create Date: 2026-05-16
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: Union[str, None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "sum_final_rules",
        sa.Column("score_mode", sa.String(length=30), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("sum_final_rules", "score_mode")
