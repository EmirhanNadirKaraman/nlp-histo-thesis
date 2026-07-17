"""drop assertion_status column from sum_map_findings and sum_normal_findings

assertion_status (confirmed/negated/uncertain) was merged into direction
(positive/negative/absent/partial/unclear).  The two fields were redundant —
both encoded claim polarity.  direction is now the single source of truth.

Revision ID: 0007
Revises: 0006
Create Date: 2026-04-13
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("sum_map_findings", "assertion_status")
    op.drop_column("sum_normal_findings", "assertion_status")


def downgrade() -> None:
    import sqlalchemy as sa
    op.add_column("sum_map_findings",
        sa.Column("assertion_status", sa.String(20), nullable=True))
    op.add_column("sum_normal_findings",
        sa.Column("assertion_status", sa.String(20), nullable=True))
