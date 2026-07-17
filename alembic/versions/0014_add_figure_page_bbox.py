"""add page_number + bbox to figures (B-075 — coordinate provenance)

Figures previously had no page/bbox columns, so they could not be located in the
source PDF even though the cropper computes both (they're in CroppedMedia and the
media JSON). Mirror the `tables` columns. Populated forward by the doc-extraction
ingester and backfilled from cached media JSON by
eval/silver/experiments/E02_provenance/backfill_media_provenance.py.

Revision ID: 0014
Revises: 0013
Create Date: 2026-06-18
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0014"
down_revision: Union[str, None] = "0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("figures", sa.Column("page_number", sa.Integer(), nullable=True))
    op.add_column("figures", sa.Column("bbox_x1", sa.Integer(), nullable=True))
    op.add_column("figures", sa.Column("bbox_y1", sa.Integer(), nullable=True))
    op.add_column("figures", sa.Column("bbox_x2", sa.Integer(), nullable=True))
    op.add_column("figures", sa.Column("bbox_y2", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("figures", "bbox_y2")
    op.drop_column("figures", "bbox_x2")
    op.drop_column("figures", "bbox_y1")
    op.drop_column("figures", "bbox_x1")
    op.drop_column("figures", "page_number")
