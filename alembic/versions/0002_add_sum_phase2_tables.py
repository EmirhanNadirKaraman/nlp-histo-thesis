"""add sum_map_findings, sum_normal_findings, sum_normal_finding_spans

Phase 2 persistence: store scored MAP findings (pre-normalization) and
normalized findings with evidence spans (post-normalization).

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-08
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── sum_map_findings ──────────────────────────────────────────────────────
    op.create_table(
        "sum_map_findings",
        sa.Column("id",                     sa.Integer,     primary_key=True, autoincrement=True),
        sa.Column("pipeline_run_id",         sa.Integer,     sa.ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("pmcid",                   sa.String(50),  nullable=False),
        sa.Column("chunk_id",                sa.String(20),  nullable=False),
        sa.Column("position_in_chunk",       sa.Integer,     nullable=False),
        sa.Column("category",                sa.String(50),  nullable=False),
        sa.Column("claim",                   sa.Text,        nullable=False),
        sa.Column("confidence",              sa.String(20),  nullable=False),
        sa.Column("verbatim_support",        sa.Text,        nullable=False),
        sa.Column("subject_entity",          sa.Text,        nullable=True),
        sa.Column("outcome_entity",          sa.Text,        nullable=True),
        sa.Column("relation_type",           sa.String(50),  nullable=False),
        sa.Column("direction",               sa.String(20),  nullable=True),
        sa.Column("assertion_status",        sa.String(20),  nullable=False),
        sa.Column("grounding_score",         sa.Float,       nullable=True),
        sa.Column("evidence_refs",           postgresql.ARRAY(sa.Text), nullable=True),
        sa.Column("scope_disease_subtype",   sa.Text,        nullable=True),
        sa.Column("scope_cohort_n",          sa.Integer,     nullable=True),
        sa.Column("scope_assay_method",      sa.Text,        nullable=True),
        sa.Column("scope_biomarker_cutoff",  sa.Text,        nullable=True),
        sa.Column("scope_tissue_site",       sa.Text,        nullable=True),
        sa.Column("scope_treatment_context", sa.Text,        nullable=True),
        sa.Column("scope_endpoint",          sa.Text,        nullable=True),
        sa.Column("scope_study_design",      sa.Text,        nullable=True),
        sa.Column("created_at",              sa.TIMESTAMP,   server_default=sa.text("now()")),
    )
    op.create_index("ix_smf_run",     "sum_map_findings", ["pipeline_run_id"])
    op.create_index("ix_smf_pmcid",   "sum_map_findings", ["pmcid"])
    op.create_index("ix_smf_subject", "sum_map_findings", ["subject_entity"])
    op.create_unique_constraint(
        "uq_sum_map_finding_pos",
        "sum_map_findings",
        ["pipeline_run_id", "chunk_id", "position_in_chunk"],
    )

    # ── sum_normal_findings ───────────────────────────────────────────────────
    op.create_table(
        "sum_normal_findings",
        sa.Column("id",                     sa.Integer,     primary_key=True, autoincrement=True),
        sa.Column("pipeline_run_id",         sa.Integer,     sa.ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("pmcid",                   sa.String(50),  nullable=False),
        sa.Column("normal_id",               sa.String(100), nullable=False),
        sa.Column("subject_entity",          sa.Text,        nullable=True),
        sa.Column("outcome_entity",          sa.Text,        nullable=True),
        sa.Column("relation_type",           sa.String(50),  nullable=False),
        sa.Column("direction",               sa.String(20),  nullable=True),
        sa.Column("assertion_status",        sa.String(20),  nullable=False),
        sa.Column("category",                sa.String(50),  nullable=False),
        sa.Column("predicate_text",          sa.Text,        nullable=False),
        sa.Column("mean_grounding_score",    sa.Float,       nullable=True),
        sa.Column("pmcids",                  postgresql.ARRAY(sa.Text), nullable=True),
        sa.Column("scope_disease_subtype",   sa.Text,        nullable=True),
        sa.Column("scope_cohort_n",          sa.Integer,     nullable=True),
        sa.Column("scope_assay_method",      sa.Text,        nullable=True),
        sa.Column("scope_biomarker_cutoff",  sa.Text,        nullable=True),
        sa.Column("scope_tissue_site",       sa.Text,        nullable=True),
        sa.Column("scope_treatment_context", sa.Text,        nullable=True),
        sa.Column("scope_endpoint",          sa.Text,        nullable=True),
        sa.Column("scope_study_design",      sa.Text,        nullable=True),
        sa.Column("created_at",              sa.TIMESTAMP,   server_default=sa.text("now()")),
    )
    op.create_index("ix_snf_run",     "sum_normal_findings", ["pipeline_run_id"])
    op.create_index("ix_snf_pmcid",   "sum_normal_findings", ["pmcid"])
    op.create_index("ix_snf_subject", "sum_normal_findings", ["subject_entity"])
    op.create_index("ix_snf_outcome", "sum_normal_findings", ["outcome_entity"])
    op.create_unique_constraint(
        "uq_sum_normal_finding",
        "sum_normal_findings",
        ["pipeline_run_id", "normal_id"],
    )

    # ── sum_normal_finding_spans ──────────────────────────────────────────────
    op.create_table(
        "sum_normal_finding_spans",
        sa.Column("id",               sa.Integer,     primary_key=True, autoincrement=True),
        sa.Column("normal_finding_id", sa.Integer,     sa.ForeignKey("sum_normal_findings.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sentence_id",       sa.String(20),  nullable=False),
        sa.Column("pmcid",             sa.String(50),  nullable=False),
        sa.Column("text_element_id",   sa.Integer,     sa.ForeignKey("text_elements.id", ondelete="SET NULL"), nullable=True),
        sa.Column("verbatim",          sa.Text,        nullable=False),
        sa.Column("created_at",        sa.TIMESTAMP,   server_default=sa.text("now()")),
    )
    op.create_index("ix_snfs_finding", "sum_normal_finding_spans", ["normal_finding_id"])
    op.create_index("ix_snfs_te",      "sum_normal_finding_spans", ["text_element_id"])
    op.create_unique_constraint(
        "uq_sum_nf_span",
        "sum_normal_finding_spans",
        ["normal_finding_id", "sentence_id"],
    )


def downgrade() -> None:
    op.drop_table("sum_normal_finding_spans")
    op.drop_table("sum_normal_findings")
    op.drop_table("sum_map_findings")
