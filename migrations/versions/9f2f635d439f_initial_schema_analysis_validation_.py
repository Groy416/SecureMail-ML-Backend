"""initial_schema_analysis_validation_records

Creates the full initial schema for SecureMail-ML:
  - analysis_records: ML inference results, with is_synthetic flag for test data
  - validation_records: pre-flight payload validation logs

Revision ID: 9f2f635d439f
Revises: (none — this is the first migration)
Create Date: 2026-09-06
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9f2f635d439f"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all tables from scratch."""

    # ------------------------------------------------------------------
    # analysis_records
    # ------------------------------------------------------------------
    op.create_table(
        "analysis_records",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("request_id", sa.String(64), nullable=False, unique=True),
        sa.Column("session_id", sa.String(128), nullable=False),
        sa.Column("client_id", sa.String(128), nullable=True),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("record_count", sa.Integer(), nullable=False, default=0),
        # ML outputs
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("final_verdict", sa.String(32), nullable=False),
        sa.Column("rule_score", sa.Float(), nullable=False),
        sa.Column("rule_triggers_count", sa.Integer(), nullable=False, default=0),
        # JSON payloads
        sa.Column("trigger_details", sa.JSON(), nullable=False),
        sa.Column("ml_scores", sa.JSON(), nullable=False),
        sa.Column("explanations", sa.JSON(), nullable=False),
        sa.Column("model_bundle", sa.JSON(), nullable=False),
        # Synthetic data flags
        sa.Column("is_synthetic", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("source_label", sa.String(64), nullable=True),
    )
    op.create_index("ix_analysis_records_request_id", "analysis_records", ["request_id"], unique=True)
    op.create_index("ix_analysis_records_session_id", "analysis_records", ["session_id"])
    op.create_index("ix_analysis_records_client_id", "analysis_records", ["client_id"])
    op.create_index("ix_analysis_records_final_verdict", "analysis_records", ["final_verdict"])
    op.create_index("ix_analysis_records_is_synthetic", "analysis_records", ["is_synthetic"])
    op.create_index("idx_analysis_session_time", "analysis_records", ["session_id", "timestamp"])
    op.create_index("idx_analysis_verdict", "analysis_records", ["final_verdict"])

    # ------------------------------------------------------------------
    # validation_records
    # ------------------------------------------------------------------
    op.create_table(
        "validation_records",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("request_id", sa.String(64), nullable=False, unique=True),
        sa.Column("session_id", sa.String(128), nullable=False),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("valid", sa.Boolean(), nullable=False),
        sa.Column("record_count", sa.Integer(), nullable=False),
        sa.Column("issues_count", sa.Integer(), nullable=False, default=0),
        sa.Column("issues", sa.JSON(), nullable=False),
    )
    op.create_index("ix_validation_records_request_id", "validation_records", ["request_id"], unique=True)


def downgrade() -> None:
    """Drop all tables (full rollback to empty database)."""
    op.drop_table("validation_records")
    op.drop_table("analysis_records")
