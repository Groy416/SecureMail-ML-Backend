"""add durable PCAP extraction jobs and session links

Revision ID: b4e5c6d7f8a9
Revises: 9f2f635d439f
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b4e5c6d7f8a9"
down_revision: Union[str, Sequence[str], None] = "9f2f635d439f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("analysis_records", sa.Column("capture_id", sa.String(128), nullable=True))
    op.add_column("analysis_records", sa.Column("protocol", sa.String(16), nullable=True))
    op.add_column("analysis_records", sa.Column("posture", sa.String(32), nullable=True))
    op.add_column(
        "analysis_records",
        sa.Column("evidence_ref_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.alter_column("analysis_records", "rule_score", existing_type=sa.Float(), nullable=True)
    op.create_index("ix_analysis_records_capture_id", "analysis_records", ["capture_id"])
    op.create_index("ix_analysis_records_protocol", "analysis_records", ["protocol"])
    op.create_index("ix_analysis_records_posture", "analysis_records", ["posture"])

    op.create_table(
        "capture_jobs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("capture_id", sa.String(64), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("pcap_sha256", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("input_path", sa.String(512), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="2"),
        sa.Column("session_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("processed_sessions", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("progress", sa.Float(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("diagnostics", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("error_code", sa.String(64), nullable=True),
    )
    op.create_index("ix_capture_jobs_capture_id", "capture_jobs", ["capture_id"], unique=True)
    op.create_index("ix_capture_jobs_status", "capture_jobs", ["status"])

    op.create_table(
        "capture_sessions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("capture_id", sa.String(64), nullable=False),
        sa.Column("session_id", sa.String(128), nullable=False),
        sa.Column("protocol", sa.String(16), nullable=False),
        sa.Column("dest_port", sa.Integer(), nullable=False),
        sa.Column("preview", sa.JSON(), nullable=False),
        sa.Column("record", sa.JSON(), nullable=False),
        sa.Column("analysis_request_id", sa.String(64), nullable=True),
    )
    op.create_index("ix_capture_sessions_capture_id", "capture_sessions", ["capture_id"])
    op.create_index("ix_capture_sessions_session_id", "capture_sessions", ["session_id"])
    op.create_index("ix_capture_sessions_analysis_request_id", "capture_sessions", ["analysis_request_id"])
    op.create_index(
        "idx_capture_session",
        "capture_sessions",
        ["capture_id", "session_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_table("capture_sessions")
    op.drop_table("capture_jobs")
    op.drop_index("ix_analysis_records_posture", table_name="analysis_records")
    op.drop_index("ix_analysis_records_protocol", table_name="analysis_records")
    op.drop_index("ix_analysis_records_capture_id", table_name="analysis_records")
    op.drop_column("analysis_records", "evidence_ref_count")
    op.drop_column("analysis_records", "posture")
    op.drop_column("analysis_records", "protocol")
    op.drop_column("analysis_records", "capture_id")
    op.alter_column("analysis_records", "rule_score", existing_type=sa.Float(), nullable=False)
