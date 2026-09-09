from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, Index, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    session_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    client_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    capture_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    protocol: Mapped[Optional[str]] = mapped_column(String(16), nullable=True, index=True)
    posture: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    record_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    evidence_ref_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    final_verdict: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    rule_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rule_triggers_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    trigger_details: Mapped[Any] = mapped_column(JSON, nullable=False)
    ml_scores: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    explanations: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    model_bundle: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    # Synthetic / test data flags — real ML runs vs hand-crafted test records
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    source_label: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    __table_args__ = (
        Index("idx_analysis_session_time", "session_id", "timestamp"),
        Index("idx_analysis_verdict", "final_verdict"),
    )


class CaptureJob(Base):
    __tablename__ = "capture_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    capture_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    pcap_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    input_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    session_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_sessions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    progress: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    diagnostics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    error_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


class CaptureSession(Base):
    __tablename__ = "capture_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    capture_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    session_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    protocol: Mapped[str] = mapped_column(String(16), nullable=False)
    dest_port: Mapped[int] = mapped_column(Integer, nullable=False)
    preview: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    record: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    analysis_request_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)

    __table_args__ = (Index("idx_capture_session", "capture_id", "session_id", unique=True),)


class ValidationRecord(Base):
    __tablename__ = "validation_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    
    valid: Mapped[bool] = mapped_column(Boolean, nullable=False)
    record_count: Mapped[int] = mapped_column(Integer, nullable=False)
    issues_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    issues: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
