import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

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
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    
    # Input summary
    record_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # ML Output
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    final_verdict: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    rule_score: Mapped[float] = mapped_column(Float, nullable=False)
    rule_triggers_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Detailed payloads stored as JSON
    trigger_details: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    ml_scores: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    explanations: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    model_bundle: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)

    # Synthetic / test data flags — real ML runs vs hand-crafted test records
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    source_label: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    __table_args__ = (
        Index("idx_analysis_session_time", "session_id", "timestamp"),
        Index("idx_analysis_verdict", "final_verdict"),
    )


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
    issues: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
