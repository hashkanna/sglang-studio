from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from studio_runner.db import Base


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    backend: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default="benchmark", index=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    parameters: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    score_input: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    mask_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    tolerance: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    repro_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    score_input_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    mask_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    result_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    artifact_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
