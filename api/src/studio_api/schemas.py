from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class RunCreate(BaseModel):
    backend: Literal["sglang-jax", "sglang-pytorch", "mock"]
    prompt: str = Field(min_length=1, max_length=20000)
    parameters: dict[str, Any] = Field(default_factory=dict)


class RunView(BaseModel):
    id: str
    backend: str
    prompt: str
    parameters: dict[str, Any]
    status: str
    result_json: dict[str, Any] | None
    artifact_key: str | None
    error: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


class CompareRequest(BaseModel):
    left_run_id: str
    right_run_id: str


class CompareResponse(BaseModel):
    left_run_id: str
    right_run_id: str
    score_abs_diff: float
    latency_ms_diff: float
    latency_pct_diff: float
    throughput_items_per_s_diff: float
