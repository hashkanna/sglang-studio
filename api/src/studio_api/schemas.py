from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ScoreInput(BaseModel):
    query: str = Field(min_length=1, max_length=20000)
    items: list[str] = Field(min_length=1)
    label_token_ids: list[int] = Field(default_factory=list)
    apply_softmax: bool = True
    item_first: bool = False

    @model_validator(mode="after")
    def validate_items(self) -> ScoreInput:
        if any(item.strip() == "" for item in self.items):
            raise ValueError("score_input.items must not contain empty strings")
        return self


class MaskConfig(BaseModel):
    preset: Literal["none", "causal", "bidirectional-prefix", "doc-isolation", "custom"] = "none"
    custom_mask: Any | None = None
    artifact_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_custom_mask(self) -> MaskConfig:
        if self.preset == "custom" and self.custom_mask is None and self.artifact_ref is None:
            raise ValueError("custom mask preset requires custom_mask or artifact_ref")
        return self


class ToleranceConfig(BaseModel):
    abs_epsilon: float = Field(default=1e-6, ge=0.0)
    rel_epsilon: float = Field(default=0.0, ge=0.0)


class ReproMetadata(BaseModel):
    backend_commit_sha: str | None = Field(default=None, max_length=128)
    model_revision: str | None = Field(default=None, max_length=256)
    tokenizer_revision: str | None = Field(default=None, max_length=256)
    config_hash: str | None = Field(default=None, max_length=128)
    branch: str | None = Field(default=None, max_length=256)
    pr_label: str | None = Field(default=None, max_length=256)
    extra: dict[str, Any] = Field(default_factory=dict)


class RunCreate(BaseModel):
    backend: Literal["sglang-jax", "sglang-pytorch", "mock"]
    mode: Literal["benchmark", "score"] = "benchmark"
    prompt: str | None = Field(default=None, max_length=20000)
    parameters: dict[str, Any] = Field(default_factory=dict)
    score_input: ScoreInput | None = None
    mask_config: MaskConfig | None = None
    tolerance: ToleranceConfig | None = None
    repro_metadata: ReproMetadata | None = None

    @model_validator(mode="after")
    def validate_mode_fields(self) -> RunCreate:
        if self.mode == "benchmark" and (self.prompt is None or self.prompt.strip() == ""):
            raise ValueError("prompt is required in benchmark mode")
        if self.mode == "score" and self.score_input is None:
            raise ValueError("score_input is required in score mode")
        return self

class RunView(BaseModel):
    id: str
    backend: str
    mode: str
    prompt: str
    parameters: dict[str, Any]
    score_input: dict[str, Any] | None
    mask_config: dict[str, Any] | None
    tolerance: dict[str, Any] | None
    repro_metadata: dict[str, Any] | None
    score_input_hash: str | None
    mask_hash: str | None
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
