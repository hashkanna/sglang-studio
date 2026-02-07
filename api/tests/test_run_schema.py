from __future__ import annotations

import pytest
from pydantic import ValidationError

from studio_api.schemas import RunCreate


def test_benchmark_mode_requires_prompt() -> None:
    with pytest.raises(ValidationError, match="prompt is required in benchmark mode"):
        RunCreate(
            backend="sglang-jax",
            mode="benchmark",
            prompt="",
            parameters={},
        )


def test_score_mode_requires_score_input() -> None:
    with pytest.raises(ValidationError, match="score_input is required in score mode"):
        RunCreate(
            backend="sglang-jax",
            mode="score",
            parameters={},
        )


def test_score_mode_accepts_structured_input() -> None:
    payload = RunCreate(
        backend="sglang-pytorch",
        mode="score",
        score_input={
            "query": "Is Paris in France?",
            "items": [" yes", " no"],
            "label_token_ids": [1, 2],
            "apply_softmax": True,
            "item_first": False,
        },
        mask_config={"preset": "none"},
        tolerance={"abs_epsilon": 1e-6, "rel_epsilon": 1e-5},
    )

    assert payload.mode == "score"
    assert payload.score_input is not None
    assert payload.score_input.query == "Is Paris in France?"
    assert payload.mask_config is not None
    assert payload.mask_config.preset == "none"


def test_custom_mask_requires_payload_or_artifact_ref() -> None:
    with pytest.raises(ValidationError, match="custom mask preset requires custom_mask or artifact_ref"):
        RunCreate(
            backend="sglang-jax",
            mode="score",
            score_input={"query": "q", "items": ["a"]},
            mask_config={"preset": "custom"},
        )
