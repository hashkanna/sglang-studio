from __future__ import annotations

from studio_runner import score_api_adapter
from studio_runner.settings import settings


def test_run_score_api_inference_normalizes_token_fields(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(settings, "local_artifacts_root", str(tmp_path))
    monkeypatch.setattr(score_api_adapter, "_score_api_url", lambda backend: "http://example/v1/score")
    monkeypatch.setattr(
        score_api_adapter,
        "_post_json",
        lambda url, payload: (
            {
                "score": -1.5,
                "tokens": ["A", "B", "C"],
                "token_logprobs": [-0.1, -0.2, -1.2],
                "token_ranks": [1, 1, 3],
            },
            200,
        ),
    )

    out = score_api_adapter.run_score_api_inference(
        run_id="run-1",
        backend="sglang-jax",
        prompt="prompt",
        score_input={"query": "Q", "items": ["A"]},
        mask_config={"preset": "none"},
        tolerance={"abs_epsilon": 1e-6, "rel_epsilon": 0.0},
    )

    assert out["mode"] == "score"
    assert out["score"] == -1.5
    assert out["token_count"] == 3
    assert out["token_nll"] == [0.1, 0.2, 1.2]
    assert out["token_ranks"] == [1, 1, 3]
    assert "raw_artifacts" in out


def test_run_score_api_inference_derives_score_from_scores_field(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(settings, "local_artifacts_root", str(tmp_path))
    monkeypatch.setattr(score_api_adapter, "_score_api_url", lambda backend: "http://example/v1/score")
    monkeypatch.setattr(
        score_api_adapter,
        "_post_json",
        lambda url, payload: (
            {
                "scores": [[0.5, 0.25], [0.1, 0.15]],
                "tokens": ["A", "B"],
                "token_logprobs": [-0.2, -0.4],
            },
            200,
        ),
    )

    out = score_api_adapter.run_score_api_inference(
        run_id="run-2",
        backend="sglang-pytorch",
        prompt="prompt",
        score_input={"query": "Q", "items": ["A", "B"]},
        mask_config=None,
        tolerance=None,
    )

    assert out["score"] == 1.0
    assert out["token_count"] == 2
