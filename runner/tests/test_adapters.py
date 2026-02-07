from __future__ import annotations

import pytest

from studio_runner import adapters
from studio_runner.adapter_errors import AdapterExecutionError
from studio_runner.settings import settings


def test_run_backend_inference_mock_backend() -> None:
    out = adapters.run_backend_inference(
        run_id="run-1",
        backend="mock",
        prompt="hello world",
        parameters={},
    )
    assert out["backend"] == "sglang-jax"
    assert out["adapter_version"] == "mock-v1"


def test_run_backend_inference_pytorch_mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "sglang_pytorch_adapter_mode", "mock")
    out = adapters.run_backend_inference(
        run_id="run-1",
        backend="sglang-pytorch",
        prompt="hello world",
        parameters={},
    )
    assert out["backend"] == "sglang-pytorch"
    assert out["adapter_version"] == "mock-v1(pytorch-mode)"


def test_run_backend_inference_jax_auto_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "sglang_jax_adapter_mode", "auto")

    monkeypatch.setattr(
        adapters,
        "run_sglang_jax_benchmark",
        lambda run_id, prompt, parameters: {
            "score": 0.5,
            "latency_ms": 12.0,
            "throughput_items_per_s": 80.0,
            "token_count": 12,
            "adapter_version": "sglang-jax-bench-wrap-v1",
            "backend": "sglang-jax",
            "notes": "ok",
        },
    )

    out = adapters.run_backend_inference(
        run_id="run-2",
        backend="sglang-jax",
        prompt="prompt",
        parameters={},
    )
    assert out["adapter_version"] == "sglang-jax-bench-wrap-v1"
    assert out["backend"] == "sglang-jax"


def test_run_backend_inference_jax_auto_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "sglang_jax_adapter_mode", "auto")

    def _raise(*args, **kwargs):
        raise AdapterExecutionError("entrypoint missing")

    monkeypatch.setattr(adapters, "run_sglang_jax_benchmark", _raise)

    out = adapters.run_backend_inference(
        run_id="run-3",
        backend="sglang-jax",
        prompt="prompt",
        parameters={},
    )
    assert out["backend"] == "sglang-jax"
    assert out["adapter_version"] == "mock-v1(jax-auto-fallback)"
    assert "adapter_error" in out


def test_run_backend_inference_jax_bench_mode_propagates_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "sglang_jax_adapter_mode", "bench")

    def _raise(*args, **kwargs):
        raise AdapterExecutionError("failed")

    monkeypatch.setattr(adapters, "run_sglang_jax_benchmark", _raise)

    with pytest.raises(AdapterExecutionError, match="failed"):
        adapters.run_backend_inference(
            run_id="run-4",
            backend="sglang-jax",
            prompt="prompt",
            parameters={},
        )


def test_run_backend_inference_pytorch_auto_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "sglang_pytorch_adapter_mode", "auto")

    monkeypatch.setattr(
        adapters,
        "run_sglang_pytorch_benchmark",
        lambda run_id, prompt, parameters: {
            "score": 0.5,
            "latency_ms": 10.0,
            "throughput_items_per_s": 120.0,
            "token_count": 12,
            "adapter_version": "sglang-pytorch-bench-wrap-v1",
            "backend": "sglang-pytorch",
            "notes": "ok",
        },
    )

    out = adapters.run_backend_inference(
        run_id="run-5",
        backend="sglang-pytorch",
        prompt="prompt",
        parameters={},
    )
    assert out["adapter_version"] == "sglang-pytorch-bench-wrap-v1"
    assert out["backend"] == "sglang-pytorch"


def test_run_backend_inference_pytorch_auto_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "sglang_pytorch_adapter_mode", "auto")

    def _raise(*args, **kwargs):
        raise AdapterExecutionError("entrypoint missing")

    monkeypatch.setattr(adapters, "run_sglang_pytorch_benchmark", _raise)

    out = adapters.run_backend_inference(
        run_id="run-6",
        backend="sglang-pytorch",
        prompt="prompt",
        parameters={},
    )
    assert out["backend"] == "sglang-pytorch"
    assert out["adapter_version"] == "mock-v1(pytorch-auto-fallback)"
    assert "adapter_error" in out


def test_run_backend_inference_pytorch_bench_mode_propagates_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "sglang_pytorch_adapter_mode", "bench")

    def _raise(*args, **kwargs):
        raise AdapterExecutionError("failed")

    monkeypatch.setattr(adapters, "run_sglang_pytorch_benchmark", _raise)

    with pytest.raises(AdapterExecutionError, match="failed"):
        adapters.run_backend_inference(
            run_id="run-7",
            backend="sglang-pytorch",
            prompt="prompt",
            parameters={},
        )


def test_run_backend_inference_score_mode_mock_backend() -> None:
    out = adapters.run_backend_inference(
        run_id="run-8",
        backend="mock",
        mode="score",
        prompt="prompt",
        parameters={},
        score_input={"query": "Is Paris in France?", "items": [" yes", " no"]},
        mask_config={"preset": "none"},
        tolerance={"abs_epsilon": 1e-6, "rel_epsilon": 0.0},
    )
    assert out["mode"] == "score"
    assert "token_logprobs" in out
    assert out["adapter_version"] == "mock-score-v1"


def test_run_backend_inference_jax_score_mode_uses_real_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "sglang_jax_adapter_mode", "auto")
    monkeypatch.setattr(
        adapters,
        "run_score_api_inference",
        lambda **kwargs: {
            "score": -1.2,
            "latency_ms": 12.0,
            "throughput_items_per_s": 30.0,
            "token_count": 3,
            "mode": "score",
            "backend": "sglang-jax",
            "adapter_version": "score-api-wrap-v1",
            "tokens": ["a", "b", "c"],
            "token_logprobs": [-0.1, -0.2, -0.3],
            "token_nll": [0.1, 0.2, 0.3],
            "token_ranks": [1, 1, 1],
        },
    )

    out = adapters.run_backend_inference(
        run_id="run-9",
        backend="sglang-jax",
        mode="score",
        prompt="prompt",
        parameters={},
        score_input={"query": "Q", "items": ["A"]},
    )
    assert out["mode"] == "score"
    assert out["adapter_version"] == "score-api-wrap-v1"


def test_run_backend_inference_jax_score_mode_requires_real_or_mock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "sglang_jax_adapter_mode", "auto")

    def _raise(**kwargs):
        raise AdapterExecutionError("missing endpoint")

    monkeypatch.setattr(adapters, "run_score_api_inference", _raise)

    with pytest.raises(AdapterExecutionError, match="missing endpoint"):
        adapters.run_backend_inference(
            run_id="run-10",
            backend="sglang-jax",
            mode="score",
            prompt="prompt",
            parameters={},
            score_input={"query": "Q", "items": ["A"]},
        )
