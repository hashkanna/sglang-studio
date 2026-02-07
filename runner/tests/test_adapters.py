from __future__ import annotations

import pytest

from studio_runner import adapters
from studio_runner.jax_bench_adapter import AdapterExecutionError
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


def test_run_backend_inference_pytorch_uses_mock_path() -> None:
    out = adapters.run_backend_inference(
        run_id="run-1",
        backend="sglang-pytorch",
        prompt="hello world",
        parameters={},
    )
    assert out["backend"] == "sglang-pytorch"
    assert out["adapter_version"] == "mock-v1"


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
