from __future__ import annotations

from typing import Any

from studio_runner.jax_bench_adapter import AdapterExecutionError, run_sglang_jax_benchmark
from studio_runner.mock_adapter import run_mock_inference
from studio_runner.settings import settings


def _run_mock_backend(backend: str, prompt: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return run_mock_inference(backend=backend, prompt=prompt, parameters=parameters)


def _run_jax_backend(run_id: str, prompt: str, parameters: dict[str, Any]) -> dict[str, Any]:
    mode = settings.sglang_jax_adapter_mode.strip().lower()
    if mode not in {"auto", "bench", "mock"}:
        raise AdapterExecutionError(
            f"Invalid STUDIO_SGLANG_JAX_ADAPTER_MODE={settings.sglang_jax_adapter_mode!r}; expected auto|bench|mock"
        )

    if mode == "mock":
        result = _run_mock_backend(backend="sglang-jax", prompt=prompt, parameters=parameters)
        result["adapter_version"] = "mock-v1(jax-mode)"
        result["notes"] = "Configured to use mock adapter for sglang-jax"
        return result

    try:
        return run_sglang_jax_benchmark(run_id=run_id, prompt=prompt, parameters=parameters)
    except AdapterExecutionError as exc:
        if mode == "bench":
            raise
        fallback = _run_mock_backend(backend="sglang-jax", prompt=prompt, parameters=parameters)
        fallback["adapter_version"] = "mock-v1(jax-auto-fallback)"
        fallback["notes"] = (
            "sglang-jax bench wrapper unavailable; returned deterministic fallback result"
        )
        fallback["adapter_error"] = str(exc)
        return fallback


def run_backend_inference(
    run_id: str,
    backend: str,
    prompt: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    if backend == "sglang-jax":
        return _run_jax_backend(run_id=run_id, prompt=prompt, parameters=parameters)
    if backend == "sglang-pytorch":
        return _run_mock_backend(backend="sglang-pytorch", prompt=prompt, parameters=parameters)
    return _run_mock_backend(backend=backend, prompt=prompt, parameters=parameters)
