from __future__ import annotations

from typing import Any

from studio_runner.adapter_errors import AdapterExecutionError
from studio_runner.jax_bench_adapter import run_sglang_jax_benchmark
from studio_runner.mock_adapter import run_mock_inference, run_mock_score
from studio_runner.pytorch_bench_adapter import run_sglang_pytorch_benchmark
from studio_runner.score_api_adapter import run_score_api_inference
from studio_runner.settings import settings


def _run_mock_backend(backend: str, prompt: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return run_mock_inference(backend=backend, prompt=prompt, parameters=parameters)


def _run_jax_backend(
    run_id: str,
    run_mode: str,
    prompt: str,
    parameters: dict[str, Any],
    score_input: dict[str, Any] | None,
    mask_config: dict[str, Any] | None,
    tolerance: dict[str, Any] | None,
) -> dict[str, Any]:
    adapter_mode = settings.sglang_jax_adapter_mode.strip().lower()
    if adapter_mode not in {"auto", "bench", "mock"}:
        raise AdapterExecutionError(
            f"Invalid STUDIO_SGLANG_JAX_ADAPTER_MODE={settings.sglang_jax_adapter_mode!r}; expected auto|bench|mock"
        )

    if run_mode == "score":
        if score_input is None:
            raise AdapterExecutionError("score_input is required for score mode")
        if adapter_mode == "mock":
            result = run_mock_score(
                backend="sglang-jax",
                score_input=score_input,
                mask_config=mask_config,
                tolerance=tolerance,
            )
            result["adapter_version"] = "mock-score-v1(jax-mode)"
            result["notes"] = "Configured to use mock score adapter for sglang-jax"
            return result

        # Score-mode parity runs must use real score execution when not explicitly mocked.
        return run_score_api_inference(
            run_id=run_id,
            backend="sglang-jax",
            prompt=prompt,
            score_input=score_input,
            mask_config=mask_config,
            tolerance=tolerance,
        )

    if adapter_mode == "mock":
        result = _run_mock_backend(backend="sglang-jax", prompt=prompt, parameters=parameters)
        result["adapter_version"] = "mock-v1(jax-mode)"
        result["notes"] = "Configured to use mock adapter for sglang-jax"
        return result

    try:
        return run_sglang_jax_benchmark(run_id=run_id, prompt=prompt, parameters=parameters)
    except AdapterExecutionError as exc:
        if adapter_mode == "bench":
            raise
        fallback = _run_mock_backend(backend="sglang-jax", prompt=prompt, parameters=parameters)
        fallback["adapter_version"] = "mock-v1(jax-auto-fallback)"
        fallback["notes"] = (
            "sglang-jax bench wrapper unavailable; returned deterministic fallback result"
        )
        fallback["adapter_error"] = str(exc)
        return fallback


def _run_pytorch_backend(
    run_id: str,
    run_mode: str,
    prompt: str,
    parameters: dict[str, Any],
    score_input: dict[str, Any] | None,
    mask_config: dict[str, Any] | None,
    tolerance: dict[str, Any] | None,
) -> dict[str, Any]:
    adapter_mode = settings.sglang_pytorch_adapter_mode.strip().lower()
    if adapter_mode not in {"auto", "bench", "mock"}:
        raise AdapterExecutionError(
            f"Invalid STUDIO_SGLANG_PYTORCH_ADAPTER_MODE={settings.sglang_pytorch_adapter_mode!r}; expected auto|bench|mock"
        )

    if run_mode == "score":
        if score_input is None:
            raise AdapterExecutionError("score_input is required for score mode")
        if adapter_mode == "mock":
            result = run_mock_score(
                backend="sglang-pytorch",
                score_input=score_input,
                mask_config=mask_config,
                tolerance=tolerance,
            )
            result["adapter_version"] = "mock-score-v1(pytorch-mode)"
            result["notes"] = "Configured to use mock score adapter for sglang-pytorch"
            return result

        # Score-mode parity runs must use real score execution when not explicitly mocked.
        return run_score_api_inference(
            run_id=run_id,
            backend="sglang-pytorch",
            prompt=prompt,
            score_input=score_input,
            mask_config=mask_config,
            tolerance=tolerance,
        )

    if adapter_mode == "mock":
        result = _run_mock_backend(backend="sglang-pytorch", prompt=prompt, parameters=parameters)
        result["adapter_version"] = "mock-v1(pytorch-mode)"
        result["notes"] = "Configured to use mock adapter for sglang-pytorch"
        return result

    try:
        return run_sglang_pytorch_benchmark(run_id=run_id, prompt=prompt, parameters=parameters)
    except AdapterExecutionError as exc:
        if adapter_mode == "bench":
            raise
        fallback = _run_mock_backend(backend="sglang-pytorch", prompt=prompt, parameters=parameters)
        fallback["adapter_version"] = "mock-v1(pytorch-auto-fallback)"
        fallback["notes"] = (
            "sglang-pytorch bench wrapper unavailable; returned deterministic fallback result"
        )
        fallback["adapter_error"] = str(exc)
        return fallback


def run_backend_inference(
    run_id: str,
    backend: str,
    prompt: str,
    parameters: dict[str, Any],
    mode: str = "benchmark",
    score_input: dict[str, Any] | None = None,
    mask_config: dict[str, Any] | None = None,
    tolerance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_mode = mode.strip().lower()
    if backend == "sglang-jax":
        return _run_jax_backend(
            run_id=run_id,
            run_mode=run_mode,
            prompt=prompt,
            parameters=parameters,
            score_input=score_input,
            mask_config=mask_config,
            tolerance=tolerance,
        )
    if backend == "sglang-pytorch":
        return _run_pytorch_backend(
            run_id=run_id,
            run_mode=run_mode,
            prompt=prompt,
            parameters=parameters,
            score_input=score_input,
            mask_config=mask_config,
            tolerance=tolerance,
        )
    if run_mode == "score":
        if score_input is None:
            raise AdapterExecutionError("score_input is required for score mode")
        return run_mock_score(
            backend=backend,
            score_input=score_input,
            mask_config=mask_config,
            tolerance=tolerance,
        )
    return _run_mock_backend(backend=backend, prompt=prompt, parameters=parameters)
