from __future__ import annotations

import hashlib


def _stable_unit_float(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(2**64)


def run_mock_inference(backend: str, prompt: str, parameters: dict) -> dict:
    if backend == "mock":
        backend = "sglang-jax"

    unit = _stable_unit_float(f"{backend}:{prompt}")
    token_count = max(4, len(prompt.split()) * 2)
    multi_item_count = max(1, int(parameters.get("multi_item_count", 1)))

    base_latency_ms = 45.0 + (token_count * 0.9)
    if backend == "sglang-pytorch":
        base_latency_ms *= 1.08
    else:
        base_latency_ms *= 0.97

    # In mock mode, multi-item workloads are modeled as amortizing per-item latency.
    latency_ms = base_latency_ms * (1.0 + (0.14 * multi_item_count))
    throughput_items_per_s = multi_item_count / (latency_ms / 1000.0)

    score = 0.1 + (0.8 * unit)

    return {
        "score": round(score, 6),
        "latency_ms": round(latency_ms, 3),
        "throughput_items_per_s": round(throughput_items_per_s, 3),
        "token_count": token_count,
        "adapter_version": "mock-v1",
        "backend": backend,
        "notes": "Deterministic mock result for Milestone 0 vertical slice",
    }
