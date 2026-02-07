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


def run_mock_score(
    backend: str,
    score_input: dict,
    mask_config: dict | None,
    tolerance: dict | None,
) -> dict:
    if backend == "mock":
        backend = "sglang-jax"

    query = str(score_input.get("query", ""))
    items = score_input.get("items") or []
    joined_items = " ".join(str(item) for item in items)
    tokens = (f"{query} {joined_items}".strip().split() or ["<empty>"])[:64]

    token_logprobs: list[float] = []
    token_ranks: list[int] = []
    for idx, token in enumerate(tokens):
        unit = _stable_unit_float(f"{backend}:{idx}:{token}")
        token_logprobs.append(round(-0.05 - (unit * 2.0), 6))
        token_ranks.append(1 + int(unit * 8))

    score = round(sum(token_logprobs), 6)
    token_nll = [round(-value, 6) for value in token_logprobs]
    token_count = len(tokens)

    base_latency_ms = 18.0 + (token_count * 0.6)
    if backend == "sglang-pytorch":
        base_latency_ms *= 1.06
    else:
        base_latency_ms *= 0.96

    item_count = max(1, len(items))
    throughput_items_per_s = item_count / (base_latency_ms / 1000.0)

    return {
        "score": score,
        "latency_ms": round(base_latency_ms, 3),
        "throughput_items_per_s": round(throughput_items_per_s, 3),
        "token_count": token_count,
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "token_nll": token_nll,
        "token_ranks": token_ranks,
        "mode": "score",
        "adapter_version": "mock-score-v1",
        "backend": backend,
        "mask_metadata": mask_config or {"preset": "none"},
        "tolerance": tolerance or {"abs_epsilon": 1e-6, "rel_epsilon": 0.0},
        "notes": "Deterministic mock score result for score-mode development",
    }
