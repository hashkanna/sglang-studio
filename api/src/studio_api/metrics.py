from __future__ import annotations


def _as_float(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Expected numeric metric but got {type(value).__name__}")


def compare_results(left: dict, right: dict) -> dict[str, float]:
    left_score = _as_float(left.get("score"))
    right_score = _as_float(right.get("score"))
    left_latency_ms = _as_float(left.get("latency_ms"))
    right_latency_ms = _as_float(right.get("latency_ms"))
    left_tp = _as_float(left.get("throughput_items_per_s"))
    right_tp = _as_float(right.get("throughput_items_per_s"))

    latency_pct_diff = 0.0
    if right_latency_ms > 0.0:
        latency_pct_diff = ((left_latency_ms - right_latency_ms) / right_latency_ms) * 100.0

    return {
        "score_abs_diff": abs(left_score - right_score),
        "latency_ms_diff": left_latency_ms - right_latency_ms,
        "latency_pct_diff": latency_pct_diff,
        "throughput_items_per_s_diff": left_tp - right_tp,
    }
