from __future__ import annotations

from typing import Any

def _as_float(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Expected numeric metric but got {type(value).__name__}")


def _as_float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        if isinstance(item, (int, float)):
            out.append(float(item))
    return out


def _as_int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        if isinstance(item, (int, float)):
            out.append(int(item))
    return out


def _get_tolerance(tolerance: dict[str, Any] | None) -> tuple[float, float]:
    tolerance = tolerance or {}
    abs_epsilon = tolerance.get("abs_epsilon", 1e-6)
    rel_epsilon = tolerance.get("rel_epsilon", 0.0)
    return _as_float(abs_epsilon), _as_float(rel_epsilon)


def _build_token_diff(
    left: dict[str, Any],
    right: dict[str, Any],
    abs_epsilon: float,
    rel_epsilon: float,
) -> dict[str, Any]:
    left_tokens = left.get("tokens") if isinstance(left.get("tokens"), list) else []
    right_tokens = right.get("tokens") if isinstance(right.get("tokens"), list) else []
    left_logprobs = _as_float_list(left.get("token_logprobs"))
    right_logprobs = _as_float_list(right.get("token_logprobs"))

    pair_count = min(len(left_logprobs), len(right_logprobs))
    token_diffs: list[dict[str, Any]] = []
    mismatch_count = 0
    first_divergence_index: int | None = None
    max_abs_diff = 0.0
    sum_abs_diff = 0.0

    for idx in range(pair_count):
        left_lp = left_logprobs[idx]
        right_lp = right_logprobs[idx]
        abs_diff = abs(left_lp - right_lp)
        denom = abs(right_lp) if abs(right_lp) > 1e-12 else 1.0
        rel_diff = abs_diff / denom

        is_match = abs_diff <= abs_epsilon or (rel_epsilon > 0.0 and rel_diff <= rel_epsilon)
        if not is_match:
            mismatch_count += 1
            if first_divergence_index is None:
                first_divergence_index = idx

        max_abs_diff = max(max_abs_diff, abs_diff)
        sum_abs_diff += abs_diff

        token_value = (
            str(left_tokens[idx])
            if idx < len(left_tokens)
            else (str(right_tokens[idx]) if idx < len(right_tokens) else f"tok_{idx}")
        )
        token_diffs.append(
            {
                "index": idx,
                "token": token_value,
                "left_logprob": left_lp,
                "right_logprob": right_lp,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
                "is_match": is_match,
            }
        )

    mean_abs_diff = (sum_abs_diff / pair_count) if pair_count > 0 else 0.0
    left_nll = _as_float_list(left.get("token_nll"))
    right_nll = _as_float_list(right.get("token_nll"))
    nll_pair_count = min(len(left_nll), len(right_nll))
    if nll_pair_count > 0:
        nll_abs_diffs = [abs(left_nll[idx] - right_nll[idx]) for idx in range(nll_pair_count)]
        token_loss_diff_summary = {
            "pair_count": nll_pair_count,
            "max_abs_nll_diff": max(nll_abs_diffs),
            "mean_abs_nll_diff": sum(nll_abs_diffs) / nll_pair_count,
        }
    else:
        token_loss_diff_summary = {
            "pair_count": 0,
            "max_abs_nll_diff": 0.0,
            "mean_abs_nll_diff": 0.0,
        }

    left_ranks = _as_int_list(left.get("token_ranks"))
    right_ranks = _as_int_list(right.get("token_ranks"))
    rank_pair_count = min(len(left_ranks), len(right_ranks))
    if rank_pair_count > 0:
        rank_deltas = [left_ranks[idx] - right_ranks[idx] for idx in range(rank_pair_count)]
        worst_rank_drop = max(rank_deltas)
        mean_abs_rank_delta = sum(abs(delta) for delta in rank_deltas) / rank_pair_count
        mrr_left = sum(1.0 / max(rank, 1) for rank in left_ranks[:rank_pair_count]) / rank_pair_count
        mrr_right = sum(1.0 / max(rank, 1) for rank in right_ranks[:rank_pair_count]) / rank_pair_count
    else:
        worst_rank_drop = 0.0
        mean_abs_rank_delta = 0.0
        mrr_left = 0.0
        mrr_right = 0.0

    rank_delta_summary = {
        "pair_count": float(rank_pair_count),
        "worst_rank_drop": float(worst_rank_drop),
        "mean_abs_rank_delta": float(mean_abs_rank_delta),
        "mrr_left": float(mrr_left),
        "mrr_right": float(mrr_right),
        "mrr_delta": float(mrr_left - mrr_right),
    }

    return {
        "token_data_available": pair_count > 0,
        "token_pair_count": pair_count,
        "token_mismatch_count": mismatch_count,
        "first_divergence_index": first_divergence_index,
        "max_token_abs_diff": max_abs_diff,
        "mean_token_abs_diff": mean_abs_diff,
        "token_diffs": token_diffs,
        "token_loss_diff_summary": token_loss_diff_summary,
        "rank_delta_summary": rank_delta_summary,
    }


def compare_results(
    left: dict[str, Any],
    right: dict[str, Any],
    tolerance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    left_score = _as_float(left.get("score"))
    right_score = _as_float(right.get("score"))
    left_latency_ms = _as_float(left.get("latency_ms"))
    right_latency_ms = _as_float(right.get("latency_ms"))
    left_tp = _as_float(left.get("throughput_items_per_s"))
    right_tp = _as_float(right.get("throughput_items_per_s"))
    abs_epsilon, rel_epsilon = _get_tolerance(tolerance)

    token_diff = _build_token_diff(left=left, right=right, abs_epsilon=abs_epsilon, rel_epsilon=rel_epsilon)

    latency_pct_diff = 0.0
    if right_latency_ms > 0.0:
        latency_pct_diff = ((left_latency_ms - right_latency_ms) / right_latency_ms) * 100.0

    score_abs_diff = abs(left_score - right_score)
    score_parity_threshold = 0.02
    latency_regression_pct_threshold = 10.0
    score_parity_pass = score_abs_diff <= score_parity_threshold
    latency_regression_pass = latency_pct_diff <= latency_regression_pct_threshold
    token_parity_pass = token_diff["token_mismatch_count"] == 0
    overall_pass = score_parity_pass and latency_regression_pass and token_parity_pass

    return {
        "score_abs_diff": score_abs_diff,
        "latency_ms_diff": left_latency_ms - right_latency_ms,
        "latency_pct_diff": latency_pct_diff,
        "throughput_items_per_s_diff": left_tp - right_tp,
        "token_abs_epsilon": abs_epsilon,
        "token_rel_epsilon": rel_epsilon,
        "token_parity_pass": token_parity_pass,
        "score_parity_pass": score_parity_pass,
        "score_parity_threshold": score_parity_threshold,
        "latency_regression_pass": latency_regression_pass,
        "latency_regression_pct_threshold": latency_regression_pct_threshold,
        "overall_pass": overall_pass,
        "token_data_available": token_diff["token_data_available"],
        "token_pair_count": token_diff["token_pair_count"],
        "token_mismatch_count": token_diff["token_mismatch_count"],
        "first_divergence_index": token_diff["first_divergence_index"],
        "max_token_abs_diff": token_diff["max_token_abs_diff"],
        "mean_token_abs_diff": token_diff["mean_token_abs_diff"],
        "token_diffs": token_diff["token_diffs"],
        "token_loss_diff_summary": token_diff["token_loss_diff_summary"],
        "rank_delta_summary": token_diff["rank_delta_summary"],
    }
