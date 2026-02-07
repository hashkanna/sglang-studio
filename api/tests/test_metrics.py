from studio_api.metrics import compare_results
import pytest


def test_compare_results_computes_expected_deltas() -> None:
    left = {"score": 0.82, "latency_ms": 120.0, "throughput_items_per_s": 8.0}
    right = {"score": 0.80, "latency_ms": 100.0, "throughput_items_per_s": 10.0}

    out = compare_results(left, right)

    assert out["score_abs_diff"] == pytest.approx(0.02)
    assert out["latency_ms_diff"] == 20.0
    assert out["latency_pct_diff"] == 20.0
    assert out["throughput_items_per_s_diff"] == -2.0
    assert out["score_parity_pass"] is True
    assert out["latency_regression_pass"] is False


def test_compare_results_token_parity_and_first_divergence() -> None:
    left = {
        "score": -1.0,
        "latency_ms": 50.0,
        "throughput_items_per_s": 10.0,
        "tokens": ["A", "B", "C"],
        "token_logprobs": [-0.1, -0.8, -0.3],
        "token_nll": [0.1, 0.8, 0.3],
        "token_ranks": [1, 5, 2],
    }
    right = {
        "score": -1.0,
        "latency_ms": 50.0,
        "throughput_items_per_s": 10.0,
        "tokens": ["A", "B", "C"],
        "token_logprobs": [-0.1, -0.2, -0.3],
        "token_nll": [0.1, 0.2, 0.3],
        "token_ranks": [1, 1, 2],
    }

    out = compare_results(left, right, tolerance={"abs_epsilon": 1e-6, "rel_epsilon": 0.0})

    assert out["token_data_available"] is True
    assert out["token_pair_count"] == 3
    assert out["token_mismatch_count"] == 1
    assert out["first_divergence_index"] == 1
    assert out["token_parity_pass"] is False
    assert out["token_loss_diff_summary"]["max_abs_nll_diff"] == pytest.approx(0.6)
    assert out["rank_delta_summary"]["worst_rank_drop"] == pytest.approx(4.0)


def test_compare_results_token_rel_tolerance_can_pass() -> None:
    left = {"score": 0.0, "latency_ms": 10.0, "throughput_items_per_s": 1.0, "token_logprobs": [-0.1001]}
    right = {"score": 0.0, "latency_ms": 10.0, "throughput_items_per_s": 1.0, "token_logprobs": [-0.1000]}

    out = compare_results(left, right, tolerance={"abs_epsilon": 0.0, "rel_epsilon": 0.02})
    assert out["token_parity_pass"] is True
    assert out["token_mismatch_count"] == 0
