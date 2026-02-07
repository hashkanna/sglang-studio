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
