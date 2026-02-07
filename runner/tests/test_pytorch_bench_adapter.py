from __future__ import annotations

import pytest

from studio_runner.adapter_errors import AdapterExecutionError
from studio_runner.pytorch_bench_adapter import parse_benchmark_metrics


def test_parse_benchmark_metrics_extracts_core_fields() -> None:
    output = """
    Overall Summary for RPS 160, Duration 60s, Item Count 10:
      Achieved RPS:          157.31
      Item count:            10
      P50 response time:     21.11 ms
      P90 response time:     39.02 ms
      P99 response time:     72.44 ms
    """

    metrics = parse_benchmark_metrics(output)
    assert metrics["achieved_rps"] == pytest.approx(157.31)
    assert metrics["item_count"] == pytest.approx(10.0)
    assert metrics["throughput_items_per_s"] == pytest.approx(1573.1)
    assert metrics["latency_ms"] == pytest.approx(21.11)
    assert metrics["latency_p99_ms"] == pytest.approx(72.44)


def test_parse_benchmark_metrics_can_use_average_latency() -> None:
    output = """
    Achieved RPS: 80.5
    Item count: 4
    Average response time: 45.7 ms
    """

    metrics = parse_benchmark_metrics(output)
    assert metrics["throughput_items_per_s"] == pytest.approx(322.0)
    assert metrics["latency_ms"] == pytest.approx(45.7)


def test_parse_benchmark_metrics_requires_achieved_rps() -> None:
    with pytest.raises(AdapterExecutionError, match="achieved RPS"):
        parse_benchmark_metrics("Item count: 8\nP50 response time: 10.1 ms")
