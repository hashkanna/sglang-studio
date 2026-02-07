from __future__ import annotations

import pytest

from studio_runner.adapter_errors import AdapterExecutionError
from studio_runner.jax_bench_adapter import parse_benchmark_metrics


def test_parse_benchmark_metrics_extracts_core_fields() -> None:
    output = """
    [Benchmark] Single Item Latency
      Throughput: 132.4 items/sec
      Latency p50: 29.7 ms
      Latency p95: 55.1 ms
      Latency p99: 82.0 ms
    """

    metrics = parse_benchmark_metrics(output)
    assert metrics["throughput_items_per_s"] == pytest.approx(132.4)
    assert metrics["latency_ms"] == pytest.approx(29.7)
    assert metrics["latency_p95_ms"] == pytest.approx(55.1)
    assert metrics["latency_p99_ms"] == pytest.approx(82.0)


def test_parse_benchmark_metrics_falls_back_to_p95_for_latency() -> None:
    output = """
    Throughput: 101.0 items/sec
    Latency p95: 64.2 ms
    """

    metrics = parse_benchmark_metrics(output)
    assert metrics["throughput_items_per_s"] == pytest.approx(101.0)
    assert metrics["latency_ms"] == pytest.approx(64.2)


def test_parse_benchmark_metrics_requires_throughput() -> None:
    with pytest.raises(AdapterExecutionError, match="throughput"):
        parse_benchmark_metrics("Latency p50: 22.0 ms")
