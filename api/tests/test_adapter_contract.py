from studio_runner.mock_adapter import run_mock_inference


def test_mock_adapter_contract_shape() -> None:
    result = run_mock_inference(
        backend="sglang-jax",
        prompt="What is the capital of France?",
        parameters={"multi_item_count": 4},
    )

    assert set(result.keys()) == {
        "score",
        "latency_ms",
        "throughput_items_per_s",
        "token_count",
        "adapter_version",
        "backend",
        "notes",
    }
    assert isinstance(result["score"], float)
    assert isinstance(result["latency_ms"], float)
    assert isinstance(result["throughput_items_per_s"], float)
    assert isinstance(result["token_count"], int)
    assert result["backend"] == "sglang-jax"
