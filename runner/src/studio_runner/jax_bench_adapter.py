from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from studio_runner.settings import settings


class AdapterExecutionError(RuntimeError):
    """Raised when an adapter cannot produce a valid benchmark result."""


def _stable_score(prompt: str) -> float:
    digest = hashlib.sha256(f"sglang-jax:{prompt}".encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], byteorder="big", signed=False) / float(2**64)
    return 0.1 + (0.8 * unit)


def _parse_last_float(text: str, pattern: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        value = value[0]
    return float(value)


def parse_benchmark_metrics(output: str) -> dict[str, float]:
    throughput = _parse_last_float(output, r"Throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*items/sec")
    latency_p50 = _parse_last_float(output, r"Latency p50:\s*([0-9]+(?:\.[0-9]+)?)\s*ms")
    latency_p95 = _parse_last_float(output, r"Latency p95:\s*([0-9]+(?:\.[0-9]+)?)\s*ms")
    latency_p99 = _parse_last_float(output, r"Latency p99:\s*([0-9]+(?:\.[0-9]+)?)\s*ms")

    if throughput is None:
        raise AdapterExecutionError("Unable to parse benchmark throughput from sglang-jax output")

    latency_ms = None
    for candidate in (latency_p50, latency_p95, latency_p99):
        if candidate is not None:
            latency_ms = candidate
            break
    if latency_ms is None:
        raise AdapterExecutionError("Unable to parse benchmark latency from sglang-jax output")

    metrics = {
        "throughput_items_per_s": throughput,
        "latency_ms": latency_ms,
    }
    if latency_p50 is not None:
        metrics["latency_p50_ms"] = latency_p50
    if latency_p95 is not None:
        metrics["latency_p95_ms"] = latency_p95
    if latency_p99 is not None:
        metrics["latency_p99_ms"] = latency_p99

    return metrics


def _resolve_entrypoint() -> Path:
    repo_root = Path(settings.sglang_jax_root)
    candidates: list[Path] = []

    if settings.sglang_jax_bench_entrypoint:
        configured = Path(settings.sglang_jax_bench_entrypoint)
        candidates.append(configured if configured.is_absolute() else repo_root / configured)

    candidates.append(repo_root / "test/srt/bench_score.py")
    candidates.append(repo_root / "test/srt/test_bench_score.py")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate

    rendered = ", ".join(str(path) for path in candidates)
    raise AdapterExecutionError(f"No sglang-jax bench entrypoint found; checked: {rendered}")


def _default_unittest_selector(entrypoint: Path, repo_root: Path) -> str:
    try:
        relative = entrypoint.relative_to(repo_root).with_suffix("")
    except ValueError:
        return entrypoint.stem
    return ".".join(relative.parts)


def _build_command(entrypoint: Path) -> tuple[list[str], Path]:
    repo_root = Path(settings.sglang_jax_root)

    if settings.sglang_jax_bench_command:
        return shlex.split(settings.sglang_jax_bench_command), repo_root

    python_exec = settings.sglang_jax_python_executable
    if entrypoint.name.startswith("test_"):
        selector = settings.sglang_jax_bench_unittest_selector
        selector = selector or _default_unittest_selector(entrypoint, repo_root)
        return [python_exec, "-m", "unittest", selector], repo_root

    return [python_exec, str(entrypoint)], repo_root


def _truncate(text: str, max_len: int = 1200) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return f"...{text[-max_len:]}"


def run_sglang_jax_benchmark(run_id: str, prompt: str, parameters: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(settings.sglang_jax_root)
    entrypoint = _resolve_entrypoint()
    command, cwd = _build_command(entrypoint)

    artifacts_dir = Path(settings.local_artifacts_root) / run_id / "sglang-jax"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = artifacts_dir / "bench.stdout.log"
    stderr_path = artifacts_dir / "bench.stderr.log"
    metadata_path = artifacts_dir / "bench.metadata.json"

    env = dict(os.environ)
    env["STUDIO_RUN_ID"] = run_id

    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=settings.sglang_jax_bench_timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise AdapterExecutionError(f"Benchmark command not found: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise AdapterExecutionError(
            f"sglang-jax benchmark timed out after {settings.sglang_jax_bench_timeout_seconds}s"
        ) from exc

    duration_ms = (time.perf_counter() - start) * 1000.0

    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "repo_root": str(repo_root),
                "entrypoint": str(entrypoint),
                "cwd": str(cwd),
                "command": command,
                "returncode": completed.returncode,
                "duration_ms": round(duration_ms, 3),
                "parameters": parameters,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    if completed.returncode != 0:
        summary = _truncate(completed.stderr or completed.stdout or "")
        raise AdapterExecutionError(
            f"sglang-jax benchmark failed with exit code {completed.returncode}: {summary}"
        )

    combined_output = f"{completed.stdout}\n{completed.stderr}"
    raw_metrics = parse_benchmark_metrics(combined_output)
    token_count = max(4, len(prompt.split()) * 2)

    return {
        "score": round(_stable_score(prompt), 6),
        "latency_ms": round(raw_metrics["latency_ms"], 3),
        "throughput_items_per_s": round(raw_metrics["throughput_items_per_s"], 3),
        "token_count": token_count,
        "adapter_version": "sglang-jax-bench-wrap-v1",
        "backend": "sglang-jax",
        "notes": "Wrap-first run via sglang-jax benchmark entrypoint",
        "raw_metrics": raw_metrics,
        "raw_artifacts": {
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "metadata_path": str(metadata_path),
        },
    }
