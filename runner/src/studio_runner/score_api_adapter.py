from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib import error, request

from studio_runner.adapter_errors import AdapterExecutionError
from studio_runner.settings import settings


def _truncate(text: str, max_len: int = 1200) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return f"...{text[-max_len:]}"


def _score_api_url(backend: str) -> str:
    if backend == "sglang-jax":
        url = settings.sglang_jax_score_api_url
    elif backend == "sglang-pytorch":
        url = settings.sglang_pytorch_score_api_url
    else:
        raise AdapterExecutionError(f"Unsupported backend for score mode: {backend}")

    if not url:
        raise AdapterExecutionError(
            f"No score API URL configured for {backend}; set STUDIO_{backend.upper().replace('-', '_')}_SCORE_API_URL"
        )
    return url


def _flatten_numbers(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_numbers(item))
        return out
    return []


def _extract_tokens(response_json: dict[str, Any], score_input: dict[str, Any]) -> list[str]:
    tokens = response_json.get("tokens")
    if isinstance(tokens, list):
        return [str(token) for token in tokens]

    query = str(score_input.get("query", ""))
    items = score_input.get("items") or []
    fallback = f"{query} {' '.join(str(item) for item in items)}".strip().split()
    return fallback


def _extract_token_logprobs(response_json: dict[str, Any]) -> list[float]:
    token_logprobs = response_json.get("token_logprobs")
    if isinstance(token_logprobs, list):
        values = _flatten_numbers(token_logprobs)
        if values:
            return values

    logprobs = response_json.get("logprobs")
    if isinstance(logprobs, list):
        if all(isinstance(item, (int, float)) for item in logprobs):
            return [float(item) for item in logprobs]
        if all(isinstance(item, dict) for item in logprobs):
            values = [
                float(item["logprob"])
                for item in logprobs
                if isinstance(item.get("logprob"), (int, float))
            ]
            if values:
                return values

    return []


def _extract_token_ranks(response_json: dict[str, Any], token_count: int) -> list[int]:
    token_ranks = response_json.get("token_ranks")
    if isinstance(token_ranks, list):
        ranks = [int(rank) for rank in token_ranks if isinstance(rank, (int, float))]
        if ranks:
            return ranks

    logprobs = response_json.get("logprobs")
    if isinstance(logprobs, list) and all(isinstance(item, dict) for item in logprobs):
        ranks = [int(item["rank"]) for item in logprobs if isinstance(item.get("rank"), (int, float))]
        if ranks:
            return ranks

    return [0] * token_count


def _derive_score(response_json: dict[str, Any], token_logprobs: list[float]) -> float:
    raw_score = response_json.get("score")
    if isinstance(raw_score, (int, float)):
        return float(raw_score)

    raw_scores = response_json.get("scores")
    flat_scores = _flatten_numbers(raw_scores)
    if flat_scores:
        return float(sum(flat_scores))

    if token_logprobs:
        return float(sum(token_logprobs))

    raise AdapterExecutionError("Score API response missing score/scores/token_logprobs")


def _post_json(url: str, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=settings.score_api_timeout_seconds) as resp:
            status = int(getattr(resp, "status", 200))
            content = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = _truncate(exc.read().decode("utf-8", errors="replace"))
        raise AdapterExecutionError(f"Score API HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise AdapterExecutionError(f"Score API request failed: {exc}") from exc

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AdapterExecutionError(f"Score API returned non-JSON response: {_truncate(content)}") from exc

    if not isinstance(parsed, dict):
        raise AdapterExecutionError("Score API response must be a JSON object")

    return parsed, status


def _apply_mask_to_payload(payload: dict[str, Any], mask_config: dict[str, Any] | None) -> None:
    if not mask_config:
        return

    preset = mask_config.get("preset", "none")
    if preset != "none":
        payload["mask_preset"] = preset

    if mask_config.get("custom_mask") is not None:
        payload["attention_mask"] = mask_config["custom_mask"]

    if mask_config.get("artifact_ref"):
        payload["mask_artifact_ref"] = mask_config["artifact_ref"]

    metadata = mask_config.get("metadata")
    if isinstance(metadata, dict) and metadata:
        payload["mask_metadata"] = metadata


def run_score_api_inference(
    run_id: str,
    backend: str,
    prompt: str,
    score_input: dict[str, Any],
    mask_config: dict[str, Any] | None,
    tolerance: dict[str, Any] | None,
) -> dict[str, Any]:
    url = _score_api_url(backend)

    payload = dict(score_input)
    payload.setdefault("return_logprobs", True)
    _apply_mask_to_payload(payload, mask_config)

    artifacts_dir = Path(settings.local_artifacts_root) / run_id / backend / "score"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    request_path = artifacts_dir / "score.request.json"
    response_path = artifacts_dir / "score.response.json"
    metadata_path = artifacts_dir / "score.metadata.json"

    request_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    start = time.perf_counter()
    response_json, status = _post_json(url, payload)
    duration_ms = (time.perf_counter() - start) * 1000.0

    response_path.write_text(json.dumps(response_json, indent=2, sort_keys=True), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "backend": backend,
                "url": url,
                "status_code": status,
                "duration_ms": round(duration_ms, 3),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    tokens = _extract_tokens(response_json, score_input)
    token_logprobs = _extract_token_logprobs(response_json)

    # Keep token debug payload bounded for UI and storage ergonomics.
    if token_logprobs:
        limit = min(settings.score_debug_max_tokens, len(token_logprobs))
        token_logprobs = token_logprobs[:limit]
        if not tokens:
            tokens = [f"tok_{idx}" for idx in range(limit)]
        else:
            tokens = tokens[:limit]
    else:
        tokens = tokens[: settings.score_debug_max_tokens]

    token_ranks = _extract_token_ranks(response_json, len(tokens))[: len(tokens)]
    if len(token_ranks) < len(tokens):
        token_ranks.extend([0] * (len(tokens) - len(token_ranks)))

    token_nll = [round(-value, 6) for value in token_logprobs]
    score = _derive_score(response_json, token_logprobs)
    item_count = max(1, len(score_input.get("items") or []))
    throughput_items_per_s = item_count / (duration_ms / 1000.0) if duration_ms > 0 else 0.0

    return {
        "score": round(score, 6),
        "latency_ms": round(duration_ms, 3),
        "throughput_items_per_s": round(throughput_items_per_s, 3),
        "token_count": len(tokens),
        "mode": "score",
        "score_source": "real-score-api",
        "adapter_version": "score-api-wrap-v1",
        "backend": backend,
        "tokens": tokens,
        "token_logprobs": [round(value, 6) for value in token_logprobs[: len(tokens)]],
        "token_nll": token_nll[: len(tokens)],
        "token_ranks": token_ranks,
        "mask_metadata": mask_config or {"preset": "none"},
        "tolerance": tolerance or {"abs_epsilon": 1e-6, "rel_epsilon": 0.0},
        "notes": "Score-mode run executed against real /v1/score endpoint",
        "raw_artifacts": {
            "request_path": str(request_path),
            "response_path": str(response_path),
            "metadata_path": str(metadata_path),
        },
    }
