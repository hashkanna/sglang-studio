#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="${ROOT_DIR}/runner/src:${PYTHONPATH:-}"

export STUDIO_DB_DSN="${STUDIO_DB_DSN:-postgresql+psycopg://studio:studio@localhost:5432/studio}"
export STUDIO_MINIO_ENDPOINT="${STUDIO_MINIO_ENDPOINT:-localhost:9000}"
export STUDIO_MINIO_ACCESS_KEY="${STUDIO_MINIO_ACCESS_KEY:-minio}"
export STUDIO_MINIO_SECRET_KEY="${STUDIO_MINIO_SECRET_KEY:-minio123}"
export STUDIO_MINIO_BUCKET="${STUDIO_MINIO_BUCKET:-studio-artifacts}"
export STUDIO_MINIO_SECURE="${STUDIO_MINIO_SECURE:-false}"
export STUDIO_POLL_INTERVAL_SECONDS="${STUDIO_POLL_INTERVAL_SECONDS:-0.5}"

export STUDIO_SGLANG_JAX_ADAPTER_MODE="${STUDIO_SGLANG_JAX_ADAPTER_MODE:-bench}"
export STUDIO_SGLANG_JAX_ROOT="${STUDIO_SGLANG_JAX_ROOT:-${SGLANG_JAX_ROOT:-${HOME}/Sandbox/sglang-all/sglang-jax}}"
if [[ -z "${STUDIO_SGLANG_JAX_BENCH_COMMAND:-}" ]] && command -v uv >/dev/null 2>&1; then
  export STUDIO_SGLANG_JAX_BENCH_COMMAND="uv run python3 -m unittest test.srt.test_bench_score"
fi

export STUDIO_SGLANG_PYTORCH_ADAPTER_MODE="${STUDIO_SGLANG_PYTORCH_ADAPTER_MODE:-mock}"
export STUDIO_SGLANG_PYTORCH_ROOT="${STUDIO_SGLANG_PYTORCH_ROOT:-${SGLANG_PYTORCH_ROOT:-${HOME}/Sandbox/sglang-all/sglang}}"
if [[ -z "${STUDIO_SGLANG_PYTORCH_BENCH_COMMAND:-}" ]] && command -v uv >/dev/null 2>&1; then
  export STUDIO_SGLANG_PYTORCH_BENCH_COMMAND="uv run python3 benchmark/prefill_only/bench_score.py"
fi

echo "[runner-host] db=${STUDIO_DB_DSN}"
echo "[runner-host] minio=${STUDIO_MINIO_ENDPOINT}"
echo "[runner-host] jax_mode=${STUDIO_SGLANG_JAX_ADAPTER_MODE} root=${STUDIO_SGLANG_JAX_ROOT}"
echo "[runner-host] pytorch_mode=${STUDIO_SGLANG_PYTORCH_ADAPTER_MODE} root=${STUDIO_SGLANG_PYTORCH_ROOT}"

if command -v uv >/dev/null 2>&1; then
  exec uv run \
    --python 3.11 \
    --with-requirements "${ROOT_DIR}/runner/requirements.txt" \
    python3 -m studio_runner.main
fi

exec python3 -m studio_runner.main
