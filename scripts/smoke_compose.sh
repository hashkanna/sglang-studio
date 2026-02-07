#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"

wait_for_api() {
  local retries=60
  local i=0
  until curl -sf "${API_URL}/health" >/dev/null; do
    i=$((i + 1))
    if [[ "$i" -ge "$retries" ]]; then
      echo "API did not become healthy in time" >&2
      exit 1
    fi
    sleep 1
  done
}

create_run() {
  local backend="$1"
  local prompt="$2"
  local payload
  payload=$(cat <<JSON
{"backend":"${backend}","prompt":"${prompt}","parameters":{"multi_item_count":4}}
JSON
)
  curl -sS -X POST "${API_URL}/api/v1/runs" \
    -H "Content-Type: application/json" \
    -d "${payload}"
}

extract_json_field() {
  local json="$1"
  local field="$2"
  python3 -c "import json,sys; print(json.loads(sys.argv[1]).get(sys.argv[2], ''))" "$json" "$field"
}

wait_for_status() {
  local run_id="$1"
  local retries=60
  local i=0

  while true; do
    local resp
    local status
    resp=$(curl -sS "${API_URL}/api/v1/runs/${run_id}")
    status=$(extract_json_field "$resp" "status")

    if [[ "$status" == "succeeded" ]]; then
      break
    fi
    if [[ "$status" == "failed" ]]; then
      echo "Run ${run_id} failed" >&2
      echo "$resp" >&2
      exit 1
    fi

    i=$((i + 1))
    if [[ "$i" -ge "$retries" ]]; then
      echo "Run ${run_id} did not finish in time" >&2
      exit 1
    fi
    sleep 1
  done
}

echo "[smoke] starting compose services"
docker compose up --build -d postgres minio api runner

echo "[smoke] waiting for api"
wait_for_api

run_a=$(create_run "sglang-jax" "Score this answer: Paris is the capital of France.")
run_a_id=$(extract_json_field "$run_a" "id")

run_b=$(create_run "sglang-pytorch" "Score this answer: Paris is the capital of France.")
run_b_id=$(extract_json_field "$run_b" "id")

if [[ -z "$run_a_id" || -z "$run_b_id" ]]; then
  echo "Failed to create runs" >&2
  echo "$run_a" >&2
  echo "$run_b" >&2
  exit 1
fi

echo "[smoke] created runs: $run_a_id $run_b_id"
wait_for_status "$run_a_id"
wait_for_status "$run_b_id"

compare_payload=$(cat <<JSON
{"left_run_id":"${run_a_id}","right_run_id":"${run_b_id}"}
JSON
)

compare_resp=$(curl -sS -X POST "${API_URL}/api/v1/compares" \
  -H "Content-Type: application/json" \
  -d "$compare_payload")

score_diff=$(extract_json_field "$compare_resp" "score_abs_diff")

if [[ -z "$score_diff" ]]; then
  echo "Comparison response missing score_abs_diff" >&2
  echo "$compare_resp" >&2
  exit 1
fi

echo "[smoke] compare response: $compare_resp"
echo "[smoke] success"
