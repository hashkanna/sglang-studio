.PHONY: up up-core down logs ps test smoke runner-local runner-local-dual-bench lint fmt

up:
	docker compose up --build -d

up-core:
	docker compose up --build -d postgres minio api

ps:
	docker compose ps

logs:
	docker compose logs -f api runner ui

down:
	docker compose down -v

test:
	docker compose run --build --rm api pytest -q /app/api/tests /app/runner/tests

smoke:
	bash scripts/smoke_compose.sh

runner-local:
	bash scripts/run_runner_host.sh

runner-local-dual-bench:
	STUDIO_SGLANG_JAX_ADAPTER_MODE=bench STUDIO_SGLANG_PYTORCH_ADAPTER_MODE=bench bash scripts/run_runner_host.sh
