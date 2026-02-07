.PHONY: up down logs ps test smoke lint fmt

up:
	docker compose up --build -d

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
