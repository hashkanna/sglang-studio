# Tech Stack Decisions (MVP)

## Decision

MVP stack is:

- UI: React + Vite + TypeScript
- API: FastAPI + Pydantic
- Metadata DB: Postgres
- Artifact Store: MinIO (S3-compatible)
- Local orchestration: Docker Compose
- Benchmark runner: Python worker service (separate container/process)

## Why This Stack

- Fast local startup and low contributor friction.
- Python-first backend aligns with existing SGLang/JAX validation tooling.
- Strong type safety across UI and API with generated schemas.
- Easy migration from local (`docker-compose`) to cloud-managed services.

## Non-Goals for MVP

- No Kubernetes dependency for day-1 contributor workflows.
- No hard requirement for TPU/GPU access.
- No additional adapter frameworks beyond SGLang JAX and SGLang PyTorch.

## API Contract Shape

- REST for control plane endpoints (create run, list runs, compare runs).
- Structured JSON payloads with explicit schema versions.
- Adapter calls go through a backend-agnostic adapter interface.

## Revisit Criteria

Re-evaluate stack after Milestone 1 if any of these occur:

- API p95 overhead becomes material relative to backend inference latency.
- Worker throughput bottlenecks cannot be solved via horizontal scaling.
- Contributor setup time exceeds the 15-minute target.
