# System Architecture

## High-Level Components

1. Web UI
- Interactive playground
- Comparison dashboard
- Benchmark campaign builder
- Run history and artifact viewer

2. Studio API
- Auth/session management
- Experiment orchestration
- Run metadata + status
- Artifact index and retrieval

3. Runner Service
- Schedules and executes benchmark jobs
- Runs locally in `docker-compose` for MVP; scales on GKE/VM pools in cloud mode
- Persists intermediate and final metrics

4. Backend Adapters
- SGLang JAX adapter
- SGLang PyTorch adapter
- Additional adapters are explicitly non-MVP

5. Telemetry Pipeline
- Request timing and token stats
- Traces/profiles ingestion
- Aggregations for dashboards

6. Storage
- Postgres for metadata
- Object storage for artifacts (profiles, logs, outputs)
- Time-series store for metrics

## Deployment Modes

### Local Development (Default)

Use a local stack first so contributors can work without cloud infrastructure.

Services in `docker-compose`:
- `studio-ui` (frontend)
- `studio-api` (control plane)
- `postgres` (metadata)
- `minio` (artifacts)
- optional `studio-runner` (local benchmark execution)
- Optional external adapters for JAX/PyTorch; when unavailable, use mock adapter.

### Cloud Deployment (GCP)

Used after local workflows are stable and for larger benchmark campaigns.

- UI + API on GKE
- Runner on GKE with autoscaling node pools
- Backend endpoints on dedicated TPU/GPU instances
- Cloud SQL for metadata
- GCS for artifacts

## Adapter Execution Model

Adapters are isolated from the API process to avoid framework dependency conflicts.

- SGLang JAX adapter runs in its own process/container.
- SGLang PyTorch adapter runs in its own process/container.
- API communicates through a stable network contract (HTTP or gRPC).
- API does not import backend frameworks directly.
- Each adapter image/environment is versioned independently.

## Design Rules

- All experiments are immutable once launched.
- Backend adapters expose a shared interface.
- Every result record includes backend commit, model revision, and config hash.
- Any comparison view requires matching input corpus and evaluation settings.
- Local development path must work without GKE/TPU/GPU dependencies.

## MVP Scope Guardrails

- MVP backend scope is only SGLang JAX and SGLang PyTorch.
- Non-MVP adapters (vLLM, TensorRT-LLM, others) are not part of initial delivery.
- Cloud deployment is not required to complete Milestone 0 or Milestone 1.
