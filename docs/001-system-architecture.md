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
- Scales workers on GKE/VM pools
- Persists intermediate and final metrics

4. Backend Adapters
- SGLang JAX adapter
- SGLang PyTorch adapter
- Future adapters (vLLM, TensorRT-LLM)

5. Telemetry Pipeline
- Request timing and token stats
- Traces/profiles ingestion
- Aggregations for dashboards

6. Storage
- Postgres for metadata
- Object storage for artifacts (profiles, logs, outputs)
- Time-series store for metrics

## Deployment Topology (GCP)

- UI + API on GKE
- Runner on GKE with autoscaling node pools
- Backend endpoints on dedicated TPU/GPU instances
- Cloud SQL for metadata
- GCS for artifacts

## Design Rules

- All experiments are immutable once launched.
- Backend adapters expose a shared interface.
- Every result record includes backend commit, model revision, and config hash.
- Any comparison view requires matching input corpus and evaluation settings.
