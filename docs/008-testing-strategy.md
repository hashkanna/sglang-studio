# Testing Strategy

## Objectives

- Verify Studio correctness independently from backend model quality.
- Ensure benchmark runs are reproducible and auditable.
- Detect regressions in parity and performance gate logic.

## Test Layers

1. Unit Tests
- Metric calculations (delta, percent regression, gate evaluation)
- Run state machine transitions
- Schema validation and serialization

2. Adapter Contract Tests
- Each adapter must satisfy request/response schema contract.
- Must return deterministic error codes and metadata fields.
- Mock adapters run in CI for fast feedback.

3. Integration Tests (Local Compose)
- UI -> API -> runner -> adapter mock -> storage path.
- Validate persisted runs, artifacts, and compare output rendering.

4. Replay Tests
- Fixed fixtures with pinned expected outputs.
- Re-run comparison logic and verify no unexpected drift in computed summaries.

5. Regression Gate Tests
- Validate threshold logic for score parity max abs diff.
- Validate threshold logic for p95 latency regression.
- Validate threshold logic for failure-rate regression.

## Determinism Requirements

- Each run stores config hash, adapter version, and input fixture id.
- Replay tests use fixed seeds and fixed input ordering.
- Any normalized metric transformation is versioned.

## MVP CI Policy

Required on every PR:

- unit tests
- adapter contract tests (mock mode)
- integration smoke test (local compose profile)

Optional/maintainer-triggered:

- real JAX backend tests
- real PyTorch backend tests
- TPU/GPU benchmark campaigns

## Exit Criteria for M1

- Single-item run passes integration and replay tests.
- Multi-item run passes integration and replay tests.
- Compare view gate evaluation tested against golden fixtures.
