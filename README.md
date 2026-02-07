# SGLang Studio

SGLang Studio is an open-source interactive lab for evaluating and debugging LLM inference behavior across backends.

It is designed for:
- SGLang PyTorch developers
- SGLang JAX developers
- Performance engineers
- Product managers validating latency/quality tradeoffs

## What It Will Do

- Run single-item and multi-item scoring interactively.
- Compare backends side-by-side (SGLang JAX vs SGLang PyTorch).
- Show latency, throughput, and token-level metrics in a modern visual dashboard.
- Execute repeatable benchmark suites and surface regressions.
- Attach traces/profiles to each run for root-cause analysis.

## Product Principles

- Reproducible by default.
- Backend-agnostic core with backend-specific adapters.
- Benchmark-first workflows, not ad-hoc scripts.
- Production-grade telemetry and run history.

## Docs

- Product vision: `docs/000-product-vision.md`
- System architecture: `docs/001-system-architecture.md`
- Benchmark model: `docs/002-benchmark-model.md`
- UI specification: `docs/003-ui-spec.md`
- MVP milestones: `docs/004-mvp-plan.md`
- Open-source strategy: `docs/005-open-source-strategy.md`
- Tech stack decisions: `docs/006-tech-stack.md`
- Studio benchmark integration RFC: `docs/007-bench-integration-rfc.md`
- Testing strategy: `docs/008-testing-strategy.md`
- Inspiration and comparables: `docs/009-inspiration-and-comparables.md`
- Runtime engineering feedback triage: `docs/010-runtime-engineering-feedback-triage.md`

## Status

Milestone 0 scaffold complete (local-first vertical slice).

Milestone 1 kickoff:
- `sglang-jax` runner adapter is wired in wrap-first mode (`STUDIO_SGLANG_JAX_ADAPTER_MODE=auto`).
- `sglang-pytorch` runner adapter is wired in wrap-first mode (`STUDIO_SGLANG_PYTORCH_ADAPTER_MODE=auto`).
- Runner attempts to execute `sglang-jax` benchmark entrypoints and parses latency/throughput from benchmark output.
- Runner attempts to execute `sglang-pytorch` benchmark entrypoints and parses latency/throughput from benchmark output.
- If the local benchmark environment is unavailable, runner falls back to deterministic mock results and records adapter error details in `result_json.adapter_error`.

## Quickstart

```bash
make up
```

Endpoints:
- UI: `http://localhost:5173`
- API: `http://localhost:8000`
- MinIO Console: `http://localhost:9001`

Run tests:

```bash
make test
```

Run modes for benchmark adapters (runner env):
- `auto` (default): try real benchmark wrapper, fallback to mock.
- `bench`: require real benchmark wrapper (run fails on adapter errors).
- `mock`: force deterministic mock results.

Run smoke validation:

```bash
make smoke
```

Run API/DB in compose and run runner on host (for real bench adapters):

```bash
make up-core
make runner-local
```

`runner-local` defaults:
- JAX adapter mode: `bench`
- PyTorch adapter mode: `mock`
- DB/MinIO endpoints: `localhost`

Use `make runner-local-dual-bench` to force `bench` mode for both adapters.

Stop and clean up:

```bash
make down
```

## License

Apache-2.0
