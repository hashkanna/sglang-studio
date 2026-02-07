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

## Status

Milestone 0 scaffold in progress (local-first vertical slice).

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

Run smoke validation:

```bash
make smoke
```

Stop and clean up:

```bash
make down
```

## License

Apache-2.0
