# MVP Plan

## Planning Inputs

- Tech stack decision: `docs/006-tech-stack.md`
- Benchmark integration decision: `docs/007-bench-integration-rfc.md`
- Testing strategy: `docs/008-testing-strategy.md`

## Milestone 0: Foundation

Deliverables
- Repo scaffolding and coding standards
- Studio API skeleton
- Adapter interface and mock adapter
- Minimal UI shell and routing
- One-command local startup (`make up`) using `docker-compose`
- Seeded sample data and one mock benchmark suite
- Example comparison run persisted in metadata store

Exit Criteria
- Local end-to-end mock run visible in UI.
- A new contributor can boot the stack and run a mock comparison in under 15 minutes.
- Local development requires no GKE setup.

## Milestone 1: Interactive Scoring

Deliverables
- SGLang JAX adapter wired
- SGLang PyTorch adapter wired
- Playground page + results page
- Run metadata persistence

Exit Criteria
- Same prompt executed on both backends from UI.
- Score and latency visible with stored run record.

## Milestone 2: Comparison + Profiles

Deliverables
- Compare page with diff calculations
- Profile artifact linking and timeline view
- Threshold-based pass/fail indicators

Exit Criteria
- Users can diagnose parity and latency regressions in one screen.

## Milestone 3: Benchmark Campaigns

Deliverables
- Benchmark suite registry
- Matrix campaign launcher
- Aggregated dashboard with trend lines

Exit Criteria
- Teams can run repeatable benchmark suites and track regressions over time.
