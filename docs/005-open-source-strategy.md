# Open Source Strategy

## Why Public

- Attract external contributors for backend adapters and benchmark suites.
- Increase transparency and reproducibility for performance claims.
- Encourage ecosystem integration with other runtimes after MVP stabilizes.

## Governance (Initial)

- Maintainers: core SGLang + SGLang JAX contributors.
- PR policy: tests required for adapter or metric changes.
- Release cadence: monthly minor releases.

## Contribution Model

- `good-first-issue` for UI and adapter enhancements.
- RFC process for benchmark schema and comparison contracts.
- Public roadmap tracked in GitHub Projects.

## Contributor Quickstart Expectations

- Contributors can run a full local stack using `docker-compose`.
- First contribution does not require GKE, TPU, or GPU access.
- Mock adapter path is always available for UI/API feature work.
- Hardware-backed validations are optional for most PRs and enabled by maintainers when needed.

## Test and CI Expectations

Required for standard PRs:
- unit tests
- contract tests for adapter interface (can use mock adapters)
- lint/format checks

Optional or maintainer-gated:
- TPU/GPU performance validation
- large benchmark campaign reruns

## Security and Privacy

- No input data retained by default unless run persistence is enabled.
- Explicit redaction hooks for logs and artifacts.
