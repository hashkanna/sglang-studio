# Runtime Engineering Feedback Triage

Date: 2026-02-07

## Summary

External feedback confirms strong product fit for:
- Runtime engineers
- Kernel engineers
- Performance engineers

Key positioning takeaway:
- Studio is most differentiated when the inference runtime is the system under test, not a black-box API.

## Relevance to Current Plan

The feedback is aligned with existing docs:
- `docs/001-system-architecture.md` (adapter isolation, runtime-level analysis)
- `docs/004-mvp-plan.md` (comparison and profiling milestones)
- `docs/007-bench-integration-rfc.md` (wrap-first benchmark parity)

It is especially relevant to Milestone 1 and Milestone 2 scope decisions.

## Feature Triage

| Feedback Item | Product Value | Priority | Proposed Milestone | Notes |
|---|---|---|---|---|
| Divergence Inspector (token-step + top-k logprobs) | Fast root cause on JAX vs PyTorch output divergence | P0 | M1 extension (next after adapter stabilization) | Highest leverage for parity debugging; recommended to pull in early |
| KV Cache and memory visualizer (cache hit rate, block usage) | Explains latency regressions from scheduler/cache behavior | P1 | M2/M3 | Requires backend-level cache metrics contract and artifact schema |
| Trace diffing timeline overlay (prefill vs decode segments) | Immediate performance attribution by phase | P1 | M2 | Builds on existing profile/trace linking direction |
| Automated bisect workflow (commit range regression hunter) | Automates manual perf regression localization | P2 | Post-MVP (M4 candidate) | High operational value but higher implementation complexity |

## Recommended Scope Decision

Recommended next addition after current M1 adapter wiring:
- Implement a minimal Divergence Inspector slice.

Minimal slice definition:
- Capture per-token output for both backends.
- Identify first divergence token index.
- Show top-5 token logprobs at divergence step for left and right runs.
- Store raw token-step artifact in object storage and link from run detail.

## Backlog Items to Track

1. Define token-step artifact schema v1 for cross-backend comparison.
2. Add adapter contract fields for optional token-step logprob traces.
3. Add compare API extension for first-divergence summary.
4. Add UI panel in Compare View for token-step inspection.
5. Define trace-segment normalization schema for future timeline overlay.
6. Define cache telemetry schema for KV reuse diagnostics.
7. Draft regression-bisect RFC (inputs, execution model, stopping criteria).

## Scope Guardrails

- Do not turn Studio into a generic prompt engineering product.
- Keep benchmark and runtime introspection as primary product surface.
- Maintain wrap-first parity with existing backend benchmark semantics.
