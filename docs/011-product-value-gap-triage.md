# Product Value Gap Triage (Post-M0/M1 Adapter Wiring)

Date: 2026-02-07

## Problem Statement

Current Studio value can still be perceived as:
- a UI wrapper around existing benchmark scripts

Reason:
- current experience emphasizes aggregate metrics and run listing, but does not yet answer the engineering questions:
  - what exactly differed in outputs?
  - which threshold failed?
  - which code/version produced the run?
  - is this getting better or worse over time?

## Assessment

Feedback is valid and relevant.

Current capabilities are useful as infrastructure scaffolding, but insufficient for daily engineering workflow adoption by runtime/performance teams.

## Prioritized Feature Triage

### P0 (Must-Have for "daily-useful" threshold)

1. Compare view shows actual backend outputs side-by-side with textual diff.
2. Threshold gates are enforced and visible as pass/fail badges (score parity + latency regression + failure rate).
3. Token-level visibility starts with divergence-focused logprob comparison (aligned with `docs/010-runtime-engineering-feedback-triage.md`).
4. Run metadata includes code version identity (commit SHA and optional branch/PR label).

### P1 (Workflow upgrades)

1. Saved prompts and reusable test suites.
2. Baseline run selection and automatic comparison against baseline.
3. Run tagging and filtering (e.g., `nightly`, `pr-review`, `experiment-*`).

### P2 (Observability and collaboration depth)

1. Embedded trace/profile viewer with timeline breakdown (not only artifact links).
2. Trend charts and regression detection over run history.
3. Shareable deep links for run and compare pages.
4. Notes/annotations and export (markdown/PDF summary).

## Suggested Implementation Sequence

### M1a (close immediate value gap)

- Output diff panel in Compare View.
- Gate evaluation in API + UI pass/fail badges.
- Commit SHA in run metadata and filters.

### M1b (bridge to repeatable workflows)

- Prompt suite model + batch launch endpoint.
- Baseline designation + auto-compare behavior.

### M2+

- Timeline trace diff view.
- Trend analytics and collaboration features.

## Scope Decision

Before expanding breadth, prioritize explaining "what happened and why" for each compare operation.

That means:
- output-level evidence
- token-level evidence at divergence
- explicit gate status
- reproducibility metadata (commit/version)
