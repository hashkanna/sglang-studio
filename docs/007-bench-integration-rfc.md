# RFC: Studio and Existing Benchmark Script Integration

## Status

Accepted for MVP.

## Problem

Studio needs benchmark execution while existing engineering workflows already use:

- `sglang-jax/test/srt/bench_score.py`

If Studio diverges too early, parity and trust will degrade.

## Options Considered

- Option A: Replace now. Reimplement benchmark logic directly in Studio. Risk is semantic drift and duplicated maintenance.
- Option B: Wrap existing scripts. Studio invokes existing script entry points through adapters. Benefit is fastest path to trustworthy parity.
- Option C: Extend existing scripts in place. Modify benchmark scripts immediately for Studio-specific output. Risk is coupling and churn before core product stabilizes.

## Decision

Use **wrap-first** for MVP, then selectively extend.

- Milestone 0-1: Studio wraps existing benchmark entry points.
- Milestone 2+: Extend shared benchmarking utilities only when needed and backward-compatible.
- No script replacement during MVP.

## Integration Contract

Studio runner invokes adapter-specific benchmark jobs with:

- immutable run id
- backend id (`sglang-jax` or `sglang-pytorch`)
- model id + tokenizer id
- input suite reference
- parameter matrix
- output artifact directory

Adapter returns:

- run summary metrics
- case-level metrics
- paths to raw artifacts (logs, traces, raw outputs)
- exit status + error detail (if failed)

## Compatibility Rules

- Existing script semantics remain source of truth during MVP.
- Any Studio-side normalization must be explicit and versioned.
- Comparison views must show both raw backend metrics and normalized metrics when they differ.

## Follow-Up

- Add equivalent stable benchmark entry point for PyTorch backend if missing.
- Publish a small adapter conformance test suite that validates output schema and status codes.
