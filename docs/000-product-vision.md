# Product Vision

## Problem

LLM inference teams currently rely on disconnected scripts and logs for validation. This makes it hard to:
- compare backends under identical inputs
- isolate performance regressions quickly
- communicate quality/perf tradeoffs with non-implementation stakeholders

## Vision

Provide a single interactive studio where users can run, compare, profile, and benchmark inference workloads across SGLang runtimes.

## Primary Users

- Runtime engineers: inspect kernel and attention-path impact.
- Model engineers: validate scoring semantics and parity.
- Performance engineers: track latency/throughput over time.
- PMs and leads: view clear readiness dashboards.

## Core Workflows

1. Interactive playground
   - Enter prompt/question.
   - Choose model, backend, and scoring mode.
   - Execute and inspect outputs plus timing breakdown.

2. Side-by-side comparison
   - Run identical input against two backends.
   - Show score delta, latency delta, throughput delta, and confidence thresholds.

3. Benchmark campaign
   - Select benchmark suite and parameter matrix.
   - Launch distributed run.
   - View run history and regression summary.

4. Profiling drill-down
   - Open run details.
   - Inspect trace segments and operator-level hotspots.

## Success Criteria

- A new feature can be validated against baseline in under 30 minutes.
- Regressions in P95 latency or score parity are visible from one dashboard.
- Benchmark results are reproducible from stored config + artifacts.
