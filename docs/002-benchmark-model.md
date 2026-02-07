# Benchmark Model

## Core Entities

- Suite: named set of cases and default thresholds.
- Case: one input workload example.
- Matrix: parameter expansion (batch size, sequence length, backend flags).
- Run: one execution of suite + matrix on target infra.
- Artifact: trace, profile, log bundle, raw output.

## Metrics

Quality metrics
- score parity delta
- ranking agreement
- tolerance pass/fail

Performance metrics
- end-to-end latency (P50/P95/P99)
- throughput (items/s, tokens/s)
- queueing and compute split

Reliability metrics
- error rate
- timeout rate
- retry count

Cost metrics
- accelerator time
- estimated cost per run

## Comparison Contract

All backend comparisons must hold constant:
- model weights and tokenizer
- input corpus and ordering
- decode/scoring parameters
- hardware class or normalized baseline class

## Gating Policy (MVP)

- Score parity max abs diff <= configured threshold.
- P95 latency regression <= configured threshold.
- No increase in failure rate beyond threshold.
