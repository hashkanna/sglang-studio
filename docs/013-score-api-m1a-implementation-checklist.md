# Score API M1a Implementation Checklist

Date: 2026-02-07

## Goal

Ship the minimum feature slice that makes Score API parity debugging between:
- `sglang-jax` vs `sglang-pytorch`
- mask A vs mask B

practically useful day-to-day.

## Definition of Done (M1a)

1. Runs use real Score API responses (no synthetic score for parity mode).
2. Request model is structured for Score API (`query`, `items/targets`, mask config, tolerance).
3. Compare view shows token-level diffs and PASS/FAIL verdict.
4. Gate evaluation works on token deltas and core aggregate thresholds.
5. Saved parity test cases can be re-run without retyping inputs.

## Task Order

### Phase 1: Data Contract and Persistence (API-first)

1. Add score-mode run schema in API:
- `mode`: `score` | `benchmark`
- `score_input`: `query`, `items`, `label_token_ids`, `apply_softmax`, `item_first`
- `mask_config`: preset/custom reference
- `tolerance`: `{abs_epsilon, rel_epsilon}`

2. Extend persisted run metadata:
- backend commit SHA, model revision, tokenizer revision, config hash
- score input hash and mask hash/shape summary

3. Add token-level result schema:
- `token_logprobs`
- `token_nll`
- `token_ranks`
- `tokens`
- `raw_response_ref`

### Phase 2: Runner Adapter Execution (Real Score API)

1. Add real score execution path in JAX and PyTorch adapters.
2. Pass through structured score/mask inputs.
3. Capture and store full raw response artifact.
4. Keep `auto|bench|mock` fallback behavior only for benchmark mode; score parity mode should default to real execution and fail loudly when unavailable.

### Phase 3: Compare API and Gate Engine

1. Extend compare endpoint output with:
- `first_divergence_index`
- `token_mismatch_count`
- `max_token_abs_diff`
- `token_parity_pass`
- `rank_delta_summary`

2. Implement token comparator:
- aligned token index comparison
- abs/rel epsilon checks
- per-token status and run-level verdict

3. Implement gate badges:
- token parity gate
- score parity gate
- p95 latency regression gate
- failure-rate gate

### Phase 4: UI Workflow (Playground + Compare)

1. Add score-mode form in Playground:
- query
- items/targets editor
- mask preset selector + custom input hook
- tolerance controls

2. Compare page:
- side-by-side token table
- per-token logprob/NLL/rank diffs
- mismatch highlighting
- PASS/FAIL summary badges

3. Run detail page:
- raw request/response links
- mask metadata and reproducibility metadata

### Phase 5: Test Case CRUD (Minimum)

1. Add saved parity test case model:
- name, score input payload, mask config, tolerance, tags

2. Add API endpoints:
- create/list/update/delete test case
- run test case

3. Add UI:
- save current score form as test case
- load and rerun a saved case

## Test Plan (Required Before Merge)

1. Unit tests:
- token comparator (abs/rel tolerance, rank delta, divergence index)
- gate evaluator (pass/fail logic)

2. Adapter contract tests:
- real score response normalization for both backends
- mask passthrough correctness

3. Integration tests:
- UI -> API -> runner -> adapter with score mode
- compare response includes token-level fields
- saved test case rerun path works

4. Replay fixture tests:
- fixed input fixtures with expected token deltas and verdict

## Out of Scope for M1a

- effective attention heatmap viewer
- trace timeline diff viewer
- automated git bisect workflow
- trend charts and collaboration annotations

## Execution Notes

- Implement in small PRs, in phase order.
- Keep backward compatibility for existing benchmark mode endpoints.
- Version new score token-debug schema explicitly (`score_debug_schema_version`).
