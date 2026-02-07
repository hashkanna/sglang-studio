# Score API and Mask Debugging Triage

Date: 2026-02-07

## Focus Clarification

Primary near-term workflow:
- compare Score API behavior between `sglang-jax` and `sglang-pytorch`
- compare Score API behavior under different attention mask configurations

Implication:
- Studio should optimize for scoring/debugging workflows, not chat-generation UX first.

## Assessment of Feedback

Feedback is directionally correct for this scope.

Current gaps for Score/Mask work:
- final aggregate score hides where divergence begins
- mask configuration is hard to iterate without custom scripts
- insufficient token-step observability for debugging kernel/runtime behavior

## Current-State Reality Check

Relative to Score API parity needs, current implementation is still missing critical pieces:

1. Adapters do not yet return real token-level Score API outputs
- current run result focuses on aggregate fields (`score`, `latency_ms`, `throughput_items_per_s`)
- score value is still synthetic/stable-hash based in current wrap adapters

2. Mask is not first-class in request UX/data model
- no dedicated fields for mask preset/custom tensor in current playground flow

3. Compare API/view is aggregate-only
- current compare response emphasizes scalar deltas
- no token-by-token mismatch table, no tolerance-based pass/fail on token deltas

4. Score API input shape is not explicit in UI
- no structured `query + items/targets + mask + score options` form yet

These points make parity debugging possible only in a limited way; day-to-day Score/Mask investigation still requires external scripts.

## Priority Features for Score/Mask Workflow

### P0 (M1 extension)

1. Scoring-mode UX in Playground
- Explicit score workflow input (`query` + `items` / completion segments).
- Disable generation-only controls when score mode is active.

2. Per-token loss diff chart
- Plot token position vs NLL/loss for left and right runs.
- Highlight first divergence region.

3. Token-level debug payload in API
- Return token logprobs and token ranks per position.
- Include first-divergence summary in compare output.

4. Tolerance-based token parity gate
- configurable epsilon (example: `1e-6`) for token score comparison
- run-level status: PASS when all compared token deltas are within tolerance

### P1 (M2 candidate)

1. Mask configuration playground
- Presets: causal, bidirectional-prefix, doc-isolation.
- Upload custom mask artifact (JSON/NPY) for advanced cases.

2. Effective mask introspection
- Export and visualize effective mask tensor (or equivalent debug representation).
- Start with short-sequence safety limits for UI rendering.

3. Rank stability metrics
- Add per-run and compare metrics based on token rank changes (including MRR delta).

## Proposed MVP Adjustment

Adjust M1 priority order for this use case:
1. Token-level compare data contract (logprobs/ranks).
2. Per-token loss diff chart in Compare View.
3. Score-mode playground UX.
4. Gate badges over score/mask-specific thresholds.

Minimum useful milestone slice (strict):
1. Structured Score API form (query/targets/mask/tolerance).
2. Real Score API adapter execution returning token-level outputs.
3. Token diff table with mismatch highlighting and PASS/FAIL verdict.
4. Save/load parity test cases for repeat runs.

Implementation checklist:
- `docs/013-score-api-m1a-implementation-checklist.md`

## Data Contract Additions (Draft)

Run result additions:
- `token_logprobs`: per-position map of candidate token scores.
- `token_ranks`: rank of ground-truth token per position.
- `token_nll`: per-position negative log likelihood.
- `mask_metadata`: mask preset/id/hash and shape summary.
- `debug_artifacts`: optional links to mask tensor export or attention debug artifact.

Compare result additions:
- `first_divergence_index`
- `token_loss_diff_summary` (max/mean over aligned positions)
- `rank_delta_summary` (MRR delta, worst rank drop)

## Risks and Guardrails

Risks:
- payload size explosion for long sequences
- backend asymmetry in debug data availability

Guardrails:
- cap default token debug length and allow opt-in expansion
- version token-debug schema explicitly
- degrade gracefully when one backend cannot emit optional debug tensors
