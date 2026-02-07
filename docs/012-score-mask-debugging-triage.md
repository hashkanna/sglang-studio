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
