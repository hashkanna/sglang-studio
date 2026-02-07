# UI Specification (MVP)

## Pages

1. Playground
- Prompt input (single query)
- Optional multi-item input builder
- Backend selector (JAX, PyTorch)
- Model selector
- Parameter controls (temperature, top-p, max tokens, scoring mode)
- Execute button and live status

2. Results View
- Output panel
- Score table
- Latency breakdown (stacked bars)
- Tokens/s and items/s cards
- Trace/profile links

3. Compare View
- Side-by-side output + score diff
- Delta cards (latency, throughput, parity)
- Threshold pass/fail badges

4. Benchmarks
- Suite selector
- Matrix controls
- Launch and monitor campaign
- Results table with filters and sparkline trends

5. Run Detail
- Metadata (commit, model, hardware)
- Timeline view
- Artifact list and download links

## UX Goals

- One-click repeat run from any previous run.
- Shareable run URL for collaboration.
- Clear red/yellow/green status for readiness.
