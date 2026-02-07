# Inspiration and Comparable Tools

This document captures external tooling used as inspiration for SGLang Studio product and UX decisions.

## Why This Exists

- Keep product direction grounded in proven workflows.
- Avoid reinventing solved UX patterns.
- Make differentiation explicit.

## Comparable Tools and Borrowed Ideas

1. vLLM
- Reference:
  - https://docs.vllm.ai/en/stable/cli/
  - https://docs.vllm.ai/en/stable/api/vllm/benchmarks/
- Inspiration:
  - CLI-first benchmark ergonomics.
  - Clear throughput/latency benchmark outputs.

2. NVIDIA Triton Tooling
- Reference:
  - https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/model_analyzer/docs/README.html
  - https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html
- Inspiration:
  - Performance profiling and model-level optimization loops.
  - Standardized benchmarking patterns for repeatability.

3. LangSmith
- Reference:
  - https://docs.langchain.com/langsmith/evaluation
  - https://docs.langchain.com/langsmith/run-evaluation-from-prompt-playground
- Inspiration:
  - Run-centric UI with shareable experiment views.
  - Prompt/evaluation loops visible in one place.

4. Arize Phoenix (OSS)
- Reference:
  - https://arize.com/phoenix-oss/
  - https://github.com/Arize-ai/phoenix
- Inspiration:
  - Open-source-first observability posture.
  - Trace-driven root-cause workflows.

5. Promptfoo
- Reference:
  - https://www.promptfoo.dev/docs/usage/web-ui/
  - https://github.com/promptfoo/promptfoo
- Inspiration:
  - Side-by-side comparison UX for fast iteration.
  - Practical eval workflows for engineering teams.

6. Weights & Biases Weave
- Reference:
  - https://docs.wandb.ai/guides/models/evaluate-models/
- Inspiration:
  - Evaluation + tracing integration mindset.
  - History and trend analysis as first-class concepts.

## What Studio Intentionally Adds

- First-class parity/performance comparison between SGLang JAX and SGLang PyTorch.
- Benchmark contracts aligned with existing SGLang development scripts.
- A single product surface for interactive scoring, diffing, benchmarking, and profiling.

## MVP Scope Reminder

- Inspiration does not expand MVP scope.
- MVP backends remain: SGLang JAX and SGLang PyTorch only.
