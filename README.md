# AETHER

Research core for the predictive dual-stream voice-agent architecture:
a Planner and a Speaker share one open-weight LLM backbone through
independent decode sessions, so the agent can act (call a tool) and speak
a safe response before the tool resolves, then continue with a factual
response only after it resolves — never before.

This repository is the **research core only** (no HTTP layer, no product
code). The product Text API is a separate repository,
[`Manifestro/aether-api`](https://github.com/Manifestro/aether-api),
which depends on this repository as a pinned package. See
[`HANDOFF.md`](HANDOFF.md) for the current state and what to work on next.

## Documentation

| | |
|---|---|
| [`HANDOFF.md`](HANDOFF.md) | Current state, invariants, next steps — **start here** |
| [`docs/plan.md`](docs/plan.md) | Living roadmap: research + product tracks, decision gates |
| [`docs/spec.md`](docs/spec.md) | Original architecture spec |
| [`docs/reports/technical_report_01.md`](docs/reports/technical_report_01.md) | First real-model proof: semantic/action lookahead on Qwen3-1.7B |
| [`docs/reports/technical_report_02.md`](docs/reports/technical_report_02.md) | Constrained grammar, tool allowlisting, plan revision, product API split |
| [`docs/invest_pitch.md`](docs/invest_pitch.md) | Pitch deck content |
| [`docs/colab.md`](docs/colab.md) | Running the real-model Colab notebooks |

## Test the dependency-free core

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

No test in this repository loads model weights, touches the network, or
requires a GPU.

## Target environment

Python 3.12 is recommended for the research environment. The package
remains installable on Python 3.10+ so hosted GPU notebook runtimes can
execute the smoke tests without replacing their kernel interpreter.

```bash
uv sync --extra dev
uv run pytest
```

ML and audio dependencies are intentionally optional:

```bash
uv sync --extra dev --extra ml --extra audio
```

## What's in the model-ready layer

- a lazy `SharedLLMBackbone` with downloads disabled by default;
- independent Planner and Speaker session ids over one backend, plus an
  `InterleavedDecodeScheduler` for token-step-level overlap;
- a constrained JSONL semantic-event grammar (`strict` mode) with an
  explicit `safe_to_say` field and tool allowlisting;
- a plan-revision protocol (`replan` events cancel only not-yet-committed
  speech; committed/played speech is never rewritten);
- LLM Planner/Speaker adapters, named generically (`llm_adapters.py`) —
  currently backed by Qwen3, but Qwen is a Planner-candidate, not an
  architectural commitment (`docs/plan.md` §3);
- deterministic fakes (`aether.testing`) for dependency-free integration
  tests, including 1,000+ synthetic-turn fuzz coverage.

No model is loaded by importing or constructing these adapters. Loading
is an explicit operation reserved for the target ML environment (see
`docs/colab.md`).
