# VOX-SYNAPSE

Research code for the predictive dual-stream architecture described in
[`spec.md`](spec.md).

The current implementation contains an instrumented sequential baseline and a
concurrent dual-session runtime. It keeps model adapters separate from the
runtime so event ordering, dependency rules and latency traces can be tested
without loading model weights.

## Test the dependency-free core

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Target environment

```bash
uv sync --extra dev
uv run pytest
```

ML and audio dependencies are intentionally optional:

```bash
uv sync --extra dev --extra ml --extra audio
```

The model-ready layer now includes:

- a lazy `SharedQwenBackbone` with downloads disabled by default;
- independent Planner and Speaker session ids over one backend;
- incremental JSONL semantic-event parsing and validation;
- Qwen Planner/Speaker adapters;
- an in-memory shared backend for dependency-free integration tests.

No model is loaded by importing or constructing these adapters. Loading is an
explicit operation reserved for the target ML environment.

