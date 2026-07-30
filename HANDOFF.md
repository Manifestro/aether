# AETHER — Handoff

**Organization:** Manifestro
**Status:** research core hardened; product API extracted to its own repository

This replaces `handoff-1.md` (removed — it predates the grammar/allowlist/
revision work and the product/research repository split, and no longer
describes the current state). Git history for the old file is preserved
under the `handoff-1.md` path in earlier commits if needed.

---

## 1. What this repository is now

This repository (`Manifestro/aether`) is the **research core only**:
Planner/Speaker/Timeline/scheduler, the semantic event grammar, tool
allowlisting, and the plan-revision protocol. It has no HTTP layer, no
FastAPI, no product code.

The product Text API lives in a separate repository,
[`Manifestro/aether-api`](https://github.com/Manifestro/aether-api),
which depends on this repository as a pinned git package
(`aether @ git+https://github.com/Manifestro/aether.git@v0.1.0`). A new
core release is a deliberate version bump in `aether-api`'s
`pyproject.toml`, not an automatic pull from `main`.

See [`docs/plan.md`](docs/plan.md) §1 for why the split exists, and
[`docs/reports/technical_report_02.md`](docs/reports/technical_report_02.md)
§2.6 for how it was executed.

## 2. What is proven, with evidence

- **Semantic/action lookahead** on a real Qwen3-1.7B (A100): the Speaker
  produces a safe response before an MCP tool result arrives, and the
  factual continuation only after. [`technical_report_01.md`](docs/reports/technical_report_01.md).
- **Constrained event grammar, tool allowlisting, plan revision**,
  validated both on real Qwen3-1.7B runs (A100 and T4) and by two fuzz
  suites (1,000 synthetic turns; 300 synthetic replan turns) with zero
  safety violations. **The one invariant that held in every single run**:
  a factual chunk never speaks before its tool result is confirmed —
  latency and hardware only affect whether the *safe* lead-in also wins
  the race, which is expected, measured behaviour, not a defect.
  [`technical_report_02.md`](docs/reports/technical_report_02.md).
- **A real-model, real-transport API smoke test** (now historical — the
  API code has moved, see §1) confirmed the same safety property holds
  end-to-end over live SSE.

## 3. Structure

```text
aether/
├── README.md
├── HANDOFF.md               (this file)
├── docs/
│   ├── spec.md                original architecture spec
│   ├── plan.md                 living roadmap — read this first for "what's next"
│   ├── invest_pitch.md
│   ├── colab.md                 how to run the Colab notebooks
│   └── reports/
│       ├── technical_report_01.md
│       └── technical_report_02.md
├── notebooks/                  stage1-3 Colab runners (research only; stage4 moved to aether-api)
├── src/aether/
│   ├── domain/                  SemanticEvent, ChunkState, Timeline (+ on_event hook)
│   ├── model/                   event grammar, LLM adapters/backbone/step-engine, scheduler
│   ├── runtime/                 SequentialBaseline, DualSessionRuntime, AllowlistToolExecutor
│   ├── testing/                 deterministic fakes — no network/GPU
│   └── experiments/             colab_stage1/2/3 runners
├── tests/                       dependency-free; see §4
└── pyproject.toml
```

## 4. Verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

No test loads model weights. Real-model experiments run from
`notebooks/aether_stage{1,2,3}_colab.ipynb` on a GPU runtime; see
`docs/colab.md`.

## 5. Invariants for the next agent

1. Never load model weights on the development machine — only in the
   target ML environment (Colab or equivalent), and only via `load()`
   called explicitly.
2. `SequentialBaseline` stays intact — it is the latency baseline every
   dual-stream claim is measured against.
3. A tool-dependent chunk never becomes `COMMITTED` before its tool
   succeeds. `COMMITTED`/`PLAYED` chunks are never rewritten or
   re-cancelled — this is enforced by `ChunkState`'s transition table,
   not by caller discipline; do not bypass it.
4. Any experiment runner that scores real-model output must separate a
   hard, scenario-agnostic safety `checks` dict from a soft,
   hardware/latency-dependent `observations` dict. Two real-model runs in
   this project's history reported `FAILED` because that split was
   missing — see `technical_report_02.md` §3.2/§5.3 before writing a new
   runner's scoring logic.
5. The Planner must always be given an explicit tool allowlist
   (`LLMPlannerAdapter(..., tools=[...])`); never assume the model will
   infer which tools exist. `AllowlistToolExecutor` is the runtime-side
   backstop, not a substitute for the prompt-side allowlist.
6. Do not add product/API/transport code to this repository — it belongs
   in `aether-api`. If a change is needed on both sides of the boundary
   (like the `on_event` hook was), land the core-only part here first,
   tag a release, then bump the tag in `aether-api`.
7. Every latency or lookahead claim must be backed by a timeline/trace,
   not an impression — and must be re-checked whenever the run environment
   (GPU, tool latency) changes; see the crossover finding in
   `technical_report_02.md` §6.

## 6. Next research steps

In priority order (see `technical_report_02.md` §8 for the full
reasoning):

1. Get a real model to actually emit a `replan` event (design a scenario
   that gives it a reason to revise), not just the fuzzed synthetic
   Planner.
2. Real MCP client adapter behind the existing `ToolExecutor` /
   `AllowlistToolExecutor` boundary, replacing `FakeWeatherTool`.
3. LNN / rule-based tempo controller (`docs/plan.md` §6) — not started.
4. Native Voice Head research track (`docs/plan.md` §7) — not started;
   remains parallel to and not blocking the product track.

Product-track next steps (rate limiting, usage logging, key rotation,
Stage 4 scenario matrix, Web Chat) now live in the `aether-api` repository.
