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
- **Phase C, Stage 4 — audio commit horizon (PASSED, real Mimi weights).**
  A single Mimi codebook, predicted by an untrained, from-scratch
  `MinimalVoiceHead` conditioned on Speaker text, was wired into
  `DualSessionRuntime` at the same commit point as text. Result: the
  existing `ChunkState` transition table governs audio with zero new
  states — a dependent chunk's audio is never buffered before its tool
  result, and `replan` cancels in-flight audio synthesis exactly like
  buffered text. `docs/colab.md` §Stage 4; `aether/experiments/colab_stage4.py`.
- **Phase C, Stage 5 — hidden-state-conditioned Voice Head (in progress).**
  `HiddenStateVoiceHead` predicts Mimi codebook-0 tokens from a Qwen3
  hidden state (not text — see `aether/model/llm_backbone.py`'s
  `encode_hidden_state`), trained against a real teacher,
  `kyutai/tts-1.6b-en_fr` (confirmed via `moshi/run_tts.py` source, not
  guessed — see the spike notebook). Latest real run: train loss
  7.86 → 0.009 (expected overfit on ~20 examples), held-out token
  agreement 0.02–0.40 (well above the ~0.05% chance rate, but not
  statistically meaningful on 4 examples). **Open/unresolved as of this
  writing:** the first decode methodology (judging audio quality from
  codebook 0 alone) was a mistake — Mimi splits semantic content
  (codebook 0) from acoustic detail (codebooks 1-31), so single-codebook
  playback is near-noise even for the real teacher. The runner has just
  been rewritten to decode the full 32-codebook teacher voice
  (`teacher_voice_full`) and a hybrid with the trained head's codebook-0
  swapped in (`hybrid_our_codebook0`) — **a fresh Colab run with this
  fixed decode has not yet been reviewed.** That review is the next
  concrete step. `aether/experiments/colab_stage5.py`;
  `aether/training/{datasets,trainer}.py`.

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
├── notebooks/                  stage1-5 + Moshi-teacher spike Colab runners
├── src/aether/
│   ├── domain/                  SemanticEvent, ChunkState, Timeline (+ on_event hook), AudioChunk
│   ├── model/                   event grammar, LLM adapters/backbone/step-engine, scheduler,
│   │                            voice_head.py (MinimalVoiceHead, HiddenStateVoiceHead)
│   ├── runtime/                 SequentialBaseline, DualSessionRuntime (+ optional voice_head),
│   │                            AllowlistToolExecutor
│   ├── audio/                   codec.py — lazy Mimi codec wrapper
│   ├── training/                datasets.py (Stage 5 phrase set), trainer.py (full-batch loop)
│   ├── testing/                 deterministic fakes — no network/GPU
│   └── experiments/             colab_stage1-5 runners, spike_moshi_teacher.py (API discovery)
├── tests/                       dependency-free; see §4
└── pyproject.toml
```

Product Stage 4 (the aether-api one, HTTP) lives in `aether-api` — do not
confuse it with this repository's Stage 4 (audio commit-horizon probe,
`aether/experiments/colab_stage4.py`), which is unrelated and stays here.

## 4. Verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

No test loads model weights. Real-model experiments run from
`notebooks/aether_stage{1,2,3,4,5}_colab.ipynb` and
`notebooks/aether_spike_moshi_teacher_colab.ipynb` on a GPU runtime; see
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
8. **Never judge Voice Head audio by decoding a single Mimi codebook.**
   Mimi splits semantic content (codebook 0, what `MinimalVoiceHead`/
   `HiddenStateVoiceHead` predict) from acoustic detail (codebooks 1-31);
   single-codebook playback is near-noise even for a real, fully-trained
   teacher. Any listening comparison must either decode all `n_q`
   codebooks (real teacher reference) or a hybrid (teacher's codebooks
   1-31 + our predicted codebook 0) — see `colab_stage5.py`'s decode step
   and `docs/colab.md`'s "Ограничения" for the exact mistake this
   invariant comes from.
9. When wiring a real external ML package whose call sequence isn't
   already used elsewhere in this repo (e.g. `moshi`'s `TTSModel`), do not
   guess the API from class signatures alone. Introspect the installed
   package (`inspect.getsource`, `dir()`) and read a real reference script
   if one ships with the package — `aether/experiments/spike_moshi_teacher.py`
   is the pattern: cheap discovery runs before committing to a training
   pipeline built on an assumption. It found the real entry point
   (`moshi.models.tts.TTSModel` + `moshi/run_tts.py`) in three iterations
   instead of one expensive wrong guess.

## 6. Next research steps

In priority order:

1. **Review the fixed Stage 5 decode.** `colab_stage5.py` now writes
   `teacher_voice_full` (real 32-codebook Kyutai voice) and
   `hybrid_our_codebook0` (same, with the trained head's codebook-0
   prediction swapped in) per train phrase. A Colab run with this fix has
   not been reviewed yet — that listening comparison is the actual
   Continue/Refine/Fallback signal for whether hidden-state conditioning
   learned anything about content (§20 `spec.md`), not the loss curve
   alone.
2. **Kazakh feasibility, before investing in a large recorded corpus.**
   There is user interest in a real (self-recorded) Kazakh speech corpus
   (hundreds to ~2000 phrases) to train a Kazakh Voice Head. Cheapest
   next check: run `SharedLLMBackbone.encode_hidden_state` on a handful of
   Kazakh phrases on the *actual* backbone in use (Qwen3-1.7B, not
   Qwen3.5-Flash or any other model not yet wired into this repo) and
   sanity-check the hidden states look meaningful before committing to
   weeks of recording. If a self-recorded corpus proceeds, this repo's
   `training/datasets.py` pattern generalizes directly (phrase list +
   paired hidden state + paired real Mimi-encoded audio, no synthetic
   teacher needed since the recordings are the ground truth).
3. Get a real model to actually emit a `replan` event (design a scenario
   that gives it a reason to revise), not just the fuzzed synthetic
   Planner.
4. Real MCP client adapter behind the existing `ToolExecutor` /
   `AllowlistToolExecutor` boundary, replacing `FakeWeatherTool`.
5. LNN / rule-based tempo controller (`docs/plan.md` §6) — not started.
6. Depth Transformer / remaining 31 Mimi codebooks for the Voice Head —
   not started; only codebook 0 is predicted so far (§2 above).

Product-track next steps (rate limiting, usage logging, key rotation,
Web Chat) live in the `aether-api` repository.
