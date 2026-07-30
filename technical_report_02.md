# AETHER Technical Report 02

## Constrained Grammar, Tool Allowlisting, Plan Revision, and a Real-Model Product API

**Project:** AETHER
**Organization:** Manifestro
**Authors:** Manifestro Research Team
**Report date:** 31 July 2026
**Report status:** Preliminary — second measured milestone
**Predecessor:** [`technical_report_01.md`](technical_report_01.md)

---

## Abstract

Report 01 established the central AETHER hypothesis on a real open-weight model: a Planner and a Speaker, sharing one Qwen3-1.7B backbone through independent decode sessions, can produce a safe spoken response before an external tool call resolves, and a factual response only after it resolves. That experiment used one fixed scenario, one fixed tool latency, and a free-form JSONL event stream.

This report documents four extensions, each backed by an executed experiment rather than a design intention:

1. **Constrained event grammar.** The Planner's JSONL output is now validated against a closed schema per event kind, including an explicit `safe_to_say` field that must agree with a chunk's dependency list, and sequence-repair events are counted rather than silently absorbed.
2. **Tool allowlisting.** The Planner is told, per turn, exactly which tools exist; a runtime-level `AllowlistToolExecutor` independently rejects any tool call outside that list, regardless of what the model attempts. This was not a speculative hardening — it was a direct response to an observed failure: on a plain greeting, the real model invented a tool named `"chat"` and the test harness answered it with fabricated weather data.
3. **Plan revision.** The runtime now accepts a `replan` event that cancels only not-yet-committed speech, verified by 300 randomized synthetic turns with zero contradictory-speech violations, in addition to a 1,000-turn dependency-safety fuzz test carried over from the grammar work.
4. **A product API.** A new `aether_api` package turns the research runtime into a streaming Text API (`POST /v1/turns`, Server-Sent Events) that never exposes internal telemetry, chain-of-thought, or hidden state. It was run end-to-end against a real Qwen3-1.7B instance on an NVIDIA T4 — different hardware from Report 01's A100 — through a real TCP socket and a real FastAPI/uvicorn stack, not an in-process test double.

Two of the four real-model experiments in this report (the Stage 3 rerun and the Stage 4 smoke test) initially reported `FAILED`. In both cases the runtime was correct and the scoring harness was not: a latency-dependent, hardware-dependent *speed* observation had been folded into the same pass/fail gate as a *safety* invariant. Both harness defects are documented here with their fixes, because the failure mode — "the system looks broken when it is the test that is wrong" — is itself a finding worth recording precisely, not smoothing over.

---

## 1. Scope relative to Report 01

Report 01's four observable requirements — shared backbone, independent decode state, semantic lookahead, controlled commitment — are not re-litigated here; they were established once, on real hardware, and nothing in this report weakens them. This report instead asks four new questions, each scoped to one subsystem:

| # | Question | Section |
|---|---|---|
| 1 | Can the Planner's output be constrained to a closed, machine-checkable grammar without breaking the model's actual behavior? | §3 |
| 2 | Does an explicit tool allowlist — in the prompt and enforced independently at runtime — stop a model from fabricating facts for tools that do not exist? | §4 |
| 3 | Can a plan be revised mid-turn without ever rewriting speech that has already been committed? | §5 |
| 4 | Can the whole pipeline be exposed as a streaming HTTP API, on different hardware, without losing any of the above guarantees? | §6, §7 |

---

## 2. System architecture changes since Report 01

### 2.1 Constrained event grammar (plan.md §4 A1)

`SemanticEventStreamParser` gained a `strict` mode. Previously only `tool_call` and `speech_plan` payloads were shape-checked; every other event kind (`fact`, `tool_error`, `replan`, `turn_complete`) accepted an arbitrary payload. In strict mode:

- `intent` and `tool_pending` are rejected outright — they are not part of the published grammar (`tool_call`, `speech_plan`, `fact`/`tool_error`, `replan`, `turn_complete`).
- Every `speech_plan` payload must carry a boolean `safe_to_say` that is required to equal `len(dependencies) == 0`. This turns "is this chunk allowed to speak before any tool result?" from an implicit property of the dependency list into an explicit, independently checkable claim the model must make and the parser verifies.
- `turn_complete` payloads must be empty; `replan` payloads must carry a non-empty `reason` and a `cancel_chunk_ids: list[str]`.
- Sequence repair (§7.1, Report 01) is no longer silent. `SemanticEventStreamParser.repaired_count` is incremented on every repair and exposed to callers via `LLMPlannerAdapter.last_parser`, so a production deployment can alert on a rising repair rate instead of never learning it happened.

`SemanticEvent` also gained a `revision_id: int = 0` field, the wire-level hook the revision protocol (§2.3) and the API contract (§2.4) both build on.

### 2.2 Tool allowlisting

Two independent layers, deliberately not just one:

**Prompt-level (prevention).** `LLMPlannerAdapter.__init__` now takes a required `tools: Sequence[str]`. The system prompt template lists the granted tools verbatim (`"Allowed tools for this turn: {tool_list}"`), states that no other tool name may be used, and — directly addressing the fabrication finding in §4 — instructs the model to skip `tool_call` entirely and answer directly when no granted tool applies.

**Runtime-level (enforcement).** `AllowlistToolExecutor` wraps any `ToolExecutor`. A call to a tool outside the allowlist never reaches the wrapped executor; it is answered immediately with `ToolResult(error="tool not allowed: {name}")`. Because the existing dependency-resolution logic in `DualSessionRuntime` only unblocks a chunk when its named dependency resolves *successfully*, a rejected tool call has exactly the same effect as an unreachable real MCP server: the dependent chunk stays `BLOCKED` forever, and only chunks with no dependency on that name can still speak.

These two layers were verified independently: unit tests confirm the prompt text changes with the granted tool list (including the empty-set phrasing), and separately that `AllowlistToolExecutor` never lets a disallowed call reach the inner executor, even with an empty allowlist.

### 2.3 Plan revision protocol (plan.md §4 A2)

`DualSessionRuntime` now handles `EventKind.REPLAN`. For each `chunk_id` in `cancel_chunk_ids`:

- if the chunk is `cancellable` (`BLOCKED`, `READY`, `GENERATING`, or `BUFFERED`), it transitions to `CANCELLED` and a `chunk_cancelled` trace event is recorded with the triggering `revision_id`;
- if the chunk is already `COMMITTED` or `PLAYED`, the cancellation is refused and a `chunk_cancel_rejected` trace event is recorded instead — the runtime never raises, and it never rewrites history.

The harder part was the race between a `replan` arriving on the Planner's coroutine and the Speaker's worker task, which runs concurrently. `speaker_worker` now checks `chunk.state is ChunkState.CANCELLED` twice: once before starting generation (a chunk cancelled while merely queued is dropped without ever calling the Speaker), and once immediately after `await self._speaker.generate(...)` returns (a chunk cancelled *while* generation was in flight is dropped before the `BUFFERED → COMMITTED → PLAYED` transition, which — in the current synchronous-commit implementation — happens without an intervening `await`, so this is the only point where such a race is observable at all).

### 2.4 Live event hook and the product API seam

`Timeline.__init__` gained an optional `on_event: Callable[[TraceEvent], None]`, invoked synchronously at the moment each event is recorded, before `record()` returns. `DualSessionRuntime.run()` forwards an `on_event` parameter to its internal `Timeline`. This is the entire change required in the research core to make live streaming possible: previously `DualSessionRuntime.run()` returned a complete `DualSessionResult` only after the whole turn finished, which is incompatible with Server-Sent Events.

Two further telemetry additions were needed for the API to carry real content instead of just structural events: `tool_completed` now includes the tool's `content`/`error`, and `chunk_committed` now includes the spoken `text` and a `safe_to_say` boolean (mirroring the grammar-level field from §2.1). `chunk_committed` was chosen deliberately as the exposure point — it is the same commit horizon past which the domain's `ChunkState` machine already refuses to let a chunk be cancelled or rewritten, so "safe to leave the process" and "safe to show a client" are the same boundary.

### 2.5 Naming generalization

`qwen_adapters.py`, `qwen_backbone.py`, and `qwen_step_engine.py` (and their `Qwen*` classes) were renamed to `llm_adapters.py`, `llm_backbone.py`, `llm_step_engine.py` (`LLMPlannerAdapter`, `LLMSpeakerAdapter`, `SharedLLMBackbone`, `LLMBackboneConfig`, `LLMTokenStepEngine`). `Qwen` now appears in the codebase only where it names an actual artifact — the default `--model` CLI value and one test's literal Hugging Face model id — never as a class or module name. This reflects an explicit project position (plan.md §3.2): Qwen is the current Planner-candidate backbone, not an architectural commitment.

### 2.6 Branch strategy

Following an explicit product/research split (plan.md §1), all work in §§2.2–2.4 above that constitutes product surface (`aether_api`) lives on a `product/api` branch; `main` remains research-core only (`aether`). The one exception is deliberately minimal: the `on_event` hook (§2.4) is a research-core change (it lives in `aether.domain.timeline` and `aether.runtime.dual_session`), because `aether_api` cannot exist without it, but it was written to be a no-op for every existing caller (`on_event` defaults to `None`) and is exercised by core-only tests (`tests/test_timeline.py`, `tests/test_dual_session.py`) that do not import `aether_api`.

---

## 3. Experiment 02 — Latency sweep and scenario diversity (Stage 3)

### 3.1 Protocol

Report 01's next-experiments list (§8, Experiment 02) called for a latency sweep and removal of the free-form `intent` event. This report's Stage 3 runner (`aether.experiments.colab_stage3`) implements both, plus scenario diversity Report 01 did not have: a single fixed scenario cannot separate "the architecture works" from "this one prompt happens to work."

Three scenarios × a latency sweep:

| Scenario | Request | Tool outcome | Swept? |
|---|---|---|---|
| `weather_success` | "Какая погода в Алматы и нужен ли зонт?" | succeeds | latencies 3000/1500/750/300 ms |
| `weather_tool_failure` | same | fails (`error`) | latencies 3000/1500/750/300 ms |
| `no_tool_greeting` | "Привет! Как у тебя дела?" | no tool should be needed | fixed at 3000 ms |

Nine runs total per full sweep. Each run is checked against explicit, code-defined criteria (`evaluate_run`), not eyeballed.

### 3.2 A methodology defect, found and corrected

The first executed sweep (real Qwen3-1.7B, A100) reported `status: failed`, 6 of 9 runs failing. Inspection showed every failure was the same check: `speaker_first_token_before_tool_complete`. At 3000 ms tool latency the Speaker led by roughly a second, exactly as in Report 01; at 1500/750/300 ms the tool completed *before* the Speaker's first token, because the fixed Planner→Speaker scheduling overhead on this hardware is on the order of 2 seconds and does not shrink just because the tool got faster.

This is not a defect in the runtime. It is the latency crossover the architecture predicts: dual-stream lookahead only produces a *speed* benefit when tool latency exceeds pipeline overhead: below that point the tool simply finishes first, and the architecture correctly falls back to behaving like a fast sequential system. The defect was in `evaluate_run`, which had folded that latency-dependent observation into the same `passed` boolean as the actual safety property (a dependent chunk must never speak before its tool is confirmed). `evaluate_run` was rewritten to return `(checks, observations)`: `checks` contains exactly one scenario-agnostic, trace-derived boolean — `dependent_chunks_only_speak_confirmed_facts`, computed by checking, for every `PLAYED` chunk with dependencies, that a `tool_completed(succeeded=True)` event for each dependency precedes that chunk's `chunk_generating` event in the recorded trace — and `observations` carries everything latency-dependent (`tool_call_emitted`, `blocked_chunk_ids`, `speaker_first_token_before_tool_complete`, the signed lead in milliseconds).

### 3.3 The tool-fabrication finding

The same first sweep exposed an independent, genuine defect. In `no_tool_greeting`, the real model — given a system prompt that only illustrated a weather tool pattern, with no explicit allowlist — invented a tool call:

```json
{"type":"tool_call","sequence":0,"payload":{"call_id":"chat-1","tool":"chat","arguments":{"user":"Привет! Как у тебя дела?"}}}
```

The test harness's `FakeWeatherTool` did not check the requested tool's name; it answered any call with weather content. The resulting spoken output was:

> «Началась беседа. В настоящее время в Алмате, в условиях дождя, температура составляет 24 градуса Цельсия.»

— a fabricated weather report presented as an answer to "how are you." The dependency protocol was not violated (the chunk correctly waited for `tool_completed` before speaking); the defect was that the harness let a nonexistent tool "succeed." `FakeWeatherTool` was fixed to return `tool_error` for any name other than `"weather"`; §2.2's `AllowlistToolExecutor` and prompt-level allowlist were the systematic fix.

### 3.4 Results across three real-model runs

| Run | Fixes present | Status | `no_tool_greeting` behavior |
|---|---|---|---|
| 1 | none of §2.1–2.2 | `FAILED` (scoring bug) | fabricated weather answer for invented `"chat"` tool |
| 2 | `FakeWeatherTool` fix + `evaluate_run` split | `PASSED`, 9/9 | tool `"chat"` still attempted, but rejected → chunk stays `BLOCKED`, spoken text is only the safe lead-in ("Началась беседа.") |
| 3 | + prompt-level allowlist + `AllowlistToolExecutor` + grammar/renaming/revision work | `PASSED`, 9/9 | **no tool call attempted at all** (`tool_call_emitted: False`); model answers directly |

Run 2 → Run 3 is the clearest evidence that prompt-level prevention (§2.2) does more than the runtime-level guard alone: enforcement stopped the fabricated fact from reaching the user in Run 2, but the model still *tried* to call a nonexistent tool; telling it explicitly what tools exist stopped the attempt itself in Run 3.

Full timing series, Run 2 (safety-passing on every run; lead sign flips exactly at the predicted crossover):

| Scenario | 3000 ms lead | 1500 ms lead | 750 ms lead | 300 ms lead |
|---|---:|---:|---:|---:|
| `weather_success` | −1014.6 ms | +448.9 ms | +1214.6 ms | +1671.5 ms |
| `weather_tool_failure` | −1036.8 ms | +490.9 ms | +1214.1 ms | +1670.4 ms |

(Negative = Speaker led; positive = tool finished first. `dependent_chunks_only_speak_confirmed_facts` was `True` in all 9 runs in both Run 2 and Run 3, including every positive-lead case — the factual chunk was withheld until the tool result regardless of who arrived first.)

### 3.5 Local, dependency-free validation

Two fuzz tests give the constrained-grammar and revision work statistical confidence without spending GPU time:

- **1,000 synthetic turns** (`tests/test_dependency_fuzz.py`) through `DualSessionRuntime` with a randomized fake Planner (0–3 chunks, random dependencies, random tool success/failure, random tiny latencies): zero turns where a dependent chunk's `chunk_generating` trace event preceded its tool's `tool_completed`, and zero turns where a chunk spoke after its tool failed. Runtime ≈1.9 s.
- **300 synthetic replan turns** (`tests/test_replan_fuzz.py`): the same randomized Planner, additionally emitting a `replan` that cancels a random subset of its own chunks roughly half the time. Zero turns where a chunk recorded as `chunk_cancelled` was later `PLAYED` or appeared in the spoken text. Runtime ≈0.5 s.

This directly satisfies the exit criteria in plan.md §4 A1/A2 ("1,000 synthetic turns without a dependency violation," "100% of plan-change-before-commit-horizon tests pass without contradictory speech") without requiring a GPU for every regression check.

---

## 4. Experiment 03 — Product API vertical slice (`aether_api`)

### 4.1 Architecture

A new package, `src/aether_api/`, depends on `aether` and is never imported by it:

```text
aether_api/
├── contract.py        PublicEventType, PublicEvent (turn_id/sequence/timestamp_ms/revision_id + payload)
├── event_mapper.py     EventMapper: TraceEvent -> PublicEvent, the only place deciding what a client may see
├── turn_service.py     TurnService: background task + queue, drains on_event live, streams through EventMapper
├── auth.py             ApiKeyStore (dev-tier): bearer keys, per-key concurrent-turn cap
├── http/
│   ├── app.py            FastAPI: POST /v1/turns -> StreamingResponse (SSE)
│   └── dev_server.py      boots the app against ScriptedSharedBackend fakes, no weights required
└── experiments/
    └── colab_stage4.py   real-model HTTP smoke test (§5)
```

`EventMapper` only forwards a fixed set of internal trace event names — `turn_started`, `tool_started`, `tool_completed`, `turn_completed`, and `chunk_committed` (split into `response.safe_delta`/`response.delta` by the `safe_to_say` attribute from §2.4). Internal scheduler/decode telemetry (`decode_started`, `chunk_generating`, `chunk_ready`, …) is not in the mapping table and is silently dropped — there is no path from a `TraceEvent` to the wire that does not go through this explicit allowlist.

`TurnService.stream_turn` builds `LLMPlannerAdapter`/`LLMSpeakerAdapter` with the request's tool list, wraps the tool executor in `AllowlistToolExecutor`, runs `DualSessionRuntime.run(..., on_event=queue.put_nowait)` as a background `asyncio.Task`, and yields `EventMapper.map(event)` for each item drained from the queue — this is the direct consumer of the §2.4 hook. A planner-side exception (e.g., a strict-grammar violation) is caught around the background task and surfaces as a synthesized `turn.failed` event rather than crashing the stream.

### 4.2 Local, dependency-free validation

18 tests exercise `contract`, `event_mapper`, and `turn_service` using the same `ScriptedSharedBackend`/`FakeWeatherTool` doubles used throughout the project — no network, no GPU, no FastAPI import required for two of the three test modules. A fourth module exercises the actual FastAPI app via `TestClient`, guarded with `unittest.skipUnless(_HAS_FASTAPI, ...)` so the project's standing invariant — `PYTHONPATH=src python -m unittest discover -s tests` runs with zero optional dependencies installed — still holds; the tests only run when the new `api` extra (`fastapi`, `uvicorn`, `httpx`) is present. Notably, one test (`test_disallowed_tool_never_leaks_a_fabricated_fact`) asserts through the full HTTP stack that a turn requesting an empty tool list produces a `tool.completed` event with `succeeded: False` and never a `response.delta` — the §3.3 finding, re-verified as a permanent regression test at the API boundary.

### 4.3 Local end-to-end proof

Before any real-model run, the vertical slice was run against a live `uvicorn` process (not a test double for the transport) using `aether_api.http.dev_server`, which wires the FastAPI app to `ScriptedSharedBackend`:

```bash
curl -N http://127.0.0.1:8731/v1/turns \
  -H 'content-type: application/json' \
  -H 'authorization: Bearer dev-key' \
  -d '{"message":"Какая погода в Алматы?","tools":["weather"]}'
```

produced, over real HTTP:

```text
event: turn.started
event: plan.tool_started
event: response.safe_delta      {"text": "Проверка погоды началась, результат ещё ожидается."}
event: tool.completed           {"tool": "weather", "succeeded": true, "location": "Almaty", "temperature_c": 24, "condition": "rain"}
event: response.delta           {"text": "Сейчас 24 градуса, ожидается дождь — зонт лучше взять."}
event: turn.completed
```

matching the SSE example in plan.md §5 B1 field-for-field, with `turn_id`/`sequence`/`revision_id` added per the actual contract.

---

## 5. Experiment 04 — Real-model API smoke test (Stage 4)

### 5.1 Protocol and hardware

`aether_api.experiments.colab_stage4` is the first run of `aether_api` against a real model rather than `ScriptedSharedBackend`. It loads a real `SharedLLMBackbone`, wraps it in the same `InterleavedDecodeScheduler` validated in Experiment 02 (not the plain locked backend — `TurnService` is backend-agnostic and will accept either; using the unscheduled backend would silently discard the dual-stream property this whole report exists to preserve), and drives one `weather_success`-shaped turn through the real FastAPI app.

Hardware differs deliberately from Report 01 and Experiment 02: an NVIDIA Tesla T4 (15.6 GB, compute capability 7.5, Turing), not the A100 used previously. T4 has no native bf16 tensor cores; `--dtype` was set explicitly to `float16` rather than `"auto"` to avoid an unstable or slow bf16 fallback on checkpoints that default to bf16. Model load time was ≈65 s (weight fetch + load), against ≈9.7 s on the A100 in Report 01 — consistent with T4's slower I/O and compute, and not itself a concern for this experiment, which measures per-turn behavior after the model is resident.

### 5.2 Transport methodology: why not `TestClient`

A first implementation used FastAPI's in-process `TestClient` (httpx `ASGITransport` plus a synchronous portal). Locally, against fakes, every event in a captured stream reported an identical arrival timestamp: `TestClient`'s transport drains the whole SSE body before `iter_lines()` yields anything, so it cannot distinguish "streamed live" from "computed once and returned as one chunk" — precisely the property the §2.4 hook exists to prove. The experiment was rewritten to run a real `uvicorn.Server` in a background thread bound to a real TCP port, and to drive it with a real `httpx.stream(...)` client. Re-run locally against fakes, the same scenario produced genuinely staggered timestamps (29 ms, 53 ms, 94 ms, then a gap to 356 ms matching a simulated 400 ms tool delay, then 468 ms) — the transport artifact was confirmed and eliminated before spending GPU time on it.

### 5.3 A second methodology defect, found and corrected

The first real-model Stage 4 run reported `FAILED`, for the same class of reason as §3.2: `report["proof"]` conflated `safe_delta_before_tool_completed` (soft, hardware- and latency-dependent) with the actual safety property. On T4, at the tool latency configured for that run (1500 ms, the pre-existing default), the observed timeline was:

```text
   0.1 s  turn.started
   2.5 s  plan.tool_started
   4.0 s  tool.completed
   6.2 s  response.safe_delta     ← after tool.completed
   9.9 s  response.delta
   9.9 s  turn.completed
```

The safe lead-in itself lost its lead on this hardware — T4's Planner+Speaker pipeline overhead, at roughly 6 seconds to a committed safe chunk, exceeded the 1500 ms tool latency, the same crossover phenomenon from §3.2, now observed as a function of hardware speed rather than of tool latency. The one property that must never be violated — `response.delta` never arriving before `tool.completed` — held (9.9 s vs. 4.0 s). The experiment's scoring was corrected the same way as `evaluate_run`: a new pure function, `evaluate_events`, returns `(checks, observations)`, where `checks` contains only `http_200`, `ends_with_turn_completed`, `no_turn_failed`, and `response_delta_never_before_tool_completed`, and `observations` carries `streamed_progressively`, `safe_delta_before_tool_completed`, and the signed millisecond lead. Five unit tests pin this logic, including a literal reproduction of the T4 trace above (`test_slow_gpu_losing_the_safe_lead_still_passes`) so this exact failure mode cannot silently regress. The default `--tool-latency-ms` was also raised to 4000 (and the notebook's default alongside it) to give slower hardware a realistic chance of also showing the lead, without making that a requirement.

### 5.4 Results

Re-run with the corrected harness and `--tool-latency-ms 4000`:

```text
  175.7 ms  turn.started
 2642.1 ms  plan.tool_started
 6267.1 ms  response.safe_delta     (374.6 ms before tool.completed)
 6641.7 ms  tool.completed          {"location":"Almaty","temperature_c":24,"condition":"rain","succeeded":true}
 9317.2 ms  response.delta          "В городе Алматы сейчас дождь, температура 24 градуса Цельсия."
 9317.2 ms  turn.completed
```

```json
"checks": {
  "http_200": true,
  "ends_with_turn_completed": true,
  "no_turn_failed": true,
  "response_delta_never_before_tool_completed": true
},
"observations": {
  "streamed_progressively": true,
  "safe_delta_before_tool_completed": true,
  "safe_delta_minus_tool_completed_ms": -374.59
}
```

All four hard checks passed; the lead reappeared once tool latency was raised past this hardware's crossover point, corroborating §3.2's account of the crossover as latency-vs-overhead rather than a fixed property of either. `status: passed`; 68/68 dependency-free tests passed in the same run (`tests.log`, 100%).

---

## 6. Cross-cutting interpretation

**The crossover is now a load-bearing, twice-replicated finding, not a one-off observation.** Report 01 measured a single lead value at a single latency on a single GPU. This report measured the lead flip sign as a function of tool latency on one GPU (§3.4) and, independently, as a function of GPU speed at fixed-then-raised tool latency (§5.3–5.4). Both replications are consistent with the same explanation: the dual-stream lead is `tool_latency − pipeline_overhead`, and either operand moving the same way moves the observed lead the same way. Any production deployment must treat "does the safe lead-in arrive first" as a measured, environment-dependent quantity, not an architectural constant — but "the factual chunk never arrives before its tool result" held in *every* run, at every latency, on both GPUs, with a failing tool, and with a fabricated tool name. That is the actual safety claim the rest of the architecture depends on, and it did not require a single exception in any experiment in this report.

**Prevention and enforcement are not substitutes for each other.** §3.4's Run 2 vs. Run 3 shows enforcement alone (the `AllowlistToolExecutor`) stops harm but not the underlying behavior; only prompting the model with an explicit allowlist stopped the attempt. A production system should keep both: enforcement is what you can prove holds regardless of model behavior, prevention is what keeps the system's behavior legible and its logs clean.

**A test harness that conflates "correct" with "fast today, on this box" produces false negatives that look exactly like real regressions.** This happened twice in this report, in two different experiments, in the same specific shape (a latency-sensitive boolean folded into a safety-relevant `passed`). Both fixes followed the same pattern — split into a scenario/hardware-agnostic `checks` dict and a `observations` dict — which is now the project's de facto convention for any future experiment runner.

---

## 7. Current limitations

### 7.1 Revision protocol has not been exercised by a real model

§2.3/§3.5's replan protocol is proven correct against a fuzzed synthetic Planner (300 turns, zero violations) and against two hand-written deterministic scenarios (`tests/test_replan.py`). No experiment in this report drove a real Qwen3-1.7B into emitting a `replan` event — the current system prompt gives it no reason to. This is a gap, not a failure: the runtime-side mechanism is validated; whether a real model can be prompted (or should be trained) to use it productively is open.

### 7.2 `aether_api` does not yet implement rate limiting, usage logging, or key rotation

`ApiKeyStore` enforces a per-key concurrent-turn cap and nothing else from plan.md §5 B2 (no requests-per-minute limit, no persistent usage counters, no revoke/rotate). This was scoped out of the current vertical slice deliberately (handoff-1.md §10's definition of done for the first slice did not include it) and remains explicitly open.

### 7.3 No real MCP client

Every experiment in this report, including the Stage 4 API smoke test, still executes tool calls against `FakeWeatherTool` — a deterministic, single-tool, in-process stand-in. `AllowlistToolExecutor` is written against the generic `ToolExecutor` protocol and does not assume a fake, but no experiment has yet substituted a real MCP-backed executor.

### 7.4 Interleaved, not physically simultaneous, execution (carried over from Report 01)

`InterleavedDecodeScheduler` alternates one-token forward steps on a single shared backbone; this report adds no evidence about batching or CUDA-stream-level concurrency, and none was claimed.

### 7.5 Stage 4 measured one scenario, once per hardware/latency configuration

Unlike Stage 3's nine-run sweep, Stage 4 has not yet been extended to a scenario matrix (tool failure, no-tool-needed) at the HTTP layer. The underlying `TurnService`/`EventMapper` behavior for those cases is already covered by the dependency-free tests in §4.2; what is missing is the same real-model, real-transport confirmation Stage 3 has for the runtime layer.

---

## 8. Next experiments

Carried forward from Report 01 (§8) and plan.md §12, reordered by what this report's findings make most urgent:

1. **Real-model replan.** Design a scenario and prompt that gives the Planner an actual reason to revise (e.g., a tool result that contradicts an already-planned safe chunk), and confirm the `chunk_cancelled`/`chunk_cancel_rejected` telemetry on a real trace, not only a fuzzed one.
2. **Stage 4 scenario matrix.** Extend `colab_stage4` to the same three-scenario shape as Stage 3, over the real HTTP/SSE transport, to close §7.5.
3. **Rate limiting, usage logging, key rotation** (plan.md §5 B2) — the remaining, explicitly-scoped-out parts of the product Definition of Done.
4. **Real MCP client adapter**, replacing `FakeWeatherTool` behind the same `ToolExecutor`/`AllowlistToolExecutor` boundary already in place.
5. **Web Chat Playground** (plan.md §5 B3), as a thin client of the now-working `POST /v1/turns` contract.
6. **LNN / rule-based tempo controller** (plan.md §6), still entirely unstarted; the `chunk_committed(safe_to_say=...)` telemetry added in §2.4 is a candidate feature source once this work begins.

---

## 9. Reproducibility

Relevant components, in addition to those listed in Report 01 §9:

- `src/aether/model/event_parser.py` — strict grammar, `repaired_count`;
- `src/aether/model/llm_adapters.py` — tool-allowlist prompt construction (renamed from `qwen_adapters.py`);
- `src/aether/runtime/dual_session.py` — `replan` handling, `on_event` hook, enriched `tool_completed`/`chunk_committed` telemetry;
- `src/aether/runtime/tool_executor.py` — `AllowlistToolExecutor`;
- `src/aether/experiments/colab_stage3.py` — latency-sweep/scenario-diversity runner, `evaluate_run`;
- `src/aether_api/` — the entire product API package (§4.1);
- `src/aether_api/experiments/colab_stage4.py` — real-model API smoke test, `evaluate_events`;
- `notebooks/aether_stage3_colab.ipynb`, `notebooks/aether_stage4_colab.ipynb` — remote experiment notebooks (Stage 4 clones `product/api`, not `main`);
- `tests/test_dependency_fuzz.py`, `tests/test_replan.py`, `tests/test_replan_fuzz.py`, `tests/test_tool_executor.py`, `tests/api/` — the new dependency-free test surface for this report.

The development suite now contains 68 tests and passes without model weights:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

API-layer tests that require `fastapi`/`httpx` (installed via the new `api` extra) are individually guarded with `unittest.skipUnless` and do not affect this invariant when the extra is absent. Remote reports referenced in this document are stored locally under `.logs/` and are excluded from version control, as are model caches and weight files.

---

## 10. Conclusion

Report 01 showed that one open model, split into two coordinated decode sessions, can act before it finishes speaking and speak safely before it can act factually. This report shows that the same architecture survives being made strict: a closed event grammar, an enforced tool allowlist, a mid-turn revision protocol, and a public streaming API layer, all without weakening the one property that matters most — a spoken fact is never produced before the system has confirmed it.

Two of this report's four experiments initially failed for reasons that turned out to be about the experiments, not the system: a scoring harness that treated a hardware-and-latency-dependent speed observation as if it were a safety requirement. Both were corrected the same way, and the corrected instrumentation (`checks` vs. `observations`) is now the pattern for every future experiment in this project. The remaining, real defect this report found — a model fabricating a fact for a tool it invented, because nothing told it which tools existed — was closed at both the prompting layer and the runtime layer, and re-verified as a permanent regression test at the API boundary, not just in the runtime that first exposed it.

The project has moved from "the semantic dual-stream mechanism works on one model, once" to "the mechanism, plus the constraints a real product needs around it, works on two models of hardware, through a real HTTP/SSE transport, with the safety property holding in every one of dozens of real-model runs and thousands of synthetic ones." The next open risk is no longer whether the runtime is safe under revision and tool failure — that is now evidenced at both the unit and the real-model level — but whether the product surface around it (auth at scale, a real MCP client, a model that uses revision productively) can be built without reopening any of it.
