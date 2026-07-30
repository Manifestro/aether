# VOX-SYNAPSE Technical Report 01

## Predictive Dual-Stream Tool Use with Interleaved LLM Decoding

**Project:** VOX-SYNAPSE  
**Organization:** Manifestro  
**Authors:** Manifestro Research Team;
**Report date:** 30 July 2026  
**Report status:** Preliminary proof-of-concept result

---

## Abstract

VOX-SYNAPSE is a research architecture for voice agents in which semantic planning and speech generation proceed as two coordinated streams. The semantic stream can initiate an external tool call while the speech stream begins a safe, tool-independent response. When the tool result arrives, the speech stream continues with facts that were previously blocked by a dependency boundary.

This report documents the first experiment using an open Qwen3-1.7B language model on an NVIDIA A100 80 GB GPU. The system loaded one shared model and maintained independent logical Planner and Speaker decoding sessions with separate decode state. An interleaved token scheduler alternated model steps between the sessions.

In the weather scenario, the first Speaker token appeared approximately **1.22 seconds before the MCP result completed**. The factual speech chunk started only after the tool result was available. This is the first experimental confirmation of the central VOX-SYNAPSE hypothesis on a real open-weight model.

The experiment does not yet demonstrate native audio generation, end-to-end speech input, or physically simultaneous CUDA execution. It establishes the semantic/action lookahead and commitment-control mechanism required for those later stages.

---

## 1. Research question

The experiment tests the following hypothesis:

> Can one open language-model backbone serve two independent decoding streams so that a Planner initiates an MCP action ahead of the spoken response, while a Speaker produces safe speech before the tool result and factual speech only after the result is available?

The hypothesis has four observable requirements:

1. **Shared backbone:** Planner and Speaker use one loaded set of model weights.
2. **Independent state:** each logical stream maintains its own decoding state.
3. **Semantic lookahead:** MCP starts before the response is complete.
4. **Controlled commitment:** tool-dependent speech cannot be generated as factual output before the tool result.

---

## 2. System architecture

```text
                         Shared Qwen3-1.7B weights
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
                  ▼                                   ▼
          Planner decode state                 Speaker decode state
          independent KV-cache                independent KV-cache
                  │                                   ▲
                  │ semantic events                  │ speech chunks
                  ▼                                   │
             Async MCP Engine ───── tool result ─────┘
                                    │
                           Commit/dependency control
```

The implementation separates the following components:

- **Planner:** generates typed JSONL semantic events;
- **MCP Engine:** starts and completes the external tool call asynchronously;
- **Speaker:** generates speech text for the current safe or factual goal;
- **Dependency graph:** marks tool-dependent chunks as blocked until facts arrive;
- **Interleaved scheduler:** selects the next token step for Planner or Speaker;
- **Timeline:** records model, tool, chunk and commitment events.

The current experiment uses text output from Speaker. Native audio-token generation is intentionally deferred until the semantic scheduler is measured and stable.

---

## 3. Model and runtime configuration

| Component | Configuration |
| --- | --- |
| Backbone | Qwen3-1.7B |
| Model weights | One shared loaded instance |
| Planner state | Independent logical decode state |
| Speaker state | Independent logical decode state |
| Scheduler policy | Speaker weight 3, Planner weight 2 |
| Tool | Deterministic synthetic weather tool |
| Tool latency | 3000 ms configured delay |
| Input | Russian text request |
| Output | Russian text chunks |
| GPU | NVIDIA A100-SXM4-80GB |
| GPU memory | Approximately 80 GB |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.13.1 |
| Accelerate | 1.14.0 |
| Safetensors | 0.8.0 |
| PEFT | 0.19.1 |

The model was loaded explicitly in the remote experiment environment. No model weights were downloaded or used on the development computer.

---

## 4. Experimental protocol

The user request was:

> «Какая погода в Алматы и нужен ли зонт?»

The expected semantic flow was:

```text
tool_call(weather, Almaty)
safe speech_plan
tool-dependent speech_plan
turn_complete
```

The Planner produced the following observable events:

```text
intent
tool_call(weather, location=Almaty)
speech_plan(lead-in, no dependencies)
speech_plan(answer, dependency=weather)
turn_complete
```

The Planner occasionally repeated a sequence number in its raw output. The adapter repaired the sequence at the model boundary so the runtime received a strictly increasing auditable event stream. This is a current model-output quality issue, not a dependency or scheduler violation.

---

## 5. Results

### 5.1. End-to-end measurements

| Measurement | Result |
| --- | ---: |
| Model load | 9701.9 ms |
| Runtime duration | 7571.2 ms |
| MCP started | 2220.0 ms |
| Safe chunk became ready | 3915.6 ms |
| Speaker first token | 4003.0 ms |
| Safe chunk played | 5076.6 ms |
| MCP completed | 5220.7 ms |
| Factual chunk started | 5980.3 ms |
| Final response completed | 7571.2 ms |

### 5.2. Primary proof

```text
Speaker first token: 4003.0 ms
MCP completion:      5220.7 ms
Difference:         -1217.7 ms
```

The first Speaker token therefore appeared approximately **1.22 seconds before the external result was available**.

The dependency boundary behaved correctly:

```text
Safe chunk:
  «Проверка погоды началась, результат ещё ожидается.»
  generated before MCP completion

Factual chunk:
  «В городе Алматы сейчас дождь, температура 24 градуса Цельсия.»
  generated after MCP completion
```

The final result was:

> «Проверка погоды началась, результат ещё ожидается. В городе Алматы сейчас дождь, температура 24 градуса Цельсия.»

### 5.3. Proof status

| Criterion | Status | Evidence |
| --- | --- | --- |
| One shared open backbone | PASS | Qwen3-1.7B loaded once |
| Independent Planner/Speaker sessions | PASS | Separate session IDs and decode states |
| MCP starts before full response | PASS | Tool starts at 2220 ms |
| Speaker starts before MCP result | PASS | First token at 4003 ms; result at 5221 ms |
| Safe chunk before factual result | PASS | Lead-in plays before tool completion |
| Factual chunk blocked until result | PASS | Answer starts at 5980 ms |
| No unsupported weather fact in lead-in | PASS | Lead-in contains no weather value |
| Native speech-to-speech | NOT TESTED | Current Speaker output is text |
| Physical simultaneous GPU kernels | NOT TESTED | Current scheduler interleaves token steps |

---

## 6. Interpretation

The experiment validates the architectural core of VOX-SYNAPSE at the semantic orchestration level. The model does not need to finish planning the whole response before the system can act. Once a usable tool call and safe speech plan are available, the scheduler allows the Speaker stream to progress while the external operation remains pending.

The result is stronger than a conventional asynchronous application wrapper because the two streams share the same model weights while maintaining independent decoding state. The scheduler controls model progress at token-step granularity rather than waiting for a complete Planner response.

The experiment also validates the concept of controlled factual commitment. The system can distinguish:

- what may be said while information is pending;
- what must remain blocked;
- what becomes speakable once the tool result arrives.

This is the mechanism required for natural filler, safe lead-ins and future audio continuation.

---

## 7. Current limitations

### 7.1. Planner output discipline

Qwen occasionally emits an unnecessary `intent` event and may repeat a sequence number. The adapter currently repairs the sequence. Future versions should use constrained decoding or a compact action grammar so the first useful event is a valid `tool_call`.

### 7.2. Normal termination telemetry

The Planner stops intentionally after `turn_complete`. In the recorded raw scheduler timeline this can appear as `decode_cancelled` because the consumer closes the stream before model EOS. The report-level proof treats a cancellation after `turn_complete` as normal completion. The runtime telemetry should be updated so this distinction is represented directly by `decode_completed(reason="turn_complete")`.

### 7.3. Interleaved, not physically simultaneous, execution

The scheduler alternates one-token forward steps. This demonstrates independent progress and shared-weight scheduling, but not two simultaneous CUDA kernels. The next optimization target is batching or CUDA-stream-aware execution where it improves latency without destabilizing KV-cache ownership.

### 7.4. Text-only Speaker

The current Speaker emits text. No claim about audio quality, TTFA in PCM, acoustic continuity or Mimi codebook generation is made in this report.

### 7.5. Input modality

The current experiment starts from text. Streaming audio input, VAD and barge-in remain future stages.

---

## 8. Next experiments

### Experiment 02 — protocol and latency sweep

- remove unnecessary `intent` from actionable tool scenarios;
- add constrained JSONL/event decoding;
- repeat with MCP delays of 3000, 1500, 750 and 300 ms;
- measure tool-call emission time, first Speaker token and safe chunk playback;
- compare sequential, locked dual-session and interleaved scheduler baselines.

### Experiment 03 — native audio output

- connect Mimi codec;
- train or adapt a Voice Head for codebook generation;
- measure first audio token and PCM playback latency;
- preserve the same semantic dependency and commit protocol.

### Experiment 04 — audio input and interruption

- add streaming audio encoder;
- add VAD and barge-in cancellation;
- measure stop-on-barge-in;
- validate that committed audio is never rewritten while buffered audio can be cancelled.

---

## 9. Reproducibility

The experiment is implemented in the VOX-SYNAPSE repository. The relevant components are:

- `src/vox/model/step_scheduler.py` — interleaved scheduler;
- `src/vox/model/qwen_step_engine.py` — one-token Qwen engine with per-session KV state;
- `src/vox/runtime/dual_session.py` — Planner/MCP/Speaker runtime;
- `src/vox/experiments/colab_stage2.py` — diagnostic runner;
- `notebooks/vox_stage2_colab.ipynb` — remote experiment notebook;
- `tests/test_step_scheduler.py` — dependency-free scheduler tests.

The development suite currently contains 21 tests and passes without model weights. The remote report used for this document is stored locally as:

`.logs/vox-colab-stage2-final/report.json`

Model caches, logs and weight files are excluded from version control.

---

## 10. Conclusion

The first real-model experiment supports the VOX-SYNAPSE thesis:

> A voice agent can initiate action and begin a safe response before the external information required for the final answer is available.

On Qwen3-1.7B and an A100, the Speaker produced its first token approximately 1.22 seconds before MCP completion, while the factual continuation remained blocked until the result arrived.

The project has now moved beyond a conceptual diagram and a mock orchestration test. The central semantic dual-stream mechanism has been demonstrated on an open model. The next research risk is no longer whether the idea can work, but whether the same scheduling and commitment protocol can be transferred to native streaming audio with sufficiently low latency and natural acoustic transitions.
