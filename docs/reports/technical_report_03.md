# AETHER Technical Report 03

## Phase C: Audio Commit Horizon, a Real Speech Teacher, and a Standalone Hidden-State Voice Head

**Project:** AETHER
**Organization:** Manifestro
**Report date:** 31 July 2026
**Report status:** Preliminary — third measured milestone
**Predecessor:** [`technical_report_02.md`](technical_report_02.md)

---

## Abstract

Reports 01 and 02 established and hardened the text-only dual-stream core: a Planner and a Speaker sharing one Qwen3-1.7B backbone, a constrained semantic event grammar, tool allowlisting, and plan revision — all validated on real weights. This report is the first to touch audio generation (Phase C, `docs/plan.md` §7; spec.md Level C), across two stages:

1. **Stage 4 — audio commit horizon (structural probe).** A single Mimi codebook, predicted by an untrained, from-scratch `MinimalVoiceHead` conditioned on Speaker text, was wired into `DualSessionRuntime` at the same commit point as text. Result: `ChunkState`'s existing transition table governs audio with zero new states — a dependent chunk's audio is never buffered before its tool result, and `replan` cancels in-flight audio synthesis exactly like buffered text. Passed, on real Mimi weights, across a full latency sweep.
2. **Stage 5 — hidden-state-conditioned, standalone Voice Head.** A small, from-scratch `MultiCodebookVoiceHead` predicts all 32 Mimi codebooks from a Qwen3 hidden state — not from decoded text — trained by distillation against a real teacher, `kyutai/tts-1.6b-en_fr`, on 20 English phrases. The finished model generates audio with zero channels borrowed from the teacher at inference time. Verified non-identical to the teacher at the byte level (ruling out a memorization artifact masquerading as a result) and judged by ear across all 24 phrases (20 train, 4 held-out): a recognizably similar voice, intelligible in places, with some words dropping out — a plausible symptom of this stage's deliberate architectural simplification (§4.4).

Getting to Stage 5's result took four real, distinct defects, each found on real hardware and each documented here with symptom and fix, per this project's practice of treating "the harness/assumption was wrong" as a finding worth recording precisely rather than smoothing over.

---

## 1. Scope relative to Report 02

Report 02's four hardening questions (grammar, allowlisting, revision, product API) are not re-litigated here. This report asks two new questions, both inside Phase C:

| # | Question | Section |
|---|---|---|
| 1 | Does the existing commit-horizon machinery (`ChunkState`), proven for text, generalize to an audio modality without modification? | §2 |
| 2 | Can a Voice Head learn to generate standalone speech from a Planner/Speaker-side *hidden state* — not from decoded text — distilled from a real open-weight teacher? | §3-§5 |

---

## 2. Stage 4 — Audio Commit Horizon

### 2.1 Hypothesis

The single most reusable claim from Reports 01-02 is `ChunkState`'s transition table: a chunk cannot become `COMMITTED` before its dependencies resolve, and `COMMITTED`/`PLAYED` chunks are never rewritten. Phase C's first open question was whether this same machinery, unmodified, would also govern an audio channel — or whether audio would need its own parallel state machine.

### 2.2 Design

- `AudioChunk` (`aether/domain/audio.py`): carries only `chunk_id`, codec tokens, and frame rate — no state of its own. Commit/cancel behavior is entirely inherited from the owning `SpeechChunk`'s `ChunkState`.
- `VoiceHead` protocol (`aether/model/protocols.py`): `synthesize(chunk, text, facts, hidden_state=None) -> AudioChunk`.
- `MinimalVoiceHead` (`aether/model/voice_head.py`): a 2-layer, `d_model=128` transformer, randomly initialized and **never trained** — this stage's pass/fail criteria are structural (timing, cancellation correctness), not perceptual, so an untrained model is sufficient and deliberate.
- `DualSessionRuntime` gained an optional `voice_head` parameter, `None` by default (zero behavior change for every existing caller/test). When present, audio synthesis is invoked at the same point as text generation, with the same two-checkpoint cancellation guard (before starting synthesis, and again after it returns) that the text path already used for `replan` races.

### 2.3 Result

Real Mimi weights (via `moshi.models.loaders.get_mimi`, restricted to `num_codebooks=1`), real Qwen3-1.7B, latency sweep 3000/1500/750/300 ms plus a no-tool scenario — 5/5 runs passed:

| Check | Result |
|---|---|
| `dependent_chunks_only_speak_confirmed_facts` (existing, text) | True, all runs |
| `dependent_chunk_audio_never_buffered_before_facts` | True, all runs |
| `audio_generated_for_every_played_chunk` | True, all runs |

No code change was needed to `ChunkState`'s transition table. This is the one clean, unambiguous "Continue" result in this report (spec.md §20).

---

## 3. Stage 5 — Finding a Teacher

### 3.1 Motivation

The next open question (Level B, `spec.md` §10) was whether a Voice Head could learn to produce audio from a Speaker's internal hidden state instead of its decoded text — an architectural precondition for lookahead to extend past the text boundary. This requires a teacher: some real system that can turn a piece of text into real Mimi codec tokens, so a hidden-state-conditioned model has something to imitate.

### 3.2 Kyutai Moshi is not a text-to-speech API

The obvious candidate, `moshi` (the package this project already depends on for the Mimi codec), is built around a full-duplex **conversational** loop — its own demo client feeds it live audio and reads live audio back, not "text in, tokens out." No one on this project had verified which, if any, of its public entry points could be repurposed as a teacher, so guessing a call chain and building a training pipeline around it would have risked a large investment on a wrong assumption.

Instead of guessing, a discovery script (`aether/experiments/spike_moshi_teacher.py`) was run three times, each cheap (load + introspect, no training):

1. **First run.** Confirmed `moshi.models.loaders.CheckpointInfo.from_hf_repo(...)` and `.get_mimi(...)` work as expected. The guessed method name for loading the conversational LM, `get_moshi_lm`, does not exist as a `CheckpointInfo` method — it turned out to exist only as an unrelated module-level function in `loaders`. The correct method is `.get_moshi(device=...)`.
2. **Second run**, with the corrected method name: `checkpoint_info.get_moshi` succeeded, and separately, a set of candidate repo IDs for a dedicated TTS checkpoint was probed. `kyutai/tts-1.6b-en_fr` resolved successfully with `tts_config` populated and `model_type: "tts"` — a real, dedicated Delayed-Streams-Modeling text-to-speech checkpoint, not the conversational model repurposed.
3. **Third run.** Rather than guess the TTS checkpoint's call sequence from class signatures, the script searched the installed `moshi` package for modules matching `tts`/`client`/`generate`/`script` and found `moshi.models.tts` and `moshi.run_tts` — a working reference CLI script. `inspect.getsource()` on `moshi.run_tts`, `TTSModel`, `get_default_tts_model`, and `script_to_entries` was dumped directly into the report, giving the real call sequence as source code rather than an inference from a signature.

The confirmed sequence (mirrored exactly in `generate_teacher_tokens`, `aether/experiments/colab_stage5.py`):

```python
checkpoint_info = CheckpointInfo.from_hf_repo("kyutai/tts-1.6b-en_fr")
tts_model = TTSModel.from_checkpoint_info(checkpoint_info, voice_repo=..., n_q=32, device=..., dtype=torch.bfloat16)
entries = tts_model.prepare_script([text], padding_between=1)
attributes = tts_model.make_condition_attributes(voices, cfg_coef_conditioning)
result = tts_model.generate([entries], [attributes], prefixes=prefixes, cfg_is_no_prefix=..., cfg_is_no_text=...)
frames = torch.cat(result.frames, dim=-1)  # (batch, 1 + n_q, time); row 0 is a text-token channel, rows 1: are the n_q Mimi codebooks
```

This confirmed `lm_config` also revealed `card: 2048` (Mimi's codebook cardinality, matching this project's existing `vocab_size=2048` default) and a `conditioners`/`fuser` block (`speaker_wavs`, `cfg`) — Kyutai's own conditioning framework, not used directly here but confirming the checkpoint's generation is conditionable in the way expected.

### 3.3 Three further defects, found and fixed on real hardware

1. **Environment: torch/torchvision ABI mismatch.** Installing the `audio` extra (`moshi`) pulled a newer `torch` without touching Colab's preinstalled `torchvision`, breaking its ABI (`operator torchvision::nms does not exist`) and — via `transformers`' unrelated vision-utils import path — masking itself as `ModuleNotFoundError: Could not import module 'Qwen3ForCausalLM'`. Fixed by reinstalling `torch`/`torchvision`/`torchaudio` together as one explicit step in the Colab notebook, immediately after the extras install.
2. **Voice resolution picked a cached embedding, not audio.** The Kyutai voice repository (`kyutai/tts-voices`) contains both `<voice>.wav` files and `<voice>.wav.<hash>@<n>.safetensors` siblings (precomputed speaker embeddings cached per model checkpoint). An early voice-selection heuristic matched any of `.wav`/`.pt`/`.safetensors` and picked the cached-embedding file; `TTSModel.get_prefix()` then called `sphn.read()` on it expecting real audio and failed (`ValueError: ... end of stream`). Fixed by restricting the heuristic to `.wav` only.
3. **An inverted conditional, not an environment problem.** Even after fixing (2), the same crash recurred. The actual defect was in this project's own code: `moshi/run_tts.py` only builds `prefixes` (and calls `get_prefix`) when the model is **not** multi-speaker; this repository's first version did the opposite, calling `get_prefix()` precisely when `multi_speaker` was `True` (the case that applies here) — a case Kyutai's own reference script never exercises that way. Voice conditioning for a multi-speaker checkpoint is meant to flow only through `make_condition_attributes`, where the cached-embedding path is exactly correct. Fixed by mirroring the reference script's condition exactly.
4. **Device mismatch.** `full_frames` (the concatenated teacher output) was moved to CPU when built; `mimi.decode()` runs on `args.tts_device` (a GPU in the target environment). Fixed by moving each decode slice to the codec's device before calling `decode()`.

### 3.4 A methodology defect in how results were judged, not just in the code

The first working end-to-end run (single codebook: `HiddenStateVoiceHead`, predicting only Mimi's codebook 0) reported `checks.train_loss_decreased: True` (loss 7.86 → 0.009) and a held-out codebook-agreement observation (6–40% across repeated runs, against a ~0.05% chance rate for a 2048-way vocabulary). To let a human actually judge the result instead of reading a loss number, the runner decoded audio two ways per phrase: the trained head's own codebook 0 (`predicted`) and the teacher's own codebook 0 (`teacher-codebook0-only`), both restricted to a single codebook.

Both were indistinguishable noise. This was not a broken pipeline — it was the wrong listening test. Mimi's architecture deliberately separates semantic content (codebook 0, the only channel this stage's head predicted) from acoustic detail (codebooks 1-31); a real, fully-formed teacher utterance decoded through codebook 0 alone is expected to sound like noise. Judging anything by ear from a single codebook was never going to be informative, for either the trained model or the ground truth.

The decode step was rewritten to reconstruct through the **real teacher's own 32 codebooks** (`teacher_voice_full`) alongside a **hybrid** — the teacher's codebooks 1-31 with codebook 0 swapped for the trained head's prediction (`hybrid_our_codebook0`). On train-set phrases, these were **byte-identical** (verified via MD5, not just "sounded the same"). This is not a bug: at a per-token cross-entropy of 0.009, greedy decoding reproduces the teacher's exact codebook-0 sequence — the head had memorized 20 examples, which the loss curve already implied. The hybrid-vs-teacher comparison on train phrases was therefore uninformative by construction, a second methodological mistake (testing the subset where the answer was already known instead of the subset that could actually be surprising).

A further correction was needed even for the *held-out* comparison: a "hybrid" (teacher's 31 real codebooks + our 1 predicted codebook) is still 31/32 the teacher's own audio. It answers "does codebook 0 matter perceptually", not "can this system speak on its own" — the actual question. The decode step was rewritten a final time to generate audio with **zero** channels borrowed from the teacher at inference (§4).

### 3.5 Confirming genuine (non-memorized) difference

Before trusting the final standalone-generation result, `teacher_voice_full` and `our_model_standalone` were compared by MD5 for both train and held-out phrases. All differed. This rules out a repeat of §3.4's memorization confound for the final architecture and confirms the model is producing its own, independently-computed output — the comparison that was actually asked for.

---

## 4. Stage 5 (final) — Standalone Multi-Codebook Generation

### 4.1 Architecture

`MultiCodebookVoiceHead` (`aether/model/voice_head.py`) predicts all `n_q=32` Mimi codebooks from a hidden state:

- Codebook 0 drives the causal, autoregressive timing exactly as in the single-codebook `HiddenStateVoiceHead` — the same projector-plus-transformer-body design, teacher-forced during training, greedily decoded at inference.
- Codebooks 1-31 are predicted **in parallel**, from the same per-frame hidden state, by 31 independent linear heads.

This is a deliberate simplification of Kyutai's real Depth Transformer, which conditions each codebook on the others within the same frame (a genuine sequential dependency). That inter-codebook consistency is real signal this design gives up, in exchange for a module small and simple enough to train from scratch in one Colab session rather than a multi-week architecture project.

### 4.2 Training

- Data: the same 20 English phrases (train) / 4 phrases (held-out) as the single-codebook stage, reusing `training/datasets.py`. Teacher tokens now cover all 32 codebooks (previously only codebook 0 was extracted; the data was already present in `generate_teacher_tokens`'s `full_frames`, just discarded).
- Loss: cross-entropy summed across all 32 codebook channels (`MultiCodebookVoiceHead.compute_training_loss`).
- `training/trainer.py`'s full-batch Adam loop needed no changes — it is agnostic to whether `target_tokens` is a flat per-frame list (1 codebook) or a nested per-frame list (32 codebooks); `torch.tensor(...)` infers the batch shape either way.

### 4.3 Results

| Run | Scope | Train loss (first → last) | Held-out codebook-0 agreement |
|---|---|---:|---|
| Single-codebook (representative) | codebook 0 only | 7.86 → 0.009 | 0.02–0.40 (varied by run) |
| Multi-codebook (final) | all 32 codebooks | 7.79 → 0.087 | 0.04–0.10 |

The multi-codebook loss floor is not directly comparable to the single-codebook one — it is averaged over 32 channels of plausibly different inherent predictability (codebook 0 is the "easy", semantic one; 1-31 are acoustic detail, likely harder to predict independently from a static hidden state with no sequential conditioning). `checks.train_loss_decreased` passed in every run.

**Listening result** (Karl, all 24 phrases, `our_model_standalone` vs `teacher_voice_full`): a recognizably similar voice; intelligible in places; some words drop out or are lost in others. Byte-level comparison (§3.5) confirms this is genuine model output, not a memorized or borrowed reconstruction.

### 4.4 Interpretation

The word-dropout is a plausible, expected symptom of §4.1's architectural simplification: where the 31 independent acoustic heads happen to disagree with each other in a way a real (sequentially-conditioned) Depth Transformer would not, the decoded audio can lose coherence at that frame. Two competing explanations were not yet separated in this report: the 20-example training set is also very small (more a memorization set than a training set, per §3.4's finding on the single-codebook stage), so underfitting from data scarcity is at least as plausible an explanation as the architecture's independence assumption. Distinguishing these is Stage 6's first order of business (see `handoff-6-stage.md`, not tracked in this repository — an internal roadmap draft, not yet a decided plan).

---

## 5. Decision (spec.md §20)

- **Stage 4 (audio commit horizon): Continue.** Unambiguous — the existing safety machinery generalizes to audio with zero modification, on real weights, across a full latency sweep.
- **Stage 5 (standalone hidden-state Voice Head): Continue, with Refine flagged.** The mechanism works: a from-scratch model conditioned only on hidden state (never text) produces audio recognizably related to real speech content, with confirmed non-memorized, independently-computed output. Quality is not yet where Karl wants it ("a proper head", his words) — whether the next lever is more training data or inter-codebook conditioning is the open question for Stage 6, not yet resolved by an experiment.

---

## 6. Next steps

See `handoff-6-stage.md` (repository-local, not tracked in git — internal scratch planning, not project history) for the detailed Stage 6 discussion. Headline candidates, none yet decided: an objective (ASR-based) quality metric to replace ear-judging at scale; scaling the English training set past 20 examples (cheap — the teacher is synthetic); and, if that alone doesn't resolve the word-dropout, real inter-codebook conditioning. Kazakh (a real, self-recorded corpus) is explicitly deferred until the English head is solid, not dropped.
