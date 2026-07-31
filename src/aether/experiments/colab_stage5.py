"""Stage 5 -- minimal hidden-state-conditioned Voice Head training probe.

Confirmed by the Moshi-teacher feasibility spikes (see docs/colab.md):
`kyutai/tts-1.6b-en_fr`, loaded via `moshi.models.loaders.CheckpointInfo` and
`moshi.models.tts.TTSModel`, is a real text-to-speech model that produces
Mimi codebook tokens for arbitrary text -- confirmed from the real
`moshi/run_tts.py` source, not guessed.

This stage:
  1. Extracts a Qwen3 hidden state for each of ~24 short English phrases
     (`aether.model.llm_backbone.SharedLLMBackbone.encode_hidden_state`).
  2. Generates the teacher's tokens for EVERY Mimi codebook (not just
     codebook 0) for the same phrases via `TTSModel` (frozen, pretrained,
     real weights).
  3. Trains a `MultiCodebookVoiceHead` (small, from-scratch) to predict
     all of those codebooks from the hidden state alone -- `text` is never
     given to it (see the class docstring). Codebook 0 drives the causal
     timing; codebooks 1-31 are predicted in parallel from the same
     per-frame state, a deliberate simplification of Kyutai's real,
     sequential Depth Transformer.
  4. Decodes audio generated ENTIRELY by this head (no channel borrowed
     from the teacher) next to the real teacher's own voice, for both
     train and held-out phrases -- a hybrid (teacher's codebooks 1-31 +
     our codebook 0) was tried first and rejected: on train phrases it was
     byte-identical to the teacher (expected memorization at this loss,
     not informative), and even on held-out phrases 31 of 32 channels
     would still be the teacher's, not evidence of what this head can do
     alone.
  5. Reports the loss curve and a soft held-out token-agreement observation.

Scope, deliberately: this is a sanity check that the mechanism is
learnable (does loss drop from a random start, does standalone-generated
audio carry any recognizable content on held-out phrases?), not a claim
about production speech quality, prosody, or readiness for any language
other than English. See docs/plan.md Phase C and spec.md Level B for the
broader context.
"""

import argparse
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from aether.experiments.colab_stage1 import environment_report, write_json
from aether.experiments.colab_stage4 import write_wav
from aether.model.llm_backbone import LLMBackboneConfig, SharedLLMBackbone
from aether.model.voice_head import MultiCodebookVoiceHead, MultiCodebookVoiceHeadConfig
from aether.training.datasets import HELD_OUT_PHRASES, PHRASES, TRAIN_PHRASES, Phrase
from aether.training.trainer import TrainingExample, train_hidden_state_voice_head


def resolve_default_voice(voice_repo: str) -> str:
    """Picks the first plausible voice file in the Kyutai voice repo.

    Not a claim that this is "the" default voice -- just a deterministic,
    inspectable choice so this runner doesn't need a human to name one.

    Only `.wav` entries qualify. A first real run found this repo also
    contains `<voice>.wav.<hash>@<n>.safetensors` siblings -- precomputed
    embeddings cached for a specific model checkpoint, not raw audio.
    `TTSModel.get_prefix()` calls `sphn.read()` on whatever path it's given,
    which needs actual audio; picking a `.safetensors` sibling by accident
    fails with `ValueError: ... end of stream`. The `TTSRequest` docstring
    (see moshi/run_tts.py) confirms a "voice name" is just a file that
    exists in the voice repository, so a plain `.wav` relative path is the
    right thing to pass through unchanged.
    """
    from huggingface_hub import list_repo_files

    files = list_repo_files(voice_repo)
    candidates = [f for f in files if f.endswith(".wav")]
    if not candidates:
        raise RuntimeError(f"no .wav voice files found in {voice_repo}: {files[:20]}")
    return sorted(candidates)[0]


def extract_all_codebook_tokens(
    frames: Any,
    idx: int,
    delay_steps: int,
    end_step: Optional[int],
    max_audio_tokens: int,
    num_codebooks: int,
) -> List[List[int]]:
    """Per-frame token vectors for every codebook (not just codebook 0).

    Returns a ``(max_audio_tokens, num_codebooks)`` nested list -- the real
    training target for `MultiCodebookVoiceHead`, which predicts standalone
    audio with no channel borrowed from the teacher at inference time.
    """
    total_len = frames.shape[-1]
    start = delay_steps
    stop = min(total_len, start + max_audio_tokens)
    if end_step is not None:
        stop = min(stop, delay_steps + end_step)
    # frames[idx, 1:1+num_codebooks, start:stop] -> (num_codebooks, T); this
    # transposes to (T, num_codebooks), one token vector per frame.
    codebook_slice = frames[idx, 1 : 1 + num_codebooks, start:stop]
    codes = [[int(value) for value in frame] for frame in codebook_slice.transpose(0, 1).tolist()]
    if len(codes) < max_audio_tokens:
        pad_frame = codes[-1] if codes else [0] * num_codebooks
        codes = codes + [pad_frame] * (max_audio_tokens - len(codes))
    else:
        codes = codes[:max_audio_tokens]
    return codes


def generate_teacher_tokens(
    phrases: List[Phrase],
    tts_hf_repo: str,
    voice_repo: str,
    tts_nq: int,
    device: str,
    max_audio_tokens: int,
    num_codebooks: int,
    diagnostics: Optional[Dict[str, Any]] = None,
    on_diagnostics_update: Optional[Any] = None,
) -> Dict[str, Any]:
    """Returns {"tokens": {phrase_id: [[int,...] * num_codebooks] * T}, "diagnostics": {...}}.

    Follows `moshi/run_tts.py`'s real call sequence (confirmed via source
    inspection, see the spike reports), not a guessed API.

    ``diagnostics``, if given, is filled in-place and ``on_diagnostics_update``
    (if given) is called after each field is added -- so a caller can persist
    partial diagnostics to disk even if a later step in this function raises,
    instead of losing everything a failed run learned.
    """
    import torch

    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import TTSModel

    if diagnostics is None:
        diagnostics = {}

    def _update(key: str, value: Any) -> None:
        diagnostics[key] = value
        if on_diagnostics_update is not None:
            on_diagnostics_update()

    checkpoint_info = CheckpointInfo.from_hf_repo(tts_hf_repo)
    tts_model = TTSModel.from_checkpoint_info(
        checkpoint_info, voice_repo=voice_repo, n_q=tts_nq, device=device, dtype=torch.bfloat16
    )
    _update("multi_speaker", bool(tts_model.multi_speaker))
    _update("valid_cfg_conditionings", bool(tts_model.valid_cfg_conditionings))
    _update("delay_steps", int(tts_model.delay_steps))

    cfg_coef_conditioning = None
    if tts_model.valid_cfg_conditionings:
        cfg_coef_conditioning = tts_model.cfg_coef
        tts_model.cfg_coef = 1.0
        cfg_is_no_text = False
        cfg_is_no_prefix = False
    else:
        cfg_is_no_text = True
        cfg_is_no_prefix = True

    # Mirrors moshi/run_tts.py exactly: `prefixes` (an audio continuation,
    # via `get_prefix`) is only a thing for single-speaker models. For a
    # multi_speaker model (this one), voice conditioning happens purely
    # through `make_condition_attributes(voices, ...)` -- `get_voice_path`'s
    # return value there is meant to be resolved to a cached speaker
    # embedding (a `.safetensors` file), which is exactly what broke when
    # the first version of this function called `get_prefix()` on it too
    # (`get_prefix` expects actual audio, not an embedding cache).
    voice_name = None
    prefixes: Optional[List[Any]] = None
    if not tts_model.multi_speaker:
        prefixes = []
    if tts_model.multi_speaker:
        voice_name = resolve_default_voice(voice_repo)
        _update("voice_name", voice_name)

    all_entries = []
    all_attributes = []
    for phrase in phrases:
        entries = tts_model.prepare_script([phrase.text], padding_between=1)
        all_entries.append(entries)
        if tts_model.multi_speaker:
            voices = [tts_model.get_voice_path(voice_name)]
        else:
            voices = []
        all_attributes.append(tts_model.make_condition_attributes(voices, cfg_coef_conditioning))
        if prefixes is not None:
            prefix_path = tts_model.get_voice_path(voice_name)
            prefixes.append(tts_model.get_prefix(prefix_path))

    result = tts_model.generate(
        all_entries, all_attributes, prefixes=prefixes,
        cfg_is_no_prefix=cfg_is_no_prefix, cfg_is_no_text=cfg_is_no_text,
    )
    frames = torch.cat(result.frames, dim=-1).cpu()
    _update("frames_shape", list(frames.shape))

    tokens: Dict[str, List[List[int]]] = {}
    for idx, phrase in enumerate(phrases):
        end_step = result.end_steps[idx]
        tokens[phrase.phrase_id] = extract_all_codebook_tokens(
            frames, idx, tts_model.delay_steps, end_step, max_audio_tokens, num_codebooks
        )
    return {
        "tokens": tokens,
        "diagnostics": diagnostics,
        # Kept alive and returned so the caller can also decode the
        # teacher's own real-voice reference for comparison.
        "tts_model": tts_model,
        "full_frames": frames,
        "end_steps": result.end_steps,
        "delay_steps": tts_model.delay_steps,
    }


def extract_hidden_states(backbone: SharedLLMBackbone, phrases: List[Phrase]) -> Dict[str, List[float]]:
    return {phrase.phrase_id: backbone.encode_hidden_state(phrase.text) for phrase in phrases}


async def run_experiment(args: argparse.Namespace, output_dir: Path) -> int:
    report: Dict[str, Any] = {
        "experiment": "voice_head_stage5_hidden_state_training_probe",
        "scope_note": (
            "Sanity check only: does a tiny hidden-state-conditioned Voice Head learn "
            "anything from ~20 examples? `held_out_wav_files` contains audio generated "
            "ENTIRELY by this head (no channel borrowed from the teacher) for phrases it "
            "never trained on -- that is the real test, not the loss curve or the "
            "train_wav_files (which are expected to sound close to the teacher; the head "
            "has effectively memorized those). Not a claim about production speech "
            "quality, prosody, or readiness for any language beyond English."
        ),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "tts_hf_repo": args.tts_hf_repo,
        "voice_repo": args.voice_repo,
        "train_phrase_ids": [p.phrase_id for p in TRAIN_PHRASES],
        "held_out_phrase_ids": [p.phrase_id for p in HELD_OUT_PHRASES],
        "environment": environment_report(),
        "status": "started",
    }
    write_json(output_dir / "report.json", report)

    backbone = SharedLLMBackbone(
        LLMBackboneConfig(
            model_path=args.model,
            device_map=args.device_map,
            dtype=args.dtype,
            allow_download=args.allow_download,
            enable_thinking=False,
        )
    )
    try:
        load_started = time.monotonic_ns()
        backbone.load()
        report["model_load_ms"] = (time.monotonic_ns() - load_started) / 1_000_000
        report["hidden_state_dim"] = backbone.hidden_size

        hidden_states = extract_hidden_states(backbone, PHRASES)
        report["hidden_states_extracted"] = len(hidden_states)
        write_json(output_dir / "report.json", report)

        teacher_diagnostics: Dict[str, Any] = {}
        report["teacher_diagnostics"] = teacher_diagnostics

        teacher = generate_teacher_tokens(
            PHRASES, args.tts_hf_repo, args.voice_repo, args.tts_nq, args.tts_device,
            args.max_audio_tokens, args.tts_nq,
            diagnostics=teacher_diagnostics,
            on_diagnostics_update=lambda: write_json(output_dir / "report.json", report),
        )
        report["teacher_diagnostics"] = teacher["diagnostics"]
        teacher_tokens = teacher["tokens"]
        write_json(output_dir / "report.json", report)

        head = MultiCodebookVoiceHead(
            MultiCodebookVoiceHeadConfig(
                hidden_state_dim=backbone.hidden_size,
                num_codebooks=args.tts_nq,
                vocab_size=args.vocab_size,
                max_audio_tokens=args.max_audio_tokens,
                device=args.voice_head_device,
            )
        )
        head.load()

        train_examples = [
            TrainingExample(p.phrase_id, hidden_states[p.phrase_id], teacher_tokens[p.phrase_id])
            for p in TRAIN_PHRASES
        ]
        held_out_examples = [
            TrainingExample(p.phrase_id, hidden_states[p.phrase_id], teacher_tokens[p.phrase_id])
            for p in HELD_OUT_PHRASES
        ]

        train_started = time.monotonic_ns()
        loss_curve = train_hidden_state_voice_head(head, train_examples, epochs=args.epochs, lr=args.lr)
        report["train_ms"] = (time.monotonic_ns() - train_started) / 1_000_000
        report["loss_curve"] = loss_curve

        # Soft, informal held-out observation -- NOT a pass/fail gate on 4
        # examples (spec.md §20: not statistically meaningful). Compares
        # each held-out phrase's greedy codebook-0 prediction against the
        # teacher's codebook-0 tokens, position agreement rate, purely as
        # an observation.
        held_out_agreement: Dict[str, float] = {}
        for example in held_out_examples:
            predicted_frames = head._forward_greedy_all_codebooks(  # noqa: SLF001 -- eval-only
                example.hidden_state
            )
            predicted_codebook0 = [frame[0] for frame in predicted_frames]
            target_codebook0 = [frame[0] for frame in example.target_tokens]
            matches = sum(1 for a, b in zip(predicted_codebook0, target_codebook0) if a == b)
            held_out_agreement[example.phrase_id] = matches / len(target_codebook0)
        report["held_out_token_agreement"] = held_out_agreement

        # Decode audio so a human can actually listen, not just read a loss
        # number. Two files per phrase:
        #   - "teacher_voice_full": the real Kyutai voice, all `num_codebooks`
        #     codebooks -- proves the pipeline produces genuine speech.
        #   - "our_model_standalone": audio generated ENTIRELY by this
        #     head (`_forward_greedy_all_codebooks`) -- zero channels
        #     borrowed from the teacher. This is the actual "does our
        #     system speak on its own" test; a hybrid (teacher codebooks
        #     1-31 + our codebook 0) is not a fair stand-in for it, since
        #     31 of 32 channels there would still be the teacher's.
        import torch

        teacher_tts_model = teacher["tts_model"]
        full_frames = teacher["full_frames"]
        end_steps = teacher["end_steps"]
        delay_steps = teacher["delay_steps"]
        phrase_index = {p.phrase_id: i for i, p in enumerate(PHRASES)}
        sample_rate = int(teacher_tts_model.mimi.sample_rate)
        num_codebooks = args.tts_nq

        wav_dir = output_dir / "wav"
        wav_dir.mkdir(parents=True, exist_ok=True)

        def decode_teacher_and_ours(examples: List[TrainingExample]) -> Dict[str, Dict[str, str]]:
            wav_files: Dict[str, Dict[str, str]] = {}
            for example in examples:
                idx = phrase_index[example.phrase_id]
                stop = full_frames.shape[-1]
                if end_steps[idx] is not None:
                    stop = min(stop, delay_steps + end_steps[idx])
                # frame[:, 1:1+num_codebooks] excludes the text-token channel
                # (index 0), same slice moshi/run_tts.py passes to
                # `mimi.decode()`. `full_frames` was moved to cpu when built
                # (generate_teacher_tokens); `mimi` lives on `args.tts_device`.
                teacher_codebooks = full_frames[
                    idx : idx + 1, 1 : 1 + num_codebooks, delay_steps:stop
                ].clone().to(args.tts_device)

                with torch.no_grad():
                    teacher_pcm = teacher_tts_model.mimi.decode(teacher_codebooks)
                teacher_path = wav_dir / f"{example.phrase_id}-teacher-voice-full.wav"
                write_wav(teacher_path, teacher_pcm[0, 0].cpu().numpy(), sample_rate)

                generated_frames = head._forward_greedy_all_codebooks(  # noqa: SLF001 -- eval-only
                    example.hidden_state
                )
                usable_len = min(teacher_codebooks.shape[-1], len(generated_frames))
                generated_tensor = torch.tensor(
                    generated_frames[:usable_len],
                    dtype=teacher_codebooks.dtype,
                    device=teacher_codebooks.device,
                ).transpose(0, 1).unsqueeze(0)  # (1, num_codebooks, T)
                with torch.no_grad():
                    ours_pcm = teacher_tts_model.mimi.decode(generated_tensor)
                ours_path = wav_dir / f"{example.phrase_id}-our-model-standalone.wav"
                write_wav(ours_path, ours_pcm[0, 0].cpu().numpy(), sample_rate)

                wav_files[example.phrase_id] = {
                    "teacher_voice_full": str(teacher_path),
                    "our_model_standalone": str(ours_path),
                }
            return wav_files

        report["train_wav_files"] = decode_teacher_and_ours(train_examples)
        write_json(output_dir / "report.json", report)
        # The actual test: phrases the head never saw during training.
        report["held_out_wav_files"] = decode_teacher_and_ours(held_out_examples)
        write_json(output_dir / "report.json", report)

        checks = {
            "train_loss_decreased": loss_curve[-1] < loss_curve[0],
        }
        report["checks"] = checks
        report["observations"] = {
            "loss_first": loss_curve[0],
            "loss_last": loss_curve[-1],
            "held_out_token_agreement": held_out_agreement,
        }
        report["passed"] = all(checks.values())
        report["status"] = "passed" if report["passed"] else "failed"
        return 0 if report["passed"] else 1
    except BaseException as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        (output_dir / "traceback.txt").write_text(report["traceback"], encoding="utf-8")
        return 1
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output_dir / "report.json", report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--tts-hf-repo", default="kyutai/tts-1.6b-en_fr")
    parser.add_argument("--voice-repo", default="kyutai/tts-voices")
    parser.add_argument("--tts-nq", type=int, default=32)
    parser.add_argument("--tts-device", default="cuda")
    parser.add_argument("--voice-head-device", default="cpu")
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--max-audio-tokens", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--output-dir", default="artifacts/colab-stage5")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if not args.allow_download:
        parser.error("pass --allow-download only in the target ML environment")
    return args


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    import asyncio

    exit_code = asyncio.run(run_experiment(args, output_dir))
    print(f"AETHER_STAGE5_STATUS={'PASSED' if exit_code == 0 else 'FAILED'}")
    print(f"AETHER_STAGE5_REPORT={output_dir / 'report.json'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
