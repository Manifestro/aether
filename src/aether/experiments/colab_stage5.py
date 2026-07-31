"""Stage 5 -- minimal hidden-state-conditioned Voice Head training probe.

Confirmed by the Moshi-teacher feasibility spikes (see docs/colab.md):
`kyutai/tts-1.6b-en_fr`, loaded via `moshi.models.loaders.CheckpointInfo` and
`moshi.models.tts.TTSModel`, is a real text-to-speech model that produces
Mimi codebook tokens for arbitrary text -- confirmed from the real
`moshi/run_tts.py` source, not guessed.

This stage:
  1. Extracts a Qwen3 hidden state for each of ~24 short English phrases
     (`aether.model.llm_backbone.SharedLLMBackbone.encode_hidden_state`).
  2. Generates the teacher's codebook-0 tokens for the same phrases via
     `TTSModel` (frozen, pretrained, real weights).
  3. Trains a `HiddenStateVoiceHead` (small, from-scratch) to predict the
     teacher's tokens from the hidden state alone -- `text` is never given
     to it (see `HiddenStateVoiceHead`'s docstring).
  4. Reports the loss curve and a soft held-out observation.

Scope, deliberately: this is a sanity check that the mechanism is
learnable (does loss drop from a random start?), not a claim about speech
quality, generalization, or readiness for any language other than English.
See docs/plan.md Phase C and spec.md Level B for the broader context.
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
from aether.model.voice_head import HiddenStateVoiceHead, HiddenStateVoiceHeadConfig
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


def extract_codebook0_tokens(
    frames: Any, idx: int, delay_steps: int, end_step: Optional[int], max_audio_tokens: int
) -> List[int]:
    total_len = frames.shape[-1]
    start = delay_steps
    stop = min(total_len, start + max_audio_tokens)
    if end_step is not None:
        stop = min(stop, delay_steps + end_step)
    codes = [int(code) for code in frames[idx, 1, start:stop].tolist()]
    if len(codes) < max_audio_tokens:
        pad_value = codes[-1] if codes else 0
        codes = codes + [pad_value] * (max_audio_tokens - len(codes))
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
    diagnostics: Optional[Dict[str, Any]] = None,
    on_diagnostics_update: Optional[Any] = None,
) -> Dict[str, Any]:
    """Returns {"tokens": {phrase_id: [int,...]}, "diagnostics": {...}}.

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

    tokens: Dict[str, List[int]] = {}
    for idx, phrase in enumerate(phrases):
        end_step = result.end_steps[idx]
        tokens[phrase.phrase_id] = extract_codebook0_tokens(
            frames, idx, tts_model.delay_steps, end_step, max_audio_tokens
        )
    return {
        "tokens": tokens,
        "diagnostics": diagnostics,
        # Kept alive and returned (not just the extracted codebook-0 ints)
        # so the caller can decode full 32-codebook audio -- codebook 0
        # alone is Mimi's semantic stream and is expected to sound like
        # noise on its own; that is not evidence of a broken pipeline, it
        # is evidence that single-codebook decode was the wrong listening
        # test. See run_experiment's wav-writing step.
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
            "anything from ~20 examples (does loss drop from a random start)? Not a "
            "claim about speech quality, generalization, or readiness for production "
            "or for any language beyond English."
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
            args.max_audio_tokens,
            diagnostics=teacher_diagnostics,
            on_diagnostics_update=lambda: write_json(output_dir / "report.json", report),
        )
        report["teacher_diagnostics"] = teacher["diagnostics"]
        teacher_tokens = teacher["tokens"]
        write_json(output_dir / "report.json", report)

        head = HiddenStateVoiceHead(
            HiddenStateVoiceHeadConfig(
                hidden_state_dim=backbone.hidden_size,
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
        # each held-out phrase's greedy prediction against its teacher
        # tokens, token-position agreement rate, purely as an observation.
        held_out_agreement: Dict[str, float] = {}
        for example in held_out_examples:
            predicted = head._forward_greedy(example.hidden_state)  # noqa: SLF001 -- eval-only introspection
            matches = sum(1 for a, b in zip(predicted, example.target_tokens) if a == b)
            held_out_agreement[example.phrase_id] = matches / len(example.target_tokens)
        report["held_out_token_agreement"] = held_out_agreement

        # Decode train-phrase audio so a human can actually listen, not just
        # read a loss number. A single-codebook decode (what an earlier
        # version of this script produced) is near-unintelligible even for
        # the *real* teacher -- Mimi splits semantic content (codebook 0)
        # from acoustic detail (codebooks 1-31) by design, so judging
        # anything by ear off codebook 0 alone is not a fair test. Instead:
        #   - "teacher_voice_full": the real Kyutai voice, all 32 codebooks
        #     -- proves the pipeline produces genuine intelligible speech
        #     at all.
        #   - "hybrid_our_codebook0": the same 32-codebook reconstruction,
        #     but with codebook 0 swapped for this trained head's greedy
        #     prediction. If the phrase is still recognizable, the head
        #     learned something about content from the hidden state; if it
        #     turns to mush, it didn't. This is the real test.
        import torch

        teacher_tts_model = teacher["tts_model"]
        full_frames = teacher["full_frames"]
        end_steps = teacher["end_steps"]
        delay_steps = teacher["delay_steps"]
        phrase_index = {p.phrase_id: i for i, p in enumerate(PHRASES)}
        sample_rate = int(teacher_tts_model.mimi.sample_rate)

        wav_dir = output_dir / "wav"
        wav_dir.mkdir(parents=True, exist_ok=True)

        train_wav_files: Dict[str, Dict[str, str]] = {}
        for example in train_examples:
            predicted = head._forward_greedy(example.hidden_state)  # noqa: SLF001 -- eval-only introspection
            idx = phrase_index[example.phrase_id]
            stop = full_frames.shape[-1]
            if end_steps[idx] is not None:
                stop = min(stop, delay_steps + end_steps[idx])
            # frame[:, 1:] excludes the text-token channel (index 0), same
            # slice moshi/run_tts.py passes to `mimi.decode()`; row 0 of
            # what remains is codebook 0, the one this head predicts.
            # `full_frames` was moved to cpu when built (generate_teacher_tokens);
            # `mimi` lives on `args.tts_device` -- move the slice to match.
            audio_codebooks = full_frames[idx : idx + 1, 1:, delay_steps:stop].clone().to(
                args.tts_device
            )

            with torch.no_grad():
                teacher_pcm = teacher_tts_model.mimi.decode(audio_codebooks)
            teacher_path = wav_dir / f"{example.phrase_id}-teacher-voice-full.wav"
            write_wav(teacher_path, teacher_pcm[0, 0].cpu().numpy(), sample_rate)

            hybrid_codebooks = audio_codebooks.clone()
            usable_len = min(hybrid_codebooks.shape[-1], len(predicted))
            hybrid_codebooks[:, 0, :usable_len] = torch.tensor(
                predicted[:usable_len], dtype=hybrid_codebooks.dtype, device=hybrid_codebooks.device
            )
            with torch.no_grad():
                hybrid_pcm = teacher_tts_model.mimi.decode(hybrid_codebooks)
            hybrid_path = wav_dir / f"{example.phrase_id}-hybrid-our-codebook0.wav"
            write_wav(hybrid_path, hybrid_pcm[0, 0].cpu().numpy(), sample_rate)

            train_wav_files[example.phrase_id] = {
                "teacher_voice_full": str(teacher_path),
                "hybrid_our_codebook0": str(hybrid_path),
            }
        report["train_wav_files"] = train_wav_files
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
