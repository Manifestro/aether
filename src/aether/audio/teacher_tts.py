"""Shared Kyutai TTS teacher client for large-scale (Stage 7+) data generation.

The exact call sequence here (`CheckpointInfo` -> `TTSModel.from_checkpoint_info`
-> `prepare_script`/`make_condition_attributes`/`generate`) was confirmed via
`moshi/run_tts.py` source inspection (see the Moshi-teacher feasibility spike
reports and `docs/reports/technical_report_03.md` §3.2-3.3), not guessed --
including two real bugs found there: the `prefixes`/`multi_speaker` condition
must match the reference exactly (inverted once, causing a real failure), and
voice resolution must pick a `.wav` file, not a `.safetensors` embedding cache.

`aether/experiments/colab_stage5.py` has its own, separate copy of this same
logic (written first, kept as-is deliberately -- Stage 5 is a closed,
already-validated result; this module is not a drop-in refactor of it, to
avoid risking that closed stage's behavior for this new, larger pipeline).
"""

from typing import Any, Dict, List, Optional


def resolve_default_voice(voice_repo: str) -> str:
    """Picks the first `.wav` voice file in the Kyutai voice repo.

    Only `.wav` entries qualify -- the repo also contains
    `<voice>.wav.<hash>@<n>.safetensors` siblings (precomputed embeddings
    cached per model checkpoint, not raw audio); `TTSModel.get_prefix()`
    needs real audio and fails on those (`ValueError: ... end of stream`).
    """
    from huggingface_hub import list_repo_files

    files = list_repo_files(voice_repo)
    candidates = [f for f in files if f.endswith(".wav")]
    if not candidates:
        raise RuntimeError(f"no .wav voice files found in {voice_repo}: {files[:20]}")
    return sorted(candidates)[0]


def load_tts_model(tts_hf_repo: str, voice_repo: str, tts_nq: int, device: str) -> Any:
    import torch

    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import TTSModel

    checkpoint_info = CheckpointInfo.from_hf_repo(tts_hf_repo)
    tts_model = TTSModel.from_checkpoint_info(
        checkpoint_info, voice_repo=voice_repo, n_q=tts_nq, device=device, dtype=torch.bfloat16
    )
    return tts_model


def extract_codebook_tokens(
    frames: Any,
    idx: int,
    delay_steps: int,
    end_step: Optional[int],
    max_audio_tokens: int,
    num_codebooks: int,
) -> List[List[int]]:
    """Per-frame token vectors for every codebook, `(max_audio_tokens, num_codebooks)`."""
    total_len = frames.shape[-1]
    start = delay_steps
    stop = min(total_len, start + max_audio_tokens)
    if end_step is not None:
        stop = min(stop, delay_steps + end_step)
    codebook_slice = frames[idx, 1 : 1 + num_codebooks, start:stop]
    codes = [[int(value) for value in frame] for frame in codebook_slice.transpose(0, 1).tolist()]
    if len(codes) < max_audio_tokens:
        pad_frame = codes[-1] if codes else [0] * num_codebooks
        codes = codes + [pad_frame] * (max_audio_tokens - len(codes))
    else:
        codes = codes[:max_audio_tokens]
    return codes


def generate_teacher_tokens_for_batch(
    tts_model: Any,
    voice_repo: str,
    phrase_texts: List[str],
    max_audio_tokens: int,
    num_codebooks: int,
) -> List[List[List[int]]]:
    """Generates teacher tokens for one batch of phrase texts, in one
    `tts_model.generate()` call. Returns one `(max_audio_tokens,
    num_codebooks)` token grid per input phrase, same order as
    `phrase_texts`. Caller is responsible for keeping batches a sane size
    (Kyutai's own `run_tts.py` defaults to 32) -- this function does not
    chunk internally.
    """
    import torch

    cfg_coef_conditioning = None
    if tts_model.valid_cfg_conditionings:
        cfg_coef_conditioning = tts_model.cfg_coef
        tts_model.cfg_coef = 1.0
        cfg_is_no_text = False
        cfg_is_no_prefix = False
    else:
        cfg_is_no_text = True
        cfg_is_no_prefix = True

    voice_name = None
    prefixes: Optional[List[Any]] = None
    if not tts_model.multi_speaker:
        prefixes = []
    # (see module docstring: this condition must match moshi/run_tts.py
    # exactly -- prefixes/get_prefix is a single-speaker-only concept)

    all_entries = []
    all_attributes = []
    for text in phrase_texts:
        entries = tts_model.prepare_script([text], padding_between=1)
        all_entries.append(entries)
        if tts_model.multi_speaker:
            if voice_name is None:
                voice_name = resolve_default_voice(voice_repo)
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

    return [
        extract_codebook_tokens(
            frames, idx, tts_model.delay_steps, result.end_steps[idx], max_audio_tokens, num_codebooks
        )
        for idx in range(len(phrase_texts))
    ]
