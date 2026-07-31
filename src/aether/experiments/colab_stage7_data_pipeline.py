"""Stage 7, part 1 -- resumable data generation for the real Depth
Transformer Voice Head training run.

Generates, for ~10,000 English phrases: a Qwen3 hidden state
(`SharedLLMBackbone.encode_hidden_state`) and the real Kyutai teacher's
tokens for every Mimi codebook (`aether.audio.teacher_tts`). Writes one
JSONL record per phrase to a cache file -- intended to live on Google
Drive (`--cache-path /content/drive/MyDrive/...`), not ephemeral Colab
storage, specifically so an interrupted multi-hour/multi-day run can be
resumed by just re-running this script: already-cached phrase_ids are
read back at startup and skipped, and the cache file is flushed after
every batch, not just at the end.

This is a data-preparation step; `colab_stage7_train.py` (not this file)
trains `DepthTransformerVoiceHead` on the cache this script produces.
"""

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

from aether.audio.teacher_tts import generate_teacher_tokens_for_batch, load_tts_model
from aether.experiments.colab_stage1 import environment_report, write_json
from aether.model.llm_backbone import LLMBackboneConfig, SharedLLMBackbone
from aether.training.datasets_large import generate_phrases
from aether.training.large_scale_trainer import format_duration


def load_cached_phrase_ids(cache_path: Path) -> Set[str]:
    if not cache_path.exists():
        return set()
    done: Set[str] = set()
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Tolerate a truncated last line from an interrupted write --
                # that one phrase just gets regenerated on this run.
                continue
            done.add(record["phrase_id"])
    return done


async def run_pipeline(args: argparse.Namespace, output_dir: Path) -> int:
    cache_path = Path(args.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "experiment": "voice_head_stage7_data_pipeline",
        "scope_note": (
            "Data generation only -- no training here. Resumable: re-running this "
            "script skips phrase_ids already present in --cache-path."
        ),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "tts_hf_repo": args.tts_hf_repo,
        "cache_path": str(cache_path),
        "phrase_count": args.phrase_count,
        "batch_size": args.batch_size,
        "environment": environment_report(),
        "status": "started",
    }
    write_json(output_dir / "report.json", report)

    try:
        phrases = generate_phrases(args.phrase_count, seed=args.seed)
        already_done = load_cached_phrase_ids(cache_path)
        report["already_cached_at_start"] = len(already_done)
        remaining = [p for p in phrases if p.phrase_id not in already_done]
        report["remaining_at_start"] = len(remaining)
        write_json(output_dir / "report.json", report)

        if not remaining:
            report["status"] = "passed"
            report["message"] = "nothing to do -- cache already complete"
            return 0

        backbone = SharedLLMBackbone(
            LLMBackboneConfig(
                model_path=args.model,
                device_map=args.device_map,
                dtype=args.dtype,
                allow_download=args.allow_download,
                enable_thinking=False,
            )
        )
        load_started = time.monotonic_ns()
        backbone.load()
        report["qwen_load_ms"] = (time.monotonic_ns() - load_started) / 1_000_000

        tts_load_started = time.monotonic_ns()
        tts_model = load_tts_model(args.tts_hf_repo, args.voice_repo, args.tts_nq, args.tts_device)
        report["tts_load_ms"] = (time.monotonic_ns() - tts_load_started) / 1_000_000
        write_json(output_dir / "report.json", report)

        total_batches = (len(remaining) + args.batch_size - 1) // args.batch_size
        print(
            f"[data] {len(remaining)} phrases remaining ({len(already_done)} already cached), "
            f"{total_batches} batches of {args.batch_size}"
        )
        processed_this_run = 0
        run_started = time.monotonic()
        with cache_path.open("a", encoding="utf-8") as cache_handle:
            for batch_index, batch_start in enumerate(range(0, len(remaining), args.batch_size)):
                batch = remaining[batch_start : batch_start + args.batch_size]
                batch_started = time.monotonic_ns()

                token_grids = generate_teacher_tokens_for_batch(
                    tts_model, args.voice_repo, [p.text for p in batch],
                    args.max_audio_tokens, args.tts_nq,
                )
                for phrase, tokens in zip(batch, token_grids):
                    hidden_state = backbone.encode_hidden_state(phrase.text)
                    record = {
                        "phrase_id": phrase.phrase_id,
                        "text": phrase.text,
                        "hidden_state": hidden_state,
                        "teacher_tokens": tokens,
                    }
                    cache_handle.write(json.dumps(record) + "\n")
                cache_handle.flush()

                processed_this_run += len(batch)
                batch_ms = (time.monotonic_ns() - batch_started) / 1_000_000
                elapsed = time.monotonic() - run_started
                phrases_per_sec = processed_this_run / elapsed if elapsed > 0 else 0.0
                remaining_after = len(remaining) - processed_this_run
                eta_seconds = remaining_after / phrases_per_sec if phrases_per_sec > 0 else None
                report["processed_this_run"] = processed_this_run
                report["remaining_now"] = remaining_after
                report["last_batch_ms"] = batch_ms
                write_json(output_dir / "report.json", report)
                eta_text = format_duration(eta_seconds) if eta_seconds is not None else "unknown"
                print(
                    f"[data] batch {batch_index + 1}/{total_batches}: "
                    f"{processed_this_run}/{len(remaining)} this run "
                    f"({batch_ms:.0f} ms/batch, {phrases_per_sec:.2f} phrases/s) "
                    f"-- cache now has {len(already_done) + processed_this_run}/{len(phrases)} "
                    f"-- elapsed {format_duration(elapsed)}, eta {eta_text}"
                )

        report["status"] = "passed"
        return 0
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
    parser.add_argument("--max-audio-tokens", type=int, default=50)
    parser.add_argument("--phrase-count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--output-dir", default="artifacts/colab-stage7-data")
    parser.add_argument(
        "--cache-path",
        default="artifacts/colab-stage7-data/cache.jsonl",
        help="Point this at Google Drive (e.g. /content/drive/MyDrive/aether/stage7_cache.jsonl) "
        "so a multi-day run survives Colab disconnects.",
    )
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

    exit_code = asyncio.run(run_pipeline(args, output_dir))
    print(f"AETHER_STAGE7_DATA_STATUS={'PASSED' if exit_code == 0 else 'FAILED'}")
    print(f"AETHER_STAGE7_DATA_REPORT={output_dir / 'report.json'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
