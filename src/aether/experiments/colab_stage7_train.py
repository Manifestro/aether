"""Stage 7, part 2 -- trains `DepthTransformerVoiceHead` on the cache
`colab_stage7_data_pipeline.py` produced.

Real scale (thousands of examples, not Stage 5's ~20), real architecture
(temporal + depth transformer, not Stage 5's parallel independent heads),
checkpointed so a multi-day run survives interruption -- point
`--checkpoint-path` and `--loss-log-path` at Google Drive, and re-running
this exact command resumes automatically instead of starting over.

Does not touch any pretrained/downloaded model -- the head here is a
from-scratch architecture (random init until trained), so there is no
`--allow-download` gate; the only input is the local/Drive JSONL cache.
"""

import argparse
import json
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from aether.experiments.colab_stage1 import environment_report, write_json
from aether.model.depth_voice_head import DepthTransformerVoiceHead, DepthTransformerVoiceHeadConfig
from aether.training.large_scale_trainer import (
    TrainingRecord,
    load_training_records,
    train_large_scale,
)


def split_records(
    records: List[TrainingRecord], val_fraction: float, seed: int
) -> "tuple[List[TrainingRecord], List[TrainingRecord]]":
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_fraction))
    return shuffled[:-val_count], shuffled[-val_count:]


def run_training(args: argparse.Namespace, output_dir: Path) -> int:
    report: Dict[str, Any] = {
        "experiment": "voice_head_stage7_depth_transformer_training",
        "scope_note": (
            "Real-scale training run, checkpointed for multi-day resumability. "
            "See docs/reports/technical_report_03.md for why the Depth Transformer "
            "(real inter-codebook conditioning) replaced Stage 5's parallel heads."
        ),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cache_path": args.cache_path,
        "checkpoint_path": args.checkpoint_path,
        "environment": environment_report(),
        "status": "started",
    }
    write_json(output_dir / "report.json", report)

    try:
        records = load_training_records(Path(args.cache_path))
        report["total_records_in_cache"] = len(records)
        if not records:
            raise RuntimeError(
                f"no records found in {args.cache_path} -- run colab_stage7_data_pipeline.py first"
            )

        train_records, val_records = split_records(records, args.val_fraction, args.split_seed)
        report["train_count"] = len(train_records)
        report["val_count"] = len(val_records)

        hidden_state_dim = len(train_records[0].hidden_state)
        num_codebooks = len(train_records[0].teacher_tokens[0])
        max_audio_tokens = len(train_records[0].teacher_tokens)
        report["hidden_state_dim"] = hidden_state_dim
        report["num_codebooks"] = num_codebooks
        report["max_audio_tokens"] = max_audio_tokens
        write_json(output_dir / "report.json", report)

        head = DepthTransformerVoiceHead(
            DepthTransformerVoiceHeadConfig(
                hidden_state_dim=hidden_state_dim,
                num_codebooks=num_codebooks,
                vocab_size=args.vocab_size,
                d_model=args.d_model,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                depth_dim=args.depth_dim,
                depth_n_layers=args.depth_n_layers,
                depth_n_heads=args.depth_n_heads,
                max_audio_tokens=max_audio_tokens,
                device=args.device,
            )
        )
        head.load()

        checkpoint_path = Path(args.checkpoint_path)
        report["resumed_from_existing_checkpoint"] = checkpoint_path.exists()
        write_json(output_dir / "report.json", report)

        train_started = time.monotonic_ns()
        result = train_large_scale(
            head,
            train_records,
            val_records,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            checkpoint_path=checkpoint_path,
            checkpoint_every_steps=args.checkpoint_every_steps,
            loss_log_path=Path(args.loss_log_path),
        )
        report["train_ms"] = (time.monotonic_ns() - train_started) / 1_000_000
        report["result"] = result
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
    parser.add_argument(
        "--cache-path",
        default="artifacts/colab-stage7-data/cache.jsonl",
        help="JSONL cache produced by colab_stage7_data_pipeline.py (point at Google Drive).",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="artifacts/colab-stage7-train/checkpoint.pt",
        help="Point this at Google Drive so an interrupted multi-day run can resume.",
    )
    parser.add_argument(
        "--loss-log-path",
        default="artifacts/colab-stage7-train/loss_log.jsonl",
        help="Appended, never overwritten -- also put this on Drive.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--depth-dim", type=int, default=256)
    parser.add_argument("--depth-n-layers", type=int, default=4)
    parser.add_argument("--depth-n-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every-steps", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="artifacts/colab-stage7-train")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exit_code = run_training(args, output_dir)
    print(f"AETHER_STAGE7_TRAIN_STATUS={'PASSED' if exit_code == 0 else 'FAILED'}")
    print(f"AETHER_STAGE7_TRAIN_REPORT={output_dir / 'report.json'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
