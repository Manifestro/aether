"""Stage 7, part 2 -- checkpointed, resumable mini-batch trainer.

Unlike `trainer.py` (full-batch, ~20 examples, one Colab session), this is
for real scale (thousands of examples) over what may be a multi-day,
interruption-prone run. Checkpoints (model + optimizer + epoch + step)
should be written to a path on Google Drive, not ephemeral Colab storage
-- re-running against the same `checkpoint_path` resumes automatically.

Logs progress verbosely (`print()`, plus a persistent JSONL log) on purpose
-- a multi-day run needs to be checkable from the Colab output at any
point without guessing whether it has stalled.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class TrainingRecord:
    phrase_id: str
    hidden_state: List[float]
    teacher_tokens: List[List[int]]


def load_training_records(cache_path: Path) -> List[TrainingRecord]:
    """Reads the JSONL cache `colab_stage7_data_pipeline.py` produces."""
    records: List[TrainingRecord] = []
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(
                TrainingRecord(data["phrase_id"], data["hidden_state"], data["teacher_tokens"])
            )
    return records


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def train_large_scale(
    head: Any,
    train_records: Sequence[TrainingRecord],
    val_records: Sequence[TrainingRecord],
    epochs: int,
    batch_size: int,
    lr: float,
    checkpoint_path: Path,
    checkpoint_every_steps: int,
    loss_log_path: Path,
    log_every_steps: int = 20,
) -> Dict[str, Any]:
    """Trains `head` (e.g. `DepthTransformerVoiceHead`) with mini-batches.

    Resumes from `checkpoint_path` if it already exists (model, optimizer,
    epoch, and step count are all restored) -- safe to just re-run this
    function/script after an interruption. `loss_log_path` is appended to,
    never overwritten, so history survives across resumed runs too.

    Prints a progress line every `log_every_steps` steps (loss, steps/sec,
    elapsed, ETA to the configured `epochs`) so a multi-day run can be
    checked from the Colab cell output alone, without opening any file.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = head.config.device
    train_hidden = torch.tensor([r.hidden_state for r in train_records], dtype=torch.float32)
    train_tokens = torch.tensor([r.teacher_tokens for r in train_records], dtype=torch.long)
    dataset = TensorDataset(train_hidden, train_tokens)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * epochs

    val_hidden: Optional[Any] = None
    val_tokens: Optional[Any] = None
    if val_records:
        val_hidden = torch.tensor([r.hidden_state for r in val_records], dtype=torch.float32, device=device)
        val_tokens = torch.tensor([r.teacher_tokens for r in val_records], dtype=torch.long, device=device)

    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    start_epoch = 0
    global_step = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        head.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        print(
            f"[train] resumed from checkpoint: epoch={start_epoch}, global_step={global_step} "
            f"({checkpoint_path})"
        )
    else:
        print(f"[train] starting fresh -- no checkpoint found at {checkpoint_path}")

    print(
        f"[train] {len(train_records)} train / {len(val_records)} val records, "
        f"{steps_per_epoch} steps/epoch, {epochs} epochs, {total_steps} steps total "
        f"({total_steps - global_step} remaining)"
    )

    def _save_checkpoint(epoch: int) -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save(
            {
                "model_state_dict": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            },
            tmp_path,
        )
        # Atomic-ish swap: a crash mid-`torch.save` must never corrupt the
        # checkpoint a multi-day run would otherwise resume from.
        tmp_path.replace(checkpoint_path)

    def _log(entry: Dict[str, Any]) -> None:
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **entry}
        loss_log_path.parent.mkdir(parents=True, exist_ok=True)
        with loss_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def _val_loss() -> Optional[float]:
        if val_hidden is None:
            return None
        head.train_mode(False)
        with torch.no_grad():
            loss = float(head.compute_training_loss(val_hidden, val_tokens).item())
        head.train_mode(True)
        return loss

    run_started_at = time.monotonic()
    steps_done_this_run = 0

    head.train_mode(True)
    for epoch in range(start_epoch, epochs):
        epoch_started_at = time.monotonic()
        for batch_hidden, batch_tokens in loader:
            batch_hidden = batch_hidden.to(device)
            batch_tokens = batch_tokens.to(device)
            optimizer.zero_grad()
            loss = head.compute_training_loss(batch_hidden, batch_tokens)
            loss.backward()
            optimizer.step()
            global_step += 1
            steps_done_this_run += 1

            if global_step % log_every_steps == 0:
                elapsed = time.monotonic() - run_started_at
                steps_per_sec = steps_done_this_run / elapsed if elapsed > 0 else 0.0
                remaining_steps = max(0, total_steps - global_step)
                eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else float("inf")
                print(
                    f"[train] step {global_step}/{total_steps} (epoch {epoch}) "
                    f"loss={loss.item():.4f} {steps_per_sec:.2f} steps/s "
                    f"elapsed={format_duration(elapsed)} "
                    f"eta={format_duration(eta) if eta != float('inf') else 'unknown'}"
                )

            if global_step % checkpoint_every_steps == 0:
                _save_checkpoint(epoch)
                _log({"global_step": global_step, "epoch": epoch, "train_loss": float(loss.item())})
                print(f"[train] checkpoint saved at step {global_step} -> {checkpoint_path}")

        val_loss = _val_loss()
        _save_checkpoint(epoch + 1)
        epoch_duration = time.monotonic() - epoch_started_at
        _log({"epoch_completed": epoch, "global_step": global_step, "val_loss": val_loss})
        print(
            f"[train] epoch {epoch} complete in {format_duration(epoch_duration)} "
            f"-- val_loss={val_loss}"
        )

    head.train_mode(False)
    final_val_loss = _val_loss()
    print(
        f"[train] finished: {epochs} epochs, global_step={global_step}, "
        f"final_val_loss={final_val_loss}"
    )
    return {"final_epoch": epochs, "final_global_step": global_step, "final_val_loss": final_val_loss}
