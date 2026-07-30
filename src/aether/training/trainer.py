from dataclasses import dataclass
from typing import Any, List, Sequence


@dataclass(frozen=True)
class TrainingExample:
    phrase_id: str
    hidden_state: List[float]
    target_tokens: List[int]


def train_hidden_state_voice_head(
    head: Any,
    examples: Sequence[TrainingExample],
    epochs: int,
    lr: float = 1e-3,
) -> List[float]:
    """Full-batch Adam training for a `HiddenStateVoiceHead`.

    No DataLoader/batching abstraction -- Stage 5 trains on ~20 examples at
    once specifically because the question is "does this tiny mechanism
    learn at all", not "does it scale". Returns the per-epoch loss curve so
    the caller can judge Continue/Refine/Fallback (spec.md §20) from a
    trace, not an impression.
    """
    import torch

    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    device = head.config.device
    hidden_state_batch = torch.tensor(
        [example.hidden_state for example in examples], dtype=torch.float32, device=device
    )
    target_tokens_batch = torch.tensor(
        [example.target_tokens for example in examples], dtype=torch.long, device=device
    )

    head.train_mode(True)
    loss_curve: List[float] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = head.compute_training_loss(hidden_state_batch, target_tokens_batch)
        loss.backward()
        optimizer.step()
        loss_curve.append(float(loss.item()))
    head.train_mode(False)
    return loss_curve
