from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ThoughtBridgeConfig:
    """Stage 6 (docs/plan.md Level B): a structural probe, not a trained bridge.

    Projects a Planner hidden state into one or more "soft prompt" token
    embeddings that get prepended to Speaker's input embeddings — a channel
    for Planner's *organized thought* to condition Speaker's generation,
    distinct from the structured `speech_plan.goal` string it already
    receives. Weights are randomly initialized and never trained here; this
    stage answers one question only: does this injection channel measurably
    change Speaker's generation at all? Whether it makes generation *better*
    is Stage 7 (training) and Stage 8 (evaluation) — see
    `docs/reports/technical_report_03.md` and the internal Stage 6 roadmap.
    """

    hidden_state_dim: int
    embedding_dim: int
    num_soft_tokens: int = 1
    device: str = "cpu"
    seed: Optional[int] = 0

    def __post_init__(self) -> None:
        if self.hidden_state_dim < 1:
            raise ValueError("hidden_state_dim must be positive")
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if self.num_soft_tokens < 1:
            raise ValueError("num_soft_tokens must be positive")


class ThoughtBridge:
    """Projects a hidden state into soft-prompt embeddings for Speaker.

    Importing this module, constructing this class and running unit tests
    never import torch or allocate a model — `load()` must be called
    explicitly, matching every other ML component in this project.
    """

    def __init__(self, config: ThoughtBridgeConfig) -> None:
        self.config = config
        self._model: Any = None
        self._torch: Any = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.loaded:
            return
        try:
            import torch
            from torch import nn
        except ImportError as error:
            raise RuntimeError(
                "torch is not installed; install the 'ml' extra in the target environment"
            ) from error

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        cfg = self.config
        self._torch = torch
        self._model = nn.Linear(cfg.hidden_state_dim, cfg.num_soft_tokens * cfg.embedding_dim)
        self._model.eval()
        self._model.to(cfg.device)

    def project(self, hidden_state: Any) -> Any:
        """Returns a `(num_soft_tokens, embedding_dim)` tensor for one hidden state."""
        if not self.loaded:
            raise RuntimeError("ThoughtBridge is not loaded; call load() explicitly")
        torch = self._torch
        cfg = self.config
        vector = list(hidden_state)
        if len(vector) != cfg.hidden_state_dim:
            raise ValueError(
                f"hidden_state has {len(vector)} dims, expected {cfg.hidden_state_dim}"
            )
        with torch.no_grad():
            input_vec = torch.tensor([vector], dtype=torch.float32, device=cfg.device)
            flat = self._model(input_vec)
        return flat.view(cfg.num_soft_tokens, cfg.embedding_dim)
