from dataclasses import dataclass
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class MimiCodecConfig:
    """Frozen, pretrained Kyutai Mimi codec, restricted to few codebooks.

    ``num_codebooks=1`` matches the minimal Voice Head experiment (Phase C,
    docs/plan.md §C1/C3): the first Mimi codebook carries the strongest
    semantic signal, so restricting to it keeps the decoded audio's
    *structure* (frame timing, streaming shape) real while the *content* is
    only as good as an untrained token predictor — which is the deliberate
    scope of this experiment.
    """

    device: str = "cpu"
    num_codebooks: int = 1
    allow_download: bool = False

    def __post_init__(self) -> None:
        if self.num_codebooks < 1:
            raise ValueError("num_codebooks must be positive")


class MimiCodec:
    """Lazily loaded wrapper around ``moshi``'s Mimi model.

    Importing this module, constructing this class and running unit tests
    never import ``moshi`` or touch model files — ``load()`` must be called
    explicitly in the target ML environment, matching
    ``aether.model.llm_backbone.SharedLLMBackbone``.
    """

    def __init__(self, config: MimiCodecConfig) -> None:
        self.config = config
        self._model: Any = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def sample_rate(self) -> float:
        if self._model is None:
            raise RuntimeError("MimiCodec is not loaded")
        return float(self._model.sample_rate)

    @property
    def frame_rate(self) -> float:
        if self._model is None:
            raise RuntimeError("MimiCodec is not loaded")
        return float(self._model.frame_rate)

    def load(self) -> None:
        if self.loaded:
            return
        if not self.config.allow_download:
            raise RuntimeError(
                "MimiCodec.load() downloads pretrained weights; pass "
                "allow_download=True only in the target ML environment"
            )
        try:
            from moshi.models import loaders
        except ImportError as error:
            raise RuntimeError(
                "moshi is not installed; install the 'audio' extra in the target environment"
            ) from error

        try:
            mimi = loaders.get_mimi(None, device=self.config.device)
            mimi.set_num_codebooks(self.config.num_codebooks)
        except Exception as error:
            # The moshi public API is small but has moved between releases.
            # Surface the real cause instead of a bare AttributeError so the
            # next agent can patch this one call site against whatever
            # version Colab installed, rather than re-deriving the failure.
            raise RuntimeError(
                "failed to load the Mimi codec via moshi.models.loaders.get_mimi(); "
                "inspect the chained traceback against the installed moshi version "
                "and adjust MimiCodec.load() accordingly"
            ) from error

        mimi.eval()
        self._model = mimi

    def decode(self, tokens: Sequence[int]) -> Any:
        """Decode one codebook's token sequence to a mono PCM waveform (numpy array)."""
        if not self.loaded:
            raise RuntimeError("MimiCodec is not loaded; call load() explicitly")
        import torch

        # shape: (batch=1, num_codebooks=1, sequence)
        codes = torch.tensor(
            [[list(tokens)]], dtype=torch.long, device=self.config.device
        )
        with torch.no_grad():
            pcm = self._model.decode(codes)
        return pcm[0, 0].cpu().numpy()
