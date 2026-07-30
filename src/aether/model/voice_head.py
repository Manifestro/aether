import asyncio
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from aether.domain.audio import AudioChunk
from aether.domain.chunks import SpeechChunk
from aether.domain.events import ToolResult


@dataclass(frozen=True)
class MinimalVoiceHeadConfig:
    """Single-codebook, from-scratch Voice Head — a structural probe, not a speech model.

    This intentionally does not aim for audio quality, prosody or emotion
    (out of scope for this experiment by design — see docs/plan.md Phase C).
    It exists to answer one question: does the existing commit horizon
    (``ChunkState``) generalize to an audio modality without modification?
    Weights are randomly initialized and never trained; the pass/fail
    criteria for the experiment that uses this class are structural
    (timing, cancellation correctness), not perceptual.
    """

    vocab_size: int = 2048
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    max_audio_tokens: int = 32
    frame_rate_hz: float = 12.5
    device: str = "cpu"
    seed: Optional[int] = 0

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must allow at least a start token and one real token")
        if self.max_audio_tokens < 1:
            raise ValueError("max_audio_tokens must be positive")
        if self.frame_rate_hz <= 0:
            raise ValueError("frame_rate_hz must be positive")


class MinimalVoiceHead:
    """Predicts one Mimi codebook's tokens from Speaker text, autoregressively.

    Importing this module and constructing this class never imports torch
    or allocates a model — matching the project rule that no ML dependency
    loads outside an explicit `load()` call in the target environment.
    """

    _START_TOKEN = 0  # reserved; real codes occupy [1, vocab_size).

    def __init__(self, config: MinimalVoiceHeadConfig = MinimalVoiceHeadConfig()) -> None:
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

        class _ByteConditionedVoiceHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.byte_embedding = nn.Embedding(256, cfg.d_model)
                self.audio_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
                self.position_embedding = nn.Embedding(
                    256 + cfg.max_audio_tokens, cfg.d_model
                )
                layer = nn.TransformerEncoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.n_heads,
                    dim_feedforward=cfg.d_model * 4,
                    batch_first=True,
                )
                self.backbone = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
                self.output_head = nn.Linear(cfg.d_model, cfg.vocab_size)

            def forward(self, byte_ids: Any, audio_ids: Any) -> Any:
                text_len = byte_ids.shape[1]
                audio_len = audio_ids.shape[1]
                text_emb = self.byte_embedding(byte_ids)
                audio_emb = self.audio_embedding(audio_ids)
                sequence = torch.cat([text_emb, audio_emb], dim=1)
                positions = torch.arange(text_len + audio_len, device=sequence.device)
                sequence = sequence + self.position_embedding(positions).unsqueeze(0)
                causal_mask = nn.Transformer.generate_square_subsequent_mask(
                    text_len + audio_len
                ).to(sequence.device)
                # Text positions may attend to each other freely; audio
                # positions are causal over the full sequence. A full causal
                # mask over the concatenation is a conservative superset that
                # keeps this module to one code path.
                hidden = self.backbone(sequence, mask=causal_mask)
                return self.output_head(hidden[:, text_len:, :])

        self._torch = torch
        self._model = _ByteConditionedVoiceHead()
        self._model.eval()
        self._model.to(self.config.device)

    def _forward_greedy(self, text: str) -> list:
        torch = self._torch
        cfg = self.config
        text_bytes = text.encode("utf-8")[:256] or b"\x00"
        byte_ids = torch.tensor([list(text_bytes)], dtype=torch.long, device=cfg.device)
        tokens = [self._START_TOKEN]
        with torch.no_grad():
            for _ in range(cfg.max_audio_tokens):
                audio_ids = torch.tensor([tokens], dtype=torch.long, device=cfg.device)
                logits = self._model(byte_ids, audio_ids)
                next_token = int(torch.argmax(logits[0, -1]).item())
                tokens.append(next_token)
        return tokens[1:]  # drop the start token; it is not a codec code

    async def synthesize(
        self,
        chunk: SpeechChunk,
        text: str,
        facts: Mapping[str, ToolResult],
        hidden_state: Optional[Any] = None,
    ) -> AudioChunk:
        # This is the Stage 4 text-conditioned probe: `hidden_state` is
        # accepted only so it satisfies the same `VoiceHead` protocol as
        # `HiddenStateVoiceHead` below and can be swapped in the runtime
        # without a signature mismatch. It is intentionally unused here —
        # see `HiddenStateVoiceHead` for the hidden-state-conditioned path.
        del hidden_state
        if not self.loaded:
            raise RuntimeError("MinimalVoiceHead is not loaded; call load() explicitly")
        loop = asyncio.get_running_loop()
        tokens = await loop.run_in_executor(None, self._forward_greedy, text)
        return AudioChunk(
            chunk_id=chunk.chunk_id,
            codebook_index=0,
            tokens=tuple(tokens),
            frame_rate_hz=self.config.frame_rate_hz,
        )


@dataclass(frozen=True)
class HiddenStateVoiceHeadConfig:
    """Stage 5: conditions on the Speaker's internal hidden state, not text.

    ``text`` is accepted by ``synthesize`` (protocol compatibility with
    ``MinimalVoiceHead``) but deliberately ignored — the whole point of this
    class is that the audio path does not go through the decoded string.
    The hidden-state vector is projected to a single conditioning position
    (a minimal stand-in for spec.md §10's gated cross-attention memory;
    full cross-attention over a hidden-state *sequence* is future work, not
    designed away).
    """

    hidden_state_dim: int
    vocab_size: int = 2048
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    max_audio_tokens: int = 32
    frame_rate_hz: float = 12.5
    device: str = "cpu"
    seed: Optional[int] = 0

    def __post_init__(self) -> None:
        if self.hidden_state_dim < 1:
            raise ValueError("hidden_state_dim must be positive")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must allow at least a start token and one real token")
        if self.max_audio_tokens < 1:
            raise ValueError("max_audio_tokens must be positive")
        if self.frame_rate_hz <= 0:
            raise ValueError("frame_rate_hz must be positive")


class HiddenStateVoiceHead:
    """Predicts one Mimi codebook's tokens from a Speaker hidden state.

    Importing this module, constructing this class and running unit tests
    never import torch or allocate a model — `load()` must be called
    explicitly, matching every other ML component in this project.
    """

    _START_TOKEN = 0

    def __init__(self, config: HiddenStateVoiceHeadConfig) -> None:
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

        class _HiddenStateConditionedVoiceHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projector = nn.Linear(cfg.hidden_state_dim, cfg.d_model)
                self.audio_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
                self.position_embedding = nn.Embedding(1 + cfg.max_audio_tokens, cfg.d_model)
                layer = nn.TransformerEncoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.n_heads,
                    dim_feedforward=cfg.d_model * 4,
                    batch_first=True,
                )
                self.backbone = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
                self.output_head = nn.Linear(cfg.d_model, cfg.vocab_size)

            def forward(self, hidden_state_vec: Any, audio_ids: Any) -> Any:
                condition = self.projector(hidden_state_vec).unsqueeze(1)  # (1, 1, d_model)
                audio_len = audio_ids.shape[1]
                audio_emb = self.audio_embedding(audio_ids)
                sequence = torch.cat([condition, audio_emb], dim=1)
                positions = torch.arange(1 + audio_len, device=sequence.device)
                sequence = sequence + self.position_embedding(positions).unsqueeze(0)
                causal_mask = nn.Transformer.generate_square_subsequent_mask(1 + audio_len).to(
                    sequence.device
                )
                hidden = self.backbone(sequence, mask=causal_mask)
                return self.output_head(hidden[:, 1:, :])

        self._torch = torch
        self._model = _HiddenStateConditionedVoiceHead()
        self._model.eval()
        self._model.to(self.config.device)

    def parameters(self) -> Any:
        if not self.loaded:
            raise RuntimeError("HiddenStateVoiceHead is not loaded; call load() explicitly")
        return self._model.parameters()

    def train_mode(self, enabled: bool = True) -> None:
        if not self.loaded:
            raise RuntimeError("HiddenStateVoiceHead is not loaded; call load() explicitly")
        self._model.train(enabled)

    def compute_training_loss(self, hidden_state_batch: Any, target_tokens_batch: Any) -> Any:
        """Teacher-forced next-token cross-entropy loss for one batch.

        ``hidden_state_batch``: float tensor ``(B, hidden_state_dim)``.
        ``target_tokens_batch``: long tensor ``(B, config.max_audio_tokens)``
        — the teacher's codebook tokens, e.g. from ``TTSModel.generate()``.

        The model input is the target shifted right by one position (start
        token prepended, last target token dropped) — the standard
        teacher-forcing setup, reusing the exact forward pass `synthesize`
        already uses at inference time, just with a real target instead of
        greedy self-generated tokens.
        """
        if not self.loaded:
            raise RuntimeError("HiddenStateVoiceHead is not loaded; call load() explicitly")
        torch = self._torch
        batch_size = target_tokens_batch.shape[0]
        start_column = torch.full(
            (batch_size, 1), self._START_TOKEN, dtype=torch.long, device=target_tokens_batch.device
        )
        audio_input = torch.cat([start_column, target_tokens_batch[:, :-1]], dim=1)
        logits = self._model(hidden_state_batch, audio_input)
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), target_tokens_batch.reshape(-1)
        )

    def _forward_greedy(self, hidden_state: Any) -> list:
        torch = self._torch
        cfg = self.config
        vector = list(hidden_state)
        if len(vector) != cfg.hidden_state_dim:
            raise ValueError(
                f"hidden_state has {len(vector)} dims, expected {cfg.hidden_state_dim}"
            )
        hidden_state_vec = torch.tensor([vector], dtype=torch.float32, device=cfg.device)
        tokens = [self._START_TOKEN]
        with torch.no_grad():
            for _ in range(cfg.max_audio_tokens):
                audio_ids = torch.tensor([tokens], dtype=torch.long, device=cfg.device)
                logits = self._model(hidden_state_vec, audio_ids)
                next_token = int(torch.argmax(logits[0, -1]).item())
                tokens.append(next_token)
        return tokens[1:]

    async def synthesize(
        self,
        chunk: SpeechChunk,
        text: str,
        facts: Mapping[str, ToolResult],
        hidden_state: Optional[Any] = None,
    ) -> AudioChunk:
        del text  # deliberately not used -- see class docstring
        if not self.loaded:
            raise RuntimeError("HiddenStateVoiceHead is not loaded; call load() explicitly")
        if hidden_state is None:
            raise ValueError(
                "HiddenStateVoiceHead requires a hidden_state; the Speaker in use does not "
                "expose one (no `last_hidden_state` attribute after generate())"
            )
        loop = asyncio.get_running_loop()
        tokens = await loop.run_in_executor(None, self._forward_greedy, hidden_state)
        return AudioChunk(
            chunk_id=chunk.chunk_id,
            codebook_index=0,
            tokens=tuple(tokens),
            frame_rate_hz=self.config.frame_rate_hz,
        )
