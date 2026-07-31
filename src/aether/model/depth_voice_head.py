from dataclasses import dataclass
from typing import Any, Mapping, Optional

from aether.domain.chunks import SpeechChunk
from aether.domain.events import ToolResult


@dataclass(frozen=True)
class DepthTransformerVoiceHeadConfig:
    """A real (if still simplified) Depth Transformer -- the fix for the
    word-dropout `MultiCodebookVoiceHead` (Stage 5) showed.

    Two stacked models, mirroring the real Mimi/Moshi design (temporal
    transformer + depth transformer), instead of Stage 5's independent
    parallel heads for codebooks 1-31:

    - **Temporal transformer**: predicts codebook 0 (the semantic/driving
      codebook) autoregressively over time, exactly as
      `MultiCodebookVoiceHead` did.
    - **Depth transformer**: a second, smaller causal transformer that
      predicts codebooks 1..num_codebooks-1 *sequentially, each
      conditioned on the previous codebooks of the same frame* (plus the
      temporal transformer's hidden state at that frame). This is the
      inter-codebook dependency Stage 5 explicitly gave up for speed --
      restoring it is this class's entire purpose.

    Sized for a real training run (default d_model=512, 8 temporal layers)
    rather than Stage 4/5's structural-probe scale (d_model=128, 2 layers).
    """

    hidden_state_dim: int
    num_codebooks: int = 32
    vocab_size: int = 2048
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    depth_dim: int = 256
    depth_n_layers: int = 4
    depth_n_heads: int = 4
    max_audio_tokens: int = 50
    frame_rate_hz: float = 12.5
    device: str = "cpu"
    seed: Optional[int] = 0

    def __post_init__(self) -> None:
        if self.hidden_state_dim < 1:
            raise ValueError("hidden_state_dim must be positive")
        if self.num_codebooks < 2:
            raise ValueError(
                "num_codebooks must be at least 2 -- with only codebook 0 there is "
                "nothing for the depth transformer to predict; use MultiCodebookVoiceHead instead"
            )
        if self.vocab_size < 2:
            raise ValueError("vocab_size must allow at least a start token and one real token")
        if self.max_audio_tokens < 1:
            raise ValueError("max_audio_tokens must be positive")
        if self.frame_rate_hz <= 0:
            raise ValueError("frame_rate_hz must be positive")


class DepthTransformerVoiceHead:
    """Standalone Voice Head with real inter-codebook conditioning.

    Importing this module, constructing this class and running unit tests
    never import torch or allocate a model — `load()` must be called
    explicitly, matching every other ML component in this project.
    """

    _START_TOKEN = 0

    def __init__(self, config: DepthTransformerVoiceHeadConfig) -> None:
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

        class _TemporalDepthModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Temporal stack -- predicts codebook 0 over time.
                self.projector = nn.Linear(cfg.hidden_state_dim, cfg.d_model)
                self.audio_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
                self.position_embedding = nn.Embedding(1 + cfg.max_audio_tokens, cfg.d_model)
                temporal_layer = nn.TransformerEncoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.n_heads,
                    dim_feedforward=cfg.d_model * 4,
                    batch_first=True,
                )
                self.temporal_backbone = nn.TransformerEncoder(temporal_layer, num_layers=cfg.n_layers)
                self.codebook0_head = nn.Linear(cfg.d_model, cfg.vocab_size)

                # Depth stack -- predicts codebooks 1..num_codebooks-1 per
                # frame, sequentially conditioned on the codebooks before
                # them in the same frame plus the temporal hidden state.
                self.depth_condition_proj = nn.Linear(cfg.d_model, cfg.depth_dim)
                self.depth_codebook_embedding = nn.Embedding(cfg.vocab_size, cfg.depth_dim)
                self.depth_position_embedding = nn.Embedding(cfg.num_codebooks, cfg.depth_dim)
                depth_layer = nn.TransformerEncoderLayer(
                    d_model=cfg.depth_dim,
                    nhead=cfg.depth_n_heads,
                    dim_feedforward=cfg.depth_dim * 4,
                    batch_first=True,
                )
                self.depth_backbone = nn.TransformerEncoder(depth_layer, num_layers=cfg.depth_n_layers)
                self.depth_output_head = nn.Linear(cfg.depth_dim, cfg.vocab_size)

            def forward_temporal(self, hidden_state_vec: Any, audio_ids: Any) -> Any:
                """Returns per-frame temporal hidden states, (B, audio_len, d_model)."""
                condition = self.projector(hidden_state_vec).unsqueeze(1)
                audio_len = audio_ids.shape[1]
                audio_emb = self.audio_embedding(audio_ids)
                sequence = torch.cat([condition, audio_emb], dim=1)
                positions = torch.arange(1 + audio_len, device=sequence.device)
                sequence = sequence + self.position_embedding(positions).unsqueeze(0)
                causal_mask = nn.Transformer.generate_square_subsequent_mask(1 + audio_len).to(
                    sequence.device
                )
                return self.temporal_backbone(sequence, mask=causal_mask)[:, 1:, :]

            def forward_depth(self, flat_temporal_hidden: Any, codebook_inputs: Any) -> Any:
                """One frame's depth pass. `flat_temporal_hidden`: (N, d_model).
                `codebook_inputs`: (N, k) teacher-forced codebook values
                (codebook 0 first). Returns logits for the next `k` codebooks,
                (N, k, vocab_size)."""
                condition = self.depth_condition_proj(flat_temporal_hidden).unsqueeze(1)
                codebook_emb = self.depth_codebook_embedding(codebook_inputs)
                sequence = torch.cat([condition, codebook_emb], dim=1)
                positions = torch.arange(sequence.shape[1], device=sequence.device)
                sequence = sequence + self.depth_position_embedding(positions).unsqueeze(0)
                causal_mask = nn.Transformer.generate_square_subsequent_mask(sequence.shape[1]).to(
                    sequence.device
                )
                hidden = self.depth_backbone(sequence, mask=causal_mask)
                return self.depth_output_head(hidden[:, 1:, :])

        self._torch = torch
        self._model = _TemporalDepthModule()
        self._model.eval()
        self._model.to(cfg.device)

    def parameters(self) -> Any:
        if not self.loaded:
            raise RuntimeError("DepthTransformerVoiceHead is not loaded; call load() explicitly")
        return self._model.parameters()

    def train_mode(self, enabled: bool = True) -> None:
        if not self.loaded:
            raise RuntimeError("DepthTransformerVoiceHead is not loaded; call load() explicitly")
        self._model.train(enabled)

    def state_dict(self) -> Any:
        if not self.loaded:
            raise RuntimeError("DepthTransformerVoiceHead is not loaded; call load() explicitly")
        return self._model.state_dict()

    def load_state_dict(self, state_dict: Any) -> None:
        if not self.loaded:
            raise RuntimeError("DepthTransformerVoiceHead is not loaded; call load() explicitly")
        self._model.load_state_dict(state_dict)

    def compute_training_loss(self, hidden_state_batch: Any, target_tokens_batch: Any) -> Any:
        """`target_tokens_batch`: (B, T, num_codebooks) long tensor -- the
        teacher's tokens for every codebook. Loss = codebook-0 temporal
        cross-entropy + depth cross-entropy over codebooks 1..num_codebooks-1,
        the latter teacher-forced on the true preceding codebooks of the
        same frame (not the model's own greedy guesses)."""
        if not self.loaded:
            raise RuntimeError("DepthTransformerVoiceHead is not loaded; call load() explicitly")
        torch = self._torch
        cfg = self.config
        batch_size, seq_len, _ = target_tokens_batch.shape

        codebook0_targets = target_tokens_batch[:, :, 0]
        start_column = torch.full(
            (batch_size, 1), self._START_TOKEN, dtype=torch.long, device=target_tokens_batch.device
        )
        audio_input = torch.cat([start_column, codebook0_targets[:, :-1]], dim=1)
        temporal_hidden = self._model.forward_temporal(hidden_state_batch, audio_input)
        codebook0_logits = self._model.codebook0_head(temporal_hidden)
        loss_codebook0 = torch.nn.functional.cross_entropy(
            codebook0_logits.reshape(-1, cfg.vocab_size), codebook0_targets.reshape(-1)
        )

        flat_hidden = temporal_hidden.reshape(batch_size * seq_len, cfg.d_model)
        flat_targets = target_tokens_batch.reshape(batch_size * seq_len, cfg.num_codebooks)
        depth_logits = self._model.forward_depth(flat_hidden, flat_targets[:, :-1])
        depth_targets = flat_targets[:, 1:]
        loss_depth = torch.nn.functional.cross_entropy(
            depth_logits.reshape(-1, cfg.vocab_size), depth_targets.reshape(-1)
        )

        return loss_codebook0 + loss_depth

    def _forward_greedy_all_codebooks(self, hidden_state: Any) -> list:
        """Returns a `(max_audio_tokens, num_codebooks)` list of lists --
        fully self-generated, codebook 0 driven by the temporal transformer,
        codebooks 1..num_codebooks-1 by the depth transformer conditioned
        on the real (just-generated) preceding codebooks of that frame."""
        torch = self._torch
        cfg = self.config
        vector = list(hidden_state)
        if len(vector) != cfg.hidden_state_dim:
            raise ValueError(
                f"hidden_state has {len(vector)} dims, expected {cfg.hidden_state_dim}"
            )
        hidden_state_vec = torch.tensor([vector], dtype=torch.float32, device=cfg.device)
        codebook0_tokens = [self._START_TOKEN]
        frames: list = []
        with torch.no_grad():
            for _ in range(cfg.max_audio_tokens):
                audio_ids = torch.tensor([codebook0_tokens], dtype=torch.long, device=cfg.device)
                temporal_hidden = self._model.forward_temporal(hidden_state_vec, audio_ids)
                last_hidden = temporal_hidden[:, -1, :]
                codebook0_logits = self._model.codebook0_head(last_hidden)
                codebook_0 = int(torch.argmax(codebook0_logits[0]).item())

                frame_tokens = [codebook_0]
                for _step in range(cfg.num_codebooks - 1):
                    codebook_inputs = torch.tensor([frame_tokens], dtype=torch.long, device=cfg.device)
                    depth_logits = self._model.forward_depth(last_hidden, codebook_inputs)
                    next_codebook = int(torch.argmax(depth_logits[0, -1]).item())
                    frame_tokens.append(next_codebook)

                frames.append(frame_tokens)
                codebook0_tokens.append(codebook_0)
        return frames

    async def synthesize(
        self,
        chunk: SpeechChunk,
        text: str,
        facts: Mapping[str, ToolResult],
        hidden_state: Optional[Any] = None,
    ) -> Any:
        del text
        if hidden_state is None:
            raise ValueError(
                "DepthTransformerVoiceHead requires a hidden_state; the Speaker in use does not "
                "expose one (no `last_hidden_state` attribute after generate())"
            )
        from aether.domain.audio import AudioChunk

        frames = self._forward_greedy_all_codebooks(hidden_state)
        flat_tokens = tuple(token for frame in frames for token in frame)
        return AudioChunk(
            chunk_id=chunk.chunk_id,
            codebook_index=0,
            tokens=flat_tokens,
            frame_rate_hz=self.config.frame_rate_hz,
        )
