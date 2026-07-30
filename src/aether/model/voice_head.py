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
    ) -> AudioChunk:
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
