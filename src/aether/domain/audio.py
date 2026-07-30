from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AudioChunk:
    """A single speech chunk's audio, expressed as codec token ids.

    Deliberately mirrors ``SpeechChunk`` at the boundary that matters for
    safety: it carries no state of its own. Its commit/cancel behaviour is
    entirely governed by the owning ``SpeechChunk``'s ``ChunkState`` — this
    is the structural claim Level C is testing (does the existing commit
    horizon generalize to audio, or does audio need its own state machine).

    ``codebook_index`` is 0 for Mimi's first ("semantic") codebook. The
    minimal Voice Head experiment predicts exactly one codebook; additional
    codebooks (acoustic detail, prosody) are deferred, not designed away —
    a chunk with more than one codebook would just carry more of these.
    """

    chunk_id: str
    codebook_index: int
    tokens: Tuple[int, ...]
    frame_rate_hz: float

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must not be empty")
        if self.codebook_index < 0:
            raise ValueError("codebook_index must be non-negative")
        if self.frame_rate_hz <= 0:
            raise ValueError("frame_rate_hz must be positive")
