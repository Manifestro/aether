from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet


class ChunkState(str, Enum):
    BLOCKED = "blocked"
    READY = "ready"
    GENERATING = "generating"
    BUFFERED = "buffered"
    COMMITTED = "committed"
    PLAYED = "played"
    CANCELLED = "cancelled"


_ALLOWED_TRANSITIONS = {
    ChunkState.BLOCKED: {ChunkState.READY, ChunkState.CANCELLED},
    ChunkState.READY: {ChunkState.GENERATING, ChunkState.CANCELLED},
    ChunkState.GENERATING: {ChunkState.BUFFERED, ChunkState.CANCELLED},
    ChunkState.BUFFERED: {ChunkState.COMMITTED, ChunkState.CANCELLED},
    ChunkState.COMMITTED: {ChunkState.PLAYED},
    ChunkState.PLAYED: set(),
    ChunkState.CANCELLED: set(),
}


@dataclass
class SpeechChunk:
    chunk_id: str
    goal: str
    dependencies: FrozenSet[str] = field(default_factory=frozenset)
    plan_version: int = 1
    state: ChunkState = ChunkState.BLOCKED

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must not be empty")
        if not self.goal.strip():
            raise ValueError("goal must not be empty")
        if self.plan_version < 1:
            raise ValueError("plan_version must be positive")
        if not self.dependencies and self.state is ChunkState.BLOCKED:
            self.state = ChunkState.READY

    def resolve(self, available_facts: FrozenSet[str]) -> bool:
        if self.state is ChunkState.BLOCKED and self.dependencies <= available_facts:
            self.transition_to(ChunkState.READY)
        return self.state is ChunkState.READY

    def transition_to(self, target: ChunkState) -> None:
        if target not in _ALLOWED_TRANSITIONS[self.state]:
            raise ValueError(f"invalid chunk transition: {self.state.value} -> {target.value}")
        self.state = target

    @property
    def cancellable(self) -> bool:
        return self.state in {
            ChunkState.BLOCKED,
            ChunkState.READY,
            ChunkState.GENERATING,
            ChunkState.BUFFERED,
        }

