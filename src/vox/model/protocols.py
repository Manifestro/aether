from typing import AsyncIterator, Mapping, Protocol, Sequence

from vox.domain.chunks import SpeechChunk
from vox.domain.events import SemanticEvent, ToolCall, ToolResult


class Planner(Protocol):
    def plan(self, turn_id: str, request: str) -> AsyncIterator[SemanticEvent]:
        ...


class Speaker(Protocol):
    async def generate(
        self,
        chunk: SpeechChunk,
        facts: Mapping[str, ToolResult],
    ) -> str:
        ...


class ToolExecutor(Protocol):
    async def execute(self, call: ToolCall) -> ToolResult:
        ...


class EventValidator(Protocol):
    def validate_sequence(self, events: Sequence[SemanticEvent]) -> None:
        ...

