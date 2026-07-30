from typing import AsyncIterator, Mapping, Protocol

from aether.domain.chunks import SpeechChunk
from aether.domain.events import SemanticEvent, ToolCall, ToolResult


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

