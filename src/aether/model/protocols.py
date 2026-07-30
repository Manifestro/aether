from typing import AsyncIterator, Mapping, Protocol

from aether.domain.audio import AudioChunk
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


class VoiceHead(Protocol):
    """Synthesizes audio codec tokens for a chunk already spoken as text.

    Takes the same ``SpeechChunk``/``facts`` the Speaker saw plus the text
    it produced, and returns codec tokens for that chunk. It participates in
    the existing commit horizon by convention, not by its own state: the
    runtime only commits a chunk once synthesis returns, and a chunk
    cancelled (by ``replan``) while synthesis is in flight is dropped by the
    runtime exactly like a cancelled text generation.
    """

    async def synthesize(
        self,
        chunk: SpeechChunk,
        text: str,
        facts: Mapping[str, ToolResult],
    ) -> AudioChunk:
        ...


class ToolExecutor(Protocol):
    async def execute(self, call: ToolCall) -> ToolResult:
        ...

