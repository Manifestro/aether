from typing import Any, AsyncIterator, Mapping, Optional, Protocol

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
    """Synthesizes audio codec tokens for a chunk the Speaker already produced.

    Takes the same ``SpeechChunk``/``facts`` the Speaker saw, the text it
    produced, and — when the Speaker exposes one (see ``last_hidden_state``
    on adapters that support it) — the internal hidden state that produced
    that text. A hidden-state-conditioned implementation is expected to
    ignore ``text`` entirely; it is kept in the signature so a
    text-conditioned implementation (the Stage 4 structural probe) and a
    hidden-state-conditioned one (Stage 5) satisfy the same protocol and are
    interchangeable at the runtime boundary.

    It participates in the existing commit horizon by convention, not by its
    own state: the runtime only commits a chunk once synthesis returns, and
    a chunk cancelled (by ``replan``) while synthesis is in flight is
    dropped by the runtime exactly like a cancelled text generation.
    """

    async def synthesize(
        self,
        chunk: SpeechChunk,
        text: str,
        facts: Mapping[str, ToolResult],
        hidden_state: Optional[Any] = None,
    ) -> AudioChunk:
        ...


class ToolExecutor(Protocol):
    async def execute(self, call: ToolCall) -> ToolResult:
        ...

