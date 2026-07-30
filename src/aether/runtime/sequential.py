from dataclasses import dataclass
from typing import Dict, List

from aether.domain.chunks import ChunkState, SpeechChunk
from aether.domain.events import EventKind, SemanticEvent, ToolCall, ToolResult
from aether.domain.timeline import Timeline
from aether.model.protocols import Planner, Speaker, ToolExecutor


@dataclass(frozen=True)
class BaselineResult:
    text: str
    semantic_events: List[SemanticEvent]
    chunks: List[SpeechChunk]
    facts: Dict[str, ToolResult]
    timeline: Timeline


class SequentialBaseline:
    """Instrumented Stage 0 baseline: plan, execute tools, then speak."""

    def __init__(self, planner: Planner, speaker: Speaker, tools: ToolExecutor) -> None:
        self._planner = planner
        self._speaker = speaker
        self._tools = tools

    async def run(self, turn_id: str, request: str) -> BaselineResult:
        timeline = Timeline()
        timeline.record("turn_started", turn_id=turn_id)

        events: List[SemanticEvent] = []
        chunks: List[SpeechChunk] = []
        facts: Dict[str, ToolResult] = {}
        last_sequence = -1

        timeline.record("planner_started")
        async for event in self._planner.plan(turn_id, request):
            if event.sequence <= last_sequence:
                raise ValueError("planner event sequence must be strictly increasing")
            last_sequence = event.sequence
            events.append(event)
            timeline.record("semantic_event", kind=event.kind.value, sequence=event.sequence)

            if event.kind is EventKind.TOOL_CALL:
                call = self._tool_call_from(event)
                timeline.record("tool_started", call_id=call.call_id, tool=call.name)
                result = await self._tools.execute(call)
                timeline.record(
                    "tool_completed",
                    call_id=call.call_id,
                    tool=call.name,
                    succeeded=result.succeeded,
                )
                facts[call.name] = result
            elif event.kind is EventKind.SPEECH_PLAN:
                chunks.append(self._chunk_from(event))

        timeline.record("planner_completed")

        available = frozenset(name for name, result in facts.items() if result.succeeded)
        spoken: List[str] = []
        timeline.record("speaker_started")
        for chunk in chunks:
            if not chunk.resolve(available):
                timeline.record("chunk_blocked", chunk_id=chunk.chunk_id)
                continue
            chunk.transition_to(ChunkState.GENERATING)
            timeline.record("chunk_generating", chunk_id=chunk.chunk_id)
            text = await self._speaker.generate(chunk, facts)
            chunk.transition_to(ChunkState.BUFFERED)
            chunk.transition_to(ChunkState.COMMITTED)
            chunk.transition_to(ChunkState.PLAYED)
            timeline.record("chunk_played", chunk_id=chunk.chunk_id)
            spoken.append(text)
        timeline.record("speaker_completed")
        timeline.record("turn_completed")

        return BaselineResult(
            text=" ".join(part.strip() for part in spoken if part.strip()),
            semantic_events=events,
            chunks=chunks,
            facts=facts,
            timeline=timeline,
        )

    @staticmethod
    def _tool_call_from(event: SemanticEvent) -> ToolCall:
        payload = event.payload
        try:
            return ToolCall(
                call_id=str(payload["call_id"]),
                name=str(payload["tool"]),
                arguments=dict(payload["arguments"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid tool_call payload") from error

    @staticmethod
    def _chunk_from(event: SemanticEvent) -> SpeechChunk:
        payload = event.payload
        try:
            return SpeechChunk(
                chunk_id=str(payload["chunk_id"]),
                goal=str(payload["goal"]),
                dependencies=frozenset(str(item) for item in payload.get("dependencies", [])),
                plan_version=int(payload.get("plan_version", 1)),
                turn_id=event.turn_id,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid speech_plan payload") from error
