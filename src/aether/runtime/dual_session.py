import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from aether.domain.chunks import ChunkState, SpeechChunk
from aether.domain.events import EventKind, SemanticEvent, ToolCall, ToolResult
from aether.domain.timeline import Timeline
from aether.model.protocols import Planner, Speaker, ToolExecutor
from aether.runtime.sequential import SequentialBaseline


@dataclass(frozen=True)
class DualSessionResult:
    text: str
    semantic_events: List[SemanticEvent]
    chunks: List[SpeechChunk]
    facts: Dict[str, ToolResult]
    timeline: Timeline


class DualSessionRuntime:
    """Stage 1 runtime that overlaps planning, tool execution and speaking.

    This runtime still uses adapter-level Planner and Speaker implementations.
    Later Qwen sessions can implement those protocols without changing the
    orchestration and dependency rules tested here.
    """

    def __init__(self, planner: Planner, speaker: Speaker, tools: ToolExecutor) -> None:
        self._planner = planner
        self._speaker = speaker
        self._tools = tools

    async def run(self, turn_id: str, request: str) -> DualSessionResult:
        timeline = Timeline()
        timeline.record("turn_started", turn_id=turn_id)

        events: List[SemanticEvent] = []
        chunks: List[SpeechChunk] = []
        facts: Dict[str, ToolResult] = {}
        spoken: List[str] = []
        dispatched: Set[str] = set()
        tool_tasks: List["asyncio.Task[None]"] = []
        speech_queue: "asyncio.Queue[Optional[SpeechChunk]]" = asyncio.Queue(maxsize=8)
        last_sequence = -1

        async def dispatch_if_ready(chunk: SpeechChunk) -> None:
            available = frozenset(
                name for name, result in facts.items() if result.succeeded
            )
            if chunk.resolve(available) and chunk.chunk_id not in dispatched:
                dispatched.add(chunk.chunk_id)
                timeline.record("chunk_ready", chunk_id=chunk.chunk_id)
                await speech_queue.put(chunk)

        async def execute_tool(call: ToolCall) -> None:
            timeline.record("tool_started", call_id=call.call_id, tool=call.name)
            try:
                result = await self._tools.execute(call)
            except Exception as error:  # Tool boundaries must become observable results.
                result = ToolResult(
                    call_id=call.call_id,
                    name=call.name,
                    content={},
                    error=f"{type(error).__name__}: {error}",
                )
            facts[call.name] = result
            timeline.record(
                "tool_completed",
                call_id=call.call_id,
                tool=call.name,
                succeeded=result.succeeded,
            )
            for chunk in chunks:
                await dispatch_if_ready(chunk)

        async def speaker_worker() -> None:
            started = False
            while True:
                chunk = await speech_queue.get()
                try:
                    if chunk is None:
                        return
                    if not started:
                        timeline.record("speaker_started")
                        started = True
                    chunk.transition_to(ChunkState.GENERATING)
                    timeline.record("chunk_generating", chunk_id=chunk.chunk_id)
                    text = await self._speaker.generate(chunk, facts)
                    chunk.transition_to(ChunkState.BUFFERED)
                    timeline.record("chunk_buffered", chunk_id=chunk.chunk_id)
                    chunk.transition_to(ChunkState.COMMITTED)
                    timeline.record("chunk_committed", chunk_id=chunk.chunk_id)
                    chunk.transition_to(ChunkState.PLAYED)
                    timeline.record("chunk_played", chunk_id=chunk.chunk_id)
                    if text.strip():
                        spoken.append(text.strip())
                finally:
                    speech_queue.task_done()

        speaker_task = asyncio.create_task(speaker_worker(), name=f"speaker:{turn_id}")

        timeline.record("planner_started")
        try:
            async for event in self._planner.plan(turn_id, request):
                if event.sequence <= last_sequence:
                    raise ValueError("planner event sequence must be strictly increasing")
                last_sequence = event.sequence
                events.append(event)
                timeline.record("semantic_event", kind=event.kind.value, sequence=event.sequence)

                if event.kind is EventKind.TOOL_CALL:
                    call = SequentialBaseline._tool_call_from(event)
                    task = asyncio.create_task(
                        execute_tool(call),
                        name=f"tool:{turn_id}:{call.call_id}",
                    )
                    tool_tasks.append(task)
                elif event.kind is EventKind.SPEECH_PLAN:
                    chunk = SequentialBaseline._chunk_from(event)
                    chunks.append(chunk)
                    await dispatch_if_ready(chunk)
        except BaseException:
            for task in tool_tasks:
                task.cancel()
            speaker_task.cancel()
            await asyncio.gather(*tool_tasks, speaker_task, return_exceptions=True)
            raise
        finally:
            timeline.record("planner_completed")

        if tool_tasks:
            await asyncio.gather(*tool_tasks)

        for chunk in chunks:
            await dispatch_if_ready(chunk)
            if chunk.state is ChunkState.BLOCKED:
                timeline.record("chunk_blocked", chunk_id=chunk.chunk_id)

        await speech_queue.put(None)
        await speaker_task
        timeline.record("speaker_completed")
        timeline.record("turn_completed")

        return DualSessionResult(
            text=" ".join(spoken),
            semantic_events=events,
            chunks=chunks,
            facts=facts,
            timeline=timeline,
        )

