import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from aether.domain.chunks import ChunkState, SpeechChunk
from aether.domain.events import EventKind, SemanticEvent, ToolCall, ToolResult
from aether.domain.timeline import Timeline
from aether.model.protocols import Planner, Speaker, ToolExecutor
from aether.runtime.converters import SequenceGuard, chunk_from, tool_call_from


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
        chunks_by_id: Dict[str, SpeechChunk] = {}
        facts: Dict[str, ToolResult] = {}
        spoken: List[str] = []
        dispatched: Set[str] = set()
        tool_tasks: List["asyncio.Task[None]"] = []
        speech_queue: "asyncio.Queue[Optional[SpeechChunk]]" = asyncio.Queue(maxsize=8)
        sequence_guard = SequenceGuard()

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
                    # A replan may have cancelled this chunk while it was
                    # queued (buffered, not yet generating). Never speak it.
                    if chunk.state is ChunkState.CANCELLED:
                        timeline.record("chunk_skip_cancelled", chunk_id=chunk.chunk_id)
                        continue
                    if not started:
                        timeline.record("speaker_started")
                        started = True
                    chunk.transition_to(ChunkState.GENERATING)
                    timeline.record("chunk_generating", chunk_id=chunk.chunk_id)
                    text = await self._speaker.generate(chunk, facts)
                    # A replan may have cancelled this chunk while `generate`
                    # was in flight. The committed prefix (already-played
                    # chunks) is untouched; only this still-buffered output
                    # is dropped.
                    if chunk.state is ChunkState.CANCELLED:
                        timeline.record("chunk_skip_cancelled_after_generate", chunk_id=chunk.chunk_id)
                        continue
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
                sequence_guard.check(event)
                events.append(event)
                timeline.record("semantic_event", kind=event.kind.value, sequence=event.sequence)

                if event.kind is EventKind.TOOL_CALL:
                    call = tool_call_from(event)
                    task = asyncio.create_task(
                        execute_tool(call),
                        name=f"tool:{turn_id}:{call.call_id}",
                    )
                    tool_tasks.append(task)
                elif event.kind is EventKind.SPEECH_PLAN:
                    chunk = chunk_from(event)
                    chunks.append(chunk)
                    chunks_by_id[chunk.chunk_id] = chunk
                    await dispatch_if_ready(chunk)
                elif event.kind is EventKind.REPLAN:
                    for chunk_id in event.payload.get("cancel_chunk_ids", []):
                        target = chunks_by_id.get(chunk_id)
                        if target is None:
                            timeline.record(
                                "replan_unknown_chunk",
                                chunk_id=chunk_id,
                                revision_id=event.revision_id,
                            )
                        elif target.cancellable:
                            target.transition_to(ChunkState.CANCELLED)
                            timeline.record(
                                "chunk_cancelled",
                                chunk_id=chunk_id,
                                revision_id=event.revision_id,
                            )
                        else:
                            # Committed/played speech is never rewritten.
                            timeline.record(
                                "chunk_cancel_rejected",
                                chunk_id=chunk_id,
                                revision_id=event.revision_id,
                                state=target.state.value,
                            )
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

