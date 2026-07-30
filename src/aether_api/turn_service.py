"""Wires the research runtime into a live, streamable turn.

`DualSessionRuntime.run()` only returns once the whole turn is done; the
`on_event` hook (aether.domain.timeline.Timeline) is what lets this service
forward events to a client while the turn is still in flight. The turn runs
as a background task; this coroutine's only job is to drain the queue that
hook feeds and translate each item through `EventMapper`.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Sequence

from aether.model.generation import TextGenerationBackend
from aether.model.llm_adapters import LLMPlannerAdapter, LLMSpeakerAdapter
from aether.model.protocols import ToolExecutor
from aether.runtime.dual_session import DualSessionRuntime
from aether.runtime.tool_executor import AllowlistToolExecutor
from aether_api.contract import PublicEvent
from aether_api.event_mapper import EventMapper

_DONE = object()


@dataclass(frozen=True)
class TurnRequest:
    message: str
    tools: Sequence[str] = field(default_factory=tuple)


class TurnService:
    """Serves turns for one shared LLM backend and one sandboxed tool executor.

    Per-key/per-tenant limits (rate limit, quota, concurrency cap) are the
    HTTP layer's job (see `aether_api.auth`), applied before a request ever
    reaches `stream_turn` — this class only knows how to run one turn safely.
    """

    def __init__(self, backend: TextGenerationBackend, tool_executor: ToolExecutor) -> None:
        self._backend = backend
        self._tool_executor = tool_executor

    async def stream_turn(
        self, request: TurnRequest, turn_id: Optional[str] = None
    ) -> AsyncIterator[PublicEvent]:
        turn_id = turn_id or uuid.uuid4().hex
        mapper = EventMapper(turn_id)
        queue: "asyncio.Queue[object]" = asyncio.Queue()

        planner = LLMPlannerAdapter(self._backend, tools=list(request.tools))
        speaker = LLMSpeakerAdapter(self._backend)
        tools = AllowlistToolExecutor(request.tools, self._tool_executor)
        runtime = DualSessionRuntime(planner, speaker, tools)

        async def run_turn() -> None:
            try:
                await runtime.run(turn_id, request.message, on_event=queue.put_nowait)
            except BaseException as error:  # noqa: BLE001 - must surface as turn.failed, not vanish
                queue.put_nowait(error)
            finally:
                queue.put_nowait(_DONE)

        runner = asyncio.create_task(run_turn(), name=f"turn:{turn_id}")
        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    return
                if isinstance(item, BaseException):
                    yield mapper.turn_failed(f"{type(item).__name__}: {item}")
                    return
                public_event = mapper.map(item)  # type: ignore[arg-type]
                if public_event is not None:
                    yield public_event
        finally:
            if not runner.done():
                runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)
