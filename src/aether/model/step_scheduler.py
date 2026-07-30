import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol

from aether.model.generation import GenerationRequest, TextGenerationBackend


@dataclass(frozen=True)
class DecodeStep:
    text: str = ""
    finished: bool = False
    token_count: int = 1


class TokenStepEngine(Protocol):
    """Low-level engine owning model-specific session/KV state."""

    async def create(self, request: GenerationRequest) -> Any:
        ...

    async def step(self, state: Any) -> DecodeStep:
        ...

    async def close(self, state: Any) -> None:
        ...


@dataclass(frozen=True)
class DecodeTraceEvent:
    name: str
    timestamp_ns: int
    absolute_timestamp_ns: int
    session_id: str
    role: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _ScheduledSession:
    request: GenerationRequest
    state: Any
    queue: "asyncio.Queue[Any]"
    token_count: int = 0
    decode_started: bool = False
    first_token_emitted: bool = False
    closed: bool = False


@dataclass(frozen=True)
class _Failure:
    error: BaseException


_END = object()


class InterleavedDecodeScheduler(TextGenerationBackend):
    """Interleaves token steps for logical sessions over one shared engine.

    When Planner and Speaker are both active, the default weighted cycle assigns
    three steps to Speaker and two to Planner. Before Speaker exists, Planner
    receives every step, so early tool decisions are not delayed.
    """

    def __init__(
        self,
        engine: TokenStepEngine,
        speaker_weight: int = 3,
        planner_weight: int = 2,
        queue_size: int = 32,
    ) -> None:
        if speaker_weight < 1 or planner_weight < 1:
            raise ValueError("scheduler weights must be positive")
        self._engine = engine
        self._speaker_weight = speaker_weight
        self._planner_weight = planner_weight
        self._queue_size = queue_size
        self._sessions: Dict[str, _ScheduledSession] = {}
        self._lock: Optional[asyncio.Lock] = None
        self._runner: Optional["asyncio.Task[None]"] = None
        self._cycle_index = 0
        self._origin_ns = time.monotonic_ns()
        self._trace: List[DecodeTraceEvent] = []

    @property
    def trace(self) -> List[DecodeTraceEvent]:
        return list(self._trace)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        lock = self._get_lock()
        state = await self._engine.create(request)
        session = _ScheduledSession(
            request=request,
            state=state,
            queue=asyncio.Queue(maxsize=self._queue_size),
        )
        async with lock:
            if request.session_id in self._sessions:
                await self._engine.close(state)
                raise ValueError(f"session already active: {request.session_id}")
            self._sessions[request.session_id] = session
            self._record("session_registered", session)
            if self._runner is None or self._runner.done():
                self._runner = asyncio.create_task(self._run(), name="aether-decode-scheduler")

        try:
            while True:
                item = await session.queue.get()
                try:
                    if item is _END:
                        return
                    if isinstance(item, _Failure):
                        raise item.error
                    yield item
                finally:
                    session.queue.task_done()
        finally:
            await self._remove_session(request.session_id, cancelled=True)

    async def complete(self, session_id: str, reason: str = "consumer_stop") -> None:
        """Mark a stream as normally complete before model EOS.

        Planner uses this when the semantic protocol emits `turn_complete`.
        It prevents an intentional early stop from being reported as a failure
        or barge-in cancellation.
        """
        async with self._get_lock():
            session = self._sessions.pop(session_id, None)
            if session is not None:
                session.closed = True
        if session is None:
            return
        self._record("decode_completed", session, token_count=session.token_count, reason=reason)
        await self._engine.close(session.state)

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def _run(self) -> None:
        while True:
            async with self._get_lock():
                session = self._pick_session()
                if session is None:
                    self._runner = None
                    return

            try:
                if not session.decode_started:
                    session.decode_started = True
                    self._record("decode_started", session)
                result = await self._engine.step(session.state)
                if session.closed:
                    continue
                session.token_count += result.token_count
                if result.text:
                    if not session.first_token_emitted:
                        session.first_token_emitted = True
                        self._record("first_token", session)
                    await session.queue.put(result.text)
                if result.finished:
                    self._record(
                        "decode_completed",
                        session,
                        token_count=session.token_count,
                    )
                    await session.queue.put(_END)
                    await self._remove_session(session.request.session_id, cancelled=False)
            except BaseException as error:
                self._record("decode_failed", session, error=repr(error))
                await session.queue.put(_Failure(error))
                await self._remove_session(session.request.session_id, cancelled=False)

            # Allow MCP tasks and stream consumers to run between model steps.
            await asyncio.sleep(0)

    def _pick_session(self) -> Optional[_ScheduledSession]:
        if not self._sessions:
            return None
        planners = [s for s in self._sessions.values() if s.request.role == "planner"]
        speakers = [s for s in self._sessions.values() if s.request.role == "speaker"]
        if not planners:
            return speakers[0]
        if not speakers:
            return planners[0]

        cycle = ["speaker"] * self._speaker_weight + ["planner"] * self._planner_weight
        role = cycle[self._cycle_index % len(cycle)]
        self._cycle_index += 1
        return speakers[0] if role == "speaker" else planners[0]

    async def _remove_session(self, session_id: str, cancelled: bool) -> None:
        async with self._get_lock():
            session = self._sessions.pop(session_id, None)
        if session is None:
            return
        session.closed = True
        if cancelled:
            self._record("decode_cancelled", session, token_count=session.token_count)
        await self._engine.close(session.state)

    def _record(self, name: str, session: _ScheduledSession, **attributes: Any) -> None:
        absolute_ns = time.monotonic_ns()
        self._trace.append(
            DecodeTraceEvent(
                name=name,
                timestamp_ns=absolute_ns - self._origin_ns,
                absolute_timestamp_ns=absolute_ns,
                session_id=session.request.session_id,
                role=session.request.role,
                attributes=attributes,
            )
        )
