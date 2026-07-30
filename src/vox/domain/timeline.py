import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass(frozen=True)
class TraceEvent:
    name: str
    timestamp_ns: int
    attributes: Dict[str, Any] = field(default_factory=dict)


class Timeline:
    """Append-only monotonic event trace for a single turn."""

    def __init__(self, clock_ns: Callable[[], int] = time.monotonic_ns) -> None:
        self._clock_ns = clock_ns
        self._origin_ns = clock_ns()
        self._events: List[TraceEvent] = []

    def record(self, name: str, **attributes: Any) -> TraceEvent:
        if not name.strip():
            raise ValueError("trace event name must not be empty")
        event = TraceEvent(
            name=name,
            timestamp_ns=self._clock_ns() - self._origin_ns,
            attributes=attributes,
        )
        if self._events and event.timestamp_ns < self._events[-1].timestamp_ns:
            raise RuntimeError("timeline clock moved backwards")
        self._events.append(event)
        return event

    @property
    def events(self) -> List[TraceEvent]:
        return list(self._events)

    def first(self, name: str) -> TraceEvent:
        for event in self._events:
            if event.name == name:
                return event
        raise KeyError(name)

    def elapsed_ms(self, start: str, end: str) -> float:
        return (self.first(end).timestamp_ns - self.first(start).timestamp_ns) / 1_000_000

