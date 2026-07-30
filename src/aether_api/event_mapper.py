"""Translates internal `TraceEvent`s into the public event contract.

This is the boundary that decides what a client is allowed to see. Internal
scheduler/decode telemetry (`decode_started`, `chunk_generating`, ...) is
never forwarded — only the semantic-protocol milestones a public API is
allowed to describe. Speech text is only exposed once a chunk reaches
`chunk_committed`: that is the commit horizon in the domain model
(`aether.domain.chunks.ChunkState`), the same point after which the
runtime itself refuses to rewrite what was said.
"""

from typing import Optional

from aether.domain.timeline import TraceEvent
from aether_api.contract import PublicEvent, PublicEventType

_SIMPLE_MAPPING = {
    "turn_started": PublicEventType.TURN_STARTED,
    "tool_started": PublicEventType.PLAN_TOOL_STARTED,
    "tool_completed": PublicEventType.TOOL_COMPLETED,
    "turn_completed": PublicEventType.TURN_COMPLETED,
}


class EventMapper:
    """Stateful per-turn mapper: assigns the public, gap-free sequence numbers."""

    def __init__(self, turn_id: str) -> None:
        self._turn_id = turn_id
        self._sequence = 0

    def map(self, event: TraceEvent) -> Optional[PublicEvent]:
        if event.name == "chunk_committed":
            is_safe = bool(event.attributes.get("safe_to_say"))
            return self._emit(
                PublicEventType.RESPONSE_SAFE_DELTA if is_safe else PublicEventType.RESPONSE_DELTA,
                {"text": event.attributes.get("text", "")},
                event.timestamp_ns,
            )

        public_type = _SIMPLE_MAPPING.get(event.name)
        if public_type is None:
            return None  # internal-only telemetry; not part of the public contract

        return self._emit(public_type, self._payload_for(event), event.timestamp_ns)

    def turn_failed(self, reason: str, timestamp_ns: int = 0) -> PublicEvent:
        return self._emit(PublicEventType.TURN_FAILED, {"reason": reason}, timestamp_ns)

    @staticmethod
    def _payload_for(event: TraceEvent) -> dict:
        if event.name == "tool_started":
            return {"tool": event.attributes.get("tool")}
        if event.name == "tool_completed":
            payload = {
                "tool": event.attributes.get("tool"),
                "succeeded": event.attributes.get("succeeded"),
            }
            if event.attributes.get("succeeded"):
                payload.update(event.attributes.get("content", {}))
            else:
                payload["error"] = event.attributes.get("error", "")
            return payload
        return {}

    def _emit(self, public_type: PublicEventType, payload: dict, timestamp_ns: int) -> PublicEvent:
        event = PublicEvent(
            turn_id=self._turn_id,
            sequence=self._sequence,
            timestamp_ms=timestamp_ns / 1_000_000,
            type=public_type,
            payload=payload,
        )
        self._sequence += 1
        return event
