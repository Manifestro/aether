from aether.domain.chunks import SpeechChunk
from aether.domain.events import SemanticEvent, ToolCall


class SequenceGuard:
    """Rejects a Planner event stream that is not strictly increasing."""

    def __init__(self) -> None:
        self._last_sequence = -1

    def check(self, event: SemanticEvent) -> None:
        if event.sequence <= self._last_sequence:
            raise ValueError("planner event sequence must be strictly increasing")
        self._last_sequence = event.sequence


def tool_call_from(event: SemanticEvent) -> ToolCall:
    payload = event.payload
    try:
        return ToolCall(
            call_id=str(payload["call_id"]),
            name=str(payload["tool"]),
            arguments=dict(payload["arguments"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid tool_call payload") from error


def chunk_from(event: SemanticEvent) -> SpeechChunk:
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
