import json
from typing import Any, Dict, List

from aether.domain.events import EventKind, SemanticEvent


class SemanticEventStreamParser:
    """Incremental JSONL parser for Planner output.

    Each non-empty line must have the shape:
    {"type": "tool_call", "sequence": 1, "payload": {...}}
    """

    def __init__(self, turn_id: str, repair_sequences: bool = False) -> None:
        if not turn_id.strip():
            raise ValueError("turn_id must not be empty")
        self._turn_id = turn_id
        self._repair_sequences = repair_sequences
        self._buffer = ""
        self._last_sequence = -1

    def feed(self, text: str) -> List[SemanticEvent]:
        self._buffer += text
        lines = self._buffer.split("\n")
        self._buffer = lines.pop()
        return [self._parse_line(line) for line in lines if line.strip()]

    def finish(self) -> List[SemanticEvent]:
        if not self._buffer.strip():
            self._buffer = ""
            return []
        line = self._buffer
        self._buffer = ""
        return [self._parse_line(line)]

    def _parse_line(self, line: str) -> SemanticEvent:
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"planner emitted invalid JSONL: {line!r}") from error
        if not isinstance(value, dict):
            raise ValueError("planner event must be a JSON object")

        event_type = value.get("type")
        sequence = value.get("sequence")
        payload = value.get("payload", {})
        if not isinstance(event_type, str):
            raise ValueError("planner event type must be a string")
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            raise ValueError("planner event sequence must be an integer")
        if sequence <= self._last_sequence:
            if not self._repair_sequences:
                raise ValueError("planner event sequence must be strictly increasing")
            sequence = self._last_sequence + 1
        if not isinstance(payload, dict):
            raise ValueError("planner event payload must be an object")

        try:
            kind = EventKind(event_type)
        except ValueError as error:
            raise ValueError(f"unsupported planner event type: {event_type}") from error
        self._validate_payload(kind, payload)
        self._last_sequence = sequence
        return SemanticEvent(self._turn_id, sequence, kind, payload)

    @staticmethod
    def _validate_payload(kind: EventKind, payload: Dict[str, Any]) -> None:
        if kind is EventKind.TOOL_CALL:
            required = {"call_id", "tool", "arguments"}
            if not required <= payload.keys() or not isinstance(payload.get("arguments"), dict):
                raise ValueError("tool_call payload is missing required fields")
        elif kind is EventKind.SPEECH_PLAN:
            required = {"chunk_id", "goal"}
            if not required <= payload.keys():
                raise ValueError("speech_plan payload is missing required fields")
            dependencies = payload.get("dependencies", [])
            if not isinstance(dependencies, list) or not all(
                isinstance(item, str) for item in dependencies
            ):
                raise ValueError("speech_plan dependencies must be a string list")
