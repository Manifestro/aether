import json
from typing import Any, Dict, List

from aether.domain.events import EventKind, SemanticEvent


_STRICT_FORBIDDEN_KINDS = {EventKind.INTENT, EventKind.TOOL_PENDING}


class SemanticEventStreamParser:
    """Incremental JSONL parser for Planner output.

    Each non-empty line must have the shape:
    {"type": "tool_call", "sequence": 1, "payload": {...}, "revision_id": 0}

    ``strict`` enables the constrained production grammar: it rejects event
    kinds that are not part of the published protocol (``intent``,
    ``tool_pending``) and validates every kind's payload shape, not just
    ``tool_call``/``speech_plan``.
    """

    def __init__(
        self,
        turn_id: str,
        repair_sequences: bool = False,
        strict: bool = False,
    ) -> None:
        if not turn_id.strip():
            raise ValueError("turn_id must not be empty")
        self._turn_id = turn_id
        self._repair_sequences = repair_sequences
        self._strict = strict
        self._buffer = ""
        self._last_sequence = -1
        self.repaired_count = 0

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
        revision_id = value.get("revision_id", 0)
        if not isinstance(event_type, str):
            raise ValueError("planner event type must be a string")
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            raise ValueError("planner event sequence must be an integer")
        if sequence <= self._last_sequence:
            if not self._repair_sequences:
                raise ValueError("planner event sequence must be strictly increasing")
            sequence = self._last_sequence + 1
            self.repaired_count += 1
        if not isinstance(payload, dict):
            raise ValueError("planner event payload must be an object")
        if not isinstance(revision_id, int) or isinstance(revision_id, bool) or revision_id < 0:
            raise ValueError("planner event revision_id must be a non-negative integer")

        try:
            kind = EventKind(event_type)
        except ValueError as error:
            raise ValueError(f"unsupported planner event type: {event_type}") from error
        if self._strict and kind in _STRICT_FORBIDDEN_KINDS:
            raise ValueError(f"event kind not allowed in strict grammar: {event_type}")
        self._validate_payload(kind, payload, strict=self._strict)
        self._last_sequence = sequence
        return SemanticEvent(self._turn_id, sequence, kind, payload, revision_id)

    @staticmethod
    def _validate_payload(kind: EventKind, payload: Dict[str, Any], strict: bool) -> None:
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
            if strict:
                safe_to_say = payload.get("safe_to_say")
                if not isinstance(safe_to_say, bool):
                    raise ValueError("speech_plan payload must declare safe_to_say as a boolean")
                if safe_to_say != (len(dependencies) == 0):
                    raise ValueError(
                        "speech_plan safe_to_say must match whether dependencies are empty"
                    )
        elif strict and kind is EventKind.FACT:
            required = {"tool", "content"}
            if not required <= payload.keys() or not isinstance(payload.get("content"), dict):
                raise ValueError("fact payload is missing required fields")
        elif strict and kind is EventKind.TOOL_ERROR:
            required = {"tool", "error"}
            if not required <= payload.keys() or not isinstance(payload.get("error"), str):
                raise ValueError("tool_error payload is missing required fields")
        elif strict and kind is EventKind.REPLAN:
            if not isinstance(payload.get("reason"), str) or not payload["reason"].strip():
                raise ValueError("replan payload must include a non-empty reason")
            cancel_chunk_ids = payload.get("cancel_chunk_ids", [])
            if not isinstance(cancel_chunk_ids, list) or not all(
                isinstance(item, str) for item in cancel_chunk_ids
            ):
                raise ValueError("replan payload cancel_chunk_ids must be a string list")
        elif strict and kind is EventKind.TURN_COMPLETE:
            if payload:
                raise ValueError("turn_complete payload must be empty in strict grammar")
