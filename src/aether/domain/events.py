from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class EventKind(str, Enum):
    INTENT = "intent"
    TOOL_CALL = "tool_call"
    TOOL_PENDING = "tool_pending"
    SPEECH_PLAN = "speech_plan"
    FACT = "fact"
    TOOL_ERROR = "tool_error"
    REPLAN = "replan"
    TURN_COMPLETE = "turn_complete"


@dataclass(frozen=True)
class SemanticEvent:
    """Observable communication from Planner to the runtime and Speaker."""

    turn_id: str
    sequence: int
    kind: EventKind
    payload: Mapping[str, Any] = field(default_factory=dict)
    revision_id: int = 0

    def __post_init__(self) -> None:
        if not self.turn_id.strip():
            raise ValueError("turn_id must not be empty")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.revision_id < 0:
            raise ValueError("revision_id must be non-negative")


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    content: Mapping[str, Any]
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return not self.error

