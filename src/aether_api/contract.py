"""The public AETHER Text API event contract (plan.md §5-6).

Only these event types ever cross the API boundary. There is deliberately
no event carrying chain-of-thought, hidden state, or system prompts —
`EventMapper` is the single place responsible for keeping it that way.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class PublicEventType(str, Enum):
    TURN_STARTED = "turn.started"
    PLAN_TOOL_STARTED = "plan.tool_started"
    RESPONSE_SAFE_DELTA = "response.safe_delta"
    TOOL_COMPLETED = "tool.completed"
    RESPONSE_DELTA = "response.delta"
    TURN_COMPLETED = "turn.completed"
    TURN_FAILED = "turn.failed"


@dataclass(frozen=True)
class PublicEvent:
    """One SSE-serializable event. See plan.md §6.1 for the field contract."""

    turn_id: str
    sequence: int
    timestamp_ms: float
    type: PublicEventType
    payload: Mapping[str, Any] = field(default_factory=dict)
    revision_id: int = 0

    def to_sse(self) -> str:
        data = {
            "turn_id": self.turn_id,
            "sequence": self.sequence,
            "timestamp_ms": self.timestamp_ms,
            "revision_id": self.revision_id,
            **self.payload,
        }
        return f"event: {self.type.value}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
