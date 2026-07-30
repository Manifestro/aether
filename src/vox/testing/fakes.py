import asyncio
from typing import AsyncIterator, Mapping

from vox.domain.chunks import SpeechChunk
from vox.domain.events import EventKind, SemanticEvent, ToolCall, ToolResult


class WeatherPlanner:
    """Deterministic stand-in for the future Qwen Planner adapter."""

    async def plan(self, turn_id: str, request: str) -> AsyncIterator[SemanticEvent]:
        yield SemanticEvent(
            turn_id=turn_id,
            sequence=0,
            kind=EventKind.INTENT,
            payload={"name": "get_weather", "request": request},
        )
        yield SemanticEvent(
            turn_id=turn_id,
            sequence=1,
            kind=EventKind.TOOL_CALL,
            payload={
                "call_id": "weather-1",
                "tool": "weather",
                "arguments": {"location": "Almaty"},
            },
        )
        yield SemanticEvent(
            turn_id=turn_id,
            sequence=2,
            kind=EventKind.SPEECH_PLAN,
            payload={
                "chunk_id": "lead-in",
                "goal": "Подтвердить проверку погоды в Алматы",
                "dependencies": [],
            },
        )
        yield SemanticEvent(
            turn_id=turn_id,
            sequence=3,
            kind=EventKind.SPEECH_PLAN,
            payload={
                "chunk_id": "weather-answer",
                "goal": "Сообщить подтверждённую погоду и рекомендацию",
                "dependencies": ["weather"],
            },
        )
        yield SemanticEvent(
            turn_id=turn_id,
            sequence=4,
            kind=EventKind.TURN_COMPLETE,
        )


class FakeWeatherTool:
    def __init__(self, latency_ms: int = 0, fail: bool = False) -> None:
        self.latency_ms = latency_ms
        self.fail = fail
        self.calls = []

    async def execute(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)
        if self.fail:
            return ToolResult(call.call_id, call.name, {}, error="weather unavailable")
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            content={"location": "Almaty", "temperature_c": 24, "condition": "rain"},
        )


class DeterministicSpeaker:
    async def generate(
        self,
        chunk: SpeechChunk,
        facts: Mapping[str, ToolResult],
    ) -> str:
        if chunk.chunk_id == "lead-in":
            return "Сейчас проверю погоду в Алматы."
        weather = facts["weather"].content
        return (
            f"Сейчас {weather['temperature_c']} градуса, ожидается дождь — "
            "зонт лучше взять."
        )

