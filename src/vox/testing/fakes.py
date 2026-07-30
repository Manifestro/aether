import asyncio
from typing import AsyncIterator, Dict, List, Mapping

from vox.domain.chunks import SpeechChunk
from vox.domain.events import EventKind, SemanticEvent, ToolCall, ToolResult
from vox.model.generation import GenerationRequest


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


class ScriptedSharedBackend:
    """In-memory stand-in for one shared Qwen model with logical sessions."""

    def __init__(self, scripts: Mapping[str, str], chunk_size: int = 13) -> None:
        self.scripts = dict(scripts)
        self.chunk_size = chunk_size
        self.requests: List[GenerationRequest] = []
        self.session_counts: Dict[str, int] = {}

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        self.requests.append(request)
        self.session_counts[request.session_id] = self.session_counts.get(request.session_id, 0) + 1
        text = self.scripts[request.role]
        if request.role == "speaker" and '"weather"' in request.messages[-1]["content"]:
            text = "Сейчас 24 градуса, ожидается дождь — зонт лучше взять."
        for offset in range(0, len(text), self.chunk_size):
            await asyncio.sleep(0)
            yield text[offset : offset + self.chunk_size]
