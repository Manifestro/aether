import json
from typing import AsyncIterator, Mapping

from vox.domain.chunks import SpeechChunk
from vox.domain.events import EventKind, SemanticEvent, ToolResult
from vox.model.event_parser import SemanticEventStreamParser
from vox.model.generation import GenerationRequest, GenerationSettings, TextGenerationBackend


_PLANNER_SYSTEM_PROMPT = """You are the VOX-SYNAPSE Planner.
Emit JSONL only: one JSON object per line, with no markdown and no prose.
Every object must contain: type, sequence, payload.
Allowed types: intent, tool_call, speech_plan, tool_error, replan, turn_complete.
Use strictly increasing integer sequence values starting at 0.
Create safe speech_plan chunks without tool dependencies when possible.
Any factual chunk requiring a tool must list that tool name in dependencies.
Do not emit private chain-of-thought. Emit only observable plans and actions.
For a weather request, follow this structural pattern with values adapted to the user:
{"type":"intent","sequence":0,"payload":{"name":"get_weather"}}
{"type":"tool_call","sequence":1,"payload":{"call_id":"weather-1","tool":"weather","arguments":{"location":"Almaty"}}}
{"type":"speech_plan","sequence":2,"payload":{"chunk_id":"lead-in","goal":"Подтвердить проверку погоды","dependencies":[]}}
{"type":"speech_plan","sequence":3,"payload":{"chunk_id":"answer","goal":"Сообщить подтверждённую погоду","dependencies":["weather"]}}
{"type":"turn_complete","sequence":4,"payload":{}}
"""

_SPEAKER_SYSTEM_PROMPT = """You are the VOX-SYNAPSE Speaker.
Produce only the short text that should be spoken for the supplied speech goal.
Use only facts explicitly supplied in the prompt.
Never invent tool results. Do not mention internal chunks, dependencies or JSON.
"""


class QwenPlannerAdapter:
    def __init__(
        self,
        backend: TextGenerationBackend,
        settings: GenerationSettings = GenerationSettings(max_new_tokens=384),
    ) -> None:
        self._backend = backend
        self._settings = settings

    async def plan(self, turn_id: str, request: str) -> AsyncIterator[SemanticEvent]:
        parser = SemanticEventStreamParser(turn_id)
        generation = GenerationRequest(
            session_id=f"planner:{turn_id}",
            role="planner",
            messages=(
                {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": request},
            ),
            settings=self._settings,
        )
        async for text in self._backend.stream(generation):
            for event in parser.feed(text):
                yield event
                if event.kind is EventKind.TURN_COMPLETE:
                    return
        for event in parser.finish():
            yield event
            if event.kind is EventKind.TURN_COMPLETE:
                return


class QwenSpeakerAdapter:
    def __init__(
        self,
        backend: TextGenerationBackend,
        settings: GenerationSettings = GenerationSettings(max_new_tokens=96),
    ) -> None:
        self._backend = backend
        self._settings = settings

    async def generate(
        self,
        chunk: SpeechChunk,
        facts: Mapping[str, ToolResult],
    ) -> str:
        fact_payload = {
            name: result.content
            for name, result in facts.items()
            if result.succeeded and name in chunk.dependencies
        }
        prompt = json.dumps(
            {
                "speech_goal": chunk.goal,
                "verified_facts": fact_payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        generation = GenerationRequest(
            session_id=f"speaker:{chunk.turn_id or 'unscoped'}",
            role="speaker",
            messages=(
                {"role": "system", "content": _SPEAKER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ),
            settings=self._settings,
        )
        parts = []
        async for text in self._backend.stream(generation):
            parts.append(text)
        return "".join(parts).strip()
