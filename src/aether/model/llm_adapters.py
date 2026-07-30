import json
from typing import AsyncIterator, Mapping, Optional, Sequence

from aether.domain.chunks import SpeechChunk
from aether.domain.events import EventKind, SemanticEvent, ToolResult
from aether.model.event_parser import SemanticEventStreamParser
from aether.model.generation import GenerationRequest, GenerationSettings, TextGenerationBackend


_PLANNER_SYSTEM_PROMPT_TEMPLATE = """You are the AETHER Planner.
Emit JSONL only: one JSON object per line, with no markdown and no prose.
Every object must contain: type, sequence, payload.
Allowed types: tool_call, speech_plan, tool_error, replan, turn_complete.
There is no separate "intent" event. Never emit one.
Use strictly increasing integer sequence values starting at 0.
Allowed tools for this turn: {tool_list}.
Never call a tool outside this list, and never invent a tool name.
If the request does not need any of the allowed tools, do not emit a
tool_call at all — answer directly with a dependency-free speech_plan.
When a tool is genuinely needed and allowed, emit tool_call as the first event.
Create safe speech_plan chunks without tool dependencies when possible.
Any factual chunk requiring a tool must list that tool name in dependencies.
Every speech_plan payload must include a boolean "safe_to_say" field that is
true only when dependencies is empty.
The turn_complete payload must be an empty object.
Do not emit private chain-of-thought. Emit only observable plans and actions.
For a weather request with "weather" allowed, follow this structural pattern
with values adapted to the user:
{{"type":"tool_call","sequence":0,"payload":{{"call_id":"weather-1","tool":"weather","arguments":{{"location":"Almaty"}}}}}}
{{"type":"speech_plan","sequence":1,"payload":{{"chunk_id":"lead-in","goal":"Сообщить, что проверка погоды началась и результат ещё ожидается","dependencies":[],"safe_to_say":true}}}}
{{"type":"speech_plan","sequence":2,"payload":{{"chunk_id":"answer","goal":"Сообщить подтверждённую погоду","dependencies":["weather"],"safe_to_say":false}}}}
{{"type":"turn_complete","sequence":3,"payload":{{}}}}
For a request that needs no tool, skip tool_call entirely and go straight to
a single safe speech_plan followed by turn_complete.
"""

_SPEAKER_SYSTEM_PROMPT = """You are the AETHER Speaker.
Produce only the short text that should be spoken for the supplied speech goal.
Use only facts explicitly supplied in the prompt.
Never invent tool results. Do not mention internal chunks, dependencies or JSON.
"""


class QwenPlannerAdapter:
    def __init__(
        self,
        backend: TextGenerationBackend,
        tools: Sequence[str],
        settings: GenerationSettings = GenerationSettings(max_new_tokens=384),
    ) -> None:
        self._backend = backend
        self._tools = tuple(tools)
        self._settings = settings
        # Exposed so callers (experiments, telemetry) can observe sequence
        # repairs instead of them being silently swallowed by the parser.
        self.last_parser: Optional[SemanticEventStreamParser] = None

    async def plan(self, turn_id: str, request: str) -> AsyncIterator[SemanticEvent]:
        # Open models occasionally repeat a sequence value while streaming.
        # Normalize that adapter-boundary defect but keep it observable via
        # `last_parser.repaired_count` rather than a fully silent repair, and
        # enforce the constrained production event grammar (strict=True).
        parser = SemanticEventStreamParser(turn_id, repair_sequences=True, strict=True)
        self.last_parser = parser
        system_prompt = _PLANNER_SYSTEM_PROMPT_TEMPLATE.format(
            tool_list=", ".join(self._tools) if self._tools else "(none — never call a tool this turn)"
        )
        generation = GenerationRequest(
            session_id=f"planner:{turn_id}",
            role="planner",
            messages=(
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request},
            ),
            settings=self._settings,
        )
        async for text in self._backend.stream(generation):
            for event in parser.feed(text):
                yield event
                if event.kind is EventKind.TURN_COMPLETE:
                    complete = getattr(self._backend, "complete", None)
                    if complete is not None:
                        await complete(generation.session_id, reason="turn_complete")
                    return
        for event in parser.finish():
            yield event
            if event.kind is EventKind.TURN_COMPLETE:
                complete = getattr(self._backend, "complete", None)
                if complete is not None:
                    await complete(generation.session_id, reason="turn_complete")
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
