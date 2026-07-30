"""Local dev entry point: `uvicorn aether_api.http.dev_server:app`.

Wires the API to deterministic fakes, not a real model — this lets anyone
run and curl the API vertical slice without downloading weights, matching
the project rule that development machines never load model weights. A
real deployment builds `create_app` with an `LLMPlannerAdapter`/
`LLMSpeakerAdapter` pair over a loaded `SharedLLMBackbone` instead.
"""

from aether.testing.fakes import FakeWeatherTool, ScriptedSharedBackend
from aether_api.auth import ApiKeyStore
from aether_api.http.app import create_app
from aether_api.turn_service import TurnService

_PLANNER_SCRIPT = (
    '{"type":"tool_call","sequence":0,"payload":{"call_id":"weather-1","tool":"weather",'
    '"arguments":{"location":"Almaty"}}}\n'
    '{"type":"speech_plan","sequence":1,"payload":{"chunk_id":"lead-in",'
    '"goal":"Сообщить, что проверка погоды началась","dependencies":[],"safe_to_say":true}}\n'
    '{"type":"speech_plan","sequence":2,"payload":{"chunk_id":"answer",'
    '"goal":"Сообщить подтверждённую погоду","dependencies":["weather"],"safe_to_say":false}}\n'
    '{"type":"turn_complete","sequence":3,"payload":{}}\n'
)

_backend = ScriptedSharedBackend(
    {
        "planner": _PLANNER_SCRIPT,
        "speaker": "Проверка погоды началась, результат ещё ожидается.",
    },
    chunk_size=24,
)
_turn_service = TurnService(_backend, FakeWeatherTool(latency_ms=800))
_api_keys = ApiKeyStore.from_env()

app = create_app(_turn_service, _api_keys)
