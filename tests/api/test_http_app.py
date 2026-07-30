import unittest

from aether.testing.fakes import FakeWeatherTool, ScriptedSharedBackend
from aether_api.auth import ApiKey, ApiKeyStore
from aether_api.turn_service import TurnService

try:
    from fastapi.testclient import TestClient

    from aether_api.http.app import create_app

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

PLANNER_SCRIPT = """{"type":"tool_call","sequence":0,"payload":{"call_id":"weather-1","tool":"weather","arguments":{"location":"Almaty"}}}
{"type":"speech_plan","sequence":1,"payload":{"chunk_id":"lead-in","goal":"lead","dependencies":[],"safe_to_say":true}}
{"type":"speech_plan","sequence":2,"payload":{"chunk_id":"answer","goal":"answer","dependencies":["weather"],"safe_to_say":false}}
{"type":"turn_complete","sequence":3,"payload":{}}
"""


def _client() -> TestClient:
    backend = ScriptedSharedBackend(
        {"planner": PLANNER_SCRIPT, "speaker": "Сейчас проверю погоду в Алматы."},
        chunk_size=7,
    )
    service = TurnService(backend, FakeWeatherTool(latency_ms=1))
    keys = ApiKeyStore({"dev-key": ApiKey(key="dev-key", owner="dev", max_concurrent_turns=1)})
    app = create_app(service, keys)
    return TestClient(app)


@unittest.skipUnless(_HAS_FASTAPI, "fastapi not installed; install the project's 'api' extra to run this")
class HttpAppTests(unittest.TestCase):
    def test_missing_auth_header_is_rejected(self) -> None:
        client = _client()

        response = client.post("/v1/turns", json={"message": "Погода?", "tools": ["weather"]})

        self.assertEqual(response.status_code, 401)

    def test_invalid_api_key_is_rejected(self) -> None:
        client = _client()

        response = client.post(
            "/v1/turns",
            json={"message": "Погода?", "tools": ["weather"]},
            headers={"authorization": "Bearer wrong-key"},
        )

        self.assertEqual(response.status_code, 401)

    def test_tool_outside_sandboxed_allowlist_is_rejected(self) -> None:
        client = _client()

        response = client.post(
            "/v1/turns",
            json={"message": "Погода?", "tools": ["arbitrary-mcp-server"]},
            headers={"authorization": "Bearer dev-key"},
        )

        self.assertEqual(response.status_code, 400)

    def test_valid_turn_streams_sse_events_ending_in_turn_completed(self) -> None:
        client = _client()

        response = client.post(
            "/v1/turns",
            json={"message": "Погода?", "tools": ["weather"]},
            headers={"authorization": "Bearer dev-key"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
        body = response.text
        self.assertIn("event: turn.started\n", body)
        self.assertIn("event: plan.tool_started\n", body)
        self.assertIn("event: tool.completed\n", body)
        self.assertIn("event: turn.completed\n", body)
        self.assertLess(body.index("event: tool.completed"), body.index("event: turn.completed"))
        # No secrets/system prompt leak into the wire payload.
        self.assertNotIn("AETHER Planner", body)
        self.assertNotIn("chain-of-thought", body)


if __name__ == "__main__":
    unittest.main()
