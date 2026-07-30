import unittest

from aether.testing.fakes import FakeWeatherTool, ScriptedSharedBackend
from aether_api.contract import PublicEventType
from aether_api.turn_service import TurnRequest, TurnService

PLANNER_SCRIPT = """{"type":"tool_call","sequence":0,"payload":{"call_id":"weather-1","tool":"weather","arguments":{"location":"Almaty"}}}
{"type":"speech_plan","sequence":1,"payload":{"chunk_id":"lead-in","goal":"lead","dependencies":[],"safe_to_say":true}}
{"type":"speech_plan","sequence":2,"payload":{"chunk_id":"answer","goal":"answer","dependencies":["weather"],"safe_to_say":false}}
{"type":"turn_complete","sequence":3,"payload":{}}
"""


class TurnServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_turn_emits_the_expected_public_sequence(self) -> None:
        backend = ScriptedSharedBackend(
            {"planner": PLANNER_SCRIPT, "speaker": "Сейчас проверю погоду в Алматы."},
            chunk_size=7,
        )
        service = TurnService(backend, FakeWeatherTool(latency_ms=5))

        events = [
            event
            async for event in service.stream_turn(TurnRequest(message="Погода?", tools=["weather"]))
        ]

        self.assertEqual(
            [event.type for event in events],
            [
                PublicEventType.TURN_STARTED,
                PublicEventType.PLAN_TOOL_STARTED,
                PublicEventType.RESPONSE_SAFE_DELTA,
                PublicEventType.TOOL_COMPLETED,
                PublicEventType.RESPONSE_DELTA,
                PublicEventType.TURN_COMPLETED,
            ],
        )
        self.assertEqual(events[3].payload["temperature_c"], 24)
        self.assertTrue(all(event.turn_id == events[0].turn_id for event in events))
        self.assertEqual([event.sequence for event in events], list(range(len(events))))

    async def test_sequence_is_monotonic_per_turn_id_when_called_twice(self) -> None:
        backend = ScriptedSharedBackend(
            {"planner": PLANNER_SCRIPT, "speaker": "Сейчас проверю погоду в Алматы."},
            chunk_size=7,
        )
        service = TurnService(backend, FakeWeatherTool(latency_ms=1))
        request = TurnRequest(message="Погода?", tools=["weather"])

        first = [event async for event in service.stream_turn(request, turn_id="turn-a")]
        second = [event async for event in service.stream_turn(request, turn_id="turn-b")]

        self.assertTrue(all(event.turn_id == "turn-a" for event in first))
        self.assertTrue(all(event.turn_id == "turn-b" for event in second))
        # Each turn's own sequence restarts at 0 - sequence is per-turn, not global.
        self.assertEqual(first[0].sequence, 0)
        self.assertEqual(second[0].sequence, 0)

    async def test_planner_protocol_violation_becomes_turn_failed(self) -> None:
        backend = ScriptedSharedBackend(
            {"planner": '{"type":"tool_call","sequence":0,"payload":{}}\n', "speaker": ""}
        )
        service = TurnService(backend, FakeWeatherTool())

        events = [
            event
            async for event in service.stream_turn(TurnRequest(message="x", tools=["weather"]))
        ]

        self.assertEqual(events[0].type, PublicEventType.TURN_STARTED)
        self.assertEqual(events[-1].type, PublicEventType.TURN_FAILED)
        self.assertIn("tool_call payload", events[-1].payload["reason"])

    async def test_disallowed_tool_never_leaks_a_fabricated_fact(self) -> None:
        # No "weather" in the granted tools: AllowlistToolExecutor must block it
        # before FakeWeatherTool ever answers, so tool.completed reports failure.
        backend = ScriptedSharedBackend(
            {"planner": PLANNER_SCRIPT, "speaker": "Сейчас проверю погоду в Алматы."},
            chunk_size=7,
        )
        service = TurnService(backend, FakeWeatherTool(latency_ms=1))

        events = [
            event async for event in service.stream_turn(TurnRequest(message="Погода?", tools=[]))
        ]

        tool_completed = next(e for e in events if e.type == PublicEventType.TOOL_COMPLETED)
        self.assertFalse(tool_completed.payload["succeeded"])
        self.assertIn("not allowed", tool_completed.payload["error"])
        self.assertNotIn(PublicEventType.RESPONSE_DELTA, [e.type for e in events])


if __name__ == "__main__":
    unittest.main()
