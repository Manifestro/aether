import unittest

from aether.runtime.dual_session import DualSessionRuntime
from aether.model.qwen_adapters import QwenPlannerAdapter, QwenSpeakerAdapter
from aether.testing.fakes import FakeWeatherTool, ScriptedSharedBackend


PLANNER_SCRIPT = """{"type":"tool_call","sequence":0,"payload":{"call_id":"weather-1","tool":"weather","arguments":{"location":"Almaty"}}}
{"type":"speech_plan","sequence":1,"payload":{"chunk_id":"lead-in","goal":"Подтвердить проверку погоды","dependencies":[],"safe_to_say":true}}
{"type":"speech_plan","sequence":2,"payload":{"chunk_id":"answer","goal":"Сообщить погоду","dependencies":["weather"],"safe_to_say":false}}
{"type":"turn_complete","sequence":3,"payload":{}}
"""


def event_index(result, name, chunk_id=None):
    for index, event in enumerate(result.timeline.events):
        if event.name == name and (
            chunk_id is None or event.attributes.get("chunk_id") == chunk_id
        ):
            return index
    raise AssertionError(f"event not found: {name}")


class QwenAdapterContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_backend_serves_independent_planner_and_speaker_sessions(self) -> None:
        backend = ScriptedSharedBackend(
            {
                "planner": PLANNER_SCRIPT,
                "speaker": "Сейчас проверю погоду в Алматы.",
            },
            chunk_size=7,
        )
        runtime = DualSessionRuntime(
            QwenPlannerAdapter(backend, tools=["weather"]),
            QwenSpeakerAdapter(backend),
            FakeWeatherTool(latency_ms=10),
        )

        result = await runtime.run("turn-model-ready", "Какая погода в Алматы?")

        self.assertIn("24 градуса", result.text)
        self.assertEqual(set(backend.session_counts), {"planner:turn-model-ready", "speaker:turn-model-ready"})
        self.assertEqual(backend.session_counts["planner:turn-model-ready"], 1)
        self.assertEqual(backend.session_counts["speaker:turn-model-ready"], 2)
        self.assertLess(
            event_index(result, "chunk_played", "lead-in"),
            event_index(result, "tool_completed"),
        )
        self.assertLess(
            event_index(result, "tool_completed"),
            event_index(result, "chunk_generating", "answer"),
        )

    async def test_planner_stops_at_turn_complete_and_ignores_trailing_garbage(self) -> None:
        backend = ScriptedSharedBackend(
            {
                "planner": PLANNER_SCRIPT + '{"unfinished',
                "speaker": "Готово.",
            },
            chunk_size=5,
        )
        planner = QwenPlannerAdapter(backend, tools=["weather"])

        events = [event async for event in planner.plan("turn-stop", "Погода?")]

        self.assertEqual(events[-1].kind.value, "turn_complete")
        self.assertEqual(len(events), 4)

    async def test_system_prompt_declares_the_granted_tool_allowlist(self) -> None:
        backend = ScriptedSharedBackend({"planner": '{"type":"turn_complete","sequence":0,"payload":{}}\n'})
        planner = QwenPlannerAdapter(backend, tools=["currency"])

        _ = [event async for event in planner.plan("turn-tools", "Курс доллара?")]

        system_message = backend.requests[0].messages[0]["content"]
        self.assertIn("Allowed tools for this turn: currency", system_message)

    async def test_system_prompt_forbids_any_tool_when_none_granted(self) -> None:
        backend = ScriptedSharedBackend({"planner": '{"type":"turn_complete","sequence":0,"payload":{}}\n'})
        planner = QwenPlannerAdapter(backend, tools=[])

        _ = [event async for event in planner.plan("turn-no-tools", "Привет!")]

        system_message = backend.requests[0].messages[0]["content"]
        self.assertIn("never call a tool this turn", system_message)


if __name__ == "__main__":
    unittest.main()
