import unittest

from vox.runtime.dual_session import DualSessionRuntime
from vox.model.qwen_adapters import QwenPlannerAdapter, QwenSpeakerAdapter
from vox.testing.fakes import FakeWeatherTool, ScriptedSharedBackend


PLANNER_SCRIPT = """{"type":"intent","sequence":0,"payload":{"name":"get_weather"}}
{"type":"tool_call","sequence":1,"payload":{"call_id":"weather-1","tool":"weather","arguments":{"location":"Almaty"}}}
{"type":"speech_plan","sequence":2,"payload":{"chunk_id":"lead-in","goal":"Подтвердить проверку погоды","dependencies":[]}}
{"type":"speech_plan","sequence":3,"payload":{"chunk_id":"answer","goal":"Сообщить погоду","dependencies":["weather"]}}
{"type":"turn_complete","sequence":4,"payload":{}}
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
            QwenPlannerAdapter(backend),
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


if __name__ == "__main__":
    unittest.main()

