import unittest

from vox.model.qwen_adapters import QwenPlannerAdapter, QwenSpeakerAdapter
from vox.model.step_scheduler import InterleavedDecodeScheduler
from vox.runtime.dual_session import DualSessionRuntime
from vox.testing.fakes import FakeTokenStepEngine, FakeWeatherTool
from tests.test_qwen_adapters import PLANNER_SCRIPT


def trace_index(trace, name, role):
    for index, event in enumerate(trace):
        if event.name == name and event.role == role:
            return index
    raise AssertionError(f"trace event not found: {name}/{role}")


class InterleavedDecodeSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_speaker_decodes_while_planner_session_is_still_active(self) -> None:
        engine = FakeTokenStepEngine(
            {
                "planner": PLANNER_SCRIPT,
                "speaker": "Сейчас проверю погоду в Алматы.",
            },
            token_size=32,
            step_delay_ms=1,
        )
        scheduler = InterleavedDecodeScheduler(engine)
        runtime = DualSessionRuntime(
            QwenPlannerAdapter(scheduler),
            QwenSpeakerAdapter(scheduler),
            FakeWeatherTool(latency_ms=80),
        )

        result = await runtime.run("turn-interleaved", "Какая погода в Алматы?")
        trace = scheduler.trace

        self.assertIn("24 градуса", result.text)
        speaker_started = trace_index(trace, "decode_started", "speaker")
        planner_completed = trace_index(trace, "decode_completed", "planner")
        self.assertLess(speaker_started, planner_completed)
        self.assertNotEqual(
            engine.created_states[0],
            engine.created_states[1],
            "logical sessions must own independent decode state",
        )

    async def test_scheduler_gives_speaker_more_steps_when_both_are_active(self) -> None:
        engine = FakeTokenStepEngine(
            {"planner": "P" * 200, "speaker": "S" * 200},
            token_size=1,
        )
        scheduler = InterleavedDecodeScheduler(engine, speaker_weight=3, planner_weight=2)

        from vox.model.generation import GenerationRequest

        async def consume(role):
            request = GenerationRequest(
                session_id=role,
                role=role,
                messages=({"role": "user", "content": role},),
            )
            output = []
            async for piece in scheduler.stream(request):
                output.append(piece)
                if len(output) == 20:
                    break

        await __import__("asyncio").gather(consume("planner"), consume("speaker"))
        starts = [event.role for event in scheduler.trace if event.name == "first_token"]
        self.assertEqual(set(starts), {"planner", "speaker"})


if __name__ == "__main__":
    unittest.main()
