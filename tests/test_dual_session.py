import unittest

from aether.domain.chunks import ChunkState
from aether.runtime.dual_session import DualSessionRuntime
from aether.runtime.sequential import SequentialBaseline
from aether.testing.fakes import DeterministicSpeaker, FakeWeatherTool, WeatherPlanner


def event_index(result, name, chunk_id=None):
    for index, event in enumerate(result.timeline.events):
        if event.name != name:
            continue
        if chunk_id is None or event.attributes.get("chunk_id") == chunk_id:
            return index
    raise AssertionError(f"event not found: {name} {chunk_id or ''}")


class DualSessionScenarioTests(unittest.IsolatedAsyncioTestCase):
    async def test_safe_chunk_plays_while_tool_is_pending(self) -> None:
        runtime = DualSessionRuntime(
            WeatherPlanner(),
            DeterministicSpeaker(),
            FakeWeatherTool(latency_ms=20),
        )

        result = await runtime.run("turn-dual-1", "Какая погода в Алматы?")

        lead_in_played = event_index(result, "chunk_played", "lead-in")
        tool_completed = event_index(result, "tool_completed")
        answer_started = event_index(result, "chunk_generating", "weather-answer")
        self.assertLess(lead_in_played, tool_completed)
        self.assertLess(tool_completed, answer_started)
        self.assertIn("24 градуса", result.text)
        self.assertEqual([chunk.state for chunk in result.chunks], [ChunkState.PLAYED] * 2)

    async def test_dual_runtime_changes_event_order_from_sequential_baseline(self) -> None:
        sequential = SequentialBaseline(
            WeatherPlanner(),
            DeterministicSpeaker(),
            FakeWeatherTool(latency_ms=10),
        )
        dual = DualSessionRuntime(
            WeatherPlanner(),
            DeterministicSpeaker(),
            FakeWeatherTool(latency_ms=10),
        )

        sequential_result = await sequential.run("turn-sequential", "Какая погода?")
        dual_result = await dual.run("turn-dual", "Какая погода?")

        self.assertLess(
            event_index(sequential_result, "tool_completed"),
            event_index(sequential_result, "speaker_started"),
        )
        self.assertLess(
            event_index(dual_result, "speaker_started"),
            event_index(dual_result, "tool_completed"),
        )

    async def test_failed_tool_keeps_dependent_chunk_blocked(self) -> None:
        runtime = DualSessionRuntime(
            WeatherPlanner(),
            DeterministicSpeaker(),
            FakeWeatherTool(latency_ms=5, fail=True),
        )

        result = await runtime.run("turn-dual-error", "Какая погода?")

        self.assertEqual(result.text, "Сейчас проверю погоду в Алматы.")
        self.assertEqual(result.chunks[0].state, ChunkState.PLAYED)
        self.assertEqual(result.chunks[1].state, ChunkState.BLOCKED)
        self.assertGreater(
            event_index(result, "chunk_blocked", "weather-answer"),
            event_index(result, "tool_completed"),
        )

    async def test_on_event_streams_events_live_before_the_turn_finishes(self) -> None:
        seen = []
        runtime = DualSessionRuntime(
            WeatherPlanner(),
            DeterministicSpeaker(),
            FakeWeatherTool(latency_ms=5),
        )

        result = await runtime.run("turn-live", "Какая погода?", on_event=seen.append)

        # Same content as the post-hoc timeline, delivered as it happened.
        self.assertEqual([event.name for event in seen], [event.name for event in result.timeline.events])
        self.assertIn("tool_completed", [event.name for event in seen])


if __name__ == "__main__":
    unittest.main()
