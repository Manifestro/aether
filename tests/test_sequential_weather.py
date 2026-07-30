import unittest

from aether.domain.chunks import ChunkState
from aether.runtime.sequential import SequentialBaseline
from aether.testing.fakes import DeterministicSpeaker, FakeWeatherTool, WeatherPlanner


class SequentialWeatherScenarioTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_weather_turn(self) -> None:
        tool = FakeWeatherTool()
        runtime = SequentialBaseline(WeatherPlanner(), DeterministicSpeaker(), tool)

        result = await runtime.run("turn-1", "Какая погода в Алматы?")

        self.assertEqual(len(tool.calls), 1)
        self.assertIn("24 градуса", result.text)
        self.assertEqual([chunk.state for chunk in result.chunks], [ChunkState.PLAYED] * 2)
        names = [event.name for event in result.timeline.events]
        self.assertLess(names.index("tool_completed"), names.index("speaker_started"))

    async def test_failed_tool_never_commits_dependent_chunk(self) -> None:
        runtime = SequentialBaseline(
            WeatherPlanner(),
            DeterministicSpeaker(),
            FakeWeatherTool(fail=True),
        )

        result = await runtime.run("turn-2", "Какая погода в Алматы?")

        self.assertEqual(result.text, "Сейчас проверю погоду в Алматы.")
        self.assertEqual(result.chunks[0].state, ChunkState.PLAYED)
        self.assertEqual(result.chunks[1].state, ChunkState.BLOCKED)
        self.assertEqual(
            result.timeline.first("chunk_blocked").attributes["chunk_id"],
            "weather-answer",
        )


if __name__ == "__main__":
    unittest.main()
