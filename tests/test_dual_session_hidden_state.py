import unittest

from aether.runtime.dual_session import DualSessionRuntime
from aether.testing.fakes import (
    FakeHiddenStateVoiceHead,
    FakeVoiceHead,
    FakeWeatherTool,
    HiddenStateSpeaker,
    WeatherPlanner,
)


class HiddenStateVoiceHeadWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_differs_by_hidden_state_even_with_identical_text(self) -> None:
        speaker = HiddenStateSpeaker(
            hidden_states={"lead-in": [1.0, 2.0, 3.0], "weather-answer": [9.0, 9.0, 9.0]},
            text="the exact same words every time",
        )
        voice_head = FakeHiddenStateVoiceHead()
        runtime = DualSessionRuntime(
            WeatherPlanner(), speaker, FakeWeatherTool(latency_ms=5), voice_head=voice_head
        )

        result = await runtime.run("turn-hidden-state", "Какая погода?")

        self.assertEqual(result.text, "the exact same words every time the exact same words every time")
        # Same text both times -- if the Voice Head were secretly reading
        # `text`, these two chunks would be identical. They must not be.
        self.assertNotEqual(result.audio["lead-in"].tokens, result.audio["weather-answer"].tokens)
        self.assertEqual(
            [call for call in voice_head.calls],
            [("lead-in", [1.0, 2.0, 3.0]), ("weather-answer", [9.0, 9.0, 9.0])],
        )

    async def test_text_conditioned_voice_head_ignores_absent_hidden_state(self) -> None:
        # DeterministicSpeaker-style fakes expose no `last_hidden_state`;
        # a text-conditioned Voice Head (Stage 4) must keep working exactly
        # as before -- `hidden_state` arrives as None and is ignored.
        from aether.testing.fakes import DeterministicSpeaker

        voice_head = FakeVoiceHead()
        runtime = DualSessionRuntime(
            WeatherPlanner(), DeterministicSpeaker(), FakeWeatherTool(latency_ms=5), voice_head=voice_head
        )

        result = await runtime.run("turn-no-hidden-state", "Какая погода?")

        self.assertIn("lead-in", result.audio)
        self.assertIn("weather-answer", result.audio)

    async def test_hidden_state_voice_head_fails_closed_when_not_loaded(self) -> None:
        from aether.model.voice_head import HiddenStateVoiceHead, HiddenStateVoiceHeadConfig

        head = HiddenStateVoiceHead(HiddenStateVoiceHeadConfig(hidden_state_dim=4))
        with self.assertRaises(RuntimeError):
            await head.synthesize(chunk=None, text="x", facts={}, hidden_state=[1.0])


if __name__ == "__main__":
    unittest.main()
