import asyncio
import unittest
from typing import AsyncIterator, Mapping

from aether.domain.chunks import ChunkState, SpeechChunk
from aether.domain.events import EventKind, SemanticEvent, ToolResult
from aether.runtime.dual_session import DualSessionRuntime
from aether.testing.fakes import (
    DeterministicSpeaker,
    FakeVoiceHead,
    FakeWeatherTool,
    WeatherPlanner,
)


def _event_index(result, name, chunk_id=None):
    for index, event in enumerate(result.timeline.events):
        if event.name != name:
            continue
        if chunk_id is None or event.attributes.get("chunk_id") == chunk_id:
            return index
    return None


class _EchoSpeaker:
    async def generate(self, chunk: SpeechChunk, facts: Mapping[str, ToolResult]) -> str:
        return f"spoke:{chunk.chunk_id}"


class _ReplanWhileAudioInFlight:
    """Cancels a dependency-free chunk shortly after it starts generating.

    The delay is tuned to land while `FakeVoiceHead`'s synthesis (given a
    latency) is in flight — the same race `test_replan.py` exercises for
    text, now for the audio path added in this experiment.
    """

    async def plan(self, turn_id: str, request: str) -> AsyncIterator[SemanticEvent]:
        yield SemanticEvent(
            turn_id, 0, EventKind.SPEECH_PLAN,
            {"chunk_id": "lead-in", "goal": "lead", "dependencies": []},
        )
        await asyncio.sleep(0.02)
        yield SemanticEvent(
            turn_id, 1, EventKind.REPLAN,
            {"reason": "cancel mid-synthesis", "cancel_chunk_ids": ["lead-in"]},
        )
        yield SemanticEvent(turn_id, 2, EventKind.TURN_COMPLETE, {})


class _ReplanBeforeChunkIsDispatched:
    """Cancels a chunk before the speaker (and voice head) ever touch it."""

    async def plan(self, turn_id: str, request: str) -> AsyncIterator[SemanticEvent]:
        yield SemanticEvent(
            turn_id, 0, EventKind.SPEECH_PLAN,
            {"chunk_id": "answer", "goal": "answer", "dependencies": ["weather"]},
        )
        yield SemanticEvent(
            turn_id, 1, EventKind.REPLAN,
            {"reason": "never needed", "cancel_chunk_ids": ["answer"]},
        )
        yield SemanticEvent(turn_id, 2, EventKind.TURN_COMPLETE, {})


class DualSessionAudioTests(unittest.IsolatedAsyncioTestCase):
    async def test_voice_head_none_is_fully_backward_compatible(self) -> None:
        runtime = DualSessionRuntime(
            WeatherPlanner(), DeterministicSpeaker(), FakeWeatherTool(latency_ms=5)
        )

        result = await runtime.run("turn-audio-none", "Какая погода?")

        self.assertEqual(result.audio, {})
        self.assertIsNone(_event_index(result, "chunk_audio_generating"))

    async def test_dependent_chunk_audio_never_buffered_before_tool_completes(self) -> None:
        voice_head = FakeVoiceHead()
        runtime = DualSessionRuntime(
            WeatherPlanner(),
            DeterministicSpeaker(),
            FakeWeatherTool(latency_ms=20),
            voice_head=voice_head,
        )

        result = await runtime.run("turn-audio-safety", "Какая погода в Алматы?")

        tool_completed = _event_index(result, "tool_completed")
        answer_audio_buffered = _event_index(result, "chunk_audio_buffered", "weather-answer")
        lead_in_audio_buffered = _event_index(result, "chunk_audio_buffered", "lead-in")

        self.assertIsNotNone(tool_completed)
        self.assertIsNotNone(answer_audio_buffered)
        self.assertIsNotNone(lead_in_audio_buffered)
        # The safety invariant: the tool-dependent chunk's audio is only
        # buffered after the tool result is confirmed. The independent
        # lead-in chunk's audio has no such ordering constraint.
        self.assertLess(tool_completed, answer_audio_buffered)
        self.assertIn("weather-answer", result.audio)
        self.assertIn("lead-in", result.audio)
        self.assertEqual(voice_head.calls, ["lead-in", "weather-answer"])

    async def test_replan_cancels_chunk_while_audio_synthesis_is_in_flight(self) -> None:
        voice_head = FakeVoiceHead(latency_ms=50)
        runtime = DualSessionRuntime(
            _ReplanWhileAudioInFlight(), _EchoSpeaker(), FakeWeatherTool(), voice_head=voice_head
        )

        result = await runtime.run("turn-audio-replan", "hello")

        chunk = next(c for c in result.chunks if c.chunk_id == "lead-in")
        self.assertEqual(chunk.state, ChunkState.CANCELLED)
        self.assertNotIn("lead-in", result.audio)
        self.assertIsNotNone(_event_index(result, "chunk_audio_generating", "lead-in"))
        self.assertIsNotNone(
            _event_index(result, "chunk_audio_skip_cancelled_after_synthesize", "lead-in")
        )
        self.assertIsNone(_event_index(result, "chunk_audio_buffered", "lead-in"))
        self.assertIsNone(_event_index(result, "chunk_committed", "lead-in"))

    async def test_replan_cancels_chunk_before_voice_head_is_ever_called(self) -> None:
        voice_head = FakeVoiceHead()
        runtime = DualSessionRuntime(
            _ReplanBeforeChunkIsDispatched(),
            _EchoSpeaker(),
            FakeWeatherTool(latency_ms=20),
            voice_head=voice_head,
        )

        result = await runtime.run("turn-audio-replan-early", "hello")

        chunk = next(c for c in result.chunks if c.chunk_id == "answer")
        self.assertEqual(chunk.state, ChunkState.CANCELLED)
        self.assertEqual(voice_head.calls, [])
        self.assertNotIn("answer", result.audio)


if __name__ == "__main__":
    unittest.main()
