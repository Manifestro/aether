import asyncio
import unittest
from typing import AsyncIterator, Mapping

from aether.domain.chunks import ChunkState, SpeechChunk
from aether.domain.events import EventKind, SemanticEvent, ToolResult
from aether.runtime.dual_session import DualSessionRuntime
from aether.testing.fakes import FakeWeatherTool


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


class _ReplanBeforeToolCompletes:
    """Planner that swaps one blocked chunk for another before the tool resolves."""

    async def plan(self, turn_id: str, request: str) -> AsyncIterator[SemanticEvent]:
        sequence = 0
        yield SemanticEvent(
            turn_id, sequence, EventKind.TOOL_CALL,
            {"call_id": "weather-1", "tool": "weather", "arguments": {"location": "Almaty"}},
        )
        sequence += 1
        yield SemanticEvent(
            turn_id, sequence, EventKind.SPEECH_PLAN,
            {"chunk_id": "lead-in", "goal": "lead", "dependencies": []},
        )
        sequence += 1
        yield SemanticEvent(
            turn_id, sequence, EventKind.SPEECH_PLAN,
            {"chunk_id": "old-answer", "goal": "old", "dependencies": ["weather"]},
        )
        sequence += 1
        yield SemanticEvent(
            turn_id, sequence, EventKind.REPLAN,
            {"reason": "better phrasing decided", "cancel_chunk_ids": ["old-answer"]},
            revision_id=1,
        )
        sequence += 1
        yield SemanticEvent(
            turn_id, sequence, EventKind.SPEECH_PLAN,
            {"chunk_id": "new-answer", "goal": "new", "dependencies": ["weather"]},
        )
        sequence += 1
        yield SemanticEvent(turn_id, sequence, EventKind.TURN_COMPLETE, {})


class _ReplanAfterChunkIsPlayed:
    """Planner that tries (and must fail) to cancel already-spoken output."""

    async def plan(self, turn_id: str, request: str) -> AsyncIterator[SemanticEvent]:
        yield SemanticEvent(
            turn_id, 0, EventKind.SPEECH_PLAN,
            {"chunk_id": "lead-in", "goal": "lead", "dependencies": []},
        )
        # Give the speaker task a chance to fully play the dependency-free
        # chunk before the replan tries to cancel it.
        await asyncio.sleep(0.02)
        yield SemanticEvent(
            turn_id, 1, EventKind.REPLAN,
            {"reason": "attempt to rewrite history", "cancel_chunk_ids": ["lead-in"]},
        )
        yield SemanticEvent(turn_id, 2, EventKind.TURN_COMPLETE, {})


class ReplanProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_replan_cancels_blocked_chunk_and_new_chunk_speaks_instead(self) -> None:
        runtime = DualSessionRuntime(
            _ReplanBeforeToolCompletes(), _EchoSpeaker(), FakeWeatherTool(latency_ms=20)
        )

        result = await runtime.run("turn-replan-1", "Какая погода?")

        by_id = {chunk.chunk_id: chunk for chunk in result.chunks}
        self.assertEqual(by_id["lead-in"].state, ChunkState.PLAYED)
        self.assertEqual(by_id["old-answer"].state, ChunkState.CANCELLED)
        self.assertEqual(by_id["new-answer"].state, ChunkState.PLAYED)

        self.assertIn("spoke:lead-in", result.text)
        self.assertNotIn("spoke:old-answer", result.text)
        self.assertIn("spoke:new-answer", result.text)

        cancel_index = _event_index(result, "chunk_cancelled", "old-answer")
        self.assertIsNotNone(cancel_index)
        self.assertIsNone(_event_index(result, "chunk_generating", "old-answer"))

    async def test_replan_cannot_cancel_an_already_played_chunk(self) -> None:
        runtime = DualSessionRuntime(
            _ReplanAfterChunkIsPlayed(), _EchoSpeaker(), FakeWeatherTool()
        )

        result = await runtime.run("turn-replan-2", "Привет")

        lead_in = next(chunk for chunk in result.chunks if chunk.chunk_id == "lead-in")
        self.assertEqual(lead_in.state, ChunkState.PLAYED)
        self.assertIn("spoke:lead-in", result.text)

        rejected_index = _event_index(result, "chunk_cancel_rejected", "lead-in")
        self.assertIsNotNone(rejected_index)
        self.assertIsNone(_event_index(result, "chunk_cancelled", "lead-in"))


if __name__ == "__main__":
    unittest.main()
