import random
import unittest
from typing import AsyncIterator, Mapping

from aether.domain.chunks import ChunkState, SpeechChunk
from aether.domain.events import EventKind, SemanticEvent, ToolResult
from aether.runtime.dual_session import DualSessionRuntime
from aether.testing.fakes import FakeWeatherTool

TURNS = 300


class _RandomReplanPlanner:
    """Randomly plans, and sometimes cancels part of the plan, before turn_complete."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    async def plan(self, turn_id: str, request: str) -> AsyncIterator[SemanticEvent]:
        sequence = 0
        has_tool = self._rng.random() < 0.7
        if has_tool:
            yield SemanticEvent(
                turn_id,
                sequence,
                EventKind.TOOL_CALL,
                {
                    "call_id": f"{turn_id}-call",
                    "tool": "weather",
                    "arguments": {"location": "Almaty"},
                },
            )
            sequence += 1

        chunk_ids = []
        for index in range(self._rng.randint(2, 4)):
            dependent = has_tool and self._rng.random() < 0.5
            chunk_id = f"{turn_id}-chunk-{index}"
            chunk_ids.append(chunk_id)
            yield SemanticEvent(
                turn_id,
                sequence,
                EventKind.SPEECH_PLAN,
                {
                    "chunk_id": chunk_id,
                    "goal": f"say something {index}",
                    "dependencies": ["weather"] if dependent else [],
                },
            )
            sequence += 1

        if chunk_ids and self._rng.random() < 0.5:
            cancel_ids = self._rng.sample(chunk_ids, self._rng.randint(1, len(chunk_ids)))
            yield SemanticEvent(
                turn_id,
                sequence,
                EventKind.REPLAN,
                {"reason": "fuzz replan", "cancel_chunk_ids": cancel_ids},
                revision_id=1,
            )
            sequence += 1

        yield SemanticEvent(turn_id, sequence, EventKind.TURN_COMPLETE, {})


class _EchoSpeaker:
    async def generate(self, chunk: SpeechChunk, facts: Mapping[str, ToolResult]) -> str:
        return f"spoke:{chunk.chunk_id}"


class ReplanFuzzTests(unittest.IsolatedAsyncioTestCase):
    """plan.md A2 exit criterion: plan changes before commit horizon never produce
    contradictory speech (a cancelled chunk speaking) and never crash the turn."""

    async def test_random_replans_never_produce_contradictory_speech(self) -> None:
        rng = random.Random(20260731)
        violations = []

        for index in range(TURNS):
            turn_id = f"replan-fuzz-{index}"
            tool = FakeWeatherTool(latency_ms=rng.choice([0, 1, 2]), fail=rng.random() < 0.2)
            runtime = DualSessionRuntime(_RandomReplanPlanner(rng), _EchoSpeaker(), tool)

            result = await runtime.run(turn_id, "synthetic replan request")

            cancelled_ids = {
                event.attributes["chunk_id"]
                for event in result.timeline.events
                if event.name == "chunk_cancelled"
            }
            for chunk in result.chunks:
                if chunk.chunk_id in cancelled_ids:
                    if chunk.state is ChunkState.PLAYED:
                        violations.append(f"{turn_id}: {chunk.chunk_id} both cancelled and played")
                    if f"spoke:{chunk.chunk_id}" in result.text:
                        violations.append(f"{turn_id}: {chunk.chunk_id} spoke after cancellation")
                if chunk.state is ChunkState.CANCELLED and chunk.chunk_id not in cancelled_ids:
                    violations.append(f"{turn_id}: {chunk.chunk_id} cancelled without a trace event")

        self.assertEqual(violations, [], f"{len(violations)}/{TURNS} turns had contradictory speech")


if __name__ == "__main__":
    unittest.main()
