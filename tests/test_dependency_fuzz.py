import random
import unittest
from typing import AsyncIterator, Mapping

from aether.domain.chunks import ChunkState, SpeechChunk
from aether.domain.events import EventKind, SemanticEvent, ToolResult
from aether.runtime.dual_session import DualSessionRuntime
from aether.testing.fakes import FakeWeatherTool

TURNS = 1000


class _RandomPlanner:
    """Emits a schema-valid but randomized event stream for one turn."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    async def plan(self, turn_id: str, request: str) -> AsyncIterator[SemanticEvent]:
        sequence = 0
        has_tool = self._rng.random() < 0.8
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

        for index in range(self._rng.randint(1, 3)):
            dependent = has_tool and self._rng.random() < 0.6
            yield SemanticEvent(
                turn_id,
                sequence,
                EventKind.SPEECH_PLAN,
                {
                    "chunk_id": f"{turn_id}-chunk-{index}",
                    "goal": f"say something {index}",
                    "dependencies": ["weather"] if dependent else [],
                },
            )
            sequence += 1

        yield SemanticEvent(turn_id, sequence, EventKind.TURN_COMPLETE, {})


class _EchoSpeaker:
    async def generate(self, chunk: SpeechChunk, facts: Mapping[str, ToolResult]) -> str:
        return f"spoke:{chunk.chunk_id}"


class DependencyInvariantFuzzTests(unittest.IsolatedAsyncioTestCase):
    """Exit criterion from plan.md A1: many synthetic turns, zero dependency violations."""

    async def test_thousand_synthetic_turns_never_violate_dependency_order(self) -> None:
        rng = random.Random(20260730)
        violations = []

        for index in range(TURNS):
            turn_id = f"fuzz-{index}"
            tool_fail = rng.random() < 0.3
            tool = FakeWeatherTool(latency_ms=rng.choice([0, 1, 2]), fail=tool_fail)
            runtime = DualSessionRuntime(_RandomPlanner(rng), _EchoSpeaker(), tool)

            result = await runtime.run(turn_id, "synthetic request")
            events = result.timeline.events

            tool_completed_index = next(
                (i for i, e in enumerate(events) if e.name == "tool_completed"), None
            )

            for chunk in result.chunks:
                if not chunk.dependencies:
                    continue
                if tool_fail:
                    if chunk.state is ChunkState.PLAYED:
                        violations.append(f"{turn_id}: {chunk.chunk_id} played after tool failure")
                    continue
                if chunk.state is not ChunkState.PLAYED:
                    continue
                generating_index = next(
                    (
                        i
                        for i, e in enumerate(events)
                        if e.name == "chunk_generating" and e.attributes.get("chunk_id") == chunk.chunk_id
                    ),
                    None,
                )
                if tool_completed_index is None or generating_index is None:
                    violations.append(f"{turn_id}: {chunk.chunk_id} missing tool/generating trace")
                elif generating_index < tool_completed_index:
                    violations.append(
                        f"{turn_id}: {chunk.chunk_id} generated before tool_completed"
                    )

        self.assertEqual(violations, [], f"{len(violations)}/{TURNS} turns violated dependency order")


if __name__ == "__main__":
    unittest.main()
