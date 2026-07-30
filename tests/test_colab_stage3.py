import unittest

from aether.domain.chunks import ChunkState, SpeechChunk
from aether.domain.events import EventKind, SemanticEvent
from aether.experiments.colab_stage3 import (
    DEFAULT_LATENCIES_MS,
    SCENARIOS,
    Scenario,
    build_plan,
    evaluate_run,
)


class _Result:
    def __init__(self, semantic_events, chunks):
        self.semantic_events = semantic_events
        self.chunks = chunks


def _event(kind: EventKind, sequence: int = 0) -> SemanticEvent:
    return SemanticEvent("turn-1", sequence, kind, {})


def _chunk(chunk_id: str, dependencies, state: ChunkState) -> SpeechChunk:
    chunk = SpeechChunk(chunk_id=chunk_id, goal="goal", dependencies=frozenset(dependencies))
    chunk.state = state
    return chunk


class EvaluateRunTests(unittest.TestCase):
    def test_dependent_chunk_after_confirmed_tool_passes(self) -> None:
        result = _Result(
            semantic_events=[_event(EventKind.TOOL_CALL, 0)],
            chunks=[
                _chunk("lead-in", [], ChunkState.PLAYED),
                _chunk("answer", ["weather"], ChunkState.PLAYED),
            ],
        )
        rt_trace = [
            {"name": "chunk_generating", "attributes": {"chunk_id": "lead-in"}, "absolute_timestamp_ns": 0},
            {
                "name": "tool_completed",
                "attributes": {"tool": "weather", "succeeded": True},
                "absolute_timestamp_ns": 10,
            },
            {"name": "chunk_generating", "attributes": {"chunk_id": "answer"}, "absolute_timestamp_ns": 20},
        ]
        dec_trace = [{"name": "first_token", "role": "speaker", "absolute_timestamp_ns": 5}]

        checks, observations = evaluate_run(result, rt_trace, dec_trace)

        self.assertTrue(checks["dependent_chunks_only_speak_confirmed_facts"])
        self.assertEqual(observations["dependency_violations"], [])
        self.assertTrue(observations["tool_call_emitted"])
        self.assertTrue(observations["speaker_first_token_before_tool_complete"])

    def test_dependent_chunk_generated_before_tool_confirmed_fails(self) -> None:
        result = _Result(
            semantic_events=[_event(EventKind.TOOL_CALL, 0)],
            chunks=[_chunk("answer", ["weather"], ChunkState.PLAYED)],
        )
        rt_trace = [
            {"name": "chunk_generating", "attributes": {"chunk_id": "answer"}, "absolute_timestamp_ns": 0},
            {
                "name": "tool_completed",
                "attributes": {"tool": "weather", "succeeded": True},
                "absolute_timestamp_ns": 10,
            },
        ]

        checks, observations = evaluate_run(result, rt_trace, [])

        self.assertFalse(checks["dependent_chunks_only_speak_confirmed_facts"])
        self.assertEqual(len(observations["dependency_violations"]), 1)

    def test_dependent_chunk_played_after_failed_tool_fails(self) -> None:
        result = _Result(
            semantic_events=[_event(EventKind.TOOL_CALL, 0)],
            chunks=[_chunk("answer", ["weather"], ChunkState.PLAYED)],
        )
        rt_trace = [
            {"name": "chunk_generating", "attributes": {"chunk_id": "answer"}, "absolute_timestamp_ns": 10},
            {
                "name": "tool_completed",
                "attributes": {"tool": "weather", "succeeded": False},
                "absolute_timestamp_ns": 0,
            },
        ]

        checks, observations = evaluate_run(result, rt_trace, [])

        self.assertFalse(checks["dependent_chunks_only_speak_confirmed_facts"])

    def test_tool_failure_keeps_dependent_chunk_blocked_and_passes(self) -> None:
        result = _Result(
            semantic_events=[_event(EventKind.TOOL_CALL, 0)],
            chunks=[
                _chunk("lead-in", [], ChunkState.PLAYED),
                _chunk("answer", ["weather"], ChunkState.BLOCKED),
            ],
        )
        rt_trace = [
            {
                "name": "tool_completed",
                "attributes": {"tool": "weather", "succeeded": False},
                "absolute_timestamp_ns": 0,
            },
        ]

        checks, observations = evaluate_run(result, rt_trace, [])

        self.assertTrue(checks["dependent_chunks_only_speak_confirmed_facts"])
        self.assertEqual(observations["blocked_chunk_ids"], ["answer"])

    def test_no_tool_scenario_with_no_dependent_chunks_passes(self) -> None:
        result = _Result(semantic_events=[], chunks=[_chunk("reply", [], ChunkState.PLAYED)])

        checks, observations = evaluate_run(result, [], [])

        self.assertTrue(checks["dependent_chunks_only_speak_confirmed_facts"])
        self.assertFalse(observations["tool_call_emitted"])

    def test_unknown_tool_answer_without_confirmation_fails(self) -> None:
        # Mirrors the real Colab finding: Planner invents a tool name (e.g. "chat")
        # that the executor never confirms as succeeded for that name.
        result = _Result(
            semantic_events=[_event(EventKind.TOOL_CALL, 0)],
            chunks=[_chunk("answer", ["chat"], ChunkState.PLAYED)],
        )
        rt_trace = [
            {
                "name": "tool_completed",
                "attributes": {"tool": "chat", "succeeded": False},
                "absolute_timestamp_ns": 0,
            },
            {"name": "chunk_generating", "attributes": {"chunk_id": "answer"}, "absolute_timestamp_ns": 10},
        ]

        checks, observations = evaluate_run(result, rt_trace, [])

        self.assertFalse(checks["dependent_chunks_only_speak_confirmed_facts"])
        self.assertIn("answer spoke before 'chat' was confirmed", observations["dependency_violations"])


class BuildPlanTests(unittest.TestCase):
    def test_plan_sweeps_latency_only_for_tool_scenarios(self) -> None:
        plan = build_plan(DEFAULT_LATENCIES_MS)

        sweeping = sum(1 for s in SCENARIOS if s.sweep_latency)
        non_sweeping = len(SCENARIOS) - sweeping
        expected = sweeping * len(DEFAULT_LATENCIES_MS) + non_sweeping
        self.assertEqual(len(plan), expected)

        no_tool_entries = [item for item in plan if item["scenario"].name == "no_tool_greeting"]
        self.assertEqual(len(no_tool_entries), 1)


if __name__ == "__main__":
    unittest.main()
