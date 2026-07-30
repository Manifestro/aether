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


_SUCCESS = Scenario("weather_success", "weather?", tool_fail=False, sweep_latency=True)
_FAILURE = Scenario("weather_tool_failure", "weather?", tool_fail=True, sweep_latency=True)
_NO_TOOL = Scenario("no_tool_greeting", "hi", tool_fail=False, sweep_latency=False)


class EvaluateRunTests(unittest.TestCase):
    def test_successful_tool_run_with_correct_ordering_passes(self) -> None:
        result = _Result(
            semantic_events=[_event(EventKind.TOOL_CALL, 0)],
            chunks=[
                _chunk("lead-in", [], ChunkState.PLAYED),
                _chunk("answer", ["weather"], ChunkState.PLAYED),
            ],
        )
        rt_trace = [
            {"name": "chunk_generating", "attributes": {"chunk_id": "lead-in"}},
            {"name": "tool_completed", "attributes": {}},
            {"name": "chunk_generating", "attributes": {"chunk_id": "answer"}},
        ]
        dec_trace = [
            {"name": "first_token", "role": "speaker", "absolute_timestamp_ns": 10},
        ]
        rt_trace_abs = [dict(event, absolute_timestamp_ns=index) for index, event in enumerate(rt_trace)]
        rt_trace_abs[1]["absolute_timestamp_ns"] = 50

        checks = evaluate_run(result, rt_trace_abs, dec_trace, _SUCCESS)

        self.assertTrue(all(checks.values()), checks)
        self.assertIn("dependent_chunk_played_after_tool_complete", checks)
        self.assertIn("speaker_first_token_before_tool_complete", checks)

    def test_factual_chunk_generated_before_tool_complete_fails(self) -> None:
        result = _Result(
            semantic_events=[_event(EventKind.TOOL_CALL, 0)],
            chunks=[_chunk("answer", ["weather"], ChunkState.PLAYED)],
        )
        rt_trace = [
            {"name": "chunk_generating", "attributes": {"chunk_id": "answer"}, "absolute_timestamp_ns": 0},
            {"name": "tool_completed", "attributes": {}, "absolute_timestamp_ns": 10},
        ]
        checks = evaluate_run(result, rt_trace, [], _SUCCESS)

        self.assertFalse(checks["dependent_chunk_played_after_tool_complete"])

    def test_tool_failure_keeps_dependent_chunk_blocked(self) -> None:
        result = _Result(
            semantic_events=[_event(EventKind.TOOL_CALL, 0)],
            chunks=[
                _chunk("lead-in", [], ChunkState.PLAYED),
                _chunk("answer", ["weather"], ChunkState.BLOCKED),
            ],
        )
        checks = evaluate_run(result, [], [], _FAILURE)

        self.assertTrue(checks["dependent_chunk_never_played"])
        self.assertTrue(checks["safe_chunk_still_played"])

    def test_tool_failure_with_fabricated_answer_fails_the_check(self) -> None:
        result = _Result(
            semantic_events=[_event(EventKind.TOOL_CALL, 0)],
            chunks=[_chunk("answer", ["weather"], ChunkState.PLAYED)],
        )
        checks = evaluate_run(result, [], [], _FAILURE)

        self.assertFalse(checks["dependent_chunk_never_played"])

    def test_no_tool_scenario_requires_nothing_blocked(self) -> None:
        result = _Result(
            semantic_events=[],
            chunks=[_chunk("reply", [], ChunkState.PLAYED)],
        )
        checks = evaluate_run(result, [], [], _NO_TOOL)

        self.assertTrue(checks["no_chunk_left_blocked"])

    def test_no_tool_scenario_flags_a_stuck_chunk(self) -> None:
        result = _Result(
            semantic_events=[],
            chunks=[_chunk("reply", ["weather"], ChunkState.BLOCKED)],
        )
        checks = evaluate_run(result, [], [], _NO_TOOL)

        self.assertFalse(checks["no_chunk_left_blocked"])


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
