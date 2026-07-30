import unittest

try:
    from aether_api.experiments.colab_stage4 import evaluate_events

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


def _event(arrived_ms: float, event_type: str) -> dict:
    return {"arrived_ms": arrived_ms, "type": event_type, "payload": {}}


@unittest.skipUnless(_HAS_FASTAPI, "fastapi not installed; install the project's 'api' extra to run this")
class EvaluateEventsTests(unittest.TestCase):
    def test_normal_lookahead_run_passes_and_observes_the_lead(self) -> None:
        events = [
            _event(0, "turn.started"),
            _event(20, "plan.tool_started"),
            _event(60, "response.safe_delta"),
            _event(500, "tool.completed"),
            _event(520, "response.delta"),
            _event(521, "turn.completed"),
        ]

        checks, observations = evaluate_events(events, http_status=200)

        self.assertTrue(all(checks.values()), checks)
        self.assertTrue(observations["safe_delta_before_tool_completed"])
        self.assertEqual(observations["safe_delta_minus_tool_completed_ms"], -440)

    def test_slow_gpu_losing_the_safe_lead_still_passes(self) -> None:
        # Reproduces the real T4 stage4 run: safe_delta arrived *after*
        # tool.completed (pipeline overhead exceeded tool latency), but the
        # factual response.delta still never arrived before tool.completed.
        events = [
            _event(110, "turn.started"),
            _event(2513, "plan.tool_started"),
            _event(4014, "tool.completed"),
            _event(6172, "response.safe_delta"),
            _event(9939, "response.delta"),
            _event(9939, "turn.completed"),
        ]

        checks, observations = evaluate_events(events, http_status=200)

        self.assertTrue(all(checks.values()), checks)
        self.assertFalse(observations["safe_delta_before_tool_completed"])
        self.assertEqual(observations["safe_delta_minus_tool_completed_ms"], 2158)

    def test_factual_delta_before_tool_completed_fails_the_hard_gate(self) -> None:
        events = [
            _event(0, "turn.started"),
            _event(10, "tool.completed"),
            _event(5, "response.delta"),  # arrived before tool.completed - fabrication
            _event(20, "turn.completed"),
        ]

        checks, _ = evaluate_events(events, http_status=200)

        self.assertFalse(checks["response_delta_never_before_tool_completed"])

    def test_turn_failed_in_the_stream_fails_the_hard_gate(self) -> None:
        events = [_event(0, "turn.started"), _event(5, "turn.failed")]

        checks, _ = evaluate_events(events, http_status=200)

        self.assertFalse(checks["no_turn_failed"])
        self.assertFalse(checks["ends_with_turn_completed"])

    def test_non_200_status_fails_the_hard_gate(self) -> None:
        checks, _ = evaluate_events([], http_status=401)

        self.assertFalse(checks["http_200"])


if __name__ == "__main__":
    unittest.main()
