import unittest

from aether.domain.timeline import TraceEvent
from aether_api.contract import PublicEventType
from aether_api.event_mapper import EventMapper


class EventMapperTests(unittest.TestCase):
    def test_internal_only_events_are_not_forwarded(self) -> None:
        mapper = EventMapper("turn-1")

        for name in ("decode_started", "chunk_generating", "chunk_ready", "planner_started"):
            self.assertIsNone(mapper.map(TraceEvent(name=name, timestamp_ns=0)))

    def test_turn_started_maps_directly(self) -> None:
        mapper = EventMapper("turn-1")

        event = mapper.map(TraceEvent(name="turn_started", timestamp_ns=1_000_000, attributes={}))

        self.assertIsNotNone(event)
        self.assertEqual(event.type, PublicEventType.TURN_STARTED)
        self.assertEqual(event.turn_id, "turn-1")
        self.assertEqual(event.timestamp_ms, 1.0)

    def test_safe_chunk_commit_becomes_response_safe_delta(self) -> None:
        mapper = EventMapper("turn-1")

        event = mapper.map(
            TraceEvent(
                name="chunk_committed",
                timestamp_ns=0,
                attributes={"chunk_id": "lead-in", "text": "Проверяю погоду.", "safe_to_say": True},
            )
        )

        self.assertEqual(event.type, PublicEventType.RESPONSE_SAFE_DELTA)
        self.assertEqual(event.payload["text"], "Проверяю погоду.")

    def test_dependent_chunk_commit_becomes_response_delta(self) -> None:
        mapper = EventMapper("turn-1")

        event = mapper.map(
            TraceEvent(
                name="chunk_committed",
                timestamp_ns=0,
                attributes={"chunk_id": "answer", "text": "24 градуса.", "safe_to_say": False},
            )
        )

        self.assertEqual(event.type, PublicEventType.RESPONSE_DELTA)

    def test_tool_started_exposes_only_the_tool_name(self) -> None:
        mapper = EventMapper("turn-1")

        event = mapper.map(
            TraceEvent(
                name="tool_started",
                timestamp_ns=0,
                attributes={"call_id": "weather-1", "tool": "weather"},
            )
        )

        self.assertEqual(event.type, PublicEventType.PLAN_TOOL_STARTED)
        self.assertEqual(event.payload, {"tool": "weather"})

    def test_successful_tool_completed_exposes_content_not_error(self) -> None:
        mapper = EventMapper("turn-1")

        event = mapper.map(
            TraceEvent(
                name="tool_completed",
                timestamp_ns=0,
                attributes={
                    "tool": "weather",
                    "succeeded": True,
                    "content": {"temperature_c": 24, "condition": "rain"},
                    "error": "",
                },
            )
        )

        self.assertEqual(event.type, PublicEventType.TOOL_COMPLETED)
        self.assertEqual(event.payload["temperature_c"], 24)
        self.assertNotIn("error", event.payload)

    def test_failed_tool_completed_exposes_error_not_fabricated_content(self) -> None:
        mapper = EventMapper("turn-1")

        event = mapper.map(
            TraceEvent(
                name="tool_completed",
                timestamp_ns=0,
                attributes={"tool": "weather", "succeeded": False, "content": {}, "error": "unavailable"},
            )
        )

        self.assertEqual(event.payload["succeeded"], False)
        self.assertEqual(event.payload["error"], "unavailable")

    def test_sequence_numbers_are_gap_free_and_increasing(self) -> None:
        mapper = EventMapper("turn-1")

        first = mapper.map(TraceEvent(name="turn_started", timestamp_ns=0))
        second = mapper.map(TraceEvent(name="turn_completed", timestamp_ns=1))

        self.assertEqual((first.sequence, second.sequence), (0, 1))

    def test_turn_failed_is_synthesized_directly(self) -> None:
        mapper = EventMapper("turn-1")

        event = mapper.turn_failed("ValueError: boom")

        self.assertEqual(event.type, PublicEventType.TURN_FAILED)
        self.assertEqual(event.payload["reason"], "ValueError: boom")

    def test_to_sse_serializes_type_and_payload(self) -> None:
        mapper = EventMapper("turn-1")
        event = mapper.map(
            TraceEvent(name="tool_started", timestamp_ns=0, attributes={"tool": "weather"})
        )

        sse = event.to_sse()

        self.assertIn("event: plan.tool_started\n", sse)
        self.assertIn('"tool": "weather"', sse)
        self.assertTrue(sse.endswith("\n\n"))


if __name__ == "__main__":
    unittest.main()
