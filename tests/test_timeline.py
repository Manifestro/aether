import unittest

from aether.domain.timeline import Timeline


class FakeClock:
    def __init__(self) -> None:
        self.value = 1_000_000_000

    def __call__(self) -> int:
        current = self.value
        self.value += 50_000_000
        return current


class TimelineTests(unittest.TestCase):
    def test_elapsed_time_is_deterministic(self) -> None:
        timeline = Timeline(clock_ns=FakeClock())
        timeline.record("planner_started")
        timeline.record("tool_started")
        self.assertEqual(timeline.elapsed_ms("planner_started", "tool_started"), 50.0)

    def test_events_property_returns_copy(self) -> None:
        timeline = Timeline(clock_ns=FakeClock())
        timeline.record("started")
        events = timeline.events
        events.clear()
        self.assertEqual(len(timeline.events), 1)


if __name__ == "__main__":
    unittest.main()

