import unittest

from vox.domain.events import EventKind
from vox.model.event_parser import SemanticEventStreamParser


class SemanticEventStreamParserTests(unittest.TestCase):
    def test_parses_json_split_across_arbitrary_chunks(self) -> None:
        parser = SemanticEventStreamParser("turn-1")
        first = parser.feed('{"type":"intent","sequence":0,"pay')
        second = parser.feed('load":{"name":"weather"}}\n')

        self.assertEqual(first, [])
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0].kind, EventKind.INTENT)
        self.assertEqual(second[0].payload["name"], "weather")

    def test_finish_parses_last_line_without_newline(self) -> None:
        parser = SemanticEventStreamParser("turn-1")
        parser.feed('{"type":"turn_complete","sequence":0,"payload":{}}')
        self.assertEqual(parser.finish()[0].kind, EventKind.TURN_COMPLETE)

    def test_rejects_non_increasing_sequences(self) -> None:
        parser = SemanticEventStreamParser("turn-1")
        parser.feed('{"type":"intent","sequence":1,"payload":{}}\n')
        with self.assertRaises(ValueError):
            parser.feed('{"type":"turn_complete","sequence":1,"payload":{}}\n')

    def test_can_repair_non_increasing_model_sequences(self) -> None:
        parser = SemanticEventStreamParser("turn-1", repair_sequences=True)
        events = parser.feed(
            '{"type":"intent","sequence":0,"payload":{}}\n'
            '{"type":"tool_call","sequence":0,"payload":{"call_id":"c","tool":"weather","arguments":{}}}\n'
        )
        self.assertEqual([event.sequence for event in events], [0, 1])

    def test_rejects_incomplete_tool_call(self) -> None:
        parser = SemanticEventStreamParser("turn-1")
        with self.assertRaises(ValueError):
            parser.feed('{"type":"tool_call","sequence":0,"payload":{}}\n')


if __name__ == "__main__":
    unittest.main()
