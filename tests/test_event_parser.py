import unittest

from aether.domain.events import EventKind
from aether.model.event_parser import SemanticEventStreamParser


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
        self.assertEqual(parser.repaired_count, 1)

    def test_rejects_incomplete_tool_call(self) -> None:
        parser = SemanticEventStreamParser("turn-1")
        with self.assertRaises(ValueError):
            parser.feed('{"type":"tool_call","sequence":0,"payload":{}}\n')

    def test_strict_grammar_rejects_intent_event(self) -> None:
        parser = SemanticEventStreamParser("turn-1", strict=True)
        with self.assertRaises(ValueError):
            parser.feed('{"type":"intent","sequence":0,"payload":{}}\n')

    def test_strict_grammar_requires_safe_to_say(self) -> None:
        parser = SemanticEventStreamParser("turn-1", strict=True)
        with self.assertRaises(ValueError):
            parser.feed(
                '{"type":"speech_plan","sequence":0,'
                '"payload":{"chunk_id":"lead-in","goal":"hi","dependencies":[]}}\n'
            )

    def test_strict_grammar_rejects_inconsistent_safe_to_say(self) -> None:
        parser = SemanticEventStreamParser("turn-1", strict=True)
        with self.assertRaises(ValueError):
            parser.feed(
                '{"type":"speech_plan","sequence":0,"payload":'
                '{"chunk_id":"answer","goal":"hi","dependencies":["weather"],"safe_to_say":true}}\n'
            )

    def test_strict_grammar_accepts_consistent_speech_plan(self) -> None:
        parser = SemanticEventStreamParser("turn-1", strict=True)
        events = parser.feed(
            '{"type":"speech_plan","sequence":0,"payload":'
            '{"chunk_id":"lead-in","goal":"hi","dependencies":[],"safe_to_say":true}}\n'
        )
        self.assertEqual(len(events), 1)

    def test_strict_grammar_rejects_nonempty_turn_complete_payload(self) -> None:
        parser = SemanticEventStreamParser("turn-1", strict=True)
        with self.assertRaises(ValueError):
            parser.feed('{"type":"turn_complete","sequence":0,"payload":{"note":"done"}}\n')

    def test_revision_id_defaults_to_zero_and_is_parsed(self) -> None:
        parser = SemanticEventStreamParser("turn-1")
        events = parser.feed('{"type":"turn_complete","sequence":0,"payload":{},"revision_id":2}\n')
        self.assertEqual(events[0].revision_id, 2)

        parser2 = SemanticEventStreamParser("turn-1")
        events2 = parser2.feed('{"type":"turn_complete","sequence":0,"payload":{}}\n')
        self.assertEqual(events2[0].revision_id, 0)


if __name__ == "__main__":
    unittest.main()
