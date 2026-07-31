import json
import tempfile
import unittest
from pathlib import Path

from aether.training.large_scale_trainer import load_training_records


class LoadTrainingRecordsTests(unittest.TestCase):
    def test_parses_jsonl_cache_into_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "cache.jsonl"
            with cache_path.open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "phrase_id": "phrase-00000",
                            "text": "hello",
                            "hidden_state": [0.1, 0.2],
                            "teacher_tokens": [[1, 2], [3, 4]],
                        }
                    )
                    + "\n"
                )

            records = load_training_records(cache_path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].phrase_id, "phrase-00000")
            self.assertEqual(records[0].hidden_state, [0.1, 0.2])
            self.assertEqual(records[0].teacher_tokens, [[1, 2], [3, 4]])

    def test_skips_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "cache.jsonl"
            with cache_path.open("w", encoding="utf-8") as handle:
                handle.write("\n")
                handle.write(
                    json.dumps(
                        {"phrase_id": "phrase-00000", "text": "hi", "hidden_state": [0.0], "teacher_tokens": [[0]]}
                    )
                    + "\n"
                )
                handle.write("\n")

            records = load_training_records(cache_path)
            self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
