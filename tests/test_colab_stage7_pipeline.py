import json
import tempfile
import unittest
from pathlib import Path

from aether.experiments.colab_stage7_data_pipeline import load_cached_phrase_ids
from aether.experiments.colab_stage7_train import split_records
from aether.training.large_scale_trainer import TrainingRecord


class LoadCachedPhraseIdsTests(unittest.TestCase):
    def test_missing_file_returns_empty_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertEqual(load_cached_phrase_ids(Path(tmp_dir) / "missing.jsonl"), set())

    def test_reads_completed_records_and_tolerates_a_truncated_last_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "cache.jsonl"
            with cache_path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"phrase_id": "phrase-00000", "text": "a"}) + "\n")
                handle.write(json.dumps({"phrase_id": "phrase-00001", "text": "b"}) + "\n")
                handle.write('{"phrase_id": "phrase-00002", "tex')  # simulated interrupted write

            done = load_cached_phrase_ids(cache_path)
            self.assertEqual(done, {"phrase-00000", "phrase-00001"})


class SplitRecordsTests(unittest.TestCase):
    def test_split_covers_all_records_without_overlap(self) -> None:
        records = [TrainingRecord(f"phrase-{i:05d}", [0.0], [[0]]) for i in range(1000)]
        train, val = split_records(records, val_fraction=0.05, seed=0)
        self.assertEqual(len(val), 50)
        self.assertEqual(len(train) + len(val), 1000)
        self.assertEqual(
            {r.phrase_id for r in train} | {r.phrase_id for r in val},
            {r.phrase_id for r in records},
        )

    def test_deterministic_given_same_seed(self) -> None:
        records = [TrainingRecord(f"phrase-{i:05d}", [0.0], [[0]]) for i in range(200)]
        first_train, first_val = split_records(records, val_fraction=0.1, seed=7)
        second_train, second_val = split_records(records, val_fraction=0.1, seed=7)
        self.assertEqual([r.phrase_id for r in first_train], [r.phrase_id for r in second_train])
        self.assertEqual([r.phrase_id for r in first_val], [r.phrase_id for r in second_val])


if __name__ == "__main__":
    unittest.main()
