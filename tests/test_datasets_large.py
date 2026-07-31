import unittest

from aether.training.datasets_large import generate_phrases, train_val_split


class DatasetsLargeTests(unittest.TestCase):
    def test_generates_requested_count_of_unique_phrases(self) -> None:
        phrases = generate_phrases(500, seed=0)
        self.assertEqual(len(phrases), 500)
        self.assertEqual(len({p.phrase_id for p in phrases}), 500)
        self.assertEqual(len({p.text for p in phrases}), 500)

    def test_deterministic_given_same_seed(self) -> None:
        first = generate_phrases(200, seed=42)
        second = generate_phrases(200, seed=42)
        self.assertEqual([p.text for p in first], [p.text for p in second])

    def test_different_seeds_produce_different_sets(self) -> None:
        first = generate_phrases(200, seed=1)
        second = generate_phrases(200, seed=2)
        self.assertNotEqual([p.text for p in first], [p.text for p in second])

    def test_train_val_split_covers_all_phrases_without_overlap(self) -> None:
        phrases = generate_phrases(1000, seed=0)
        train, val = train_val_split(phrases, val_fraction=0.05)
        self.assertEqual(len(val), 50)
        self.assertEqual(len(train) + len(val), len(phrases))
        self.assertEqual(
            {p.phrase_id for p in train} | {p.phrase_id for p in val},
            {p.phrase_id for p in phrases},
        )

    def test_train_val_split_rejects_invalid_fraction(self) -> None:
        phrases = generate_phrases(10, seed=0)
        with self.assertRaises(ValueError):
            train_val_split(phrases, val_fraction=0)
        with self.assertRaises(ValueError):
            train_val_split(phrases, val_fraction=1)


if __name__ == "__main__":
    unittest.main()
