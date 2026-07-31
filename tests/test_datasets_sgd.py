import unittest

from aether.training.datasets_sgd import _find_key


class FindKeyTests(unittest.TestCase):
    def test_finds_first_matching_candidate(self) -> None:
        record = {"speaker": "SYSTEM", "utterance": "Hello"}
        self.assertEqual(_find_key(record, ["speaker", "author", "role"]), "speaker")

    def test_finds_later_candidate_if_first_absent(self) -> None:
        record = {"role": "SYSTEM"}
        self.assertEqual(_find_key(record, ["speaker", "author", "role"]), "role")

    def test_raises_specific_error_listing_actual_keys_when_none_match(self) -> None:
        record = {"totally_different_key": "value"}
        with self.assertRaises(RuntimeError) as context:
            _find_key(record, ["speaker", "author", "role"])
        message = str(context.exception)
        self.assertIn("totally_different_key", message)
        self.assertIn("speaker", message)


if __name__ == "__main__":
    unittest.main()
