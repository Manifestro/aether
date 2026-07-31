import io
import json
import unittest
import zipfile
from unittest.mock import patch

from aether.training.datasets_sgd import extract_sgd_phrases


def _build_fake_sgd_zip() -> bytes:
    """A minimal in-memory zip mirroring the real repo's layout/schema
    (confirmed against a real file -- see the module docstring), so
    `extract_sgd_phrases` can be tested without a network call."""
    dialogues = [
        {
            "dialogue_id": "1_00000",
            "services": ["Restaurants_2"],
            "turns": [
                {"speaker": "USER", "utterance": "Hi, could you book a table?"},
                {"speaker": "SYSTEM", "utterance": "Sure, which restaurant would you like?"},
                {"speaker": "USER", "utterance": "Anywhere nice downtown."},
                {"speaker": "SYSTEM", "utterance": "I found a table for you at seven."},
                # Too short -- must be filtered out by min_words.
                {"speaker": "SYSTEM", "utterance": "Ok."},
            ],
        }
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "dstc8-schema-guided-dialogue-master/train/dialogues_001.json",
            json.dumps(dialogues),
        )
        # A non-dialogue file in the repo (e.g. schema.json) must be ignored.
        archive.writestr(
            "dstc8-schema-guided-dialogue-master/train/schema.json", json.dumps({}),
        )
    return buffer.getvalue()


class ExtractSgdPhrasesTests(unittest.TestCase):
    def test_extracts_only_system_turns_within_word_bounds(self) -> None:
        fake_zip = _build_fake_sgd_zip()
        with patch("aether.training.datasets_sgd.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.read.return_value = fake_zip
            phrases = extract_sgd_phrases(target_count=2, min_words=3, max_words=30, seed=0)

        texts = {p.text for p in phrases}
        self.assertEqual(len(phrases), 2)
        self.assertTrue(texts.issubset({
            "Sure, which restaurant would you like?",
            "I found a table for you at seven.",
        }))
        # Neither a USER turn nor the too-short "Ok." should ever appear.
        self.assertNotIn("Hi, could you book a table?", texts)
        self.assertNotIn("Ok.", texts)

    def test_raises_when_not_enough_phrases_survive_filtering(self) -> None:
        fake_zip = _build_fake_sgd_zip()
        with patch("aether.training.datasets_sgd.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.read.return_value = fake_zip
            with self.assertRaises(RuntimeError):
                extract_sgd_phrases(target_count=1000, min_words=3, max_words=30, seed=0)


if __name__ == "__main__":
    unittest.main()
