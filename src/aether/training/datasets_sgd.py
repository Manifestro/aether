"""Real, natural assistant utterances from Schema-Guided Dialogue (SGD) --
replacing `datasets_large.py`'s template-generated phrases with actual
human-written dialogue turns across 16+ real domains (restaurants, hotels,
travel, media, weather, banking, ...), not just weather paraphrases.

Downloads the raw JSON directly from Google's original GitHub release
(`google-research-datasets/dstc8-schema-guided-dialogue`), not via Hugging
Face `datasets.load_dataset()` -- that repo is a legacy "dataset script"
repo, and recent `datasets` versions refuse to run those at all
(`RuntimeError: Dataset scripts are no longer supported`, hit on a real
run; see docs/reports/technical_report_03.md). The raw-JSON schema here
was confirmed directly (one real file fetched and inspected) rather than
guessed: each dialogue is `{"dialogue_id", "services", "turns"}`, each turn
is `{"frames", "speaker", "utterance"}` with `speaker` exactly `"USER"` or
`"SYSTEM"`.
"""

import io
import json
import random
import urllib.request
import zipfile
from typing import List

from aether.training.datasets import Phrase

_REPO_ZIP_URL = (
    "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/"
    "archive/refs/heads/master.zip"
)
_SPLIT_DIRS = ("train", "dev", "test")


def extract_sgd_phrases(
    target_count: int = 18_000,
    min_words: int = 3,
    max_words: int = 30,
    seed: int = 0,
) -> List[Phrase]:
    """Extracts real, deduplicated SYSTEM-turn utterances from SGD.

    Downloads the whole repo as a zip (once per call) and reads every
    `train/dev/test/dialogues_*.json` file directly -- no HF `datasets`
    dependency, no dataset-script execution.
    """
    with urllib.request.urlopen(_REPO_ZIP_URL, timeout=120) as response:
        zip_bytes = response.read()

    seen = set()
    phrases: List[Phrase] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for name in archive.namelist():
            parts = name.split("/")
            # e.g. "dstc8-schema-guided-dialogue-master/train/dialogues_001.json"
            if len(parts) < 3 or parts[1] not in _SPLIT_DIRS:
                continue
            if not parts[-1].startswith("dialogues_") or not parts[-1].endswith(".json"):
                continue
            with archive.open(name) as handle:
                dialogues = json.load(handle)
            for dialogue in dialogues:
                for turn in dialogue.get("turns", []):
                    if turn.get("speaker") != "SYSTEM":
                        continue
                    text = str(turn.get("utterance", "")).strip()
                    word_count = len(text.split())
                    if not (min_words <= word_count <= max_words):
                        continue
                    if text in seen:
                        continue
                    seen.add(text)
                    phrases.append(Phrase(phrase_id=f"sgd-{len(phrases):05d}", text=text))

    if len(phrases) < target_count:
        raise RuntimeError(
            f"only found {len(phrases)}/{target_count} unique SYSTEM utterances in SGD "
            f"after filtering (min_words={min_words}, max_words={max_words}) -- loosen "
            "the word-count bounds"
        )

    random.Random(seed).shuffle(phrases)
    selected = phrases[:target_count]
    # Re-number ids after shuffling+truncating so ids stay contiguous
    # (downstream code assumes phrase_id order matches list order).
    return [Phrase(phrase_id=f"sgd-{i:05d}", text=phrase.text) for i, phrase in enumerate(selected)]
