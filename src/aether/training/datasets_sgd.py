"""Real, natural assistant utterances from Schema-Guided Dialogue (SGD) --
replacing `datasets_large.py`'s template-generated phrases with actual
human-written dialogue turns across 16+ real domains (restaurants, hotels,
travel, media, weather, banking, ...), not just weather paraphrases.

The dataset (`google-research-datasets/schema_guided_dstc8`, loadable via
Hugging Face `datasets`) is confirmed to exist and be the right shape
(dialogues of turns, each turn tagged with a speaker and an utterance) --
but the exact nested field names were not verified offline. This module
discovers them defensively at runtime (tries the documented/likely key
names, raises a specific error listing what it actually found if none
match) instead of assuming, matching this project's practice for any
external data source not already used elsewhere in the repo (see the
Moshi-teacher spikes in `docs/reports/technical_report_03.md` for why).
"""

import random
from typing import Any, Dict, List

from aether.training.datasets import Phrase

_LIKELY_SPEAKER_KEYS = ["speaker", "author", "role"]
_LIKELY_UTTERANCE_KEYS = ["utterance", "text"]
_SYSTEM_SPEAKER_VALUES = {"SYSTEM", "system", "System", 1, "1"}


def _find_key(record: Dict[str, Any], candidates: List[str]) -> str:
    for key in candidates:
        if key in record:
            return key
    raise RuntimeError(
        f"none of the expected keys {candidates} found in an SGD turn record; "
        f"actual keys: {list(record.keys())} -- inspect this and update "
        "_LIKELY_SPEAKER_KEYS/_LIKELY_UTTERANCE_KEYS in this module"
    )


def extract_sgd_phrases(
    target_count: int = 18_000,
    min_words: int = 3,
    max_words: int = 30,
    seed: int = 0,
    dataset_id: str = "google-research-datasets/schema_guided_dstc8",
) -> List[Phrase]:
    """Extracts real, deduplicated SYSTEM-turn utterances from SGD.

    Raises with a specific, actionable message (not a bare KeyError/IndexError)
    if the installed `datasets` version's SGD schema doesn't match what this
    function expects.
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_id)

    seen = set()
    phrases: List[Phrase] = []
    speaker_key = None
    utterance_key = None

    for split_name in dataset.keys():
        for example in dataset[split_name]:
            turns = example.get("turns")
            if turns is None:
                raise RuntimeError(
                    f"no 'turns' field on an SGD example (split={split_name}); "
                    f"actual keys: {list(example.keys())}"
                )
            # `turns` may be a dict-of-lists (HF's columnar nesting) or a
            # list-of-dicts, depending on the installed datasets version --
            # handle both rather than assuming one.
            if isinstance(turns, dict):
                num_turns = len(next(iter(turns.values())))
                turn_records = [{key: value[i] for key, value in turns.items()} for i in range(num_turns)]
            else:
                turn_records = list(turns)

            for turn in turn_records:
                if speaker_key is None:
                    speaker_key = _find_key(turn, _LIKELY_SPEAKER_KEYS)
                if utterance_key is None:
                    utterance_key = _find_key(turn, _LIKELY_UTTERANCE_KEYS)
                if turn[speaker_key] not in _SYSTEM_SPEAKER_VALUES:
                    continue
                text = str(turn[utterance_key]).strip()
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
            "the word-count bounds or double check the speaker-value filter"
        )

    random.Random(seed).shuffle(phrases)
    selected = phrases[:target_count]
    # Re-number ids after shuffling+truncating so ids stay contiguous
    # (downstream code assumes phrase_id order matches list order).
    return [Phrase(phrase_id=f"sgd-{i:05d}", text=phrase.text) for i, phrase in enumerate(selected)]
