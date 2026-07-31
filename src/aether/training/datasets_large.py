"""Large-scale (~10k) English phrase generation for the real Voice Head run.

Deliberately combinatorial/templated rather than hand-authored: reaching
this order of magnitude by hand isn't practical, and the point (per the
20/24-phrase memorization finding in earlier runs) is genuine lexical and
structural diversity across the assistant domain, not just volume in one
narrow topic. Deterministic given a seed, so the same run can be
reproduced or resumed by phrase_id.
"""

import itertools
import random
from typing import List

from aether.training.datasets import Phrase

_CITIES = [
    "Almaty", "Astana", "London", "Paris", "Tokyo", "Berlin", "Madrid",
    "Rome", "Toronto", "Seattle", "Chicago", "Dubai", "Singapore", "Seoul",
    "Amsterdam", "Vienna", "Prague", "Lisbon", "Warsaw", "Helsinki",
    "Osaka", "Bangkok", "Nairobi", "Cairo", "Istanbul", "Athens",
    "Barcelona", "Munich", "Zurich", "Dublin", "Auckland", "Vancouver",
    "Denver", "Austin", "Boston", "Miami", "Montreal", "Oslo",
    "Stockholm", "Copenhagen",
]
_CONDITIONS = [
    "sunny", "rainy", "cloudy", "snowy", "windy", "foggy", "clear",
    "humid", "stormy", "mild",
]
_ITEMS = [
    "an umbrella", "a warm coat", "sunglasses", "a light jacket",
    "an extra layer",
]
_TIMES_OF_DAY = ["morning", "afternoon", "evening", "day"]
_ACTIONS = [
    "check the weather", "set a reminder", "send a message",
    "look that up", "schedule a meeting", "find a restaurant",
    "book a table", "play some music", "turn off the lights",
    "add that to your calendar", "call a taxi", "order dinner",
    "check the traffic", "read the news", "translate that",
    "set an alarm", "start a timer", "check your email",
    "find a nearby cafe", "look up directions",
]
_ACTIONS_PAST = [
    "checked the weather", "set the reminder", "sent the message",
    "looked that up", "scheduled the meeting", "found a restaurant",
    "booked the table", "started the music", "turned off the lights",
    "added that to your calendar", "called a taxi", "placed the order",
    "checked the traffic", "pulled up the news", "translated that",
    "set the alarm", "started the timer", "checked your email",
    "found a nearby cafe", "looked up directions",
]
_TOPICS = [
    "the weather", "your schedule", "that restaurant", "the news",
    "your reminders", "traffic conditions", "nearby events",
    "your calendar", "the forecast", "local restaurants",
    "public transport", "flight status", "the score", "the meeting time",
    "your messages",
]
_EVENTS = [
    "meeting", "appointment", "flight", "reservation", "call",
    "reminder", "interview", "checkup", "delivery", "pickup",
]
_TIME_VALUES = [
    "9 AM", "10:30 AM", "noon", "1 PM", "3 PM", "5:30 PM", "7 PM",
    "8:15 PM", "6 AM", "11 AM",
]
_FACTS = [
    "its historic architecture", "its vibrant food scene",
    "its beautiful parks", "its music festivals", "its old town",
    "its museums", "its coastline", "its nightlife",
    "its street markets", "its skyline",
]
_TASKS = [
    "scheduling", "reminders", "weather updates", "general questions",
    "finding places nearby", "translations", "basic math",
    "setting alarms", "reading the news",
]
_TEMPS = list(range(-10, 36, 2))

_TEMPLATES = [
    lambda r: f"What's the weather like in {r.choice(_CITIES)} right now?",
    lambda r: f"It's {r.choice(_CONDITIONS)} and {r.choice(_TEMPS)} degrees in {r.choice(_CITIES)}.",
    lambda r: f"It's {r.choice(_CONDITIONS)} in {r.choice(_CITIES)}, you might want {r.choice(_ITEMS)}.",
    lambda r: f"Hello! How are you doing this {r.choice(_TIMES_OF_DAY)}?",
    lambda r: "I'm doing well, thank you for asking!",
    lambda r: f"Sure, I'll {r.choice(_ACTIONS)} right away.",
    lambda r: f"I've {r.choice(_ACTIONS_PAST)} for you.",
    lambda r: f"Let me check {r.choice(_TOPICS)} for you.",
    lambda r: "One moment, please.",
    lambda r: f"Your {r.choice(_EVENTS)} is set for {r.choice(_TIME_VALUES)}.",
    lambda r: f"I've set a reminder for your {r.choice(_EVENTS)} at {r.choice(_TIME_VALUES)}.",
    lambda r: f"Could you tell me more about {r.choice(_TOPICS)}?",
    lambda r: f"Just to confirm, you'd like me to {r.choice(_ACTIONS)}?",
    lambda r: f"That's interesting, tell me more about {r.choice(_TOPICS)}.",
    lambda r: f"I'm sorry, I couldn't find information about {r.choice(_TOPICS)}.",
    lambda r: "Apologies, let me try that again.",
    lambda r: f"{r.choice(_CITIES)} is known for {r.choice(_FACTS)}.",
    lambda r: f"Have a great {r.choice(_TIMES_OF_DAY)}, talk to you soon!",
    lambda r: "You're welcome! Let me know if you need anything else.",
    lambda r: f"I can help you with {r.choice(_TASKS)}, just let me know.",
]


def generate_phrases(count: int = 10_000, seed: int = 0) -> List[Phrase]:
    """Deterministically generates up to `count` unique phrases.

    Draws repeatedly from the template set (each template fills its own
    slots randomly from a fixed RNG stream), deduplicating by exact text,
    until `count` unique phrases are collected or attempts are exhausted.
    """
    rng = random.Random(seed)
    seen = set()
    phrases: List[Phrase] = []
    attempts = 0
    max_attempts = count * 50
    while len(phrases) < count and attempts < max_attempts:
        attempts += 1
        template = rng.choice(_TEMPLATES)
        text = template(rng)
        if text in seen:
            continue
        seen.add(text)
        phrases.append(Phrase(phrase_id=f"phrase-{len(phrases):05d}", text=text))
    if len(phrases) < count:
        raise RuntimeError(
            f"only generated {len(phrases)}/{count} unique phrases after {max_attempts} "
            "attempts -- widen the template/fill-value pools"
        )
    return phrases


def train_val_split(phrases: List[Phrase], val_fraction: float = 0.05) -> "tuple[list, list]":
    """Deterministic split by position -- the phrase list itself is already
    in a random draw order (not sorted by template), so a straight slice is
    an unbiased sample, not "the last few alphabetically" or similar."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    val_count = max(1, int(len(phrases) * val_fraction))
    return phrases[:-val_count], phrases[-val_count:]
