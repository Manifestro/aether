from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Phrase:
    phrase_id: str
    text: str


# Stage 5 minimal training set (docs/plan.md Phase C / spec.md Level B).
# Deliberately small -- ~20 examples is enough to check whether the
# mechanism (hidden state -> projector -> Voice Head) is learnable at all
# (does loss drop from a random start?), not to teach fluent English speech.
# English per this stage's scope decision (the confirmed teacher,
# kyutai/tts-1.6b-en_fr, covers English/French; Russian is a later step).
# Phrases reuse this project's actual weather/greeting domain (the same
# register as DeterministicSpeaker's chunks) rather than synthetic filler.
PHRASES: List[Phrase] = [
    Phrase("phrase-00", "I'm checking the weather in Almaty now."),
    Phrase("phrase-01", "It's rainy and twenty four degrees, you might want an umbrella."),
    Phrase("phrase-02", "Let me check the forecast for you."),
    Phrase("phrase-03", "It's sunny and warm today."),
    Phrase("phrase-04", "The temperature is dropping quickly this evening."),
    Phrase("phrase-05", "Hello! How are you doing today?"),
    Phrase("phrase-06", "Good morning, I hope you slept well."),
    Phrase("phrase-07", "It looks like snow is expected tonight."),
    Phrase("phrase-08", "The wind is quite strong right now."),
    Phrase("phrase-09", "I'm looking up the weather for Astana."),
    Phrase("phrase-10", "It's cloudy with a light breeze."),
    Phrase("phrase-11", "Expect thunderstorms later this afternoon."),
    Phrase("phrase-12", "The forecast says clear skies all week."),
    Phrase("phrase-13", "It's freezing outside, wear a warm coat."),
    Phrase("phrase-14", "Humidity is high, so it might feel warmer."),
    Phrase("phrase-15", "The weather in Almaty is rainy and twenty four degrees."),
    Phrase("phrase-16", "There's a chance of light showers tomorrow."),
    Phrase("phrase-17", "It's a beautiful and sunny afternoon."),
    Phrase("phrase-18", "The skies are clearing up nicely now."),
    Phrase("phrase-19", "Please bring an umbrella just in case."),
    # Held out -- not used for gradient updates; checked only as a soft
    # "does this look different from noise" observation (spec.md §20: not
    # a statistically meaningful generalization claim on 4 examples).
    Phrase("phrase-20", "It's mild and pleasant this morning."),
    Phrase("phrase-21", "Heavy rain is expected this weekend."),
    Phrase("phrase-22", "The air feels crisp and cold today."),
    Phrase("phrase-23", "Looks like a perfect day for a walk."),
]

TRAIN_PHRASES: List[Phrase] = PHRASES[:20]
HELD_OUT_PHRASES: List[Phrase] = PHRASES[20:]
