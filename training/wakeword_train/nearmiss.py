"""Near-miss phrases for any wake phrase, so a phrase typed into Wake Word Studio gets hard negatives
automatically (as hand-written ones were for "Hey UNO").

Uses the CMU Pronouncing Dictionary (BSD-2-Clause, downloaded by the setup step): for each word, words
one phoneme away (substituted, dropped or added), plus the phrase's own fragments and common speech.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np

COMMON = [
    "hey", "hello", "okay", "you know", "hey there", "hello there", "excuse me", "thank you",
    "what time is it", "I don't know", "see you later", "come here",
]

_WORD = re.compile(r"^[a-z][a-z']*$")


class Pronunciations:
    def __init__(self, entries: dict[str, list[tuple[str, ...]]]):
        self.entries = entries
        self.by_phones: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for word, prons in entries.items():
            for p in prons:
                self.by_phones[p].append(word)
        self.phones = sorted({ph for prons in entries.values() for p in prons for ph in p})

    @staticmethod
    def load(path: Path) -> "Pronunciations":
        entries: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            word, *phones = line.split()
            word = re.sub(r"\(\d+\)$", "", word.lower())
            if _WORD.match(word):
                entries[word].append(tuple(re.sub(r"\d", "", p) for p in phones))  # drop stress marks
        return Pronunciations(dict(entries))

    def neighbours(self, word: str) -> list[str]:
        """Words whose pronunciation is one phoneme away from any pronunciation of `word`."""
        found: set[str] = set()
        for p in self.entries.get(word.lower(), []):
            for i in range(len(p) + 1):
                for ph in self.phones:
                    found.update(self.by_phones.get(p[:i] + (ph,) + p[i:], []))  # insert
                if i < len(p):
                    found.update(self.by_phones.get(p[:i] + p[i + 1:], []))  # drop
                    for ph in self.phones:
                        if ph != p[i]:
                            found.update(self.by_phones.get(p[:i] + (ph,) + p[i + 1:], []))  # substitute
        found.discard(word.lower())
        return sorted(w for w in found if len(w) >= 2)


def near_misses(phrase: str, pron: Pronunciations, per_word: int = 6, limit: int = 40, seed: int = 0) -> list[str]:
    words = phrase.lower().split()
    rng = np.random.default_rng(seed)
    out: list[str] = []

    # Fragments: each word alone, every prefix and suffix ("hey", "computer", "hey" + ...).
    out += words if len(words) > 1 else []
    out += [" ".join(words[:k]) for k in range(1, len(words))]
    out += [" ".join(words[k:]) for k in range(1, len(words))]

    # One word swapped for a word that sounds almost the same ("hey commuter").
    for i, w in enumerate(words):
        candidates = pron.neighbours(w)
        if i == len(words) - 1:
            # "computers", "computer's": a sound added at the very end is too close to the phrase itself
            # to train as a negative without hurting recall.
            candidates = [c for c in candidates if not c.startswith(w)]
        if not candidates:
            continue
        picks = rng.choice(candidates, size=min(per_word, len(candidates)), replace=False)
        out += [" ".join(words[:i] + [str(c)] + words[i + 1:]) for c in picks]

    out += COMMON
    seen, result = {phrase.lower()}, []
    for text in out:
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result[:limit]


def phrase_id(phrase: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_") or "phrase"


def phrase_config(phrase: str, pron: Pronunciations) -> dict:
    """A `phrases` entry for the pipeline config."""
    words = phrase.split()
    text = " ".join(words).lower()
    spellings = [text, f"{text}!", f"{text}?"]
    if len(words) > 1:
        spellings.append(f"{words[0].lower()}, {' '.join(words[1:]).lower()}")
    return {
        "id": phrase_id(phrase),
        "display": " ".join(words),
        "tts_texts": spellings,
        "adversarial_texts": near_misses(phrase, pron),
        "transcript_variants": [text],
    }
