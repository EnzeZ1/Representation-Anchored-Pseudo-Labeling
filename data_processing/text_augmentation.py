"""Deterministic shared text augmentation for semi-supervised regression."""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass
from typing import Callable, Iterable

AUGMENTATION_VERSION = "rapl-text-augmentation-v1"
PROTECTED_NEGATIONS = frozenset({"no", "not", "never", "n't"})
TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)?|[^\w\s]", re.UNICODE)
CONTENT_POS_PREFIXES = ("NN", "VB", "JJ", "RB")


def augmentation_seed(formal_seed: int, epoch: int, sample_id: str, sentence_side: int) -> int:
    key = f"{formal_seed}\0{epoch}\0{sample_id}\0{sentence_side}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def _default_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except (ImportError, LookupError):
        return {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "were", "will", "with",
        }


def _wordnet_synonyms(token: str, pos_tag: str) -> list[str]:
    try:
        from nltk.corpus import wordnet
    except ImportError as error:
        raise RuntimeError("nltk with WordNet is required for strong text augmentation") from error
    wn_pos = None
    if pos_tag.startswith("NN"):
        wn_pos = wordnet.NOUN
    elif pos_tag.startswith("VB"):
        wn_pos = wordnet.VERB
    elif pos_tag.startswith("JJ"):
        wn_pos = wordnet.ADJ
    elif pos_tag.startswith("RB"):
        wn_pos = wordnet.ADV
    values = set()
    try:
        synsets = wordnet.synsets(token, pos=wn_pos)
    except LookupError as error:
        raise RuntimeError("NLTK WordNet data is required for strong text augmentation") from error
    for synset in synsets:
        for lemma in synset.lemmas():
            candidate = lemma.name().replace("_", " ").strip()
            if (
                candidate
                and candidate.casefold() != token.casefold()
                and not any(character.isdigit() for character in candidate)
            ):
                values.add(candidate)
    return sorted(values, key=lambda value: (value.casefold(), value))


def _pos_tags(tokens: list[str]) -> list[str]:
    try:
        import nltk
        return [tag for _, tag in nltk.pos_tag(tokens)]
    except ImportError as error:
        raise RuntimeError("nltk is required for POS-aware text augmentation") from error
    except LookupError as error:
        raise RuntimeError("NLTK POS-tagger data is required for strong text augmentation") from error


@dataclass(frozen=True)
class TextAugmenter:
    replacement_rate: float = 0.10
    insertion_rate: float = 0.05
    edit_cap: float = 0.20
    synonym_provider: Callable[[str, str], list[str]] = _wordnet_synonyms
    pos_tagger: Callable[[list[str]], list[str]] = _pos_tags
    stopwords: frozenset[str] = frozenset()

    def __post_init__(self):
        if self.replacement_rate < 0 or self.insertion_rate < 0 or not 0 <= self.edit_cap <= 1:
            raise ValueError("Invalid augmentation rates")
        if not self.stopwords:
            object.__setattr__(self, "stopwords", frozenset(_default_stopwords()))

    def augment_sentence(
        self, sentence: str, formal_seed: int, epoch: int, sample_id: str, sentence_side: int
    ) -> str:
        original_tokens = TOKEN_PATTERN.findall(sentence)
        if not original_tokens:
            return sentence
        tags = self.pos_tagger(original_tokens)
        eligible = []
        synonyms: dict[int, list[str]] = {}
        for index, (token, tag) in enumerate(zip(original_tokens, tags)):
            folded = token.casefold()
            if (
                not tag.startswith(CONTENT_POS_PREFIXES)
                or folded in self.stopwords
                or folded in PROTECTED_NEGATIONS
                or token.isnumeric()
                or not any(character.isalpha() for character in token)
            ):
                continue
            candidates = self.synonym_provider(token, tag)
            if candidates:
                eligible.append(index)
                synonyms[index] = candidates
        if not eligible:
            return sentence
        rng = random.Random(augmentation_seed(formal_seed, epoch, sample_id, sentence_side))
        cap = max(1, int(math.floor(len(eligible) * self.edit_cap)))
        replacement_count = min(cap, int(math.ceil(len(eligible) * self.replacement_rate)))
        insertion_count = min(
            cap - replacement_count, int(math.ceil(len(eligible) * self.insertion_rate))
        )
        shuffled = eligible.copy()
        rng.shuffle(shuffled)
        replacements = shuffled[:replacement_count]
        insertions = shuffled[replacement_count : replacement_count + insertion_count]
        output = list(original_tokens)
        for index in replacements:
            output[index] = rng.choice(synonyms[index])
        insertion_values = [(index, rng.choice(synonyms[index])) for index in insertions]
        for _, value in insertion_values:
            position = rng.randrange(len(output) + 1)
            output.insert(position, value)
        return " ".join(output) if output else sentence

    def augment_pair(
        self, sentence1: str, sentence2: str, formal_seed: int, epoch: int, sample_id: str
    ) -> tuple[str, str]:
        return (
            self.augment_sentence(sentence1, formal_seed, epoch, sample_id, 1),
            self.augment_sentence(sentence2, formal_seed, epoch, sample_id, 2),
        )


def weak_view(sentence1: str, sentence2: str) -> tuple[str, str]:
    return sentence1, sentence2
