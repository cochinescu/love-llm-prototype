"""Shared text surface: style groups, topic pseudo-words, marker tokens.

This module is the only vocabulary both sides may share. The persona GENERATOR
(personas.py) and the other-model ESTIMATOR (othermodel.py) each build their
own representations on top of it; the estimator never sees the generator's
schedules (schema independence, pytest-enforced).

Style features are function-word GROUP rates — they do not depend on which
content words appear. Group 4 (intensifiers) is the mood-modulated group: the
generator's mechanical map raises the intensifier rate with mood, which is the
signal the estimator's `warmth` EWMA reads.
"""

from __future__ import annotations

import numpy as np

STYLE_DIM = 6
TOPIC_DIM = 8
K = STYLE_DIM + TOPIC_DIM
INTENSIFIER_GROUP = 4

FUNCTION_WORD_GROUPS: list[list[str]] = [
    ["the", "a", "an", "this", "that"],           # determiners
    ["and", "but", "or", "so", "yet"],            # conjunctions
    ["of", "in", "on", "at", "by"],               # prepositions
    ["is", "was", "be", "are", "been"],           # copulas
    ["quite", "rather", "very", "really", "truly"],  # intensifiers (mood surface)
    ["it", "they", "we", "you", "she"],           # pronouns
]
FUNCTION_WORDS: dict[str, int] = {
    w: g for g, group in enumerate(FUNCTION_WORD_GROUPS) for w in group
}

_SYLLABLES = ["ka", "re", "mi", "to", "su", "ne", "lo", "da", "vi", "bu"]
WORDS_PER_TOPIC = 40


def _content_word(topic: int, index: int) -> str:
    h = (topic * 7919 + index * 104729) % (10**6)
    s = [_SYLLABLES[(h // (10**i)) % 10] for i in range(3)]
    return "".join(s) + str(topic)


TOPIC_WORDS: list[list[str]] = [
    [_content_word(t, i) for i in range(WORDS_PER_TOPIC)] for t in range(TOPIC_DIM)
]
_CONTENT_TOPIC: dict[str, int] = {
    w: t for t, words in enumerate(TOPIC_WORDS) for w in words
}

# Style-neutral, topic-neutral filler vocabulary for token parity padding.
NEUTRAL_FILLER: list[str] = [f"flb{i}" for i in range(40)]

MARKER_PREFIX = "mk"


def make_marker(user_id: int, event_id: int) -> str:
    """Unique shared-history marker token for (user, event)."""
    return f"{MARKER_PREFIX}{user_id}x{event_id}"


def is_marker(token: str) -> bool:
    return token.startswith(MARKER_PREFIX) and "x" in token[len(MARKER_PREFIX):]


def tokens(text: str) -> list[str]:
    return [w for w in text.lower().replace(".", " ").split() if w]


def marker_tokens(text: str) -> set[str]:
    return {w for w in tokens(text) if is_marker(w)}


def style_vector(text: str) -> np.ndarray:
    """Function-word group rates, scaled to use [0,1]."""
    toks = tokens(text)
    if not toks:
        return np.zeros(STYLE_DIM)
    counts = np.zeros(STYLE_DIM)
    for w in toks:
        g = FUNCTION_WORDS.get(w)
        if g is not None:
            counts[g] += 1
    return np.clip(counts / len(toks) / 0.25, 0.0, 1.0)


def intensifier_rate(text: str) -> float:
    """Raw per-token rate of the intensifier group (the mood surface)."""
    toks = tokens(text)
    if not toks:
        return 0.0
    n = sum(1 for w in toks if FUNCTION_WORDS.get(w) == INTENSIFIER_GROUP)
    return n / len(toks)


def topic_vector(text: str) -> np.ndarray:
    """Topic proportions over content words (simplex, or zeros)."""
    toks = tokens(text)
    counts = np.zeros(TOPIC_DIM)
    for w in toks:
        t = _CONTENT_TOPIC.get(w)
        if t is not None:
            counts[t] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def text_features(text: str) -> np.ndarray:
    return np.concatenate([style_vector(text), topic_vector(text)])


def response_depth(text: str) -> float:
    """The depth observable: token count of a response."""
    return float(len(tokens(text)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pad_to_parity(context_tokens: list[str], target_len: int, rng: np.random.Generator) -> list[str]:
    """Pad a context token list with neutral filler to exactly target_len tokens.

    The token-parity rule (PROTOCOL): divergence probes always compare contexts
    of identical token length, so measured divergence cannot be attributed to
    prompt-length sensitivity.
    """
    if len(context_tokens) > target_len:
        raise ValueError("parity target shorter than context")
    n = target_len - len(context_tokens)
    filler = [NEUTRAL_FILLER[int(rng.integers(len(NEUTRAL_FILLER)))] for _ in range(n)]
    return context_tokens + filler
