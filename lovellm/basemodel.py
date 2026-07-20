"""Base-model interface + the synthetic stand-in.

`BaseModel.respond()` is the seam where a real pinned open-weight model plugs
in later (build stage W2): given a conditioning vector in the shared feature
space, the retrieved memory items (with weights), and a response-length hint,
return one utterance. All arms share ONE base model instance — the plan's
matched-base-model identification.

`SyntheticBaseModel` emulates a companion LLM's response surface so the
pipeline, arms, probes, and metrics validate end-to-end without weights:
  - topic/style word choice follows the conditioning vector;
  - marker tokens present in the supplied memory items are emitted with
    probability proportional to their weight (mechanical — a marker can NEVER
    appear unless the layer put it in context: no leakage);
  - response length follows depth_hint (the depth_budget observable path).
Numbers measured against it are PIPELINE VALIDATION, not paper results
(results/RESULTS.md carries the banner).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .features import (
    FUNCTION_WORD_GROUPS,
    STYLE_DIM,
    TOPIC_DIM,
    TOPIC_WORDS,
    WORDS_PER_TOPIC,
)

MARKER_EMIT_COEF = 0.35  # per-marker emission probability = min(0.9, coef * weight)


class BaseModel(Protocol):
    def respond(
        self,
        conditioning: np.ndarray,
        memory_markers: list[tuple[str, float]],
        depth_hint: float,
        temperature: float,
        rng: np.random.Generator,
        context_tokens: list[str] | None = None,
    ) -> str: ...


def _tempered(p: np.ndarray, temperature: float) -> np.ndarray:
    q = np.power(np.maximum(p, 1e-9), 1.0 / max(temperature, 0.05))
    return q / q.sum()


class SyntheticBaseModel:
    """Seeded synthetic responder over the shared vocabulary."""

    def respond(
        self,
        conditioning: np.ndarray,
        memory_markers: list[tuple[str, float]],
        depth_hint: float,
        temperature: float,
        rng: np.random.Generator,
        context_tokens: list[str] | None = None,
    ) -> str:
        # context_tokens are accepted for seam parity with the real model
        # (which consumes them); the synthetic stand-in reads only the
        # conditioning vector and marker list — padding is structural here.
        style = np.clip(conditioning[:STYLE_DIM], 0.0, 1.0)
        topic_w = np.maximum(conditioning[STYLE_DIM : STYLE_DIM + TOPIC_DIM], 1e-9)
        topic_p = _tempered(topic_w / topic_w.sum(), temperature)
        style_p = np.maximum(style, 1e-9)
        style_p = style_p / style_p.sum()
        fw_rate = 0.2 + 0.2 * float(style.mean())

        length = max(4, int(round(rng.normal(depth_hint, 1.5))))
        words: list[str] = []
        for _ in range(length):
            if rng.random() < fw_rate:
                g = int(rng.choice(STYLE_DIM, p=style_p))
                words.append(FUNCTION_WORD_GROUPS[g][int(rng.integers(5))])
            else:
                t = int(rng.choice(TOPIC_DIM, p=topic_p))
                words.append(TOPIC_WORDS[t][int(rng.integers(WORDS_PER_TOPIC))])

        # Shared-history markers: emitted only from supplied memory, with
        # probability proportional to the layer-assigned weight.
        for marker, weight in memory_markers:
            if rng.random() < min(0.9, MARKER_EMIT_COEF * weight):
                pos = int(rng.integers(len(words) + 1))
                words.insert(pos, marker)
        return " ".join(words) + "."
