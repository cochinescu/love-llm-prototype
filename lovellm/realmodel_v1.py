"""The ORIGINAL (pre-R1) prompt renderer, restored 2026-10-06 (PROTOCOL Amendment A9).

The initial real-model grid (`results/real/`, scripts/reproduce_real.py) was
generated with this renderer, recovered verbatim in behavior from the paper
repository's history (commit 744ba1a):

- a FIXED word count per topic: the top N_PROMPT_TOPICS topics each
  contribute WORDS_PER_TOPIC_IN_PROMPT = 6 words regardless of their
  conditioning weight (this flattening is one of the two defects R1 fixed);
- a 60-token context excerpt cap (R1 reduced it to CONTEXT_EXCERPT = 12).

Everything else — pinned model and revision, decoding, the "exactly n tokens"
instruction, the generation cap 2·n+16, the marker instruction — is shared
with `realmodel.MLXBaseModel`. `scripts/reproduce_real.py` must use THIS
class; `reproduce_real_v2.py` onward use the R1 renderer.
"""

from __future__ import annotations

import numpy as np

from .features import STYLE_DIM, TOPIC_DIM, TOPIC_WORDS
from .realmodel import N_PROMPT_TOPICS, PINNED_MODEL, PINNED_REVISION, MLXBaseModel

WORDS_PER_TOPIC_IN_PROMPT = 6   # original renderer: fixed per-topic word count
CONTEXT_EXCERPT_V1 = 60         # original renderer: context cap


class MLXBaseModelV1(MLXBaseModel):
    """Original renderer over the same pinned model and decoding."""

    def __init__(self, model_id: str = PINNED_MODEL, revision: str = PINNED_REVISION):
        super().__init__(model_id, revision)

    def _vocab_from_conditioning(self, conditioning: np.ndarray) -> list[str]:
        topic_w = np.maximum(conditioning[STYLE_DIM : STYLE_DIM + TOPIC_DIM], 0.0)
        top = np.argsort(topic_w)[::-1][:N_PROMPT_TOPICS]
        words: list[str] = []
        for t in top:
            if topic_w[t] <= 0:
                continue
            words.extend(TOPIC_WORDS[int(t)][:WORDS_PER_TOPIC_IN_PROMPT])
        return words

    def respond(
        self,
        conditioning: np.ndarray,
        memory_markers: list[tuple[str, float]],
        depth_hint: float,
        temperature: float,
        rng: np.random.Generator,
        context_tokens: list[str] | None = None,
    ) -> str:
        import mlx.core as mx
        from mlx_lm import generate

        n = max(4, int(round(depth_hint)))
        words = self._vocab_from_conditioning(conditioning)
        markers = [m for m, _ in memory_markers]
        parts = []
        if context_tokens:
            parts.append(
                "Conversation context tokens: " + " ".join(context_tokens[:CONTEXT_EXCERPT_V1]) + "."
            )
        parts.append(
            f"Write one line of exactly {n} tokens, each copied EXACTLY from this list "
            f"(repeats allowed): {' '.join(words)}."
        )
        if markers:
            parts.append("Also include these memory tokens somewhere: " + " ".join(markers) + ".")
        parts.append("Output only the tokens separated by spaces, nothing else.")
        user = " ".join(parts)

        prompt = self.tok.apply_chat_template(
            [{"role": "user", "content": user}], add_generation_prompt=True
        )
        mx.random.seed(int(rng.integers(2**31)))
        out = generate(
            self.model,
            self.tok,
            prompt=prompt,
            max_tokens=int(2 * n + 16),
            sampler=self._sampler,
            verbose=False,
        )
        return out.strip() + "."
