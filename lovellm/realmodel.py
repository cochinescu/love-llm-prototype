"""MLXBaseModel — the pinned real open-weight base model (build stage W2).

Pinned model (PROTOCOL "W2 freeze"): mlx-community/Qwen2.5-0.5B-Instruct-4bit
at revision a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3; decoding uses temperature 0.5,
top_p 0.95; per-call determinism via mx.random.seed derived from the pipeline's
threaded numpy rng (paired conditions share seeds, as in the synthetic run).

Prompt rendering: the shared feature space is a pseudo-word world, so the
prompt instructs the model to compose its reply from an explicit token list
derived from the conditioning vector (top topics by weight), plus the
layer-supplied context tokens and shared-memory markers, with the response
length set by depth_hint. The metrics remain fully mechanical on the emitted
text (topic-word counts, marker detection, token counts) — nothing here scores
anything. A marker can only be emitted if the layer put it in the prompt
(the no-leakage property, same as the synthetic stand-in).
"""

from __future__ import annotations

import numpy as np

from .features import STYLE_DIM, TOPIC_DIM, TOPIC_WORDS

PINNED_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
PINNED_REVISION = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
TEMP = 0.5
TOP_P = 0.95
N_PROMPT_TOPICS = 3
LIST_WORDS_TOTAL = 16   # word-list length, apportioned by conditioning weight (R1 renderer)
CONTEXT_EXCERPT = 12    # context tokens included in the prompt (same cap for ALL conditions)


class MLXBaseModel:
    """Real base model behind the same seam as SyntheticBaseModel."""

    def __init__(self, model_id: str = PINNED_MODEL, revision: str = PINNED_REVISION):
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        self.model_id = model_id
        self.revision = revision
        self.model, self.tok = load(model_id, revision=revision)
        self._sampler = make_sampler(temp=TEMP, top_p=TOP_P)

    def _vocab_from_conditioning(self, conditioning: np.ndarray) -> list[str]:
        """Word list apportioned by conditioning weight.

        The list is the renderer's faithful image of the conditioning vector:
        each of the top topics contributes words IN PROPORTION to its weight
        (largest-remainder over LIST_WORDS_TOTAL), so a guarded mix actually
        guards the emitted surface — a fixed per-topic word count would
        silently flatten the weights the mechanism set."""
        topic_w = np.maximum(conditioning[STYLE_DIM : STYLE_DIM + TOPIC_DIM], 0.0)
        top = [int(t) for t in np.argsort(topic_w)[::-1][:N_PROMPT_TOPICS] if topic_w[t] > 0]
        w = np.array([topic_w[t] for t in top])
        shares = LIST_WORDS_TOTAL * w / w.sum()
        counts = np.floor(shares).astype(int)
        rem = LIST_WORDS_TOTAL - int(counts.sum())
        order = sorted(range(len(top)), key=lambda i: -(shares[i] - counts[i]))
        for i in order[:rem]:
            counts[i] += 1
        words: list[str] = []
        for t, c in zip(top, counts):
            words.extend(TOPIC_WORDS[t][: max(1, int(c))])
        return words

    def respond(
        self,
        conditioning: np.ndarray,
        memory_markers: list[tuple[str, float]],
        depth_hint: float,
        temperature: float,  # decoding temp is PINNED; kept for seam compatibility
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
            # Same excerpt cap for every condition (known / padded stranger),
            # so token parity survives into the rendered prompt.
            parts.append(
                "Conversation context tokens: " + " ".join(context_tokens[:CONTEXT_EXCERPT]) + "."
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
