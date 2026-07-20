"""ESTIMATOR schema: the per-user other-model, in its OWN representation.

Schema-independence rule (PROTOCOL, pytest-enforced): this module never
imports personas.py. It reads emitted text only, through the shared surface
(features.py). Its labels (`EST_NEEDS`) and its state (`warmth`, a style-cue
EWMA) are its own; the frozen scoring map that aligns them with the
generator's ground truth for accuracy scoring lives in metrics.py.

The other-model's outputs are ESTIMATES IN A DATA STRUCTURE (the stated scope
boundary) — nothing here is empathic access, and the paper never calls it
theory of mind.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import TOPIC_DIM, intensifier_rate, topic_vector

EWMA_WARMTH = 0.15   # PROVISIONAL (PROTOCOL "Formal model constants")
EWMA_NEED = 0.25     # PROVISIONAL
CONF_TAU = 10.0      # confidence saturation constant

# The estimator's OWN need labels and its OWN topic->need reading.
EST_NEEDS = ("comfort", "answers", "fun", "listening")
_EST_TOPIC_NEED = np.zeros((TOPIC_DIM, 4))
for _t in range(TOPIC_DIM):
    _EST_TOPIC_NEED[_t, _t // 2] = 1.0

# Valence bins from the warmth EWMA. With the observable intensifier surface
# spanning roughly [0.08, 0.18], these thresholds bracket the neutral band.
WARMTH_NEG = 0.1225
WARMTH_POS = 0.1375
NEUTRAL_WARMTH = 0.13


@dataclass(frozen=True)
class Estimate:
    valence_bin: str   # "neg" | "neu" | "pos"
    need_label: str    # one of EST_NEEDS
    conf: float


@dataclass
class OtherModel:
    warmth: float = NEUTRAL_WARMTH
    need_belief: np.ndarray = field(default_factory=lambda: np.full(4, 0.25))
    n_obs: int = 0


def prior() -> OtherModel:
    """The empty-R estimator: no accumulated observations."""
    return OtherModel()


def observe(om: OtherModel, text: str) -> None:
    """Per-turn update from one emitted user signal (text only)."""
    om.warmth = (1 - EWMA_WARMTH) * om.warmth + EWMA_WARMTH * intensifier_rate(text)
    tv = topic_vector(text)
    if tv.sum() > 0:
        need_obs = tv @ _EST_TOPIC_NEED
        need_obs = need_obs / need_obs.sum()
        om.need_belief = (1 - EWMA_NEED) * om.need_belief + EWMA_NEED * need_obs
        om.need_belief = om.need_belief / om.need_belief.sum()
    om.n_obs += 1


def estimate(om: OtherModel) -> Estimate:
    if om.warmth < WARMTH_NEG:
        valence = "neg"
    elif om.warmth > WARMTH_POS:
        valence = "pos"
    else:
        valence = "neu"
    need = EST_NEEDS[int(np.argmax(om.need_belief))]
    conf = float(1.0 - np.exp(-om.n_obs / CONF_TAU))
    return Estimate(valence, need, conf)


def need_topics(om: OtherModel) -> np.ndarray:
    """The estimator's need belief mapped back to a topic boost vector —
    what the conditioning uses to attend to the user's CURRENT state."""
    return _EST_TOPIC_NEED @ om.need_belief
