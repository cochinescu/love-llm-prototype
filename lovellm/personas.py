"""GENERATOR schema: scripted personas with constructed ground-truth hidden states.

Everything here is ground truth BY CONSTRUCTION (plan §2): mood/need/engagement
schedules are generated with known, recorded parameters, so other-model accuracy
is scored against construction, not judged. The mechanical signal map is the
contract the estimator can exploit:

  - mood -> intensifier rate:  rate = 0.08 + 0.10 * (mood + 1) / 2
  - need -> topic mix:         topics = 0.5 * pref + 0.5 * uniform(NEED_TOPICS[need])
  - engagement -> signal length

Shared-history events inject unique marker tokens (features.make_marker) at
known times — the constructed reference set for the marker-emission metric.
The estimator (othermodel.py) never imports this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import (
    FUNCTION_WORD_GROUPS,
    INTENSIFIER_GROUP,
    STYLE_DIM,
    TOPIC_DIM,
    TOPIC_WORDS,
    WORDS_PER_TOPIC,
    make_marker,
)

NEEDS = ("support", "info", "play", "vent")
# Generator's need -> topic map (two topics per need, disjoint).
NEED_TOPICS: dict[str, tuple[int, int]] = {
    "support": (0, 1),
    "info": (2, 3),
    "play": (4, 5),
    "vent": (6, 7),
}

MOOD_AR_PHI = 0.7
MOOD_AR_SIGMA = 0.15
MOOD_AMPLITUDE = 0.7
INTENSIFIER_BASE = 0.08
INTENSIFIER_SLOPE = 0.10


@dataclass(frozen=True)
class GroundTruth:
    mood: float          # [-1, 1]
    need: str            # one of NEEDS
    engagement: float    # [0, 1]


@dataclass(frozen=True)
class PersonaParams:
    user_id: int
    pref: np.ndarray             # topic preference simplex
    style_target: np.ndarray     # style coordinates in [0,1]^STYLE_DIM
    mood_period: float
    mood_phase: float
    need_transitions: np.ndarray  # (4,4) row-stochastic


class Persona:
    """Scripted persona: schedules precomputed from the seed (deterministic)."""

    def __init__(self, params: PersonaParams, n_turns: int, seed: int):
        self.params = params
        self.user_id = params.user_id
        rng = np.random.default_rng(seed)
        # Mood: sinusoid + AR(1) noise, clipped.
        noise = np.zeros(n_turns)
        for t in range(1, n_turns):
            noise[t] = MOOD_AR_PHI * noise[t - 1] + rng.normal(0.0, MOOD_AR_SIGMA)
        base = MOOD_AMPLITUDE * np.sin(
            2 * np.pi * np.arange(n_turns) / params.mood_period + params.mood_phase
        )
        self._mood = np.clip(base + noise, -1.0, 1.0)
        # Need: seeded Markov chain over NEEDS.
        needs = [int(rng.integers(len(NEEDS)))]
        for _ in range(1, n_turns):
            p = params.need_transitions[needs[-1]]
            needs.append(int(rng.choice(len(NEEDS), p=p)))
        self._need = [NEEDS[i] for i in needs]
        # Engagement tracks mood with small noise (recorded mechanical map).
        eng = 0.5 + 0.3 * self._mood + rng.normal(0.0, 0.05, size=n_turns)
        self._engagement = np.clip(eng, 0.0, 1.0)
        self._seed = seed

    def hidden_state(self, t: int) -> GroundTruth:
        return GroundTruth(float(self._mood[t]), self._need[t], float(self._engagement[t]))

    def topic_mix(self, t: int) -> np.ndarray:
        """The persona's current topic mix: 0.5*pref + 0.5*need topics.

        Also the target of the pref_alignment observable (attentiveness to the
        user's CURRENT state, not just their standing preference).
        """
        gt = self.hidden_state(t)
        need_vec = np.zeros(TOPIC_DIM)
        for topic in NEED_TOPICS[gt.need]:
            need_vec[topic] = 0.5
        return 0.5 * self.params.pref + 0.5 * need_vec

    def signal(self, t: int) -> str:
        """One user turn: mechanically modulated by the ground-truth state.

        Deterministic in t (fresh rng per call), so the same persona object
        yields identical signal streams across arms and runs."""
        gt = self.hidden_state(t)
        rng = np.random.default_rng(self._seed + 7919 * t)
        length = max(6, int(round(10 + 14 * gt.engagement + rng.normal(0, 2))))
        fw_rate = 0.30
        intens = INTENSIFIER_BASE + INTENSIFIER_SLOPE * (gt.mood + 1.0) / 2.0
        topic_p = self.topic_mix(t)
        topic_p = topic_p / topic_p.sum()
        style_p = np.maximum(self.params.style_target, 1e-9)
        # Zero out the intensifier group in the style draw: its rate is
        # controlled separately (the mood surface must stay mechanical).
        style_p[INTENSIFIER_GROUP] = 0.0
        style_p = style_p / style_p.sum()
        words: list[str] = []
        for _ in range(length):
            r = rng.random()
            if r < intens:
                words.append(FUNCTION_WORD_GROUPS[INTENSIFIER_GROUP][int(rng.integers(5))])
            elif r < intens + fw_rate:
                g = int(rng.choice(STYLE_DIM, p=style_p))
                words.append(FUNCTION_WORD_GROUPS[g][int(rng.integers(5))])
            else:
                topic = int(rng.choice(TOPIC_DIM, p=topic_p))
                words.append(TOPIC_WORDS[topic][int(rng.integers(WORDS_PER_TOPIC))])
        return " ".join(words) + "."

    def shared_event(self, session: int) -> str:
        """Marker token introduced at the start of each session (ground truth)."""
        return make_marker(self.user_id, session)


def make_pool(n_users: int, n_turns: int, seed: int) -> list[Persona]:
    rng = np.random.default_rng(seed)
    pool = []
    for u in range(n_users):
        pref = rng.dirichlet(np.ones(TOPIC_DIM) * 1.5)
        style = rng.uniform(0.2, 0.8, size=STYLE_DIM)
        period = float(rng.uniform(10.0, 22.0))
        phase = float(rng.uniform(0, 2 * np.pi))
        trans = rng.dirichlet(np.ones(4) * 2.0, size=4)
        # Sticky needs: boost the diagonal, renormalize.
        trans = trans + 1.5 * np.eye(4)
        trans = trans / trans.sum(axis=1, keepdims=True)
        params = PersonaParams(u, pref, style, period, phase, trans)
        pool.append(Persona(params, n_turns, seed + 71 * u + 13))
    return pool
