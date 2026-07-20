"""Relationship state R: history, preference estimate, shared references, trust.

R is a DATA STRUCTURE with defined update rules (plan §2 "claims NOT made":
`trust` is a stored scalar, never psychological trust). The layer sees only
emitted text — the engagement proxy below is computed from observables, never
from the generator's ground truth.

Constants here are PROVISIONAL protocol items (PROTOCOL.md "Formal model
constants"); changing them after the real grid starts is a protocol violation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import TOPIC_DIM, cosine, marker_tokens, text_features, tokens

ETA_TRUST = 0.04       # trust update step (PROVISIONAL)
EWMA_PREF = 0.20       # preference-estimate EWMA (PROVISIONAL)
SALIENCE_DECAY = 0.98  # per-turn salience decay (PROVISIONAL)
MARKER_SALIENCE_BONUS = 1.0
RECENCY_BASE = 0.98    # retrieval recency factor per turn of age (persistence: old
                       # salient memories must stay retrievable across sessions)


@dataclass
class MemoryItem:
    features: np.ndarray
    toks: list[str]
    markers: set[str]
    session: int
    turn: int
    salience: float


@dataclass
class RState:
    history: list[MemoryItem] = field(default_factory=list)
    pref_est: np.ndarray = field(default_factory=lambda: np.full(TOPIC_DIM, 1.0 / TOPIC_DIM))
    shared_refs: dict[str, float] = field(default_factory=dict)
    trust: float = 0.0
    depth: int = 0


def empty() -> RState:
    """The stranger / empty-R constructor."""
    return RState()


def engagement_proxy(text: str) -> float:
    """Observable-only proxy: signal length + shared-reference mention."""
    toks = tokens(text)
    length_part = min(len(toks) / 30.0, 1.0)
    marker_part = 0.3 if marker_tokens(text) else 0.0
    return float(np.clip(0.7 * length_part + marker_part, 0.0, 1.0))


def update(r: RState, text: str, session: int, turn: int) -> None:
    """The R update rule (one user turn)."""
    marks = marker_tokens(text)
    proxy = engagement_proxy(text)
    r.trust = float(np.clip(r.trust + ETA_TRUST * proxy, 0.0, 1.0))
    feats = text_features(text)
    topic_part = feats[-TOPIC_DIM:]
    if topic_part.sum() > 0:
        r.pref_est = (1 - EWMA_PREF) * r.pref_est + EWMA_PREF * topic_part
        r.pref_est = r.pref_est / r.pref_est.sum()
    for item in r.history:
        item.salience *= SALIENCE_DECAY
    for m in marks:
        r.shared_refs[m] = r.shared_refs.get(m, 0.0) + 1.0
    salience = 1.0 + (MARKER_SALIENCE_BONUS if marks else 0.0)
    r.history.append(MemoryItem(feats, tokens(text), marks, session, turn, salience))
    r.depth += 1


def retrieve(r: RState, query_features: np.ndarray, k: int) -> list[MemoryItem]:
    """Memory-weighted retrieval: salience x recency x relevance, deterministic."""
    if not r.history or k <= 0:
        return []
    age_last = r.history[-1].turn if r.history else 0
    scored = []
    for i, item in enumerate(r.history):
        recency = RECENCY_BASE ** max(0, age_last - item.turn)
        relevance = 0.5 + 0.5 * cosine(item.features, query_features)
        scored.append((-(item.salience * recency * relevance), i))
    scored.sort()
    return [r.history[i] for _, i in scored[:k]]


def retrieve_generic(r: RState, query_features: np.ndarray, k: int) -> list[MemoryItem]:
    """Relevance-only retrieval — the generic-personalization control's
    retriever (no salience weighting, no recency shaping, no allocation)."""
    if not r.history or k <= 0:
        return []
    scored = [(-cosine(item.features, query_features), i) for i, item in enumerate(r.history)]
    scored.sort()
    return [r.history[i] for _, i in scored[:k]]
