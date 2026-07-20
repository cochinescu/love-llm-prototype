"""Divergence channels, marker emission, quality floor, and the frozen scoring map.

Divergence aggregates EXACTLY the channels named by the allocation→observable
map (plan §3): initiative_rate, response_depth, reference_rate, pref_alignment.
Channel weights and normalization constants are FROZEN protocol items.

SCORING_MAP is the frozen alignment between the estimator's own schema
(othermodel.py) and the generator's ground truth (personas.py): the same
scorer is applied to ALL arms, so alignment bias cancels in the empty-R
contrast (plan C3).
"""

from __future__ import annotations

import numpy as np

from .features import cosine, marker_tokens, response_depth, topic_vector

CHANNELS = ("initiative_rate", "response_depth", "reference_rate", "pref_alignment")
CHANNEL_WEIGHTS = {c: 0.25 for c in CHANNELS}  # FROZEN
NORM = {  # FROZEN normalization constants
    "initiative_rate": 1.0,
    "response_depth": 24.0,
    "reference_rate": 1.0,
    "pref_alignment": 1.0,
}

# Frozen scoring map (estimator schema -> generator schema).
SCORING_MAP_NEED = {
    "comfort": "support",
    "answers": "info",
    "fun": "play",
    "listening": "vent",
}
MOOD_DEAD_ZONE = 0.15  # |mood| below this scores as "neu"


REF_CAP = 2.0  # FROZEN: per-response reference channel = min(1, emitted gt markers / REF_CAP)


def channels_of(response: str, initiative_rate: float, gt_markers: set[str], target_mix: np.ndarray) -> dict:
    """The four observables for one probe response.

    reference_rate is count-based (capped emitted-marker count per response),
    not a fraction of ALL ground-truth markers — a fraction denominator that
    grows with the relationship would dilute the channel as history deepens."""
    emitted = marker_tokens(response) & gt_markers
    return {
        "initiative_rate": float(initiative_rate),
        "response_depth": response_depth(response),
        "reference_rate": min(1.0, len(emitted) / REF_CAP),
        "pref_alignment": cosine(topic_vector(response), target_mix),
    }


def divergence(known: dict, stranger: dict) -> dict:
    """Per-channel normalized |delta| + the aggregate D (equal weights)."""
    out = {}
    for c in CHANNELS:
        out[c] = abs(known[c] - stranger[c]) / NORM[c]
    out["D"] = float(sum(CHANNEL_WEIGHTS[c] * out[c] for c in CHANNELS))
    return out


def marker_emission_rate(texts: list[str], gt_markers: set[str]) -> float:
    """Fraction of the constructed ground-truth markers emitted anywhere in texts.

    The metric's content is the EMPTY-R vs POPULATED-R contrast (and the
    generic-personalization comparison), never marker detectability per se —
    detectability is by construction (plan C3 framing)."""
    if not gt_markers:
        return 0.0
    emitted = set()
    for t in texts:
        emitted |= marker_tokens(t) & gt_markers
    return len(emitted) / len(gt_markers)


def quality_score(probe_conditioning: np.ndarray, response: str) -> float:
    """The §8 floor metric: topical relevance to the probe + length adequacy."""
    from .features import STYLE_DIM

    probe_topics = probe_conditioning[STYLE_DIM:]
    relevance = cosine(topic_vector(response), probe_topics)
    adequacy = min(response_depth(response) / 12.0, 1.0)
    return 0.5 * relevance + 0.5 * adequacy


def floor_check(stranger_scores: list[float], base_scores: list[float], margin: float) -> dict:
    """UNPADDED stranger arm vs unmodified base model. Breach = invalid run."""
    delta = abs(float(np.mean(stranger_scores)) - float(np.mean(base_scores)))
    return {"delta": delta, "margin": margin, "pass": delta <= margin}


def score_valence(gt_mood: float) -> str:
    if gt_mood < -MOOD_DEAD_ZONE:
        return "neg"
    if gt_mood > MOOD_DEAD_ZONE:
        return "pos"
    return "neu"


def accuracy(pairs: list[tuple]) -> dict:
    """pairs: (Estimate, GroundTruth) at probe checkpoints. Frozen scoring map."""
    if not pairs:
        return {"valence_acc": 0.0, "need_acc": 0.0, "n": 0}
    v_hits = sum(1 for est, gt in pairs if est.valence_bin == score_valence(gt.mood))
    n_hits = sum(1 for est, gt in pairs if SCORING_MAP_NEED[est.need_label] == gt.need)
    return {
        "valence_acc": v_hits / len(pairs),
        "need_acc": n_hits / len(pairs),
        "n": len(pairs),
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation with average ranks for ties, without scipy."""

    def average_ranks(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            stop = start + 1
            while stop < len(values) and values[order[stop]] == values[order[start]]:
                stop += 1
            ranks[order[start:stop]] = (start + stop - 1) / 2.0
            start = stop
        return ranks

    ra = average_ranks(a)
    rb = average_ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def tost_equivalence(
    a: list[float], b: list[float], margin: float, n_boot: int = 2000, seed: int = 0
) -> dict:
    """Bootstrap TOST-style equivalence of mean(a) vs mean(b) within +/- margin.

    a and b are per-run (cluster-level) means. Each arm is resampled
    independently, so clustering is respected but any cross-arm pairing is not
    used. Equivalence holds when the central 90% interval of the mean
    difference lies entirely inside (-margin, +margin). This is a bootstrap
    analogue of two one-sided tests at alpha = 0.05.
    """
    rng = np.random.default_rng(seed)
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = (
            a_arr[rng.integers(len(a_arr), size=len(a_arr))].mean()
            - b_arr[rng.integers(len(b_arr), size=len(b_arr))].mean()
        )
    lo, hi = float(np.quantile(diffs, 0.05)), float(np.quantile(diffs, 0.95))
    return {
        "mean_diff": float(a_arr.mean() - b_arr.mean()),
        "ci90": (lo, hi),
        "margin": margin,
        "equivalent": bool(-margin < lo and hi < margin),
    }


def marker_naturalism(texts: list[str]) -> dict:
    """Report whether marker tokens remain below the flooding cap."""
    from .features import is_marker, tokens as _tokens

    all_toks = [w for t in texts for w in _tokens(t)]
    if not all_toks:
        return {"marker_density": 0.0, "band": (0.0, 0.08), "pass": False}
    n_markers = sum(1 for w in all_toks if is_marker(w))
    marker_density = n_markers / len(all_toks)
    # Reporting band: at least one marker must be present, without flooding.
    band = (0.0, 0.08)
    return {
        "marker_density": marker_density,
        "band": band,
        "pass": band[0] < marker_density < band[1],
    }
