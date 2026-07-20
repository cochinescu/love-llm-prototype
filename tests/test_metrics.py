import numpy as np

from lovellm.features import STYLE_DIM, TOPIC_DIM
from lovellm.metrics import (
    CHANNELS,
    accuracy,
    divergence,
    floor_check,
    marker_emission_rate,
    marker_naturalism,
    quality_score,
    score_valence,
    spearman,
)
from lovellm.othermodel import Estimate
from lovellm.personas import GroundTruth


def test_divergence_zero_on_identical():
    ch = {"initiative_rate": 0.3, "response_depth": 12.0, "reference_rate": 0.2, "pref_alignment": 0.5}
    d = divergence(ch, dict(ch))
    assert d["D"] == 0.0
    assert all(d[c] == 0.0 for c in CHANNELS)


def test_divergence_normalization():
    known = {"initiative_rate": 0.5, "response_depth": 24.0, "reference_rate": 0.0, "pref_alignment": 0.0}
    stranger = {"initiative_rate": 0.0, "response_depth": 0.0, "reference_rate": 0.0, "pref_alignment": 0.0}
    d = divergence(known, stranger)
    assert abs(d["response_depth"] - 1.0) < 1e-9  # 24 / NORM 24
    assert abs(d["D"] - 0.25 * (0.5 + 1.0)) < 1e-9


def test_floor_check_negative_control():
    """The check must be able to fail (degraded stranger treatment)."""
    ok = floor_check([0.70, 0.72], [0.71, 0.73], margin=0.05)
    assert ok["pass"]
    bad = floor_check([0.40, 0.42], [0.70, 0.72], margin=0.05)
    assert not bad["pass"]


def test_marker_emission_rate():
    gt = {"mk0x0", "mk0x1"}
    assert marker_emission_rate(["hello mk0x0 world"], gt) == 0.5
    assert marker_emission_rate(["nothing here"], gt) == 0.0
    assert marker_emission_rate([], set()) == 0.0


def test_scoring_map_dead_zone():
    assert score_valence(0.05) == "neu"
    assert score_valence(-0.5) == "neg"
    assert score_valence(0.5) == "pos"
    pairs = [
        (Estimate("pos", "comfort", 1.0), GroundTruth(0.6, "support", 0.5)),
        (Estimate("neg", "answers", 1.0), GroundTruth(0.6, "play", 0.5)),
    ]
    acc = accuracy(pairs)
    assert acc["valence_acc"] == 0.5 and acc["need_acc"] == 0.5


def test_spearman():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(spearman(a, a * 10 + 3) - 1.0) < 1e-9
    assert abs(spearman(a, -a) + 1.0) < 1e-9


def test_spearman_handles_ties_and_constant_vectors():
    assert spearman(np.array([1.0, 2.0, 3.0]), np.zeros(3)) == 0.0
    assert abs(spearman(np.array([1.0, 1.0, 2.0]), np.array([3.0, 3.0, 4.0])) - 1.0) < 1e-9


def test_quality_score_range():
    probe = np.concatenate([np.full(STYLE_DIM, 0.5), np.full(TOPIC_DIM, 1.0 / TOPIC_DIM)])
    q = quality_score(probe, "kare0 remito2 " * 8)
    assert 0.0 <= q <= 1.0


def test_marker_naturalism_bands():
    natural = marker_naturalism(["kare0 " * 20 + "mk0x0"])
    assert natural["pass"]
    flooded = marker_naturalism(["mk0x0 mk0x1 mk0x2 kare0"])
    assert not flooded["pass"]
    assert not marker_naturalism(["kare0 remito2"])["pass"]
