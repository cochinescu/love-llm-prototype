import numpy as np

from lovellm.curve import agreement, predicted_divergence, shape_features


DEPTHS = np.array([8, 16, 24, 32, 40, 48])


def test_deterministic():
    a = predicted_divergence(DEPTHS)
    b = predicted_divergence(DEPTHS)
    assert np.array_equal(a, b)  # freezability: same input, same bytes


def test_monotone_saturating():
    pred = predicted_divergence(DEPTHS)
    sf = shape_features(pred)
    assert sf.monotone_fraction == 1.0
    assert sf.curvature_class == "saturating"
    assert 0.0 < sf.half_full_ratio < 1.0
    assert pred[-1] > 0.15  # the signature's minimum final divergence is reachable


def test_agreement_self():
    pred = predicted_divergence(DEPTHS)
    sf = shape_features(pred)
    checks = agreement(sf, sf, rho=1.0)
    assert checks["overall"]


def test_agreement_fails_on_flat():
    pred = predicted_divergence(DEPTHS)
    sf = shape_features(pred)
    flat = shape_features(np.full(len(DEPTHS), 0.2))
    checks = agreement(sf, flat, rho=0.0)
    assert not checks["overall"]
