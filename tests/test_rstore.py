import numpy as np

from lovellm import rstore
from lovellm.features import K, text_features
from lovellm.personas import make_pool


def test_trust_bounded_and_monotone():
    r = rstore.empty()
    pool = make_pool(1, 48, 3)
    prev = 0.0
    for t in range(48):
        rstore.update(r, pool[0].signal(t), t // 8, t)
        assert 0.0 <= r.trust <= 1.0
        assert r.trust >= prev
        prev = r.trust
    assert r.depth == 48
    assert r.trust > 0.2
    assert abs(r.pref_est.sum() - 1.0) < 1e-9


def test_retrieval_salience_and_determinism():
    r = rstore.empty()
    rstore.update(r, "mk0x0 kare0 kare0 kare0", 0, 0)
    rstore.update(r, "remito2 remito2 remito2", 0, 1)
    q = text_features("kare0 remito2")
    a = rstore.retrieve(r, q, 1)
    b = rstore.retrieve(r, q, 1)
    assert a == b
    # The marker-bearing item has the salience bonus and outranks at equal recency-ish.
    assert "mk0x0" in a[0].markers or a[0].salience > 1.0


def test_empty_retrieves_nothing():
    r = rstore.empty()
    assert rstore.retrieve(r, np.zeros(K), 5) == []
    assert rstore.retrieve_generic(r, np.zeros(K), 5) == []
