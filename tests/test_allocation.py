import numpy as np

from lovellm.allocation import (
    Budget,
    K_DEPTH,
    Alloc,
    allocate,
    caring_weight,
    decision_table,
    depth_hint_tokens,
    floor_alloc,
)
from lovellm.rstore import RState


def _r(depth: int, trust: float) -> RState:
    r = RState()
    r.depth = depth
    r.trust = trust
    return r


def test_zero_sum_invariant_and_floor():
    """The F1 mechanic: floor unconditional; surplus sums EXACTLY."""
    b = Budget()
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(2, 7))
        weights = list(rng.uniform(0.0, 1.0, size=n))
        if rng.random() < 0.1:
            weights = [0.0] * n  # all-stranger pool
        allocs = allocate(b, weights)
        assert sum(a.initiative for a in allocs) == b.surplus_initiative
        assert sum(a.context - b.floor_context for a in allocs) == b.surplus_context
        assert sum(a.depth_units - b.floor_depth for a in allocs) == b.surplus_depth
        for a in allocs:
            assert a.context >= b.floor_context
            assert a.depth_units >= b.floor_depth


def test_caring_weight_monotone_bounded():
    for trust in (0.1, 0.5, 0.9):
        gs = [caring_weight(_r(d, trust)) for d in range(0, 60, 4)]
        assert all(b >= a for a, b in zip(gs, gs[1:]))
    for depth in (4, 16, 40):
        gs = [caring_weight(_r(depth, t)) for t in np.linspace(0, 1, 8)]
        assert all(b >= a for a, b in zip(gs, gs[1:]))
    assert 0.0 <= caring_weight(_r(1000, 1.0)) <= 1.0


def test_decision_table_consistent():
    for row in decision_table():
        dlo, dhi = (int(x) for x in row["depth_band"].split("-"))
        tlo, thi = (float(x) for x in row["trust_band"].split("-"))
        mid = caring_weight(_r(int((dlo + dhi) / 2), (tlo + thi) / 2))
        assert abs(mid - row["caring_weight"]) < 0.01


def test_floor_alloc_matches_base_envelope():
    b = Budget()
    f = floor_alloc(b)
    assert f == Alloc(initiative=0, context=b.floor_context, depth_units=b.floor_depth)
    assert depth_hint_tokens(f.depth_units) == 4.0 + 2.0 * b.floor_depth
