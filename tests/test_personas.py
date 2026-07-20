import numpy as np

from lovellm.features import marker_tokens, topic_vector, intensifier_rate
from lovellm.personas import NEED_TOPICS, make_pool

N_TURNS = 48


def test_determinism():
    a = make_pool(2, N_TURNS, 42)
    b = make_pool(2, N_TURNS, 42)
    for pa, pb in zip(a, b):
        for t in (0, 7, 23, 47):
            assert pa.signal(t) == pb.signal(t)
            assert pa.hidden_state(t) == pb.hidden_state(t)
    # Repeated calls on the SAME object are also identical (t-determinism).
    assert a[0].signal(5) == a[0].signal(5)


def test_markers_unique_and_detectable():
    pool = make_pool(3, N_TURNS, 7)
    marks = [p.shared_event(s) for p in pool for s in range(6)]
    assert len(set(marks)) == len(marks)
    text = marks[0] + " " + pool[0].signal(0)
    assert marks[0] in marker_tokens(text)
    assert marker_tokens(pool[0].signal(1)) == set()


def test_mood_moves_intensifier_rate():
    pool = make_pool(4, N_TURNS, 11)
    hi, lo = [], []
    for p in pool:
        for t in range(N_TURNS):
            gt = p.hidden_state(t)
            rate = intensifier_rate(p.signal(t))
            if gt.mood > 0.5:
                hi.append(rate)
            elif gt.mood < -0.5:
                lo.append(rate)
    assert len(hi) > 10 and len(lo) > 10
    assert np.mean(hi) > np.mean(lo) + 0.02


def test_need_moves_topics():
    pool = make_pool(4, N_TURNS, 13)
    mass_on_need, mass_off_need = [], []
    for p in pool:
        for t in range(N_TURNS):
            gt = p.hidden_state(t)
            tv = topic_vector(p.signal(t))
            need_mass = sum(tv[topic] for topic in NEED_TOPICS[gt.need])
            mass_on_need.append(need_mass)
            mass_off_need.append(1.0 - need_mass)
    # Two of eight topics carry the need boost; uniform would give 0.25.
    assert np.mean(mass_on_need) > 0.30
