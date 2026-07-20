"""Integration: the arm grid end-to-end on a small seeded configuration.

Synthetic-validation directions asserted here (the plan's C2–C5 mechanics):
ordering known > mismatched > stranger; accuracy full > empty_r; markers
populated > empty; the §8 floor holds between the unpadded stranger arm and
the unmodified base model; parity and probe constancy are structural.
"""

import numpy as np
import pytest

from lovellm.allocation import Budget
from lovellm.arms import (
    BASE_UNMODIFIED,
    CARING_ONLY,
    EMPTY_R,
    FULL,
    GENERIC_PERSONALIZATION,
    MISMATCHED_R,
)
from lovellm.basemodel import SyntheticBaseModel
from lovellm.metrics import accuracy, floor_check, marker_emission_rate
from lovellm.personas import make_pool
from lovellm.pipeline import PROBES, N_PROBES, run_relationship

N_SESSIONS = 4
TURNS = 8
N_SEEDS = 4
SEED = 909


@pytest.fixture(scope="module")
def grid():
    bm = SyntheticBaseModel()
    budget = Budget()
    pool = make_pool(4, N_SESSIONS * TURNS, SEED)
    runs = {}
    for arm in (FULL, EMPTY_R, MISMATCHED_R, CARING_ONLY, GENERIC_PERSONALIZATION, BASE_UNMODIFIED):
        runs[arm.name] = [
            run_relationship(
                arm, bm, pool[s % 4], pool[(s + 1) % 4], budget, N_SESSIONS, TURNS, SEED + 331 * s
            )
            for s in range(N_SEEDS)
        ]
    return pool, runs


def _final_d(runs):
    return float(np.median([r.checkpoints[-1].D for r in runs]))


def test_probe_battery_frozen():
    assert len(PROBES) == N_PROBES
    assert all(p.shape == PROBES[0].shape for p in PROBES)


def test_ordering_known_mismatched_stranger(grid):
    _, runs = grid
    d_full = _final_d(runs["full"])
    d_mis = _final_d(runs["mismatched_r"])
    d_str = _final_d(runs["empty_r"])
    assert d_full > d_mis + 0.02
    assert d_mis > d_str + 0.02


def test_empty_r_divergence_is_null(grid):
    """Paired seeds + identical floor conditioning: the stranger baseline's
    self-divergence is exactly zero (the instrument's null)."""
    _, runs = grid
    assert _final_d(runs["empty_r"]) == 0.0


def test_accuracy_full_beats_empty(grid):
    _, runs = grid
    def acc(name):
        pairs = [r.checkpoints[-1].accuracy_pair for r in runs[name]]
        a = accuracy(pairs)
        return (a["valence_acc"] + a["need_acc"]) / 2
    assert acc("full") > acc("empty_r")
    assert acc("caring_only") == acc("empty_r")  # prior-only estimator in both


def test_marker_contrast(grid):
    pool, runs = grid
    def emission(name):
        vals = []
        for i, r in enumerate(runs[name]):
            gt = {pool[i % 4].shared_event(s) for s in range(N_SESSIONS)}
            vals.append(marker_emission_rate(r.checkpoints[-1].responses_known, gt))
        return float(np.mean(vals))
    assert emission("full") > 0.0
    assert emission("empty_r") == 0.0
    assert emission("mismatched_r") == 0.0  # wrong user's markers never match focal ground truth


def test_floor_holds_and_padding_structural(grid):
    _, runs = grid
    stranger_q = [ck.quality_known for r in runs["empty_r"] for ck in r.checkpoints]
    base_q = [ck.quality_known for r in runs["base_unmodified"] for ck in r.checkpoints]
    verdict = floor_check(stranger_q, base_q, margin=0.05)
    assert verdict["pass"], f"floor breached: {verdict}"
    # Padded vs unpadded stranger quality is identical on the synthetic
    # stand-in (padding is structural here; consumed by a real model at W2).
    for r in runs["full"]:
        for ck in r.checkpoints:
            assert abs(ck.quality_stranger_padded - ck.quality_stranger_unpadded) < 1e-12


def test_divergence_grows_with_depth(grid):
    _, runs = grid
    med = np.median(np.array([[ck.D for ck in r.checkpoints] for r in runs["full"]]), axis=0)
    assert med[-1] > med[0]
