"""Caring as allocation under constraint: guaranteed floor + contested surplus.

The budget structure is the plan's F1 reconciliation (round-2 review): every
user is unconditionally allocated the floor (resources sufficient for
unmodified-base-model service quality — what makes the §8 stranger-treatment
floor satisfiable BY CONSTRUCTION), and caring operates as genuinely zero-sum
allocation of the SURPLUS ONLY. `allocate` carries the zero-sum invariant as a
tested property: surplus allocations sum exactly to the surplus.

The allocation→observable map (plan §3) is verbatim in OBSERVABLE_MAP: each
allocated resource maps to a named, measurable output channel, and the
divergence metric aggregates exactly those channels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rstore import RState

K_DEPTH = 12.0  # caring-weight depth constant (PROVISIONAL)


@dataclass(frozen=True)
class Budget:
    # Guaranteed per-user floor (unconditional).
    floor_context: int = 4    # memory slots
    floor_depth: int = 2      # depth units (token target = 4 + 2*units)
    floor_response: int = 1   # response slot per cycle
    # Contested surplus (zero-sum across the pool). Granularity matters: slots
    # must be numerous enough that the focal user's integer share can GROW with
    # g(R) rather than pin at the rounding floor.
    surplus_initiative: int = 8
    surplus_context: int = 8
    surplus_depth: int = 12


@dataclass(frozen=True)
class Alloc:
    initiative: int      # surplus initiative slots
    context: int         # floor + surplus memory slots
    depth_units: int     # floor + surplus depth units


OBSERVABLE_MAP = {
    "initiative_slots": "initiative_rate",
    "context_share": "reference_and_alignment",
    "depth_budget": "response_depth",
}


def caring_weight(r: RState) -> float:
    """g(R) = trust * (1 - exp(-depth / K_DEPTH)) — monotone in both, bounded."""
    return float(r.trust * (1.0 - np.exp(-r.depth / K_DEPTH)))


def _largest_remainder(total: int, weights: np.ndarray) -> np.ndarray:
    """Integer split of `total` proportional to weights; sums to total exactly."""
    n = len(weights)
    if total == 0 or n == 0:
        return np.zeros(n, dtype=int)
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones(n)
    shares = total * w / w.sum()
    base = np.floor(shares).astype(int)
    rem = total - int(base.sum())
    if rem > 0:
        frac = shares - base
        order = sorted(range(n), key=lambda i: (-frac[i], i))
        for i in order[:rem]:
            base[i] += 1
    return base


def allocate(budget: Budget, weights: list[float]) -> list[Alloc]:
    """Floor unconditional; surplus zero-sum, proportional to caring weights."""
    w = np.asarray(weights, dtype=float)
    init = _largest_remainder(budget.surplus_initiative, w)
    ctx = _largest_remainder(budget.surplus_context, w)
    dep = _largest_remainder(budget.surplus_depth, w)
    return [
        Alloc(
            initiative=int(init[i]),
            context=budget.floor_context + int(ctx[i]),
            depth_units=budget.floor_depth + int(dep[i]),
        )
        for i in range(len(weights))
    ]


def floor_alloc(budget: Budget) -> Alloc:
    """The floor-only allocation: what a stranger (or a flat-allocation arm)
    receives — identical to the unmodified base model's service envelope."""
    return Alloc(initiative=0, context=budget.floor_context, depth_units=budget.floor_depth)


def depth_hint_tokens(depth_units: int) -> float:
    """depth units -> response token target (the depth_budget observable path)."""
    return 4.0 + 2.0 * depth_units


# Explicit decision table (depth band x trust band -> caring weight at the band
# midpoint). Documentation of g(R)'s behavior; a pytest test recomputes it.
DEPTH_BANDS = [(0, 8), (9, 24), (25, 60)]
TRUST_BANDS = [(0.0, 0.33), (0.33, 0.66), (0.66, 1.0)]


def decision_table() -> list[dict]:
    rows = []
    for dlo, dhi in DEPTH_BANDS:
        for tlo, thi in TRUST_BANDS:
            mid_depth = (dlo + dhi) / 2.0
            mid_trust = (tlo + thi) / 2.0
            g = mid_trust * (1.0 - np.exp(-mid_depth / K_DEPTH))
            rows.append(
                {
                    "depth_band": f"{dlo}-{dhi}",
                    "trust_band": f"{tlo:.2f}-{thi:.2f}",
                    "caring_weight": round(float(g), 4),
                }
            )
    return rows
