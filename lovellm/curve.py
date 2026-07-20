"""The LLM-free predicted divergence-vs-R curve (Stage A, frozen).

C2's discipline (plan + PROTOCOL): the predicted curve is an ANALYTIC
composition of the formal model's constants — trust update, caring weight,
allocation shares, salience accumulation, estimator convergence. It is never
fit to measured runs; reproduce.py writes it FIRST and sha256-freezes it, so
"predicted vs. measured" is a test, not goodness-of-fit.

Derivation of each term (constants imported from their owning modules):
- trust_hat(d)  = min(1, ETA_TRUST * E_PROXY * d): expected trust after d turns
  at the mean engagement proxy E_PROXY.
- g_hat(d)      = trust_hat * (1 - exp(-d / K_DEPTH)): the caring weight.
- share(d)      = g_hat / (g_hat + SHARE_SMOOTHING): the focal user's expected
  surplus share against the reference pool (SHARE_SMOOTHING = the pool's total
  caring weight, (N_POOL-1) * pipeline.G_REF).
- initiative(d) = share * surplus_initiative / turns_per_session (rate delta;
  the stranger floor has 0 initiative).
- depth(d)      = share * surplus_depth * TOKENS_PER_UNIT / NORM_depth (token
  delta over the stranger floor).
- reference(d)  = P_REF_MAX * (1 - exp(-d / K_REF)): the capped per-response
  emitted-marker channel saturates as markers accumulate in the retrievable
  context (P_REF_MAX composes the per-marker emission probability with the
  context share's marker capacity and the REF_CAP normalization).
- alignment(d)  = A_ALIGN * (1 - exp(-d * EWMA_PREF)): the conditioning's
  pref/need components converge at the preference-EWMA rate; A_ALIGN is the
  mix weight available to them (MIX_PREF + MIX_NEED) times an expected cosine
  gain of ~0.7.

D(d) = sum of channel weights x channel terms (weights FROZEN in metrics.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .allocation import Budget, K_DEPTH, depth_hint_tokens
from .metrics import CHANNEL_WEIGHTS, NORM
from .rstore import ETA_TRUST, EWMA_PREF

E_PROXY = 0.5          # expected engagement proxy (documented modeling constant)
SHARE_SMOOTHING = 0.75  # = (N_POOL-1) * G_REF: the reference pool's total caring weight
K_REF = 16.0            # marker-availability saturation constant (turns)
TURNS_PER_SESSION = 8


@dataclass(frozen=True)
class CurveConstants:
    """Mechanism-version-specific analytic terms.

    V1: original mechanism — a_align = (mix_pref+mix_need)·0.7 = 0.5·0.7;
    p_ref_max from uncapped marker emission. R1 ("probe-guarded partiality",
    PROTOCOL W2 revision R1): guarded mix 0.75/0.15/0.10 → a_align =
    0.25·0.7 = 0.175; marker cap 1 → p_ref_max ≈ 1·p_emit / REF_CAP = 0.35.
    """

    a_align: float
    p_ref_max: float


V1_CONSTS = CurveConstants(a_align=0.35, p_ref_max=0.6)
R1_CONSTS = CurveConstants(a_align=0.175, p_ref_max=0.35)


def predicted_channels(
    depths: np.ndarray, budget: Budget | None = None, consts: CurveConstants = V1_CONSTS
) -> dict[str, np.ndarray]:
    b = budget or Budget()
    d = np.asarray(depths, dtype=float)
    trust_hat = np.minimum(1.0, ETA_TRUST * E_PROXY * d)
    g_hat = trust_hat * (1.0 - np.exp(-d / K_DEPTH))
    share = g_hat / (g_hat + SHARE_SMOOTHING)
    initiative = share * b.surplus_initiative / TURNS_PER_SESSION / NORM["initiative_rate"]
    depth_delta = (
        share * (depth_hint_tokens(b.floor_depth + b.surplus_depth) - depth_hint_tokens(b.floor_depth))
    ) / NORM["response_depth"]
    reference = consts.p_ref_max * (1.0 - np.exp(-d / K_REF)) / NORM["reference_rate"]
    alignment = consts.a_align * (1.0 - np.exp(-d * EWMA_PREF)) / NORM["pref_alignment"]
    return {
        "initiative_rate": initiative,
        "response_depth": depth_delta,
        "reference_rate": reference,
        "pref_alignment": alignment,
    }


def predicted_divergence(
    depths: np.ndarray, budget: Budget | None = None, consts: CurveConstants = V1_CONSTS
) -> np.ndarray:
    ch = predicted_channels(depths, budget, consts)
    return sum(CHANNEL_WEIGHTS[c] * ch[c] for c in ch)


@dataclass(frozen=True)
class ShapeFeatures:
    monotone_fraction: float
    half_full_ratio: float
    curvature_class: str

    def to_dict(self) -> dict:
        return {
            "monotone_fraction": self.monotone_fraction,
            "half_full_ratio": self.half_full_ratio,
            "curvature_class": self.curvature_class,
        }


# Curvature classifier boundaries (PROVISIONAL). With few grid points and
# integer allocation steps the classifier must not be noise-fragile: a process
# is "saturating" when the second-half slope falls below SATURATING_MAX x the
# first-half slope — a genuinely linear/flat process sits at ratio ~1.0 and
# still fails.
SATURATING_MAX = 0.75
ACCELERATING_MIN = 1.4


def shape_features(curve: np.ndarray) -> ShapeFeatures:
    c = np.asarray(curve, dtype=float)
    rng_span = float(c.max() - c.min())
    tol = 0.02 * rng_span  # plateau wobble is not non-monotonicity
    diffs = np.diff(c)
    monotone = float(np.mean(diffs >= -tol)) if len(diffs) else 1.0
    mid = len(c) // 2
    ratio = float(c[mid] / c[-1]) if c[-1] > 0 else 0.0
    first = float(np.mean(diffs[:mid])) if mid > 0 else 0.0
    second = float(np.mean(diffs[mid:])) if mid < len(diffs) else 0.0
    if first > 0 and second < SATURATING_MAX * first:
        curvature = "saturating"
    elif first > 0 and second > ACCELERATING_MIN * first:
        curvature = "accelerating"
    else:
        curvature = "linear"
    return ShapeFeatures(monotone, ratio, curvature)


# PROVISIONAL agreement thresholds (PROTOCOL "shape thresholds").
MONOTONE_MIN = 0.80
RATIO_TOL = 0.20
RHO_MIN = 0.8


def agreement(predicted: ShapeFeatures, measured: ShapeFeatures, rho: float) -> dict:
    checks = {
        "monotone": measured.monotone_fraction >= MONOTONE_MIN,
        "half_full_ratio": abs(measured.half_full_ratio - predicted.half_full_ratio) <= RATIO_TOL,
        "curvature_class": measured.curvature_class == predicted.curvature_class,
        "spearman_rho": rho >= RHO_MIN,
    }
    checks["overall"] = all(checks.values())
    return checks
