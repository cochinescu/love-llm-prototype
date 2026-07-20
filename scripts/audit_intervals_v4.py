"""Recompute the R1-XL near-threshold bootstrap intervals from shipped run-level data.

The battery-A ordering-margin and monotone-fraction intervals are functions of
the per-run divergence values in results/real_v4/divergence.csv and the seeded
bootstrap stream alone (reproduce_real_v4.boot draws a fresh
default_rng(MASTER_SEED) per statistic), so both are recomputable LLM-free.
divergence.csv stores D rounded to four decimals while the runner used
unrounded values, so recomputation can differ from decision_stats.csv by at
most that rounding step. The script reproduces the shipped points, bootstrap
medians, and 90% intervals within 1e-4 and aborts on any larger mismatch;
only then does it write the 95% intervals of the same bootstrap draws to
results/real_v4/decision_stats_ci95.csv.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lovellm.curve import shape_features  # noqa: E402
from scripts.reproduce import MASTER_SEED  # noqa: E402

V4 = ROOT / "results" / "real_v4"
N_RUNS = 16
N_BOOT = 2000
FINAL_DEPTH = 48


def boot_vals(stat_fn, n_runs: int, seed: int = MASTER_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([stat_fn(rng.integers(n_runs, size=n_runs)) for _ in range(N_BOOT)])


def load_curves() -> dict[str, np.ndarray]:
    rows = list(csv.DictReader((V4 / "divergence.csv").open()))
    curves: dict[str, dict[int, list[float]]] = {}
    for r in rows:
        curves.setdefault(r["arm"], {}).setdefault(int(r["seed"]), []).append(float(r["D"]))
    return {a: np.array([by_seed[s] for s in range(N_RUNS)]) for a, by_seed in curves.items()}


ROUNDING_TOL = 1.01e-4  # divergence.csv retains four decimals


def check(name: str, got: float, shipped: str) -> None:
    if abs(got - float(shipped)) > ROUNDING_TOL:
        raise SystemExit(f"MISMATCH {name}: recomputed {got:.4f} != shipped {shipped}")


def main() -> None:
    curves = load_curves()
    finals = {a: c[:, -1] for a, c in curves.items()}
    shipped = {r["stat"]: r for r in csv.DictReader((V4 / "decision_stats.csv").open())}

    margin_vals = boot_vals(
        lambda i: float(np.median(finals["full"][i]) - np.median(finals["mismatched_r"][i])), N_RUNS
    )
    mono_vals = boot_vals(
        lambda i: shape_features(np.median(curves["full"][i], axis=0)).monotone_fraction, N_RUNS
    )

    margin_point = float(np.median(finals["full"]) - np.median(finals["mismatched_r"]))
    mono_point = shape_features(np.median(curves["full"], axis=0)).monotone_fraction
    check("A_ordering_margin point", margin_point, shipped["A_ordering_margin"]["point"])
    check("A_full_monotone point", mono_point, shipped["A_full_monotone"]["point"])
    for name, vals in (("A_ordering_margin", margin_vals), ("A_full_monotone", mono_vals)):
        check(f"{name} boot_median", float(np.median(vals)), shipped[name]["boot_median"])
        check(f"{name} ci90_lo", float(np.quantile(vals, 0.05)), shipped[name]["ci90_lo"])
        check(f"{name} ci90_hi", float(np.quantile(vals, 0.95)), shipped[name]["ci90_hi"])

    out = V4 / "decision_stats_ci95.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stat", "ci90_lo", "ci90_hi", "ci95_lo", "ci95_hi"])
        w.writeheader()
        w.writerows(
            [
                {
                    "stat": name,
                    "ci90_lo": f"{np.quantile(vals, 0.05):.4f}",
                    "ci90_hi": f"{np.quantile(vals, 0.95):.4f}",
                    "ci95_lo": f"{np.quantile(vals, 0.025):.4f}",
                    "ci95_hi": f"{np.quantile(vals, 0.975):.4f}",
                }
                for name, vals in (("A_ordering_margin", margin_vals), ("A_full_monotone", mono_vals))
            ]
        )
    print("shipped decision statistics reproduced within the 4-decimal rounding step; written", out)


if __name__ == "__main__":
    main()
