"""Paired right-user-vs-wrong-user accuracy contrast for the R1-XL grid.

The estimator's accuracy depends only on persona signals (deterministic per
(persona, t)) and never on base-model generations, so the per-run paired
contrast is recomputable exactly, LLM-free. Writes
results/real_v4/right_vs_wrong_contrast.csv (summary statistics) and
results/real_v4/right_vs_wrong_runs.csv (the 16 per-run paired values).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lovellm import othermodel as om_mod  # noqa: E402
from lovellm.metrics import SCORING_MAP_NEED, score_valence  # noqa: E402
from lovellm.personas import make_pool  # noqa: E402
from scripts.reproduce import MASTER_SEED, REPORTING_SEED  # noqa: E402

POOL_SEED_XL = REPORTING_SEED + 888_000
N_RUNS = 16


def run_acc(source, focal) -> float:
    om = om_mod.prior()
    for t in range(48):
        om_mod.observe(om, source.signal(t))
    est = om_mod.estimate(om)
    gt = focal.hidden_state(47)
    v = 1.0 if est.valence_bin == score_valence(gt.mood) else 0.0
    n = 1.0 if SCORING_MAP_NEED[est.need_label] == gt.need else 0.0
    return (v + n) / 2


def main() -> None:
    pool = make_pool(20, 48, POOL_SEED_XL)
    full = np.array([run_acc(pool[s], pool[s]) for s in range(N_RUNS)])
    mism = np.array([run_acc(pool[(s + 10) % 20], pool[s]) for s in range(N_RUNS)])
    diff = full - mism
    rng = np.random.default_rng(MASTER_SEED)
    boots = np.array([diff[rng.integers(N_RUNS, size=N_RUNS)].mean() for _ in range(2000)])
    out = ROOT / "results" / "real_v4" / "right_vs_wrong_contrast.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stat", "value"])
        w.writeheader()
        w.writerows(
            [
                {"stat": "paired_mean_diff", "value": f"{diff.mean():.4f}"},
                {"stat": "ci90_lo", "value": f"{np.quantile(boots, 0.05):.4f}"},
                {"stat": "ci90_hi", "value": f"{np.quantile(boots, 0.95):.4f}"},
                {"stat": "ci95_lo", "value": f"{np.quantile(boots, 0.025):.4f}"},
                {"stat": "ci95_hi", "value": f"{np.quantile(boots, 0.975):.4f}"},
                {"stat": "full_mean", "value": f"{full.mean():.4f}"},
                {"stat": "mismatched_mean", "value": f"{mism.mean():.4f}"},
            ]
        )
    runs_out = ROOT / "results" / "real_v4" / "right_vs_wrong_runs.csv"
    with runs_out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["seed", "full_acc", "mismatched_acc", "diff"])
        w.writeheader()
        w.writerows(
            [
                {
                    "seed": s,
                    "full_acc": f"{full[s]:.4f}",
                    "mismatched_acc": f"{mism[s]:.4f}",
                    "diff": f"{diff[s]:.4f}",
                }
                for s in range(N_RUNS)
            ]
        )
    print("written", out)
    print("written", runs_out)


if __name__ == "__main__":
    main()
