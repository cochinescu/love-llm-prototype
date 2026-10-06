"""Manuscript Figure 1, re-rendered from the stored R1-grid run-level data.

Reads results/real_v2/divergence.csv and results/real_v2/model_predictions.csv
(the four-run R1 grid, scripts/reproduce_real_v2.py) and writes
results/real_v2/fig1_divergence_curve_r1_manuscript.png. Lines are per-depth
medians across the four runs; bands are the 2.5-97.5% empirical quantiles of
the four per-run values (descriptive spread, not confidence intervals). The
medians are identical to those in the runner-generated
fig1_divergence_curve_r1.png (2026-07-17); only labels and layout differ.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results" / "real_v2"


def main() -> None:
    pred = {int(r["depth"]): float(r["D"]) for r in csv.DictReader(open(V2 / "model_predictions.csv"))}
    runs = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(V2 / "divergence.csv")):
        runs[r["arm"]][int(r["seed"])][int(r["depth"])] = float(r["D"])
    depths = sorted(pred)

    def band(arm):
        m = np.array([[runs[arm][s][d] for d in depths] for s in sorted(runs[arm])])
        return np.median(m, 0), np.quantile(m, 0.025, 0), np.quantile(m, 0.975, 0), m.shape[0]

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.plot(depths, [pred[d] for d in depths], "k--", lw=1.6, label="recorded analytical curve (R1 constants)")
    spec = {
        "full": ("#1f5fa8", "known user (full mechanism)", 2.2),
        "mismatched_r": ("#d9792b", "mismatched-$R$ (another user's history)", 1.6),
        "empty_r": ("#6a9f58", "empty-$R$ (stranger baseline)", 1.6),
    }
    n_runs = None
    for arm, (c, lab, lw) in spec.items():
        med, lo, hi, n_runs = band(arm)
        ax.plot(depths, med, color=c, lw=lw, marker="o", ms=3.5, label=lab)
        ax.fill_between(depths, lo, hi, color=c, alpha=0.15, lw=0)
    ax.set_xlabel("relationship depth (turns)")
    ax.set_ylabel("divergence $D$")
    ax.set_title(f"Measured divergence tracks the analytical shape, not its level (R1 grid, n = {n_runs} runs)", loc="left")
    ax.margins(0.04)
    ax.set_ylim(-0.015, 0.44)
    ax.legend(frameon=False, loc="upper left")
    ax.text(depths[-1], 0.004, "exactly 0 by construction", color="#6a9f58", ha="right", va="bottom", fontsize=7)
    fig.tight_layout()
    out = V2 / "fig1_divergence_curve_r1_manuscript.png"
    fig.savefig(out, dpi=200)
    print("written", out)


if __name__ == "__main__":
    main()
