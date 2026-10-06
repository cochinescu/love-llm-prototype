"""Post-hoc verification of the COMPLETE curve-fidelity rule (PROTOCOL Amendment A8).

The grid runners' coded `crit_curve` checks Spearman rho >= 0.8, monotone
fraction >= 0.80, and final D >= 0.15. The protocol's curve-fidelity bound
ALSO requires the mid-to-final ratio (D at depth 32 / D at depth 48) within
+-0.20 of the recorded curve's ratio and a matching curvature class; the
runners report those two features for the full arm but do not fold them into
the per-arm joint verdict. This LLM-free script recomputes the complete rule
for every arm on every stored grid from the run-level `divergence.csv` files
and writes results/joint_rule_full_check.csv. As of 2026-10-06 the complete
rule changes NO verdict on any grid.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lovellm.curve import RATIO_TOL, RHO_MIN, shape_features  # noqa: E402
from lovellm.metrics import spearman  # noqa: E402

MIN_FINAL_D = 0.15
ARMS = ["full", "empty_r", "mismatched_r", "caring_only", "particularity_only", "generic_personalization"]
GRIDS = [  # (results subdir, prediction file)
    (".", "model_predictions.csv"),
    ("real", "model_predictions.csv"),
    ("real_v2", "real_v2/model_predictions.csv"),
    ("real_v3", "real_v2/model_predictions.csv"),
    ("real_v4", "real_v2/model_predictions.csv"),
]


def median_curve(rows, arm):
    seeds = sorted({int(r["seed"]) for r in rows if r["arm"] == arm})
    m = np.array(
        [
            [float(r["D"]) for r in sorted((x for x in rows if x["arm"] == arm and int(x["seed"]) == s), key=lambda x: int(x["depth"]))]
            for s in seeds
        ]
    )
    return np.median(m, axis=0)


def main() -> None:
    out_rows = []
    for grid, predf in GRIDS:
        pred = np.array([float(r["D"]) for r in csv.DictReader(open(ROOT / "results" / predf))])
        ps = shape_features(pred)
        rows = list(csv.DictReader(open(ROOT / "results" / grid / "divergence.csv")))
        for arm in ARMS:
            med = median_curve(rows, arm)
            sf = shape_features(med)
            rho = spearman(pred, med)
            coded = bool(rho >= RHO_MIN and sf.monotone_fraction >= 0.80 and med[-1] >= MIN_FINAL_D)
            ratio_ok = bool(abs(sf.half_full_ratio - ps.half_full_ratio) <= RATIO_TOL)
            curv_ok = bool(sf.curvature_class == ps.curvature_class)
            full = bool(coded and ratio_ok and curv_ok)
            out_rows.append(
                {
                    "grid": "synthetic" if grid == "." else grid, "arm": arm,
                    "rho": f"{rho:.4f}", "monotone": f"{sf.monotone_fraction:.3f}", "final_D": f"{med[-1]:.4f}",
                    "ratio": f"{sf.half_full_ratio:.4f}", "ratio_pred": f"{ps.half_full_ratio:.4f}", "ratio_ok": ratio_ok,
                    "curvature": sf.curvature_class, "curvature_pred": ps.curvature_class, "curvature_ok": curv_ok,
                    "coded_curve_criterion": coded, "complete_curve_criterion": full, "verdict_changed": coded != full,
                }
            )
    out = ROOT / "results" / "joint_rule_full_check.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    changed = [r for r in out_rows if r["verdict_changed"]]
    print("written", out, "| verdicts changed by the complete rule:", len(changed))


if __name__ == "__main__":
    main()
