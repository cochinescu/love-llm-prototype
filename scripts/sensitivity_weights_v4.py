"""Channel-weighting sensitivity of the R1-XL divergence decisions.

The divergence D averages four channels with equal weights. This script
recomputes the R1-XL battery-A decision quantities under each leave-one-
channel-out weighting, applying the same reweighting to the recorded R1
analytical per-channel curves, entirely from shipped run-level data
(results/real_v4/divergence.csv and results/real_v2/model_predictions.csv).
The equal-weight row must reproduce the shipped decision statistics within
the CSVs' 4-decimal rounding step; the script aborts otherwise. Writes
results/real_v4/weight_sensitivity.csv.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lovellm.curve import shape_features  # noqa: E402
from lovellm.metrics import spearman  # noqa: E402

V4 = ROOT / "results" / "real_v4"
CHANNELS = ["initiative_rate", "response_depth", "reference_rate", "pref_alignment"]
N_RUNS = 16
N_SESSIONS = 6
ROUNDING_TOL = 1.01e-4


def load_measured() -> dict[str, np.ndarray]:
    rows = list(csv.DictReader((V4 / "divergence.csv").open()))
    out: dict[str, dict[int, list[list[float]]]] = {}
    for r in rows:
        out.setdefault(r["arm"], {}).setdefault(int(r["seed"]), []).append(
            [float(r[f"d_{c}"]) for c in CHANNELS]
        )
    return {  # arm -> (seed, session, channel)
        a: np.array([by_seed[s] for s in range(N_RUNS)]) for a, by_seed in out.items()
    }


def load_predicted() -> np.ndarray:  # (session, channel)
    rows = list(csv.DictReader((ROOT / "results" / "real_v2" / "model_predictions.csv").open()))
    return np.array([[float(r[c]) for c in CHANNELS] for r in rows])


def main() -> None:
    measured = load_measured()
    predicted = load_predicted()
    weightings = {"equal": np.ones(4) / 4}
    for i, c in enumerate(CHANNELS):
        w = np.ones(4) / 3
        w[i] = 0.0
        weightings[f"drop_{c}"] = w

    out_rows = []
    for name, w in weightings.items():
        curves = {a: (m @ w) for a, m in measured.items()}  # (seed, session)
        med_full = np.median(curves["full"], axis=0)
        final_full = float(med_full[-1])
        margin = final_full - float(np.median(curves["mismatched_r"], axis=0)[-1])
        pred_curve = predicted @ w
        rho = spearman(pred_curve, med_full)
        sf = shape_features(med_full)
        if name == "equal":
            for label, got, shipped in (
                ("final D", final_full, 0.2303),
                ("ordering margin", margin, 0.0228),
                ("monotone", sf.monotone_fraction, 0.8000),
            ):
                if abs(got - shipped) > ROUNDING_TOL:
                    raise SystemExit(f"MISMATCH equal-weight {label}: {got:.4f} != {shipped}")
        out_rows.append(
            {
                "weighting": name,
                "full_final_D": f"{final_full:.4f}",
                "ordering_margin": f"{margin:.4f}",
                "rho_vs_reweighted_prediction": f"{rho:.3f}",
                "monotone_fraction": f"{sf.monotone_fraction:.3f}",
                "curvature": sf.curvature_class,
            }
        )

    out = V4 / "weight_sensitivity.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    for r in out_rows:
        print(r)
    print("equal-weight row reproduces shipped statistics; written", out)


if __name__ == "__main__":
    main()
