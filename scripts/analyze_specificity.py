"""Paired right-user-vs-wrong-user accuracy contrast for the R1-XL grid.

The estimator's accuracy depends only on persona signals (deterministic per
(persona, t)) and never on base-model generations, so the per-run paired
contrast is recomputable exactly, LLM-free.

AMENDMENT 2026-10-06. The original version of this script fed the estimator
`source.signal(t)` alone and omitted the shared-event marker that the pipeline
prepends on the first turn of every session (see `pipeline.estimator_input`).
That stream differs from the experimental stream, and it gave a paired mean
difference of 0.3438 (full mean 0.5938). The corrected version builds the
stream with `pipeline.estimator_input`, matching the pipeline exactly, and
gives 0.3750 (full mean 0.6250 — which also matches the pipeline's own
final-depth full-arm accuracy in results/real_v4/ablations.csv). Both outputs
are retained:

  results/real_v4/right_vs_wrong_contrast.csv          (ORIGINAL, signal-only stream)
  results/real_v4/right_vs_wrong_runs.csv              (ORIGINAL, signal-only stream)
  results/real_v4/right_vs_wrong_contrast_pipeline_stream.csv  (CORRECTED)
  results/real_v4/right_vs_wrong_runs_pipeline_stream.csv      (CORRECTED)

Run with `--legacy` to regenerate the original files.
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
from lovellm.pipeline import estimator_input  # noqa: E402
from scripts.reproduce import MASTER_SEED, REPORTING_SEED  # noqa: E402

POOL_SEED_XL = REPORTING_SEED + 888_000
N_RUNS = 16
N_SESSIONS = 6
TURNS_PER_SESSION = 8
N_TURNS = N_SESSIONS * TURNS_PER_SESSION  # 48


def run_acc(source, focal, legacy: bool = False) -> float:
    """Final-depth combined accuracy of an estimator that observed `source`'s
    stream, scored against `focal`'s ground truth at the last turn."""
    om = om_mod.prior()
    for t in range(N_TURNS):
        sig = source.signal(t) if legacy else estimator_input(source, t, TURNS_PER_SESSION)
        om_mod.observe(om, sig)
    est = om_mod.estimate(om)
    gt = focal.hidden_state(N_TURNS - 1)
    v = 1.0 if est.valence_bin == score_valence(gt.mood) else 0.0
    n = 1.0 if SCORING_MAP_NEED[est.need_label] == gt.need else 0.0
    return (v + n) / 2


def compute(legacy: bool = False) -> dict:
    pool = make_pool(20, N_TURNS, POOL_SEED_XL)
    full = np.array([run_acc(pool[s], pool[s], legacy) for s in range(N_RUNS)])
    mism = np.array([run_acc(pool[(s + 10) % 20], pool[s], legacy) for s in range(N_RUNS)])
    diff = full - mism
    rng = np.random.default_rng(MASTER_SEED)
    boots = np.array([diff[rng.integers(N_RUNS, size=N_RUNS)].mean() for _ in range(2000)])
    return {"full": full, "mism": mism, "diff": diff, "boots": boots}


def main(legacy: bool = False) -> None:
    res = compute(legacy)
    full, mism, diff, boots = res["full"], res["mism"], res["diff"], res["boots"]
    suffix = "" if legacy else "_pipeline_stream"
    out = ROOT / "results" / "real_v4" / f"right_vs_wrong_contrast{suffix}.csv"
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
            + (
                []  # --legacy reproduces the ORIGINAL file byte-for-byte: no extra rows
                if legacy
                else [
                    {"stat": "n_positive", "value": str(int((diff > 0).sum()))},
                    {"stat": "n_zero", "value": str(int((diff == 0).sum()))},
                    {"stat": "n_negative", "value": str(int((diff < 0).sum()))},
                    {"stat": "estimator_stream", "value": "pipeline.estimator_input"},
                ]
            )
        )
    runs_out = ROOT / "results" / "real_v4" / f"right_vs_wrong_runs{suffix}.csv"
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
    main(legacy="--legacy" in sys.argv[1:])
