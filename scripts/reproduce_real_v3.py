"""R1-L: the larger characterization grid (PROTOCOL 'R1-L').

Mechanism, constants, criteria, margins, instrument, decoding, and the frozen
R1 prediction are UNCHANGED from v2. New: 8 fresh personas not used in
calibration x 6 seeds/arm; every decision statistic carries a
run-level bootstrap 90% CI (2000 draws, resampling relationship runs, paired
across arms by seed); full-vs-mismatched final-depth divergence is decomposed
per channel. Writes results/real_v3/.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lovellm.allocation import Budget  # noqa: E402
from lovellm.arms import BASE_UNMODIFIED, CORE_ARMS  # noqa: E402
from lovellm.curve import R1_CONSTS, agreement, predicted_divergence, shape_features  # noqa: E402
from lovellm.metrics import (  # noqa: E402
    CHANNELS,
    accuracy,
    floor_check,
    marker_emission_rate,
    marker_naturalism,
    spearman,
    tost_equivalence,
)
from lovellm.personas import make_pool  # noqa: E402
from lovellm.pipeline import R1_MECH, run_relationship  # noqa: E402
from lovellm.realmodel import PINNED_MODEL, PINNED_REVISION, TEMP, TOP_P, MLXBaseModel  # noqa: E402
from scripts.reproduce import (  # noqa: E402
    C3_ACC_MARGIN,
    REPORTING_SEED,
    DEPTHS,
    FLOOR_MARGIN,
    MASTER_SEED,
    MIN_FINAL_D,
    N_SESSIONS,
    ORDERING_MARGIN,
    TURNS_PER_SESSION,
    band,
    median_curve,
    write_csv,
)
from scripts.reproduce_real import N_GEN_REAL, TOST_MARGIN  # noqa: E402

V3 = ROOT / "results" / "real_v3"
N_USERS_L = 8
N_SEEDS_L = 6
POOL_SEED_L = REPORTING_SEED + 777_000  # fresh personas not used in calibration
N_BOOT = 2000


def boot_ci(stat_fn, n_runs: int, seed: int = MASTER_SEED) -> tuple[float, float]:
    """Central 90% CI of stat_fn(resampled run indices); paired across arms."""
    rng = np.random.default_rng(seed)
    vals = np.array([stat_fn(rng.integers(n_runs, size=n_runs)) for _ in range(N_BOOT)])
    return float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))


def main() -> None:
    V3.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # Stage A: verify the untouched frozen R1 prediction.
    pred = predicted_divergence(DEPTHS, consts=R1_CONSTS)
    p2 = ROOT / "results" / "real_v2" / "model_predictions.csv"
    frozen = ROOT / "results" / "real_v2" / "FROZEN_model_predictions.sha256"
    digest = hashlib.sha256(p2.read_bytes()).hexdigest()
    if digest != frozen.read_text().split()[0]:
        raise RuntimeError("R1 frozen prediction drifted")
    pred_shape = shape_features(pred)
    print(f"Stage A: R1 frozen prediction verified ({digest[:12]}…)")

    n_turns = N_SESSIONS * TURNS_PER_SESSION
    pool = make_pool(N_USERS_L, n_turns, POOL_SEED_L)
    bm = MLXBaseModel()
    budget = Budget()
    all_arms = CORE_ARMS + [BASE_UNMODIFIED]

    runs_by_arm: dict[str, list] = {}
    div_rows, om_rows, ref_rows, ov_rows = [], [], [], []
    for arm in all_arms:
        t_arm = time.time()
        runs = []
        for s in range(N_SEEDS_L):
            focal = pool[s % N_USERS_L]
            distractor = pool[(s + 3) % N_USERS_L]
            run = run_relationship(
                arm, bm, focal, distractor, budget, N_SESSIONS, TURNS_PER_SESSION,
                REPORTING_SEED + 331 * s, n_gen=N_GEN_REAL, mech=R1_MECH,
            )
            runs.append(run)
            for ck in run.checkpoints:
                div_rows.append(
                    {
                        "arm": arm.name,
                        "seed": s,
                        "session": ck.session,
                        "depth": ck.depth,
                        **{f"d_{c}": f"{ck.div[c]:.4f}" for c in ck.div if c != "D"},
                        "D": f"{ck.D:.4f}",
                    }
                )
        runs_by_arm[arm.name] = runs
        for si in range(N_SESSIONS):
            acc = accuracy([r.checkpoints[si].accuracy_pair for r in runs])
            om_rows.append(
                {
                    "arm": arm.name,
                    "depth": int(DEPTHS[si]),
                    "valence_acc": f"{acc['valence_acc']:.4f}",
                    "need_acc": f"{acc['need_acc']:.4f}",
                    "n": acc["n"],
                }
            )
        rates = []
        for r_i, run in enumerate(runs):
            focal = pool[r_i % N_USERS_L]
            gt = {focal.shared_event(s) for s in range(N_SESSIONS)}
            rates.append(marker_emission_rate(run.checkpoints[-1].responses_known, gt))
        ref_rows.append({"arm": arm.name, "marker_emission_rate": band(rates)})
        ov_rows.append(
            {
                "arm": arm.name,
                "layer_us_per_turn": band([r.layer_us_per_turn for r in runs]),
                "generation_us_per_turn": band([r.generation_us_per_turn for r in runs]),
            }
        )
        print(f"  arm {arm.name}: {time.time()-t_arm:.0f}s", flush=True)

    write_csv(V3 / "divergence.csv", div_rows)
    write_csv(V3 / "othermodel.csv", om_rows)
    write_csv(V3 / "references.csv", ref_rows)
    write_csv(V3 / "overhead.csv", ov_rows)

    # Per-run final D and curves.
    finals = {a: np.array([r.checkpoints[-1].D for r in runs_by_arm[a]]) for a in runs_by_arm}
    curves = {
        a: np.array([[ck.D for ck in r.checkpoints] for r in runs_by_arm[a]]) for a in runs_by_arm
    }

    # Decision statistics + bootstrap CIs (paired by seed across arms).
    def ci_final(arm):
        return boot_ci(lambda idx: float(np.median(finals[arm][idx])), N_SEEDS_L)

    def ci_margin(idx):
        return float(np.median(finals["full"][idx]) - np.median(finals["mismatched_r"][idx]))

    def ci_monotone(idx):
        med = np.median(curves["full"][idx], axis=0)
        return shape_features(med).monotone_fraction

    margin_pt = float(np.median(finals["full"]) - np.median(finals["mismatched_r"]))
    margin_ci = boot_ci(ci_margin, N_SEEDS_L)
    mono_pt = shape_features(np.median(curves["full"], axis=0)).monotone_fraction
    mono_ci = boot_ci(ci_monotone, N_SEEDS_L)
    full_final_ci = ci_final("full")

    def acc_contrast(idx):
        pf = [runs_by_arm["full"][i].checkpoints[-1].accuracy_pair for i in idx]
        pe = [runs_by_arm["empty_r"][i].checkpoints[-1].accuracy_pair for i in idx]
        af, ae = accuracy(pf), accuracy(pe)
        return (af["valence_acc"] + af["need_acc"]) / 2 - (ae["valence_acc"] + ae["need_acc"]) / 2

    acc_pt = acc_contrast(np.arange(N_SEEDS_L))
    acc_ci = boot_ci(acc_contrast, N_SEEDS_L)

    stranger_q = [ck.quality_known for r in runs_by_arm["empty_r"] for ck in r.checkpoints]
    base_q = [ck.quality_known for r in runs_by_arm["base_unmodified"] for ck in r.checkpoints]
    floor = floor_check(stranger_q, base_q, FLOOR_MARGIN)
    full_q_runs = [float(np.mean([ck.quality_known for ck in r.checkpoints])) for r in runs_by_arm["full"]]
    base_q_runs = [float(np.mean([ck.quality_known for ck in r.checkpoints])) for r in runs_by_arm["base_unmodified"]]
    tost = tost_equivalence(full_q_runs, base_q_runs, TOST_MARGIN, seed=MASTER_SEED)

    measured = np.median(curves["full"], axis=0)
    rho = spearman(pred, measured)
    meas_shape = shape_features(measured)
    checks = agreement(pred_shape, meas_shape, rho)
    texts = [t for r in runs_by_arm["full"] for ck in r.checkpoints for t in ck.responses_known]
    naturalism = marker_naturalism(texts)
    pad_deltas = [
        abs(ck.quality_stranger_padded - ck.quality_stranger_unpadded)
        for r in runs_by_arm["full"]
        for ck in r.checkpoints
    ]

    # Channel decomposition at final depth (median per run set).
    def chan_median(arm, c):
        return float(np.median([r.checkpoints[-1].div[c] for r in runs_by_arm[arm]]))

    decomp = {
        c: {"full": chan_median("full", c), "mismatched_r": chan_median("mismatched_r", c)}
        for c in CHANNELS
    }

    # Joint-signature verdicts (protocol-defined point statistics).
    om_final = {
        (row["arm"], row["depth"]): (float(row["valence_acc"]) + float(row["need_acc"])) / 2
        for row in om_rows
    }
    empty_acc = om_final[("empty_r", int(DEPTHS[-1]))]
    ref_by_arm = {row["arm"]: float(row["marker_emission_rate"].split()[0]) for row in ref_rows}
    verdicts = {}
    for arm in [a.name for a in CORE_ARMS]:
        curve_arm = np.median(curves[arm], axis=0)
        rho_arm = spearman(pred, curve_arm)
        sf = shape_features(curve_arm)
        crit_curve = rho_arm >= 0.8 and sf.monotone_fraction >= 0.80 and float(curve_arm[-1]) >= MIN_FINAL_D
        crit_acc = (om_final[(arm, int(DEPTHS[-1]))] - empty_acc) >= C3_ACC_MARGIN
        crit_ref = ref_by_arm[arm] > ref_by_arm["empty_r"]
        verdicts[arm] = {
            "curve": bool(crit_curve),
            "accuracy": bool(crit_acc),
            "markers": bool(crit_ref),
            "joint": bool(crit_curve and crit_acc and crit_ref),
            "final_D": float(curve_arm[-1]),
            "rho": float(rho_arm),
            "monotone": float(sf.monotone_fraction),
            "acc_combined": om_final[(arm, int(DEPTHS[-1]))],
        }
    write_csv(
        V3 / "ablations.csv",
        [
            {
                "arm": a,
                **{k: (f"{v[k]:.4f}" if isinstance(v[k], float) else v[k]) for k in v},
            }
            for a, v in verdicts.items()
        ],
    )
    stats_rows = [
        {"stat": "full_final_D", "point": f"{float(np.median(finals['full'])):.4f}", "ci90_lo": f"{full_final_ci[0]:.4f}", "ci90_hi": f"{full_final_ci[1]:.4f}", "bar": MIN_FINAL_D},
        {"stat": "ordering_margin_full_vs_mismatched", "point": f"{margin_pt:.4f}", "ci90_lo": f"{margin_ci[0]:.4f}", "ci90_hi": f"{margin_ci[1]:.4f}", "bar": ORDERING_MARGIN},
        {"stat": "full_monotone_fraction", "point": f"{mono_pt:.4f}", "ci90_lo": f"{mono_ci[0]:.4f}", "ci90_hi": f"{mono_ci[1]:.4f}", "bar": 0.80},
        {"stat": "accuracy_contrast_full_vs_empty", "point": f"{acc_pt:.4f}", "ci90_lo": f"{acc_ci[0]:.4f}", "ci90_hi": f"{acc_ci[1]:.4f}", "bar": C3_ACC_MARGIN},
        {"stat": "tost_quality_diff_full_vs_base", "point": f"{tost['mean_diff']:.4f}", "ci90_lo": f"{tost['ci90'][0]:.4f}", "ci90_hi": f"{tost['ci90'][1]:.4f}", "bar": f"±{TOST_MARGIN}"},
    ]
    write_csv(V3 / "decision_stats.csv", stats_rows)
    write_csv(
        V3 / "channel_decomposition.csv",
        [
            {"channel": c, "full": f"{decomp[c]['full']:.4f}", "mismatched_r": f"{decomp[c]['mismatched_r']:.4f}",
             "person_specific_delta": f"{decomp[c]['full'] - decomp[c]['mismatched_r']:.4f}"}
            for c in CHANNELS
        ],
    )
    write_csv(
        V3 / "capability.csv",
        [
            {
                "comparison": "full_known_vs_base_unmodified_quality",
                "mean_diff": f"{tost['mean_diff']:.4f}",
                "ci90_lo": f"{tost['ci90'][0]:.4f}",
                "ci90_hi": f"{tost['ci90'][1]:.4f}",
                "margin": TOST_MARGIN,
                "equivalent": tost["equivalent"],
            }
        ],
    )

    lines = [
        "# R1-L larger characterization grid (scripts/reproduce_real_v3.py — PROTOCOL 'R1-L')",
        "",
        f"Pinned base model: `{PINNED_MODEL}` · decoding: temp {TEMP}, top_p {TOP_P} · master seed `{MASTER_SEED}` · "
        f"{N_USERS_L} fresh personas (pool seed {POOL_SEED_L}) · {N_SESSIONS} sessions × {TURNS_PER_SESSION} turns · "
        f"{N_SEEDS_L} seeds/arm · {N_GEN_REAL} paired generations · mechanism R1 (unchanged) · frozen R1 prediction verified",
        "",
        "## Decision statistics with run-level bootstrap 90% CIs (2000 draws, paired by seed)",
        "",
        "| statistic | point | 90% CI | bar | interval clears bar |",
        "|---|---|---|---|---|",
    ]
    def clears(lo, hi, bar, kind):
        if kind == "ge":
            return "YES" if lo >= bar else ("point only" if hi >= bar else "NO")
        return "YES" if (lo > -bar and hi < bar) else "NO"
    lines.append(f"| full final D | {np.median(finals['full']):.3f} | [{full_final_ci[0]:.3f}, {full_final_ci[1]:.3f}] | ≥ {MIN_FINAL_D} | {clears(full_final_ci[0], full_final_ci[1], MIN_FINAL_D, 'ge')} |")
    lines.append(f"| ordering margin (full−mismatched) | {margin_pt:.3f} | [{margin_ci[0]:.3f}, {margin_ci[1]:.3f}] | ≥ {ORDERING_MARGIN} | {clears(margin_ci[0], margin_ci[1], ORDERING_MARGIN, 'ge')} |")
    lines.append(f"| full monotone fraction | {mono_pt:.3f} | [{mono_ci[0]:.3f}, {mono_ci[1]:.3f}] | ≥ 0.80 | {clears(mono_ci[0], mono_ci[1], 0.80, 'ge')} |")
    lines.append(f"| accuracy contrast (full−empty) | {acc_pt:.3f} | [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}] | ≥ {C3_ACC_MARGIN} | {clears(acc_ci[0], acc_ci[1], C3_ACC_MARGIN, 'ge')} |")
    lines.append(f"| TOST quality diff (full−base) | {tost['mean_diff']:.4f} | [{tost['ci90'][0]:.4f}, {tost['ci90'][1]:.4f}] | within ±{TOST_MARGIN} | {clears(tost['ci90'][0], tost['ci90'][1], TOST_MARGIN, 'within')} |")
    lines += [
        "",
        "## Curve vs frozen R1 prediction (full arm)",
        "",
        f"- predicted: {[round(float(x),3) for x in pred]}",
        f"- measured median: {[round(float(x),3) for x in measured]}",
        f"- rho {rho:.3f}; monotone {meas_shape.monotone_fraction:.3f}; ratio {meas_shape.half_full_ratio:.3f} "
        f"(pred {pred_shape.half_full_ratio:.3f} ± 0.20); curvature {meas_shape.curvature_class} — "
        f"**shape agreement: {'YES' if checks['overall'] else 'NO'}** (level agreement NOT claimed: measured final "
        f"{float(measured[-1]):.3f} vs predicted {float(pred[-1]):.3f})",
        "",
        "## Channel decomposition at final depth (full vs mismatched-R)",
        "",
        "| channel | full | mismatched-R | person-specific delta |",
        "|---|---|---|---|",
    ]
    for c in CHANNELS:
        lines.append(f"| {c} | {decomp[c]['full']:.3f} | {decomp[c]['mismatched_r']:.3f} | {decomp[c]['full']-decomp[c]['mismatched_r']:+.3f} |")
    lines += [
        "",
        "## Ablations (protocol-defined point statistics)",
        "",
        "| arm | final D | rho | monotone | acc(final) | curve | accuracy | markers | JOINT |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for a, v in verdicts.items():
        lines.append(
            f"| {a} | {v['final_D']:.3f} | {v['rho']:.3f} | {v['monotone']:.2f} | {v['acc_combined']:.2f} | "
            f"{v['curve']} | {v['accuracy']} | {v['markers']} | **{'PASS' if v['joint'] else 'FAIL'}** |"
        )
    lines += [
        "",
        "## Protocol checks",
        "",
        f"- Floor: Δ = {floor['delta']:.4f} vs {FLOOR_MARGIN} → " + ("**PASS**" if floor["pass"] else "**BREACH**"),
        f"- Padding sensitivity: mean {float(np.mean(pad_deltas)):.4f}, max {float(np.max(pad_deltas)):.4f}",
        f"- Marker naturalism: {naturalism['marker_density']:.4f} in {naturalism['band']} → " + ("PASS" if naturalism["pass"] else "FAIL"),
        "",
        f"Total wall time: {time.time()-t_start:.0f}s",
        "",
    ]
    (V3 / "RESULTS.md").write_text("\n".join(lines))
    (V3 / "run_meta.json").write_text(
        json.dumps(
            {
                "grid": "R1-L larger characterization grid (PROTOCOL R1-L)",
                "mechanism_revision": "R1 (unchanged)",
                "base_model": PINNED_MODEL,
                "base_model_revision": PINNED_REVISION,
                "n_users": N_USERS_L,
                "pool_seed": POOL_SEED_L,
                "n_seeds_per_arm": N_SEEDS_L,
                "n_sessions": N_SESSIONS,
                "turns_per_session": TURNS_PER_SESSION,
                "n_gen": N_GEN_REAL,
                "n_boot": N_BOOT,
                "frozen_r1_prediction_sha256": digest,
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            indent=2,
        )
    )
    print(
        "reproduce_real_v3: DONE.",
        "ShapeAgree:", "YES" if checks["overall"] else "NO",
        "| Floor:", "PASS" if floor["pass"] else "BREACH",
        f"| margin {margin_pt:.3f} CI [{margin_ci[0]:.3f},{margin_ci[1]:.3f}]",
        "| TOST:", "EQUIV" if tost["equivalent"] else "NOT-EQUIV",
        f"| {time.time()-t_start:.0f}s",
    )


if __name__ == "__main__":
    main()
