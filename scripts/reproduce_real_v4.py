"""R1-XL: expanded grid + second probe battery (PROTOCOL 'R1-XL').

16 runs per arm (distinct fresh focal personas from a pool of 20 not used in
calibration, one seed each) on battery A for every protocol-defined criterion, plus battery B
(probe seed 778, odd-topic loci, never used in any calibration) on the
equivalence/floor/ordering arms — the probe-leakage test. Mechanism (R1),
criteria, margins, instrument, decoding, and the frozen R1 prediction are all
UNCHANGED. Decisions stay point-statistic-based; bootstrap intervals are
DESCRIPTIVE (percentile + bootstrap median reported). Writes results/real_v4/.
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
from lovellm.arms import BASE_UNMODIFIED, CORE_ARMS, EMPTY_R, FULL, MISMATCHED_R  # noqa: E402
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
from lovellm.pipeline import PROBES_B, R1_MECH, run_relationship  # noqa: E402
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
    write_csv,
)
from scripts.reproduce_real import N_GEN_REAL, TOST_MARGIN  # noqa: E402

V4 = ROOT / "results" / "real_v4"
N_RUNS = 16
POOL_N = 20
POOL_SEED_XL = REPORTING_SEED + 888_000
N_BOOT = 2000
B_ARMS = [FULL, EMPTY_R, MISMATCHED_R, BASE_UNMODIFIED]


def boot(stat_fn, n_runs: int, seed: int = MASTER_SEED):
    rng = np.random.default_rng(seed)
    vals = np.array([stat_fn(rng.integers(n_runs, size=n_runs)) for _ in range(N_BOOT)])
    return (
        float(np.quantile(vals, 0.05)),
        float(np.quantile(vals, 0.95)),
        float(np.median(vals)),
    )


def run_grid(bm, pool, budget, arms, probes, tag):
    runs_by_arm = {}
    for arm in arms:
        t_arm = time.time()
        runs = []
        for s in range(N_RUNS):
            focal = pool[s]
            distractor = pool[(s + 10) % POOL_N]
            runs.append(
                run_relationship(
                    arm, bm, focal, distractor, budget, N_SESSIONS, TURNS_PER_SESSION,
                    REPORTING_SEED + 331 * s, n_gen=N_GEN_REAL, mech=R1_MECH, probes=probes,
                )
            )
        runs_by_arm[arm.name] = runs
        print(f"  [{tag}] arm {arm.name}: {time.time()-t_arm:.0f}s", flush=True)
    return runs_by_arm


def main() -> None:
    V4.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    pred = predicted_divergence(DEPTHS, consts=R1_CONSTS)
    p2 = ROOT / "results" / "real_v2" / "model_predictions.csv"
    digest = hashlib.sha256(p2.read_bytes()).hexdigest()
    recorded = (ROOT / "results" / "real_v2" / "FROZEN_model_predictions.sha256").read_text().split()[0]
    if digest != recorded:
        raise RuntimeError("R1 frozen prediction drifted")
    pred_shape = shape_features(pred)
    print(f"Stage A: R1 frozen prediction verified ({digest[:12]}…)")

    n_turns = N_SESSIONS * TURNS_PER_SESSION
    pool = make_pool(POOL_N, n_turns, POOL_SEED_XL)
    bm = MLXBaseModel()
    budget = Budget()
    all_arms = CORE_ARMS + [BASE_UNMODIFIED]

    runs_a = run_grid(bm, pool, budget, all_arms, None, "batteryA")
    runs_b = run_grid(bm, pool, budget, B_ARMS, PROBES_B, "batteryB")

    # ---- battery A: full criteria set ----
    div_rows, om_rows, ref_rows = [], [], []
    for arm_name, runs in runs_a.items():
        for s, run in enumerate(runs):
            for ck in run.checkpoints:
                div_rows.append(
                    {
                        "arm": arm_name, "seed": s, "session": ck.session, "depth": ck.depth,
                        **{f"d_{c}": f"{ck.div[c]:.4f}" for c in ck.div if c != "D"},
                        "D": f"{ck.D:.4f}",
                    }
                )
        for si in range(N_SESSIONS):
            acc = accuracy([r.checkpoints[si].accuracy_pair for r in runs])
            om_rows.append(
                {
                    "arm": arm_name, "depth": int(DEPTHS[si]),
                    "valence_acc": f"{acc['valence_acc']:.4f}",
                    "need_acc": f"{acc['need_acc']:.4f}", "n": acc["n"],
                }
            )
        rates = []
        for s, run in enumerate(runs):
            gt = {pool[s].shared_event(x) for x in range(N_SESSIONS)}
            rates.append(marker_emission_rate(run.checkpoints[-1].responses_known, gt))
        ref_rows.append({"arm": arm_name, "marker_emission_rate": band(rates)})
    write_csv(V4 / "divergence.csv", div_rows)
    write_csv(V4 / "othermodel.csv", om_rows)
    write_csv(V4 / "references.csv", ref_rows)

    finals = {a: np.array([r.checkpoints[-1].D for r in runs_a[a]]) for a in runs_a}
    curves = {a: np.array([[ck.D for ck in r.checkpoints] for r in runs_a[a]]) for a in runs_a}
    measured = np.median(curves["full"], axis=0)
    rho = spearman(pred, measured)
    meas_shape = shape_features(measured)
    checks = agreement(pred_shape, meas_shape, rho)

    margin_runs = finals["full"] - finals["mismatched_r"]  # paired by seed
    m_lo, m_hi, m_med = boot(lambda i: float(np.median(finals["full"][i]) - np.median(finals["mismatched_r"][i])), N_RUNS)
    f_lo, f_hi, f_med = boot(lambda i: float(np.median(finals["full"][i])), N_RUNS)
    mono_lo, mono_hi, mono_med = boot(lambda i: shape_features(np.median(curves["full"][i], axis=0)).monotone_fraction, N_RUNS)

    def acc_contrast(idx):
        af = accuracy([runs_a["full"][i].checkpoints[-1].accuracy_pair for i in idx])
        ae = accuracy([runs_a["empty_r"][i].checkpoints[-1].accuracy_pair for i in idx])
        return (af["valence_acc"] + af["need_acc"]) / 2 - (ae["valence_acc"] + ae["need_acc"]) / 2

    acc_pt = acc_contrast(np.arange(N_RUNS))
    a_lo, a_hi, a_med = boot(acc_contrast, N_RUNS)

    def q_runs(runs):
        return [float(np.mean([ck.quality_known for ck in r.checkpoints])) for r in runs]

    tost_a = tost_equivalence(q_runs(runs_a["full"]), q_runs(runs_a["base_unmodified"]), TOST_MARGIN, seed=MASTER_SEED)
    floor_a = floor_check(
        [ck.quality_known for r in runs_a["empty_r"] for ck in r.checkpoints],
        [ck.quality_known for r in runs_a["base_unmodified"] for ck in r.checkpoints],
        FLOOR_MARGIN,
    )
    texts = [t for r in runs_a["full"] for ck in r.checkpoints for t in ck.responses_known]
    naturalism = marker_naturalism(texts)
    pad = [abs(ck.quality_stranger_padded - ck.quality_stranger_unpadded) for r in runs_a["full"] for ck in r.checkpoints]

    om_final = {(r["arm"], r["depth"]): (float(r["valence_acc"]) + float(r["need_acc"])) / 2 for r in om_rows}
    om_comp = {(r["arm"], r["depth"]): (float(r["valence_acc"]), float(r["need_acc"])) for r in om_rows}
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
            "curve": bool(crit_curve), "accuracy": bool(crit_acc), "markers": bool(crit_ref),
            "joint": bool(crit_curve and crit_acc and crit_ref),
            "final_D": float(curve_arm[-1]), "rho": float(rho_arm),
            "monotone": float(sf.monotone_fraction), "acc_combined": om_final[(arm, int(DEPTHS[-1]))],
        }
    write_csv(
        V4 / "ablations.csv",
        [{"arm": a, **{k: (f"{v[k]:.4f}" if isinstance(v[k], float) else v[k]) for k in v}} for a, v in verdicts.items()],
    )

    # ---- battery B: leakage test on equivalence / floor / ordering ----
    finals_b = {a: np.array([r.checkpoints[-1].D for r in runs_b[a]]) for a in runs_b}
    tost_b = tost_equivalence(q_runs(runs_b["full"]), q_runs(runs_b["base_unmodified"]), TOST_MARGIN, seed=MASTER_SEED)
    floor_b = floor_check(
        [ck.quality_known for r in runs_b["empty_r"] for ck in r.checkpoints],
        [ck.quality_known for r in runs_b["base_unmodified"] for ck in r.checkpoints],
        FLOOR_MARGIN,
    )
    ord_b = {
        "full": float(np.median(finals_b["full"])),
        "mismatched_r": float(np.median(finals_b["mismatched_r"])),
        "empty_r": float(np.median(finals_b["empty_r"])),
    }
    mb_lo, mb_hi, mb_med = boot(lambda i: float(np.median(finals_b["full"][i]) - np.median(finals_b["mismatched_r"][i])), N_RUNS)

    stats_rows = [
        {"stat": "A_full_final_D", "point": f"{float(np.median(finals['full'])):.4f}", "boot_median": f"{f_med:.4f}", "ci90_lo": f"{f_lo:.4f}", "ci90_hi": f"{f_hi:.4f}", "bar": MIN_FINAL_D},
        {"stat": "A_ordering_margin", "point": f"{float(np.median(finals['full']) - np.median(finals['mismatched_r'])):.4f}", "boot_median": f"{m_med:.4f}", "ci90_lo": f"{m_lo:.4f}", "ci90_hi": f"{m_hi:.4f}", "bar": ORDERING_MARGIN},
        {"stat": "A_full_monotone", "point": f"{shape_features(measured).monotone_fraction:.4f}", "boot_median": f"{mono_med:.4f}", "ci90_lo": f"{mono_lo:.4f}", "ci90_hi": f"{mono_hi:.4f}", "bar": 0.80},
        {"stat": "A_accuracy_contrast", "point": f"{acc_pt:.4f}", "boot_median": f"{a_med:.4f}", "ci90_lo": f"{a_lo:.4f}", "ci90_hi": f"{a_hi:.4f}", "bar": C3_ACC_MARGIN},
        {"stat": "A_tost_quality_diff", "point": f"{tost_a['mean_diff']:.4f}", "boot_median": "-", "ci90_lo": f"{tost_a['ci90'][0]:.4f}", "ci90_hi": f"{tost_a['ci90'][1]:.4f}", "bar": f"±{TOST_MARGIN}"},
        {"stat": "B_tost_quality_diff_heldout_probes", "point": f"{tost_b['mean_diff']:.4f}", "boot_median": "-", "ci90_lo": f"{tost_b['ci90'][0]:.4f}", "ci90_hi": f"{tost_b['ci90'][1]:.4f}", "bar": f"±{TOST_MARGIN}"},
        {"stat": "B_ordering_margin_heldout_probes", "point": f"{ord_b['full'] - ord_b['mismatched_r']:.4f}", "boot_median": f"{mb_med:.4f}", "ci90_lo": f"{mb_lo:.4f}", "ci90_hi": f"{mb_hi:.4f}", "bar": ORDERING_MARGIN},
        {"stat": "B_floor_delta_heldout_probes", "point": f"{floor_b['delta']:.4f}", "boot_median": "-", "ci90_lo": "-", "ci90_hi": "-", "bar": FLOOR_MARGIN},
    ]
    write_csv(V4 / "decision_stats.csv", stats_rows)
    write_csv(
        V4 / "run_level_margins.csv",
        [{"seed": s, "battA_margin": f"{margin_runs[s]:.4f}", "battB_margin": f"{(finals_b['full'] - finals_b['mismatched_r'])[s]:.4f}"} for s in range(N_RUNS)],
    )

    vA = om_comp[("full", int(DEPTHS[-1]))]
    vE = om_comp[("empty_r", int(DEPTHS[-1]))]
    decomp = {
        c: float(np.median([r.checkpoints[-1].div[c] for r in runs_a["full"]]))
        - float(np.median([r.checkpoints[-1].div[c] for r in runs_a["mismatched_r"]]))
        for c in CHANNELS
    }

    lines = [
        "# R1-XL expanded grid + second probe battery (scripts/reproduce_real_v4.py — PROTOCOL 'R1-XL')",
        "",
        f"Pinned base model: `{PINNED_MODEL}` · decoding: temp {TEMP}, top_p {TOP_P} · master seed `{MASTER_SEED}` · "
        f"{N_RUNS} runs/arm, each a distinct fresh focal persona (pool {POOL_N}, seed {POOL_SEED_XL}) · "
        f"{N_SESSIONS} sessions × {TURNS_PER_SESSION} turns · {N_GEN_REAL} paired generations · mechanism R1 (unchanged) · "
        f"frozen R1 prediction verified ({digest[:12]}…)",
        "",
        "## Decision statistics (battery A) — decided on points; intervals DESCRIPTIVE (percentile, 2000 draws, 16 clusters)",
        "",
        "| statistic | point | boot median | 90% interval | bar |",
        "|---|---|---|---|---|",
    ]
    for r in stats_rows:
        lines.append(f"| {r['stat']} | {r['point']} | {r['boot_median']} | [{r['ci90_lo']}, {r['ci90_hi']}] | {r['bar']} |")
    lines += [
        "",
        "## Curve vs frozen R1 prediction (battery A, full arm)",
        "",
        f"- predicted: {[round(float(x),3) for x in pred]}",
        f"- measured median: {[round(float(x),3) for x in measured]}",
        f"- rho {rho:.3f}; monotone {meas_shape.monotone_fraction:.3f}; ratio {meas_shape.half_full_ratio:.3f} "
        f"(pred {pred_shape.half_full_ratio:.3f} ± 0.20); curvature {meas_shape.curvature_class} — "
        f"shape agreement: {'YES' if checks['overall'] else 'NO'} (level not claimed: {float(measured[-1]):.3f} vs {float(pred[-1]):.3f})",
        "",
        "## Ablations (battery A, protocol-defined point statistics)",
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
        f"Per-component accuracy at final depth: full valence {vA[0]:.2f} / need {vA[1]:.2f}; empty_r valence {vE[0]:.2f} / need {vE[1]:.2f}.",
        f"Channel decomposition (full − mismatched, final depth): " + ", ".join(f"{c} {decomp[c]:+.3f}" for c in CHANNELS) + ".",
        "",
        "## Battery B — probes not used in R1 calibration",
        "",
        f"- TOST (full − base): {tost_b['mean_diff']:+.4f}, 90% CI [{tost_b['ci90'][0]:.4f}, {tost_b['ci90'][1]:.4f}] vs ±{TOST_MARGIN} → "
        + ("**EQUIVALENT**" if tost_b["equivalent"] else "**NOT SHOWN EQUIVALENT**"),
        f"- Floor: Δ = {floor_b['delta']:.4f} vs {FLOOR_MARGIN} → " + ("**PASS**" if floor_b["pass"] else "**BREACH**"),
        f"- Ordering: full {ord_b['full']:.3f} > mismatched {ord_b['mismatched_r']:.3f} > empty {ord_b['empty_r']:.3f} "
        f"(margin {ord_b['full']-ord_b['mismatched_r']:.3f}, interval [{mb_lo:.3f}, {mb_hi:.3f}])",
        "",
        "## Protocol checks (battery A)",
        "",
        f"- Floor: Δ = {floor_a['delta']:.4f} vs {FLOOR_MARGIN} → " + ("**PASS**" if floor_a["pass"] else "**BREACH**"),
        f"- Padding sensitivity: mean {float(np.mean(pad)):.4f}, max {float(np.max(pad)):.4f}",
        f"- Marker naturalism: {naturalism['marker_density']:.4f} in {naturalism['band']} → " + ("PASS" if naturalism["pass"] else "FAIL"),
        "",
        f"Total wall time: {time.time()-t_start:.0f}s",
        "",
    ]
    (V4 / "RESULTS.md").write_text("\n".join(lines))
    (V4 / "run_meta.json").write_text(
        json.dumps(
            {
                "grid": "R1-XL expanded + battery B not used in calibration (PROTOCOL R1-XL)",
                "mechanism_revision": "R1 (unchanged)",
                "base_model": PINNED_MODEL,
                "base_model_revision": PINNED_REVISION,
                "n_runs_per_arm": N_RUNS,
                "pool_n": POOL_N,
                "pool_seed": POOL_SEED_XL,
                "battery_b": {"probe_seed": 778, "topic_offset": 1, "arms": [a.name for a in B_ARMS]},
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
        "reproduce_real_v4: DONE.",
        "ShapeAgree:", "YES" if checks["overall"] else "NO",
        "| A-TOST:", "EQUIV" if tost_a["equivalent"] else "NOT-EQUIV",
        "| B-TOST:", "EQUIV" if tost_b["equivalent"] else "NOT-EQUIV",
        f"| A-margin {float(np.median(finals['full']) - np.median(finals['mismatched_r'])):.3f} [{m_lo:.3f},{m_hi:.3f}]",
        f"| B-margin {ord_b['full']-ord_b['mismatched_r']:.3f}",
        f"| {time.time()-t_start:.0f}s",
    )


if __name__ == "__main__":
    main()
