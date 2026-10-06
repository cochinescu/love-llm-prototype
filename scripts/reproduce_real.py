"""W2: the real-base-model grid (pinned MLX model), writing results/real/.

Evidential order is inherited from scripts/reproduce.py and enforced here:
Stage A only VERIFIES the frozen prediction (it must already exist and match
its recorded sha256 — this script never freezes or re-freezes); Stages B-E
mirror the synthetic run against the pinned real model, with two W2-only
additions: real padding sensitivity (the padded context is actually consumed)
and the capability-equivalence TOST (full arm's known-condition quality vs
the unmodified base model, cluster bootstrap, specified margin).

Grid sizes and decoding parameters are the PROTOCOL "W2 freeze" values.
This runner uses the ORIGINAL prompt renderer (`lovellm.realmodel_v1`);
the R1 renderer in `lovellm.realmodel` is used by reproduce_real_v2.py onward.
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
from lovellm.curve import agreement, predicted_divergence, shape_features  # noqa: E402
from lovellm.metrics import (  # noqa: E402
    accuracy,
    floor_check,
    marker_emission_rate,
    marker_naturalism,
    spearman,
    tost_equivalence,
)
from lovellm.personas import make_pool  # noqa: E402
from lovellm.pipeline import run_relationship  # noqa: E402
from lovellm.realmodel import PINNED_MODEL, PINNED_REVISION, TEMP, TOP_P  # noqa: E402
from lovellm.realmodel_v1 import MLXBaseModelV1  # noqa: E402  (ORIGINAL renderer; PROTOCOL A9)
from scripts.plot import (  # noqa: E402
    fig_ablations,
    fig_divergence_curve,
    fig_floor_and_ordering,
    fig_othermodel_accuracy,
)
from scripts.reproduce import (  # noqa: E402
    C3_ACC_MARGIN,
    REPORTING_SEED,
    DEPTHS,
    FLOOR_MARGIN,
    MASTER_SEED,
    MIN_FINAL_D,
    N_SESSIONS,
    N_USERS,
    ORDERING_MARGIN,
    TURNS_PER_SESSION,
    band,
    median_curve,
    write_csv,
)

RESULTS = ROOT / "results"
REAL = RESULTS / "real"
N_SEEDS_REAL = 4  # PROTOCOL W2 freeze
N_GEN_REAL = 2    # PROTOCOL W2 freeze
TOST_MARGIN = 0.05  # PROTOCOL W2 freeze (quality-score units)


def stage_a_verify() -> dict:
    pred = predicted_divergence(DEPTHS)
    path = RESULTS / "model_predictions.csv"
    frozen = RESULTS / "FROZEN_model_predictions.sha256"
    if not (path.exists() and frozen.exists()):
        raise RuntimeError("frozen prediction missing — run scripts/reproduce.py first")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    recorded = frozen.read_text().split()[0]
    if digest != recorded:
        raise RuntimeError(
            f"model_predictions.csv does not match its frozen hash: {recorded} != {digest}"
        )
    return {"pred": pred, "shape": shape_features(pred), "digest": digest}


def main() -> None:
    REAL.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    a_out = stage_a_verify()
    print(f"Stage A: frozen prediction verified ({a_out['digest'][:12]}…)")

    n_turns = N_SESSIONS * TURNS_PER_SESSION
    pool = make_pool(N_USERS, n_turns, REPORTING_SEED)
    # The initial grid was generated with the ORIGINAL renderer (fixed 6 words
    # per topic, 60-token context cap). Restored 2026-10-06 so this grid is
    # regenerable as run; v2+ use the R1 renderer (lovellm.realmodel).
    bm = MLXBaseModelV1()
    budget = Budget()
    all_arms = CORE_ARMS + [BASE_UNMODIFIED]

    runs_by_arm: dict[str, list] = {}
    div_rows, om_rows, ref_rows, ov_rows = [], [], [], []
    for arm in all_arms:
        t_arm = time.time()
        runs = []
        for s in range(N_SEEDS_REAL):
            focal = pool[s % N_USERS]
            distractor = pool[(s + 1) % N_USERS]
            run = run_relationship(
                arm, bm, focal, distractor, budget, N_SESSIONS, TURNS_PER_SESSION,
                REPORTING_SEED + 331 * s, n_gen=N_GEN_REAL,
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
            focal = pool[r_i % N_USERS]
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
        print(f"  arm {arm.name}: {time.time()-t_arm:.0f}s")

    write_csv(REAL / "divergence.csv", div_rows)
    write_csv(REAL / "othermodel.csv", om_rows)
    write_csv(REAL / "references.csv", ref_rows)
    write_csv(REAL / "overhead.csv", ov_rows)

    # ---- Stage D: protocol checks on the real grid ----
    stranger_q = [ck.quality_known for r in runs_by_arm["empty_r"] for ck in r.checkpoints]
    base_q = [ck.quality_known for r in runs_by_arm["base_unmodified"] for ck in r.checkpoints]
    floor = floor_check(stranger_q, base_q, FLOOR_MARGIN)
    floor_info = {
        "stranger_q": float(np.mean(stranger_q)),
        "base_q": float(np.mean(base_q)),
        "margin": FLOOR_MARGIN,
        **floor,
    }
    final_d = {a: float(median_curve(runs_by_arm[a])[-1]) for a in ("full", "mismatched_r", "empty_r")}
    ordering = {
        "values": final_d,
        "pass": (
            final_d["full"] >= final_d["mismatched_r"] + ORDERING_MARGIN
            and final_d["mismatched_r"] >= final_d["empty_r"] + ORDERING_MARGIN
        ),
        "margin": ORDERING_MARGIN,
    }
    pad_deltas = [
        abs(ck.quality_stranger_padded - ck.quality_stranger_unpadded)
        for r in runs_by_arm["full"]
        for ck in r.checkpoints
    ]
    padding = {"mean_abs_delta": float(np.mean(pad_deltas)), "max_abs_delta": float(np.max(pad_deltas))}
    measured = median_curve(runs_by_arm["full"])
    rho = spearman(a_out["pred"], measured)
    meas_shape = shape_features(measured)
    checks = agreement(a_out["shape"], meas_shape, rho)
    texts = [t for r in runs_by_arm["full"] for ck in r.checkpoints for t in ck.responses_known]
    naturalism = marker_naturalism(texts)

    # Capability equivalence (TOST): full arm's known-condition quality vs the
    # unmodified base model, per-run means as clusters.
    full_q_runs = [float(np.mean([ck.quality_known for ck in r.checkpoints])) for r in runs_by_arm["full"]]
    base_q_runs = [float(np.mean([ck.quality_known for ck in r.checkpoints])) for r in runs_by_arm["base_unmodified"]]
    tost = tost_equivalence(full_q_runs, base_q_runs, TOST_MARGIN, seed=MASTER_SEED)

    om_final = {
        (row["arm"], row["depth"]): (float(row["valence_acc"]) + float(row["need_acc"])) / 2
        for row in om_rows
    }
    empty_acc = om_final[("empty_r", int(DEPTHS[-1]))]
    ref_by_arm = {row["arm"]: float(row["marker_emission_rate"].split()[0]) for row in ref_rows}
    verdicts = {}
    for arm in [a.name for a in CORE_ARMS]:
        curve_arm = median_curve(runs_by_arm[arm])
        rho_arm = spearman(a_out["pred"], curve_arm)
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
        REAL / "ablations.csv",
        [
            {
                "arm": a,
                "crit_curve": v["curve"],
                "crit_accuracy": v["accuracy"],
                "crit_markers": v["markers"],
                "joint_signature": v["joint"],
                "final_D": f"{v['final_D']:.4f}",
                "rho_vs_predicted": f"{v['rho']:.4f}",
                "monotone_fraction": f"{v['monotone']:.3f}",
                "acc_combined_final": f"{v['acc_combined']:.4f}",
            }
            for a, v in verdicts.items()
        ],
    )
    write_csv(
        REAL / "capability.csv",
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

    # ---- Stage E: figures + RESULTS.md + run_meta ----
    def curve_band(arm):
        m = np.array([[ck.D for ck in r.checkpoints] for r in runs_by_arm[arm]])
        return (np.median(m, axis=0), np.quantile(m, 0.025, axis=0), np.quantile(m, 0.975, axis=0))

    fig_divergence_curve(
        REAL / "fig1_divergence_curve_real.png",
        DEPTHS,
        a_out["pred"],
        {a: curve_band(a) for a in ("full", "mismatched_r", "empty_r")},
        title="Divergence vs relationship depth: recorded curve and measured medians",
    )
    acc_by_arm = {}
    for arm in ("full", "empty_r", "caring_only", "particularity_only"):
        rows = [r for r in om_rows if r["arm"] == arm]
        acc_by_arm[arm] = {
            "valence_acc": [float(r["valence_acc"]) for r in rows],
            "need_acc": [float(r["need_acc"]) for r in rows],
        }
    fig_othermodel_accuracy(REAL / "fig2_othermodel_accuracy_real.png", DEPTHS, acc_by_arm)
    arm_names = [a.name for a in CORE_ARMS]
    fig_ablations(
        REAL / "fig3_ablations_real.png",
        arm_names,
        [verdicts[a]["final_D"] for a in arm_names],
        verdicts,
    )
    fig_floor_and_ordering(REAL / "fig4_floor_and_ordering_real.png", floor_info, ordering)

    lines = [
        "# REAL-MODEL grid results (generated by scripts/reproduce_real.py — build stage W2)",
        "",
        f"Pinned base model: `{PINNED_MODEL}` · decoding: temp {TEMP}, top_p {TOP_P} · "
        f"seed `{MASTER_SEED}` · pool {N_USERS} personas · {N_SESSIONS} sessions × {TURNS_PER_SESSION} turns · "
        f"{N_SEEDS_REAL} seeds/arm · {N_GEN_REAL} paired generations/(probe,condition)",
        "",
        "These are measurements from the pinned open-weight base model. The",
        "base model consumes rendered prompts (context tokens, marker tokens, length",
        "target) and all metrics remain mechanical on the emitted text. The frozen",
        f"prediction was verified untouched before any run (sha256 `{a_out['digest'][:16]}…`).",
        "",
        "## Divergence vs frozen prediction (full arm)",
        "",
        f"- measured median D over depths {list(map(int, DEPTHS))}: {[round(float(x), 3) for x in measured]}",
        f"- Spearman rho vs frozen prediction: {rho:.3f} (bar 0.8)",
        f"- shape: monotone {meas_shape.monotone_fraction:.3f} (bar 0.80), ratio {meas_shape.half_full_ratio:.3f} "
        f"(predicted {a_out['shape'].half_full_ratio:.3f} ± 0.20), curvature {meas_shape.curvature_class} "
        f"(predicted {a_out['shape'].curvature_class})",
        f"- **agreement: {'YES' if checks['overall'] else 'NO'}** ({checks})",
        "",
        "## Ablations (protocol-defined joint signature)",
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
        f"- **Floor (unpadded empty_r vs base_unmodified):** Δ = {floor_info['delta']:.4f} vs margin {FLOOR_MARGIN} → "
        + ("**PASS**" if floor_info["pass"] else "**BREACH — RUN INVALID**"),
        f"- **Ordering:** full {final_d['full']:.3f} > mismatched {final_d['mismatched_r']:.3f} > empty {final_d['empty_r']:.3f} "
        f"(margin {ORDERING_MARGIN}) → " + ("**PASS**" if ordering["pass"] else "**FAIL**"),
        f"- **Padding sensitivity (REAL: padded context consumed):** mean |Δq| = {padding['mean_abs_delta']:.4f}, "
        f"max = {padding['max_abs_delta']:.4f}",
        f"- **Marker naturalism:** density {naturalism['marker_density']:.4f} in band {naturalism['band']} → "
        + ("PASS" if naturalism["pass"] else "FAIL"),
        f"- **Capability equivalence (TOST, margin ±{TOST_MARGIN}):** mean quality diff (full − base) = "
        f"{tost['mean_diff']:.4f}, 90% CI [{tost['ci90'][0]:.4f}, {tost['ci90'][1]:.4f}] → "
        + ("**EQUIVALENT**" if tost["equivalent"] else "**NOT SHOWN EQUIVALENT**"),
        "",
        "### Overhead (layer vs generation, µs/turn)",
        "",
        "| arm | layer | generation |",
        "|---|---|---|",
    ]
    for row in ov_rows:
        lines.append(f"| {row['arm']} | {row['layer_us_per_turn']} | {row['generation_us_per_turn']} |")
    lines += [
        "",
        f"Total wall time: {time.time()-t_start:.0f}s",
        "",
    ]
    (REAL / "RESULTS.md").write_text("\n".join(lines))
    (REAL / "run_meta.json").write_text(
        json.dumps(
            {
                "base_model": PINNED_MODEL,
                "base_model_revision": PINNED_REVISION,
                "decoding": {"temperature": TEMP, "top_p": TOP_P},
                "master_seed": MASTER_SEED,
                "reporting_seed": REPORTING_SEED,
                "n_users": N_USERS,
                "n_sessions": N_SESSIONS,
                "turns_per_session": TURNS_PER_SESSION,
                "n_seeds_per_arm": N_SEEDS_REAL,
                "n_gen": N_GEN_REAL,
                "tost_margin": TOST_MARGIN,
                "frozen_prediction_sha256": a_out["digest"],
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            indent=2,
        )
    )
    print(
        "reproduce_real: DONE.",
        "Agreement:", "YES" if checks["overall"] else "NO",
        "| Floor:", "PASS" if floor_info["pass"] else "BREACH",
        "| Ordering:", "PASS" if ordering["pass"] else "FAIL",
        "| TOST:", "EQUIV" if tost["equivalent"] else "NOT-EQUIV",
        f"| {time.time()-t_start:.0f}s",
    )


if __name__ == "__main__":
    main()
