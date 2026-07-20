"""Regenerate every CSV, figure, and RESULTS.md from one fixed seed.

Execution order preserves the intended dependency structure:
  Stage A  formal model -> results/model_predictions.csv + predicted shape,
           FROZEN (sha256 recorded; regeneration must match).
  Stage B  persona pool + marker ground truth (reporting split).
  Stage C  the arm grid: divergence, other-model accuracy, marker emission,
           joint-signature ablation verdicts, overhead.
  Stage D  protocol checks: stranger-treatment floor (UNPADDED vs
           base_unmodified — breach = INVALID RUN), ordering
           (known > mismatched > stranger), padding sensitivity,
           measured-vs-frozen shape agreement, marker naturalism.
  Stage E  figures + RESULTS.md + run_meta.json.

Everything here runs against the SYNTHETIC base model: outputs are pipeline
validation, not open-weight-model measurements. Stage A is LLM-free by design.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lovellm.allocation import Budget, decision_table  # noqa: E402
from lovellm.arms import BASE_UNMODIFIED, CORE_ARMS, EMPTY_R, FULL, MISMATCHED_R  # noqa: E402
from lovellm.basemodel import SyntheticBaseModel  # noqa: E402
from lovellm.curve import agreement, predicted_divergence, shape_features  # noqa: E402
from lovellm.metrics import (  # noqa: E402
    accuracy,
    floor_check,
    marker_emission_rate,
    marker_naturalism,
    spearman,
)
from lovellm.personas import make_pool  # noqa: E402
from lovellm.pipeline import run_relationship  # noqa: E402
from scripts.plot import (  # noqa: E402
    fig_ablations,
    fig_divergence_curve,
    fig_floor_and_ordering,
    fig_othermodel_accuracy,
)

RESULTS = ROOT / "results"
MASTER_SEED = 20260712
CAL_SEED = MASTER_SEED + 1_000_000   # calibration split (tuning only; unused this run)
REPORTING_SEED = MASTER_SEED + 2_000_000  # all reported synthetic-grid numbers

N_USERS = 4
N_SESSIONS = 6
TURNS_PER_SESSION = 8
N_SEEDS = 6
DEPTHS = np.array([(s + 1) * TURNS_PER_SESSION for s in range(N_SESSIONS)])

# Signature criteria (PROTOCOL: joint-signature definition; PROVISIONAL values).
C3_ACC_MARGIN = 0.10
MIN_FINAL_D = 0.15
FLOOR_MARGIN = 0.05
ORDERING_MARGIN = 0.02


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def band(vals) -> str:
    v = np.asarray(vals, dtype=float)
    return f"{np.median(v):.3f} [{np.quantile(v, 0.025):.3f}, {np.quantile(v, 0.975):.3f}]"


# ---------------------------------------------------------------- Stage A

def stage_a() -> dict:
    pred = predicted_divergence(DEPTHS)
    from lovellm.curve import predicted_channels

    ch = predicted_channels(DEPTHS)
    rows = []
    for i, d in enumerate(DEPTHS):
        rows.append(
            {
                "depth": int(d),
                **{c: f"{ch[c][i]:.6f}" for c in ch},
                "D": f"{pred[i]:.6f}",
            }
        )
    path = RESULTS / "model_predictions.csv"
    write_csv(path, rows)
    shape = shape_features(pred)
    (RESULTS / "predicted_shape.json").write_text(json.dumps(shape.to_dict(), indent=2))

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    frozen = RESULTS / "FROZEN_model_predictions.sha256"
    if frozen.exists():
        recorded = frozen.read_text().split()[0]
        if recorded != digest:
            raise RuntimeError(
                "model_predictions.csv no longer matches its frozen hash — the "
                "recorded analytical curve may have been altered. "
                f"frozen={recorded} current={digest}"
            )
        freshly = False
    else:
        frozen.write_text(f"{digest}  model_predictions.csv\n")
        freshly = True
    return {"pred": pred, "shape": shape, "digest": digest, "freshly_frozen": freshly}


# ---------------------------------------------------------------- Stage B

def stage_b():
    n_turns = N_SESSIONS * TURNS_PER_SESSION
    pool = make_pool(N_USERS, n_turns, REPORTING_SEED)
    meta = {
        "n_users": N_USERS,
        "n_sessions": N_SESSIONS,
        "turns_per_session": TURNS_PER_SESSION,
        "markers_per_user": N_SESSIONS,
        "decision_table": decision_table(),
    }
    (RESULTS / "personas_meta.json").write_text(json.dumps(meta, indent=2))
    return pool


# ---------------------------------------------------------------- Stage C

def stage_c(pool) -> dict:
    bm = SyntheticBaseModel()
    budget = Budget()
    all_arms = CORE_ARMS + [BASE_UNMODIFIED]
    runs_by_arm: dict[str, list] = {}
    div_rows, om_rows, ref_rows, ov_rows = [], [], [], []

    for arm in all_arms:
        runs = []
        for s in range(N_SEEDS):
            focal = pool[s % N_USERS]
            distractor = pool[(s + 1) % N_USERS]
            run = run_relationship(
                arm, bm, focal, distractor, budget, N_SESSIONS, TURNS_PER_SESSION, REPORTING_SEED + 331 * s
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

        # Other-model accuracy per checkpoint (pooled over seeds at each depth).
        for si in range(N_SESSIONS):
            pairs = [r.checkpoints[si].accuracy_pair for r in runs]
            acc = accuracy(pairs)
            om_rows.append(
                {
                    "arm": arm.name,
                    "depth": int(DEPTHS[si]),
                    "valence_acc": f"{acc['valence_acc']:.4f}",
                    "need_acc": f"{acc['need_acc']:.4f}",
                    "n": acc["n"],
                }
            )
        # Marker emission over the final checkpoint's known responses.
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

    write_csv(RESULTS / "divergence.csv", div_rows)
    write_csv(RESULTS / "othermodel.csv", om_rows)
    write_csv(RESULTS / "references.csv", ref_rows)
    write_csv(RESULTS / "overhead.csv", ov_rows)
    return {"runs": runs_by_arm, "om_rows": om_rows, "ref_rows": ref_rows, "ov_rows": ov_rows}


def median_curve(runs) -> np.ndarray:
    return np.median(np.array([[ck.D for ck in r.checkpoints] for r in runs]), axis=0)


# ---------------------------------------------------------------- Stage D

def stage_d(a_out: dict, c_out: dict) -> dict:
    runs = c_out["runs"]

    # Floor check: UNPADDED empty_r known-condition quality vs base_unmodified.
    stranger_q = [ck.quality_known for r in runs["empty_r"] for ck in r.checkpoints]
    base_q = [ck.quality_known for r in runs["base_unmodified"] for ck in r.checkpoints]
    floor = floor_check(stranger_q, base_q, FLOOR_MARGIN)
    floor_info = {
        "stranger_q": float(np.mean(stranger_q)),
        "base_q": float(np.mean(base_q)),
        "margin": FLOOR_MARGIN,
        **floor,
    }

    # Ordering check (C5 hardening): known > mismatched > stranger at final depth.
    final_d = {a: float(median_curve(runs[a])[-1]) for a in ("full", "mismatched_r", "empty_r")}
    ordering = {
        "values": final_d,
        "pass": (
            final_d["full"] >= final_d["mismatched_r"] + ORDERING_MARGIN
            and final_d["mismatched_r"] >= final_d["empty_r"] + ORDERING_MARGIN
        ),
        "margin": ORDERING_MARGIN,
    }

    # Padding sensitivity: padded vs unpadded stranger quality (full-arm probes).
    pad_deltas = [
        abs(ck.quality_stranger_padded - ck.quality_stranger_unpadded)
        for r in runs["full"]
        for ck in r.checkpoints
    ]
    padding = {"mean_abs_delta": float(np.mean(pad_deltas)), "max_abs_delta": float(np.max(pad_deltas))}

    # Measured-vs-frozen shape agreement on the full arm.
    measured = median_curve(runs["full"])
    rho = spearman(a_out["pred"], measured)
    meas_shape = shape_features(measured)
    checks = agreement(a_out["shape"], meas_shape, rho)

    # Marker naturalism on the full arm's known responses.
    texts = [t for r in runs["full"] for ck in r.checkpoints for t in ck.responses_known]
    naturalism = marker_naturalism(texts)

    # Joint-signature verdicts per arm (PROTOCOL-defined conjunction).
    om_final = {
        (row["arm"], row["depth"]): (float(row["valence_acc"]) + float(row["need_acc"])) / 2
        for row in c_out["om_rows"]
    }
    empty_acc = om_final[("empty_r", int(DEPTHS[-1]))]
    ref_by_arm = {row["arm"]: float(row["marker_emission_rate"].split()[0]) for row in c_out["ref_rows"]}
    verdicts = {}
    for arm in [a.name for a in CORE_ARMS]:
        curve_arm = median_curve(runs[arm])
        rho_arm = spearman(a_out["pred"], curve_arm)
        sf = shape_features(curve_arm)
        crit_curve = (
            rho_arm >= 0.8 and sf.monotone_fraction >= 0.80 and float(curve_arm[-1]) >= MIN_FINAL_D
        )
        crit_acc = (om_final[(arm, int(DEPTHS[-1]))] - empty_acc) >= C3_ACC_MARGIN
        crit_ref = ref_by_arm[arm] > ref_by_arm["empty_r"]
        verdicts[arm] = {
            "curve": bool(crit_curve),
            "accuracy": bool(crit_acc),
            "markers": bool(crit_ref),
            "joint": bool(crit_curve and crit_acc and crit_ref),
            "final_D": float(curve_arm[-1]),
            "rho": float(rho_arm),
        }
    write_csv(
        RESULTS / "ablations.csv",
        [
            {
                "arm": a,
                "crit_curve": v["curve"],
                "crit_accuracy": v["accuracy"],
                "crit_markers": v["markers"],
                "joint_signature": v["joint"],
                "final_D": f"{v['final_D']:.4f}",
                "rho_vs_predicted": f"{v['rho']:.4f}",
            }
            for a, v in verdicts.items()
        ],
    )

    return {
        "floor": floor_info,
        "ordering": ordering,
        "padding": padding,
        "rho": rho,
        "measured_shape": meas_shape,
        "checks": checks,
        "naturalism": naturalism,
        "verdicts": verdicts,
        "measured_curve": measured,
    }


# ---------------------------------------------------------------- Stage E

def stage_e(a_out: dict, c_out: dict, d_out: dict) -> None:
    runs = c_out["runs"]

    def curve_band(arm):
        m = np.array([[ck.D for ck in r.checkpoints] for r in runs[arm]])
        return (
            np.median(m, axis=0),
            np.quantile(m, 0.025, axis=0),
            np.quantile(m, 0.975, axis=0),
        )

    fig_divergence_curve(
        RESULTS / "fig1_divergence_curve.png",
        DEPTHS,
        a_out["pred"],
        {a: curve_band(a) for a in ("full", "mismatched_r", "empty_r")},
    )
    acc_by_arm = {}
    for arm in ("full", "empty_r", "caring_only", "particularity_only"):
        rows = [r for r in c_out["om_rows"] if r["arm"] == arm]
        acc_by_arm[arm] = {
            "valence_acc": [float(r["valence_acc"]) for r in rows],
            "need_acc": [float(r["need_acc"]) for r in rows],
        }
    fig_othermodel_accuracy(RESULTS / "fig2_othermodel_accuracy.png", DEPTHS, acc_by_arm)
    arm_names = [a.name for a in CORE_ARMS]
    fig_ablations(
        RESULTS / "fig3_ablations.png",
        arm_names,
        [d_out["verdicts"][a]["final_D"] for a in arm_names],
        d_out["verdicts"],
    )
    fig_floor_and_ordering(RESULTS / "fig4_floor_and_ordering.png", d_out["floor"], d_out["ordering"])

    pred_shape, meas_shape, checks = a_out["shape"], d_out["measured_shape"], d_out["checks"]
    floor = d_out["floor"]
    ordering = d_out["ordering"]
    lines = [
        "# Synthetic validation results (generated by scripts/reproduce.py)",
        "",
        f"Seed: `{MASTER_SEED}` · pool: {N_USERS} personas · {N_SESSIONS} sessions × "
        f"{TURNS_PER_SESSION} turns · {N_SEEDS} seeds/arm",
        "",
        "> **Synthetic-validation scope.** Every number below was produced against the bundled",
        "> SYNTHETIC base model (a seeded stand-in emulating a companion LLM's",
        "> response surface). These are PIPELINE-VALIDATION numbers — they",
        "> check that the R store, other-model, floor-plus-surplus allocation,",
        "> arms, probes, and metrics execute end-to-end and have the expected",
        "> directional behavior. They are not open-weight-model measurements.",
        "> Capability equivalence is not reported for the synthetic stand-in.",
        "> `model_predictions.csv` contains the formal model's",
        "> divergence-vs-R curve is LLM-free by design and FROZEN (sha256 below)",
        "> before any base-model stage runs.",
        "",
        "## Stage A — frozen model prediction (C2, LLM-free)",
        "",
        f"- `model_predictions.csv` sha256: `{a_out['digest']}`"
        + (" (frozen this run)" if a_out["freshly_frozen"] else " (matches existing freeze)"),
        f"- Predicted shape: monotone_fraction={pred_shape.monotone_fraction:.3f}, "
        f"half_full_ratio={pred_shape.half_full_ratio:.3f}, curvature={pred_shape.curvature_class}",
        "",
        "## Stage C — arm grid (reporting split)",
        "",
        "| arm | final D | rho vs predicted | curve | accuracy | markers | JOINT |",
        "|---|---|---|---|---|---|---|",
    ]
    for a, v in d_out["verdicts"].items():
        lines.append(
            f"| {a} | {v['final_D']:.3f} | {v['rho']:.3f} | {v['curve']} | {v['accuracy']} | "
            f"{v['markers']} | **{'PASS' if v['joint'] else 'FAIL'}** |"
        )
    lines += [
        "",
        "Joint signature = the protocol-defined conjunction (curve fidelity AND C3 accuracy",
        "margin AND marker contrast); `degrade` means failing at least one criterion",
        f"(C3 margin {C3_ACC_MARGIN}, min final D {MIN_FINAL_D}, rho ≥ 0.8).",
        "",
        "### Overhead (layer vs generation, µs/turn)",
        "",
        "| arm | layer | generation |",
        "|---|---|---|",
    ]
    for row in c_out["ov_rows"]:
        lines.append(f"| {row['arm']} | {row['layer_us_per_turn']} | {row['generation_us_per_turn']} |")
    lines += [
        "",
        "## Stage D — protocol checks",
        "",
        f"- **Stranger-treatment floor (UNPADDED empty_r vs base_unmodified):** "
        f"Δ = {floor['delta']:.4f} vs margin {floor['margin']} → "
        + ("**PASS**" if floor["pass"] else "**BREACH — INVALID RUN**"),
        f"- **Ordering (known > mismatched > stranger, margin {ordering['margin']}):** "
        f"full {ordering['values']['full']:.3f} > mismatched_r {ordering['values']['mismatched_r']:.3f} "
        f"> empty_r {ordering['values']['empty_r']:.3f} → "
        + ("**PASS**" if ordering["pass"] else "**FAIL**"),
        f"- **Padding sensitivity (padded vs unpadded stranger quality):** mean |Δ| = "
        f"{d_out['padding']['mean_abs_delta']:.4f}, max = {d_out['padding']['max_abs_delta']:.4f} "
        "(structural on the synthetic stand-in; a real model consumes the padded context — W2).",
        f"- **Marker naturalism (full arm):** density = {d_out['naturalism']['marker_density']:.4f} "
        f"in band {d_out['naturalism']['band']} → " + ("PASS" if d_out["naturalism"]["pass"] else "FAIL"),
        "",
        "### Measured vs frozen shape (full arm)",
        "",
        "| feature | predicted | measured | criterion | pass |",
        "|---|---|---|---|---|",
        f"| monotone_fraction | {pred_shape.monotone_fraction:.3f} | {meas_shape.monotone_fraction:.3f} | ≥ 0.80 | {checks['monotone']} |",
        f"| half_full_ratio | {pred_shape.half_full_ratio:.3f} | {meas_shape.half_full_ratio:.3f} | ± 0.20 | {checks['half_full_ratio']} |",
        f"| curvature_class | {pred_shape.curvature_class} | {meas_shape.curvature_class} | equal | {checks['curvature_class']} |",
        f"| spearman rho | — | {d_out['rho']:.3f} | ≥ 0.8 | {checks['spearman_rho']} |",
        "",
        f"**Model/measurement agreement (synthetic validation): {'YES' if checks['overall'] else 'NO'}**",
        "",
        "## Files",
        "",
        "`model_predictions.csv` `predicted_shape.json` `FROZEN_model_predictions.sha256` "
        "`personas_meta.json` `divergence.csv` `othermodel.csv` `references.csv` "
        "`ablations.csv` `overhead.csv` `fig1_divergence_curve.png` "
        "`fig2_othermodel_accuracy.png` `fig3_ablations.png` `fig4_floor_and_ordering.png` "
        "`run_meta.json`",
        "",
    ]
    (RESULTS / "RESULTS.md").write_text("\n".join(lines))


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    a_out = stage_a()
    pool = stage_b()
    c_out = stage_c(pool)
    d_out = stage_d(a_out, c_out)
    stage_e(a_out, c_out, d_out)
    (RESULTS / "run_meta.json").write_text(
        json.dumps(
            {
                "master_seed": MASTER_SEED,
                "calibration_seed": CAL_SEED,
                "reporting_seed": REPORTING_SEED,
                "n_users": N_USERS,
                "n_sessions": N_SESSIONS,
                "turns_per_session": TURNS_PER_SESSION,
                "n_seeds_per_arm": N_SEEDS,
                "base_model": "SyntheticBaseModel (pipeline validation only)",
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            indent=2,
        )
    )
    if not d_out["floor"]["pass"]:
        print("reproduce: FLOOR BREACH — RUN INVALID (see RESULTS.md)")
    print(
        "reproduce: OK — results/ regenerated.",
        "Agreement:", "YES" if d_out["checks"]["overall"] else "NO",
        "| Floor:", "PASS" if d_out["floor"]["pass"] else "BREACH",
        "| Ordering:", "PASS" if d_out["ordering"]["pass"] else "FAIL",
    )


if __name__ == "__main__":
    main()
