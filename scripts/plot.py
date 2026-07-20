"""Figures from the reproduce run (CSV-backed, one function per figure)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fig_divergence_curve(
    path, depths, predicted, measured_by_arm: dict,
    title: str = "Divergence vs relationship depth: recorded curve and measured medians",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(depths, predicted, "k--", lw=2, label="recorded analytical curve")
    colors = {"full": "C0", "mismatched_r": "C1", "empty_r": "C2"}
    for arm, (med, lo, hi) in measured_by_arm.items():
        c = colors.get(arm, "C3")
        ax.plot(depths, med, color=c, lw=1.8, label=f"measured: {arm}")
        ax.fill_between(depths, lo, hi, color=c, alpha=0.15)
    ax.set_xlabel("relationship depth (turns)")
    ax.set_ylabel("divergence D")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_othermodel_accuracy(path, depths, acc_by_arm: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, title in zip(axes, ("valence_acc", "need_acc"), ("valence accuracy", "need accuracy")):
        for arm, series in acc_by_arm.items():
            ax.plot(depths, series[key], marker="o", ms=3, label=arm)
        ax.set_xlabel("relationship depth (turns)")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
    fig.suptitle("Other-model accuracy vs depth (frozen scoring map)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_ablations(path, arm_names, final_d, verdicts) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["C0" if verdicts[a]["joint"] else "C3" for a in arm_names]
    ax.bar(arm_names, final_d, color=colors)
    ax.set_ylabel("final-depth divergence D")
    ax.set_title("Necessity ablations: final D per arm (blue = joint signature PASS)")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_floor_and_ordering(path, floor_info: dict, ordering: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ax = axes[0]
    ax.bar(["empty_r (unpadded)", "base_unmodified"], [floor_info["stranger_q"], floor_info["base_q"]], color=["C2", "C7"])
    ax.axhline(floor_info["base_q"] - floor_info["margin"], color="r", ls=":", lw=1, label="floor margin")
    ax.axhline(floor_info["base_q"] + floor_info["margin"], color="r", ls=":", lw=1)
    ax.set_ylabel("quality score")
    ax.set_title(f"Stranger-treatment floor: {'PASS' if floor_info['pass'] else 'BREACH — INVALID RUN'}")
    ax.legend(fontsize=8)
    ax = axes[1]
    names = list(ordering["values"])
    ax.bar(names, [ordering["values"][n] for n in names], color=["C0", "C1", "C2"])
    ax.set_ylabel("final-depth D")
    ax.set_title(f"Ordering known > mismatched > stranger: {'PASS' if ordering['pass'] else 'FAIL'}")
    ax.tick_params(axis="x", rotation=15, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
