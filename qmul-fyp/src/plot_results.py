"""Generate all qmul-fyp figures from results CSVs."""
from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
FIGS = os.path.join(RESULTS, "figures")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
})

COLOURS = {"lowbits": "#7f7f7f", "baseline": "#1f77b4", "optimised": "#ff7f0e"}
LABELS = {"lowbits": "Low-bits", "baseline": "Baseline (full)", "optimised": "Optimised (full)"}


def plot_metrics():
    df = pd.read_csv(os.path.join(RESULTS, "metrics.csv"))
    bl = df[df["mode"] == "baseline"]
    op = df[df["mode"] == "optimised"]

    for metric, ylabel in [("depth", "Transpiled Depth"),
                           ("cx", "CNOT Gate Count"),
                           ("qubits", "Qubit Count")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(bl))
        w = 0.35
        ax.bar(x - w / 2, bl[metric].values, w, label="Baseline", color=COLOURS["baseline"])
        ax.bar(x + w / 2, op[metric].values, w, label="Optimised", color=COLOURS["optimised"])
        ax.set_xticks(x)
        ax.set_xticklabels([f"n={n}" for n in bl["n"].values])
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Bit-width")
        ax.set_title(f"{ylabel}: Baseline vs Optimised Multiplier")
        ax.legend()

        # Add ratio annotations
        for i, (b, o) in enumerate(zip(bl[metric].values, op[metric].values)):
            ratio = b / o if o else 0
            if metric == "qubits":
                ratio = o / b  # qubits: optimised uses more
                ax.annotate(f"{ratio:.1f}×", (i + w / 2, o), ha="center", va="bottom", fontsize=9)
            else:
                ax.annotate(f"{ratio:.1f}×", (i - w / 2, b), ha="center", va="bottom", fontsize=9)

        fig.savefig(os.path.join(FIGS, f"metrics_{metric}.png"))
        plt.close(fig)
        print(f"  saved metrics_{metric}.png")


def plot_depth_reduction():
    """Depth reduction ratio plot."""
    df = pd.read_csv(os.path.join(RESULTS, "metrics.csv"))
    bl = df[df["mode"] == "baseline"].set_index("n")
    op = df[df["mode"] == "optimised"].set_index("n")

    ns = bl.index.values
    depth_ratio = bl["depth"].values / op["depth"].values
    cx_ratio = bl["cx"].values / op["cx"].values

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(ns))
    w = 0.35
    ax.bar(x - w / 2, depth_ratio, w, label="Depth reduction", color="#2ca02c")
    ax.bar(x + w / 2, cx_ratio, w, label="CNOT reduction", color="#d62728")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in ns])
    ax.set_ylabel("Ratio (Baseline / Optimised)")
    ax.set_xlabel("Bit-width")
    ax.set_title("Resource Reduction: Optimised vs Baseline")
    ax.legend()
    for i, (d, c) in enumerate(zip(depth_ratio, cx_ratio)):
        ax.annotate(f"{d:.1f}×", (i - w / 2, d), ha="center", va="bottom", fontsize=9)
        ax.annotate(f"{c:.1f}×", (i + w / 2, c), ha="center", va="bottom", fontsize=9)
    fig.savefig(os.path.join(FIGS, "reduction_ratio.png"))
    plt.close(fig)
    print("  saved reduction_ratio.png")


def plot_noise():
    df = pd.read_csv(os.path.join(RESULTS, "noise.csv"))

    for n_val in df["n"].unique():
        sub = df[df["n"] == n_val]
        fig, ax = plt.subplots(figsize=(6, 4))
        for mode in ["baseline", "optimised"]:
            part = sub[sub["mode"] == mode]
            ax.plot(range(len(part)), part["accuracy"].values,
                    "o-", label=LABELS.get(mode, mode), color=COLOURS[mode], linewidth=2)

        ax.set_xticks(range(len(part)))
        ax.set_xticklabels(part["noise_level"].values, rotation=0)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Noise Level")
        ax.set_title(f"Accuracy Under Noise (n={n_val})")
        ax.set_ylim(-0.05, 1.1)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.savefig(os.path.join(FIGS, f"noise_n{n_val}.png"))
        plt.close(fig)
        print(f"  saved noise_n{n_val}.png")


def plot_noise_combined():
    """Combined noise plot with both n values side by side."""
    df = pd.read_csv(os.path.join(RESULTS, "noise.csv"))
    noise_levels = df["noise_level"].unique()
    ns = sorted(df["n"].unique())

    fig, axes = plt.subplots(1, len(ns), figsize=(12, 4.5), sharey=True)
    if len(ns) == 1:
        axes = [axes]

    for idx, n_val in enumerate(ns):
        ax = axes[idx]
        sub = df[df["n"] == n_val]
        for mode in ["baseline", "optimised"]:
            part = sub[sub["mode"] == mode]
            ax.plot(range(len(part)), part["accuracy"].values,
                    "o-", label=LABELS.get(mode, mode), color=COLOURS[mode], linewidth=2, markersize=7)
        ax.set_xticks(range(len(noise_levels)))
        ax.set_xticklabels(noise_levels, rotation=0)
        ax.set_xlabel("Noise Level")
        ax.set_title(f"n = {n_val}")
        ax.grid(axis="y", alpha=0.3)
        if idx == 0:
            ax.set_ylabel("Accuracy")
            ax.legend()

    fig.suptitle("Accuracy Under Depolarizing Noise: Baseline vs Optimised", fontsize=13, y=1.02)
    fig.savefig(os.path.join(FIGS, "noise_combined.png"))
    plt.close(fig)
    print("  saved noise_combined.png")


def plot_scaling():
    """All three modes: qubit count, depth, CX vs n."""
    df = pd.read_csv(os.path.join(RESULTS, "metrics.csv"))

    for metric, ylabel, title in [
        ("depth", "Transpiled Depth", "Circuit Depth Scaling"),
        ("cx", "CNOT Gates", "CNOT Gate Count Scaling"),
        ("qubits", "Qubits", "Qubit Count Scaling"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for mode in ["lowbits", "baseline", "optimised"]:
            part = df[df["mode"] == mode]
            ax.plot(part["n"], part[metric], "o-", label=LABELS[mode],
                    color=COLOURS[mode], linewidth=2, markersize=7)
        ax.set_xlabel("Bit-width (n)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(FIGS, f"scaling_{metric}.png"))
        plt.close(fig)
        print(f"  saved scaling_{metric}.png")


def main():
    print("Generating qmul-fyp figures...")
    plot_metrics()
    plot_depth_reduction()
    plot_noise()
    plot_noise_combined()
    plot_scaling()
    print(f"\nAll figures saved to {FIGS}")


if __name__ == "__main__":
    main()
