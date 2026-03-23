"""Generate all qkern-fyp figures from sweep_results.csv."""
from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
FIGS = os.path.join(RESULTS, "figures")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
})


def load():
    return pd.read_csv(os.path.join(RESULTS, "sweep_results.csv"))


def plot_accuracy_by_dataset():
    """Bar chart: mean accuracy per dataset for QK, RBF, Poly, Linear."""
    df = load()
    g = df.groupby("dataset")[["qk_acc", "rbf_acc", "poly_acc", "lin_acc"]].mean()
    g = g.sort_values("qk_acc", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(g))
    w = 0.2
    labels = ["Quantum Kernel", "RBF", "Polynomial", "Linear"]
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (col, lab, c) in enumerate(zip(
            ["qk_acc", "rbf_acc", "poly_acc", "lin_acc"], labels, colours)):
        ax.bar(x + i * w, g[col].values, w, label=lab, color=c)

    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(g.index, rotation=15)
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Classification Accuracy: Quantum vs Classical Kernels")
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(FIGS, "accuracy_by_dataset.png"))
    plt.close(fig)
    print("  saved accuracy_by_dataset.png")


def plot_accuracy_vs_qubits():
    """Line plot: QK accuracy vs qubits, one line per dataset, averaged over seeds/depths/fmaps."""
    df = load()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        g = sub.groupby("qubits")["qk_acc"].agg(["mean", "std"]).reset_index()
        ax.errorbar(g["qubits"], g["mean"], yerr=g["std"], fmt="o-",
                    label=ds, linewidth=2, markersize=6, capsize=3)

    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Quantum Kernel Accuracy")
    ax.set_title("Quantum Kernel Accuracy vs Qubit Count")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.savefig(os.path.join(FIGS, "accuracy_vs_qubits.png"))
    plt.close(fig)
    print("  saved accuracy_vs_qubits.png")


def plot_feature_map_comparison():
    """Grouped bar: ZZ vs IQP accuracy per dataset."""
    df = load()
    g = df.groupby(["dataset", "feature_map"])["qk_acc"].mean().unstack("feature_map")
    g = g.sort_values("iqp", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(g))
    w = 0.35
    ax.bar(x - w / 2, g["zz"].values, w, label="ZZ Feature Map", color="#1f77b4")
    ax.bar(x + w / 2, g["iqp"].values, w, label="IQP Feature Map", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(g.index, rotation=15)
    ax.set_ylabel("Mean Quantum Kernel Accuracy")
    ax.set_title("Feature Map Comparison: ZZ vs IQP")
    ax.set_ylim(0.5, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(FIGS, "feature_map_comparison.png"))
    plt.close(fig)
    print("  saved feature_map_comparison.png")


def plot_alignment_vs_accuracy():
    """Scatter: kernel alignment vs QK accuracy."""
    df = load()
    fig, ax = plt.subplots(figsize=(6, 5))
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        ax.scatter(sub["align"], sub["qk_acc"], alpha=0.5, s=30, label=ds)
    ax.set_xlabel("Kernel Alignment (cHSIC)")
    ax.set_ylabel("Quantum Kernel Accuracy")
    ax.set_title("Kernel Alignment vs Classification Accuracy")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(FIGS, "alignment_vs_accuracy.png"))
    plt.close(fig)
    print("  saved alignment_vs_accuracy.png")


def plot_depth_effect():
    """Line plot: accuracy vs circuit depth, one line per dataset."""
    df = load()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        g = sub.groupby("depth")["qk_acc"].agg(["mean", "std"]).reset_index()
        ax.errorbar(g["depth"], g["mean"], yerr=g["std"], fmt="o-",
                    label=ds, linewidth=2, markersize=6, capsize=3)
    ax.set_xlabel("Feature Map Depth")
    ax.set_ylabel("Quantum Kernel Accuracy")
    ax.set_title("Effect of Feature Map Depth on Accuracy")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.savefig(os.path.join(FIGS, "depth_effect.png"))
    plt.close(fig)
    print("  saved depth_effect.png")


def plot_heatmap_best_configs():
    """Heatmap: best QK accuracy for each dataset × feature_map × qubits combo."""
    df = load()
    g = df.groupby(["dataset", "feature_map", "qubits"])["qk_acc"].mean().reset_index()
    piv = g.pivot_table(index="dataset", columns=["feature_map", "qubits"],
                        values="qk_acc")
    # Flatten column names
    piv.columns = [f"{fm.upper()} q={q}" for fm, q in piv.columns]

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    # Annotate cells
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if val > 0.85 else "black")
    plt.colorbar(im, ax=ax, label="Accuracy")
    ax.set_title("Quantum Kernel Accuracy Heatmap (Dataset × Configuration)")
    fig.savefig(os.path.join(FIGS, "accuracy_heatmap.png"))
    plt.close(fig)
    print("  saved accuracy_heatmap.png")


def plot_timing():
    """Bar chart: mean QK computation time per dataset."""
    df = load()
    g = df.groupby("dataset")["qk_time_s"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(range(len(g)), g.values, color="#1f77b4")
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g.index)
    ax.set_xlabel("Mean Computation Time (s)")
    ax.set_title("Quantum Kernel Computation Time by Dataset")
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(os.path.join(FIGS, "timing.png"))
    plt.close(fig)
    print("  saved timing.png")


def main():
    print("Generating qkern-fyp figures...")
    plot_accuracy_by_dataset()
    plot_accuracy_vs_qubits()
    plot_feature_map_comparison()
    plot_alignment_vs_accuracy()
    plot_depth_effect()
    plot_heatmap_best_configs()
    plot_timing()
    print(f"\nAll figures saved to {FIGS}")


if __name__ == "__main__":
    main()
