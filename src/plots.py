"""Shared plotting helpers. Run from the project root:
    python -m src.plots
Reads results/baseline_val.csv and writes figures to results/figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Fixed colors per band group (same color = same band group in every figure)
BAND_STYLE = {
    "rgb":     {"color": "#2a78d6", "marker": "o", "label": "RGB"},
    "rgb_nir": {"color": "#eb6834", "marker": "s", "label": "RGB+NIR"},
    "all":     {"color": "#1baf7a", "marker": "^", "label": "All 13 bands"},
}
MODEL_NAMES = {"random_forest": "Random forest", "logreg": "Logistic regression"}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def plot_accuracy_vs_budget(csv_path="results/baseline_val.csv",
                            out_path="results/figures/baseline_acc_vs_budget.png"):
    df = pd.read_csv(csv_path)
    summary = (df.groupby(["model", "bands", "budget"])["accuracy"]
                 .agg(["mean", "std"]).reset_index())

    models = [m for m in MODEL_NAMES if m in summary["model"].unique()]
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.2), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        for bands, style in BAND_STYLE.items():
            d = summary[(summary.model == model) & (summary.bands == bands)].sort_values("budget")
            if d.empty:
                continue
            ax.errorbar(d["budget"], d["mean"], yerr=d["std"],
                        color=style["color"], marker=style["marker"], markersize=7,
                        linewidth=2, capsize=3, label=style["label"])
            last = d.iloc[-1]
            ax.annotate(style["label"], (last["budget"], last["mean"]),
                        xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=TEXT)

        ax.set_xscale("log")
        budgets = sorted(summary["budget"].unique())
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.minorticks_off()
        ax.set_xlim(budgets[0] * 0.8, budgets[-1] * 2.6)
        ax.set_title(MODEL_NAMES[model], fontsize=11, color=TEXT, loc="left")
        ax.set_xlabel("Labeled training images per class", color=MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED)

    axes[0].set_ylabel("Validation accuracy (mean ± std, 3 seeds)", color=MUTED)
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("Reference baseline: accuracy vs. label budget by band group",
                 x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")
    return summary


if __name__ == "__main__":
    summary = plot_accuracy_vs_budget()
    print(summary.round(3).to_string(index=False))
