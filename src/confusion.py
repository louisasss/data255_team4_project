"""Confusion matrices for the reference baseline (validation set).
Run from the project root:
    python -m src.confusion
Retrains ONE setting (default: logreg, 50 images/class, seed 1) for each band group,
saves the predictions, and draws row-normalized confusion matrices side by side.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from src.baseline import extract_features, select_bands, make_model
from src.metrics import compute_metrics

SHORT_NAMES = ["AnnualCrop", "Forest", "HerbVeg", "Highway", "Industrial",
               "Pasture", "PermCrop", "Residential", "River", "SeaLake"]
TITLES = {"rgb": "RGB", "rgb_nir": "RGB+NIR", "all": "All 13 bands"}
CMAP = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def plot_confusion_grid(results, title, out_path):
    """results: {band_set: (confusion_matrix, accuracy)}"""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 6.0))
    if n == 1:
        axes = [axes]

    for ax, (band_set, (cm, acc)) in zip(axes, results.items()):
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        ax.imshow(cm_pct, cmap=CMAP, vmin=0, vmax=100)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = cm_pct[i, j]
                if v >= 1:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if v > 55 else "#0b0b0b")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(SHORT_NAMES, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(SHORT_NAMES, fontsize=8)
        ax.set_xlabel("Predicted class", color="#52514e")
        ax.set_ylabel("True class", color="#52514e")
        ax.set_title(f"{TITLES.get(band_set, band_set)}  (accuracy {acc:.3f})",
                     fontsize=11, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)

    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def run(get_dataset, model_name="logreg", budget=50, seed=1,
        band_sets=("rgb", "rgb_nir", "all"), split="val"):
    print(f"Extracting {split} features (takes a few minutes)...")
    X_eval_all, y_eval = extract_features(get_dataset(split, "all"))
    X_train_all, y_train = extract_features(get_dataset("train", "all", budget, seed))

    pred_dir = Path("results/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for band_set in band_sets:
        model = make_model(model_name, seed)
        model.fit(select_bands(X_train_all, band_set), y_train)
        y_pred = model.predict(select_bands(X_eval_all, band_set))

        np.savez(pred_dir / f"{model_name}_{band_set}_b{budget}_s{seed}_{split}.npz",
                 y_true=y_eval, y_pred=y_pred)
        m = compute_metrics(y_eval, y_pred)
        results[band_set] = (m["confusion_matrix"], m["accuracy"])
        print(f"{band_set:8s} accuracy={m['accuracy']:.3f}  "
              f"(should match results/baseline_val.csv for this setting)")

    plot_confusion_grid(
        results,
        title=f"Reference baseline confusion matrices (row %, {split} set) - "
              f"{model_name}, {budget} images/class, seed {seed}",
        out_path=f"results/figures/confusion_{model_name}_b{budget}_s{seed}_{split}.png")


if __name__ == "__main__":
    from src.data import get_dataset
    run(get_dataset)
