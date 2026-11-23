"""Representative examples: one validation image per class, shown as
true-color RGB (B04, B03, B02) and as the near-infrared band (B08).
Run from the project root:
    python -m src.examples
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.metrics import CLASSES

RGB_IDX = [3, 2, 1]   # B04, B03, B02
NIR_IDX = 7           # B08


def stretch(x, lo, hi):
    """Scale to 0-1 for display only. The same lo/hi is used for every image,
    so brightness can be compared across classes."""
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)


def pick_one_per_class(dataset, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.asarray(dataset.targets, dtype=int)
    return [int(rng.choice(np.flatnonzero(labels == c))) for c in range(len(CLASSES))]


def plot_examples(dataset, indices, out_path="results/figures/examples_rgb_nir_val.png"):
    n = len(indices)
    samples = [dataset[i] for i in indices]
    imgs = [s["image"].numpy().astype(np.float32) for s in samples]   # raw values, (13, 64, 64)
    # one shared display range per band, from the 2nd-98th percentile of all shown images
    rng_ = {b: np.percentile(np.stack([im[b] for im in imgs]), [2, 98])
            for b in RGB_IDX + [NIR_IDX]}
    fig, axes = plt.subplots(2, n, figsize=(1.7 * n, 4.0))
    for col, idx in enumerate(indices):
        img = imgs[col]
        label = int(samples[col]["label"])

        rgb = np.stack([stretch(img[b], *rng_[b]) for b in RGB_IDX], axis=-1)
        nir = stretch(img[NIR_IDX], *rng_[NIR_IDX])

        axes[0, col].imshow(rgb)
        axes[0, col].set_title(CLASSES[label], fontsize=8)
        axes[1, col].imshow(nir, cmap="gray")
        for r in range(2):
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])
            axes[r, col].set_xlabel(f"val #{idx}" if r == 1 else "", fontsize=6, color="#52514e")

    axes[0, 0].set_ylabel("RGB\n(B4, B3, B2)", fontsize=9)
    axes[1, 0].set_ylabel("Near-infrared\n(B8)", fontsize=9)
    fig.suptitle("EuroSAT validation examples, one per class "
                 "(same display scale for all images in a row)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    from src.data import load_eurosat
    val = load_eurosat("val")
    indices = pick_one_per_class(val, seed=0)
    print("Validation indices used:", indices)
    plot_examples(val, indices)
