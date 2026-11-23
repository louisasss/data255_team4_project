"""Reference baseline: Random Forest and Logistic Regression on per-band statistics.
Run from the project root:  python -m src.baseline
Evaluates on the VALIDATION set. The test set is not used until Week 6.
"""
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.metrics import compute_metrics, bootstrap_accuracy_ci, CLASSES

BUDGETS = [10, 50, 250]
SEEDS = [1, 2, 3]
BAND_SETS = {
    "rgb": [3, 2, 1],          # B04, B03, B02
    "all": list(range(13)),
}


def extract_features(dataset):
    """Mean, std, min, max of each of the 13 bands -> (N, 13, 4)."""
    feats, labels = [], []
    for i in range(len(dataset)):
        image, label = dataset[i]
        x = image.numpy().reshape(image.shape[0], -1)
        stats = np.stack([x.mean(1), x.std(1), x.min(1), x.max(1)], axis=1)
        feats.append(stats)
        labels.append(label)
    return np.stack(feats), np.array(labels)


def select_bands(feats, band_set):
    idx = BAND_SETS[band_set]
    return feats[:, idx, :].reshape(len(feats), -1)


def make_model(name, seed):
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    raise ValueError(name)


def run(get_dataset, eval_split="val", out_path="results/baseline_val.csv"):
    print(f"Extracting {eval_split} features (takes a few minutes)...")
    X_eval_all, y_eval = extract_features(get_dataset(eval_split, "all"))

    rows = []
    for budget in BUDGETS:
        for seed in SEEDS:
            X_train_all, y_train = extract_features(get_dataset("train", "all", budget, seed))
            for band_set in BAND_SETS:
                X_train = select_bands(X_train_all, band_set)
                X_eval = select_bands(X_eval_all, band_set)
                for model_name in ["random_forest", "logreg"]:
                    model = make_model(model_name, seed)
                    start = time.perf_counter()
                    model.fit(X_train, y_train)
                    train_time = time.perf_counter() - start

                    y_pred = model.predict(X_eval)
                    m = compute_metrics(y_eval, y_pred)
                    low, high = bootstrap_accuracy_ci(y_eval, y_pred)

                    rows.append({
                        "model": model_name, "bands": band_set,
                        "budget": budget, "seed": seed, "eval_split": eval_split,
                        "accuracy": m["accuracy"], "acc_ci_low": low, "acc_ci_high": high,
                        "macro_f1": m["macro_f1"],
                        **{f"recall_{c}": m[f"recall_{c}"] for c in CLASSES},
                        "n_features": X_train.shape[1], "train_time_s": train_time,
                    })
                    print(f"{model_name:13s} {band_set:3s} budget={budget:3d} seed={seed} "
                          f"acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    from src.data import get_dataset
    run(get_dataset)
