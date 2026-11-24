"""Check whether confusion-matrix patterns hold across ALL budgets and seeds
(reference baseline, validation set).
Run from the project root:
    python -m src.confusion_check
Saves results/confusion_check_val.csv and prints a summary.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline import extract_features, select_bands, make_model, BUDGETS, SEEDS, BAND_SETS
from src.metrics import compute_metrics, CLASSES

# (true class, predicted class) mistakes to track, as % of the true class
PAIRS = [("Forest", "SeaLake"), ("Highway", "River"), ("PermanentCrop", "AnnualCrop")]


def run(get_dataset, model_names=("logreg", "random_forest"), split="val",
        out_path="results/confusion_check_val.csv"):
    print(f"Extracting {split} features (takes a few minutes)...")
    X_eval_all, y_eval = extract_features(get_dataset(split, "all"))
    idx = {c: i for i, c in enumerate(CLASSES)}

    rows = []
    for budget in BUDGETS:
        for seed in SEEDS:
            X_train_all, y_train = extract_features(get_dataset("train", "all", budget, seed))
            for model_name in model_names:
                for band_set in BAND_SETS:
                    model = make_model(model_name, seed)
                    model.fit(select_bands(X_train_all, band_set), y_train)
                    y_pred = model.predict(select_bands(X_eval_all, band_set))
                    m = compute_metrics(y_eval, y_pred)
                    cm = m["confusion_matrix"]
                    row_pct = cm / cm.sum(axis=1, keepdims=True) * 100

                    row = {"model": model_name, "bands": band_set, "budget": budget,
                           "seed": seed, "accuracy": m["accuracy"]}
                    for t, p in PAIRS:
                        row[f"{t}->{p} %"] = row_pct[idx[t], idx[p]]
                    for c in CLASSES:
                        row[f"recall_{c}"] = m[f"recall_{c}"]
                    rows.append(row)
            print(f"done budget={budget} seed={seed}")

    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}\n")

    cols = [f"{t}->{p} %" for t, p in PAIRS]
    summary = (df.groupby(["model", "budget", "bands"])[cols]
                 .agg(["mean", "min", "max"]).round(1))
    pd.set_option("display.width", 200)
    print("Mistake rates (% of the true class), across 3 seeds:")
    print(summary.to_string())


if __name__ == "__main__":
    from src.data import get_dataset
    run(get_dataset)
