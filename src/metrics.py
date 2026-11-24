import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(CLASSES)))

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }

    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for name, r in zip(CLASSES, recalls):
        results[f"recall_{name}"] = r

    results["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels)
    return results


def bootstrap_accuracy_ci(y_true, y_pred, n_boot=1000, seed=0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores.append((y_true[idx] == y_pred[idx]).mean())

    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


if __name__ == "__main__":
    # Fake data only, to test that the code runs. NOT real results.
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 10, size=200)
    y_pred = y_true.copy()
    wrong = rng.random(200) < 0.3
    y_pred[wrong] = rng.integers(0, 10, size=wrong.sum())

    res = compute_metrics(y_true, y_pred)
    for k, v in res.items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.3f}")
    print(res["confusion_matrix"])
    print("95% CI for accuracy:", bootstrap_accuracy_ci(y_true, y_pred))
