"""Paired hypothesis evidence for the RGB-versus-13-band comparison.

The analysis uses the same seed for RGB and all-band runs, then computes
delta = accuracy(all 13 bands) - accuracy(RGB). 
It doesn't train models, and uses the agreed 10/50/250 budgets
"""
from pathlib import Path

import numpy as np
import pandas as pd


BUDGETS = (10, 50, 250)
REQUIRED_COLUMNS = {"model", "bands", "budget", "seed", "accuracy"}


def paired_deltas(results):
    # Return one RGB/all accuracy delta per model, budget, and seed
    missing = REQUIRED_COLUMNS - set(results.columns)
    if missing:
        raise ValueError(f"Result file is missing columns: {sorted(missing)}")

    comparison = results[results["bands"].isin(["rgb", "all"])].copy()
    paired = comparison.pivot_table(
        index=["model", "budget", "seed"],
        columns="bands",
        values="accuracy",
        aggfunc="first",
    ).dropna(subset=["rgb", "all"])
    if paired.empty:
        raise ValueError("No paired rgb/all accuracy results were found")

    paired = paired.reset_index()
    paired["delta_all_minus_rgb"] = paired["all"] - paired["rgb"]
    return paired


def evaluate_hypotheses(results):
    # Return per-budget deltas and transparent H0/H1/H2 evidence flags
    paired = paired_deltas(results)
    summary = (
        paired.groupby(["model", "budget"], as_index=False)["delta_all_minus_rgb"]
        .agg(delta_mean="mean", delta_std="std", n_seeds="count",
             delta_min="min", delta_max="max")
    )
    summary["h0_within_run_variation"] = (
        summary["delta_mean"].abs() <= summary["delta_std"].fillna(0)
    )
    summary["all_seeds_positive"] = summary["delta_min"] > 0

    hypothesis_rows = []
    for model, group in summary.groupby("model"):
        means = group.set_index("budget")["delta_mean"].to_dict()
        available = set(means)
        has_minimum_grid = set(BUDGETS).issubset(available)
        h1 = (
            has_minimum_grid
            and means[10] > means[50] > means[250]
            and all(means[budget] > 0 for budget in BUDGETS)
        )
        h2 = (
            has_minimum_grid
            and means[50] > means[10]
            and means[50] > means[250]
        )
        hypothesis_rows.append({
            "model": model,
            "h1_positive_and_grows_at_smaller_budgets": h1,
            "h2_mid_budget_peak_exploratory": h2,
            "h2_note": "25 and 100 budgets are absent from the minimum grid",
        })

    return summary, pd.DataFrame(hypothesis_rows), paired


def run(input_path="results/baseline_val.csv", output_dir="results"):
    # Analyze a saved result CSV and write hypothesis evidence tables
    results = pd.read_csv(input_path)
    summary, hypotheses, paired = evaluate_hypotheses(results)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "hypothesis_deltas.csv", index=False)
    hypotheses.to_csv(output_dir / "hypothesis_summary.csv", index=False)
    paired.to_csv(output_dir / "hypothesis_paired_deltas.csv", index=False)
    return summary, hypotheses, paired


if __name__ == "__main__":
    run()
