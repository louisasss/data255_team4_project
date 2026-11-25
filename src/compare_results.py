"""Summarize the RGB-versus-13-band main comparison.

Reads saved result CSVs only and doesn't load data, build models,
or start training. """
from pathlib import Path

import pandas as pd


COMPARISON_BANDS = ("rgb", "all")
REQUIRED_COLUMNS = {"model", "bands", "budget", "seed", "accuracy", "macro_f1"}

def summarize_results(input_path="results/baseline_val.csv", output_dir="results"):
    # Write overall, paired-seed, and per-class RGB/all summaries
    df = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Result file is missing columns: {sorted(missing)}")

    comparison = df[df["bands"].isin(COMPARISON_BANDS)].copy()
    if comparison.empty:
        raise ValueError("Result file contains no rgb/all comparison rows")

    summary = (
        comparison.groupby(["model", "bands", "budget"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            n_seeds=("seed", "nunique"),
        )
    )

    paired = comparison.pivot_table(
        index=["model", "budget", "seed"],
        columns="bands",
        values=["accuracy", "macro_f1"],
        aggfunc="first",
    ).dropna()
    paired.columns = [f"{metric}_{bands}" for metric, bands in paired.columns]
    paired = paired.reset_index()
    paired["accuracy_all_minus_rgb"] = paired["accuracy_all"] - paired["accuracy_rgb"]
    paired["macro_f1_all_minus_rgb"] = paired["macro_f1_all"] - paired["macro_f1_rgb"]

    recall_columns = [column for column in comparison.columns if column.startswith("recall_")]
    per_class = pd.DataFrame()
    if recall_columns:
        per_class = (
            comparison.groupby(["model", "bands", "budget"], as_index=False)[recall_columns]
            .agg(["mean", "std"])
        )
        per_class.columns = [
            "_".join(part for part in column if part)
            if isinstance(column, tuple) else column
            for column in per_class.columns
        ]
        per_class = per_class.reset_index(drop=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "rgb_vs_all_summary.csv"
    paired_path = output_dir / "rgb_vs_all_paired.csv"
    per_class_path = output_dir / "per_class_recall_summary.csv"
    summary.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)
    if not per_class.empty:
        per_class.to_csv(per_class_path, index=False)

    print(f"Saved {summary_path}")
    print(f"Saved {paired_path}")
    if not per_class.empty:
        print(f"Saved {per_class_path}")
    return summary, paired, per_class


if __name__ == "__main__":
    summarize_results()
