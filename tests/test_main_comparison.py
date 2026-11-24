# tests experiment_grid and compare_results modules for the main comparison
import pandas as pd
import pytest

from src.compare_results import summarize_results
from src.experiment_grid import main_comparison_configs


def test_main_comparison_grid_has_18_unique_runs():
    configs = main_comparison_configs()

    assert len(configs) == 18
    assert len({config.run_id for config in configs}) == 18
    assert {config.bands for config in configs} == {"rgb", "all"}
    assert {config.budget for config in configs} == {10, 50, 250}
    assert {config.seed for config in configs} == {1, 2, 3}


def test_compare_results_writes_rgb_all_summaries(tmp_path):
    rows = []
    for model in ["test_model"]:
        for bands in ["rgb", "all"]:
            for budget in [10, 50, 250]:
                for seed in [1, 2, 3]:
                    rows.append({
                        "model": model,
                        "bands": bands,
                        "budget": budget,
                        "seed": seed,
                        "accuracy": 0.5 if bands == "rgb" else 0.6,
                        "macro_f1": 0.4 if bands == "rgb" else 0.55,
                        "recall_AnnualCrop": 0.5 if bands == "rgb" else 0.7,
                    })

    input_path = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    summary, paired, per_class = summarize_results(input_path, tmp_path / "out")

    assert len(summary) == 6
    assert len(paired) == 9
    assert paired["accuracy_all_minus_rgb"].tolist() == pytest.approx([0.1] * 9)
    assert "recall_AnnualCrop_mean" in per_class.columns
    assert (tmp_path / "out" / "rgb_vs_all_summary.csv").exists()
