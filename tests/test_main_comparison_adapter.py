from src.experiment_grid import ExperimentConfig
from src.main_comparison import channels_for, prepare_run, run_one


def test_channels_for_rgb_and_all_bands():
    assert channels_for("rgb") == 3
    assert channels_for("all") == 13


def test_run_adapter_prepares_inputs_without_training():
    config = ExperimentConfig("all", 50, 1)
    calls = {}

    def fake_dataset(split, bands, budget=None, seed=1):
        return {"split": split, "bands": bands, "budget": budget, "seed": seed}

    def fake_model(**kwargs):
        calls["model"] = kwargs
        return "fake-model"

    prepared = prepare_run(config, fake_dataset, fake_model)
    assert prepared["run_id"] == "all_b50_s1"
    assert prepared["train_data"]["budget"] == 50
    assert prepared["val_data"]["budget"] is None
    assert calls["model"]["in_channels"] == 13

    result = run_one(
        config,
        lambda **kwargs: kwargs["config"].run_id,
        fake_dataset,
        fake_model,
    )
    assert result == "all_b50_s1"
