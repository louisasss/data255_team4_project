#Adapter for connecting the experiment grid to the shared training runner.
from src.experiment_grid import ExperimentConfig

def channels_for(bands):
    if bands == "rgb":
        return 3
    if bands == "all":
        return 13
    raise ValueError(f"Unsupported band setting: {bands}")

def prepare_run(config, dataset_loader=None, model_builder=None):
    # Build one run's datasets and model; the caller still owns training.
    if dataset_loader is None:
        from src.data import get_dataset
        dataset_loader = get_dataset
    if model_builder is None:
        from src.models import build_model
        model_builder = build_model

    train_data = dataset_loader("train", config.bands, config.budget, config.seed)
    val_data = dataset_loader(config.eval_split, config.bands)
    model = model_builder(
        in_channels=channels_for(config.bands),
        init_strategy="mean_rgb_rescaled",
        pretrained=True,
    )
    return {
        "run_id": config.run_id,
        "config": config,
        "model": model,
        "train_data": train_data,
        "val_data": val_data,
    }


def run_one(config: ExperimentConfig, runner, dataset_loader=None, model_builder=None):
    """Pass one prepared run to the shared runner.
    The runner should accept keyword arguments named ``model``, ``train_data``,
    ``val_data``, and ``config`` and return its result row. This is the small
    contract to confirm with the runner owner.
    """
    prepared = prepare_run(config, dataset_loader, model_builder)
    return runner(
        model=prepared["model"],
        train_data=prepared["train_data"],
        val_data=prepared["val_data"],
        config=config,
    )
