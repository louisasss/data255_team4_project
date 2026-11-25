# Reproducible experiment definitions for the main comparison that defines the RGB-versus-13-band experiment matrix. Training remains
# in the shared runner once the model and runner interfaces are finalized.
from dataclasses import asdict, dataclass

BANDS = ("rgb", "all")
BUDGETS = (10, 50, 250)
SEEDS = (1, 2, 3)

@dataclass(frozen=True)
class ExperimentConfig:
    bands: str
    budget: int
    seed: int
    eval_split: str = "val"

    @property
    def run_id(self):
        return f"{self.bands}_b{self.budget}_s{self.seed}"

    def as_dict(self):
        values = asdict(self)
        values["run_id"] = self.run_id
        return values

def main_comparison_configs():
    #Return all 18 RGB/13-band comparison configurations
    return [
        ExperimentConfig(bands=bands, budget=budget, seed=seed)
        for bands in BANDS
        for budget in BUDGETS
        for seed in SEEDS
    ]


if __name__ == "__main__":
    for config in main_comparison_configs():
        print(config.run_id, config.as_dict())
