import json
from pathlib import Path

import numpy as np
from torch.utils.data import Subset

BUDGETS = [10, 50, 250]
SEEDS = [1, 2, 3]
NUM_CLASSES = 10


def create_balanced_subset(dataset, budget, seed, manifest_dir="artifacts/manifests"):
    #Select a reproducible number of training images per class
    if budget not in BUDGETS:
        raise ValueError(f"Budget must be one of {BUDGETS}")

    if seed not in SEEDS:
        raise ValueError(f"Seed must be one of {SEEDS}")

    rng = np.random.default_rng(seed)
    labels = np.asarray(dataset.targets, dtype=int)

    selected_indices = []

    for class_id in range(NUM_CLASSES):
        class_indices = np.flatnonzero(labels == class_id)

        if len(class_indices) < budget:
            raise ValueError(f"Class {class_id} has fewer than {budget} images")

        chosen = rng.choice(class_indices, size=budget, replace=False)

        selected_indices.extend(chosen.tolist())

    # Shuffle selected images reproducibly
    rng.shuffle(selected_indices)
    selected_indices = [int(i) for i in selected_indices]

    # Save the selected indices for reproducibility
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / f"train_{budget}_seed{seed}.json"

    if manifest_path.exists():
        with manifest_path.open() as f:
            saved_indices = json.load(f)

        if saved_indices != selected_indices:
            raise ValueError(f"Existing manifest differs: {manifest_path}")
    else:
        with manifest_path.open("w") as f:
            json.dump(selected_indices, f, indent=2)

    return Subset(dataset, selected_indices)
