import torch
from src.data import get_dataset, RGB_IDX, load_eurosat
import json
from pathlib import Path
from src.subsets import create_balanced_subset

# Load the same training subset in two formats
rgb_dataset = get_dataset("train", "rgb", 50, 1)
ms_dataset = get_dataset("train", "all", 50, 1)

rgb_image, rgb_label = rgb_dataset[0]
ms_image, ms_label = ms_dataset[0]

print("RGB dataset size:", len(rgb_dataset))
print("13-band dataset size:", len(ms_dataset))
print("RGB image shape:", rgb_image.shape)
print("13-band image shape:", ms_image.shape)
print("Labels match:", rgb_label == ms_label)

assert len(rgb_dataset) == 500
assert len(ms_dataset) == 500
assert rgb_image.shape == (3, 64, 64)
assert ms_image.shape == (13, 64, 64)
assert rgb_label == ms_label

# RGB channels should match the corresponding 13-band channels
assert torch.allclose(rgb_image, ms_image[RGB_IDX])

print("\nShared dataset interface works")

# Check validation and test datasets
for split in ["val", "test"]:
    dataset = get_dataset(
        split=split,
        bands="all",
        budget=None,
        seed=1
    )

    assert len(dataset) == 5400
    image, label = dataset[0]
    assert image.shape == (13, 64, 64)
    assert 0 <= label < 10
    print(f"{split.upper()} split: PASSED")

# Verify reproducibility of saved subsets
train = load_eurosat("train")

for budget in [10, 50, 250]:
    for seed in [1, 2, 3]:
        subset = create_balanced_subset(train, budget, seed)

        path = Path(f"artifacts/manifests/train_{budget}_seed{seed}.json")

        with path.open() as f:
            saved_indices = json.load(f)

        assert list(subset.indices) == saved_indices
        print(f"Manifest {budget}, seed {seed}: PASSED")

print("\nAll saved manifests verified")