
from torchgeo.datasets import EuroSAT
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from src.subsets import create_balanced_subset

def load_eurosat(split="train", root="data"):
    #  Load the EuroSAT multispectral dataset and split into train, validation, and test sets
    dataset = EuroSAT(root=root, split=split, download=True)
    return dataset

# RGB uses Red (B4), Green (B3), Blue (B2), so mapping correct indices for the 13-band image to RGB is [3, 2, 1]
RGB_IDX = [3, 2, 1]

def select_bands(image, bands="rgb"):
    if bands == "rgb":
        return image[RGB_IDX]

    elif bands == "all":
        return image

    else:
        raise ValueError(f"Unsupported band selection: {bands}")

def compute_normalization_stats(dataset, batch_size=64):
    # Compute per-band mean and std from training images

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    channel_sum = torch.zeros(13, dtype=torch.float64)
    channel_squared_sum = torch.zeros(13, dtype=torch.float64)
    total_pixels = 0

    for batch in loader:
        images = batch["image"].to(torch.float64)

        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += (images ** 2).sum(dim=(0, 2, 3))

        total_pixels += images.shape[0] * images.shape[2] * images.shape[3]

    mean = channel_sum / total_pixels
    variance = channel_squared_sum / total_pixels - mean ** 2
    std = torch.sqrt(variance.clamp(min=0))

    return mean.tolist(), std.tolist()

def save_normalization_stats(mean, std, path="artifacts/normalization.json"):
    # normalization statistics for reuse

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump({"mean": mean, "std": std}, f, indent=2)

class EuroSATProcessed(Dataset):
    # Select bands and normalize EuroSAT images
    def __init__(self, dataset, bands, mean, std):
        self.dataset = dataset
        self.bands = bands

        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        image = sample["image"].float()
        label = int(sample["label"])

        # Select RGB or all 13 bands
        image = select_bands(image, self.bands)

        # Select the matching normalization statistics
        if self.bands == "rgb":
            mean = self.mean[RGB_IDX]
            std = self.std[RGB_IDX]
        else:
            mean = self.mean
            std = self.std

        # Normalize the image
        image = (image - mean[:, None, None]) / std[:, None, None]
        return image, label


def get_dataset(split, bands, budget=None, seed=1):
    # Return a normalized EuroSAT dataset

    if split not in ("train", "val", "test"):
        raise ValueError(f"Unsupported split: {split}")

    if split != "train" and budget is not None:
        raise ValueError("Label budgets can only be used with training data")

    dataset = load_eurosat(split=split)

    # Select balanced training subset
    if split == "train" and budget is not None:
        dataset = create_balanced_subset(
            dataset,
            budget=budget,
            seed=seed
        )

    # Load normalization statistics
    with open("artifacts/normalization.json") as f:
        stats = json.load(f)

    return EuroSATProcessed(
        dataset=dataset,
        bands=bands,
        mean=stats["mean"],
        std=stats["std"]
    )

if __name__ == "__main__":
    train_dataset = load_eurosat(split="train")

    mean, std = compute_normalization_stats(train_dataset)

    save_normalization_stats(mean, std)

    print("Band means:", mean)
    print("Band stds:", std)
    print("Saved to data/normalization.json")