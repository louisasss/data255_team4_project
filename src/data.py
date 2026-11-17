
from torchgeo.datasets import EuroSAT

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


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        dataset = load_eurosat(split=split)

        sample = dataset[0]
        image = sample["image"]
        label = sample["label"]

        print(f"\n{split.upper()} SPLIT")
        print("Number of images:", len(dataset))
        print("Image shape:", image.shape)
        print("Label:", label.item())