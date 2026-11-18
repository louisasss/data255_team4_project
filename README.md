# Beyond RGB: Using Extra Satellite Bands for Land-Use Classification

**DATA255 Deep Learning Final Project — Team 4**

## Project Overview

This project investigates whether using all 13 Sentinel-2 satellite bands improves land-use classification compared to standard RGB images, particularly when labeled training data is limited.

Using the EuroSAT multispectral dataset, we compare ImageNet-pretrained ResNet models trained on RGB and 13-band inputs across different label budgets. The goal is to evaluate how additional spectral information affects classification accuracy when training data is limited.

## Team Members

- Sang Ah Lee
- Natalie Leung
- Kazi Samina Maraj Mumu
- Louisa Stumpf

## Dataset

**Dataset:** EuroSAT Multispectral (EuroSAT-MS)  
**Source:** https://zenodo.org/records/7711810

The dataset contains approximately 27,000 satellite images across 10 land-use classes.

- **Image dimensions:** 64 × 64 pixels
- **RGB inputs:** B4 (Red), B3 (Green), B2 (Blue)
- **Multispectral inputs:** All 13 Sentinel-2 bands

### Dataset Splits

We use the predefined TorchGeo EuroSAT train/validation/test splits:

| Split | Number of Images |
|---|---:|
| Training | 16,200 |
| Validation | 5,400 |
| Test | 5,400 |
| **Total** | **27,000** |

The splits were verified to contain no overlapping image file paths.

Balanced training subsets are generated using **10, 50, and 250 images per class**, with random seeds **1, 2, and 3**. RGB and 13-band models use identical selected samples for each budget and seed.

Per-band normalization statistics are calculated using the training split only and applied consistently across all splits.

## Experimental Setup

The project compares three approaches:

1. **Reference baseline:** Random Forest and/or Logistic Regression using per-band statistics.
2. **Strong baseline:** ImageNet-pretrained ResNet using RGB inputs.
3. **Main investigation:** ImageNet-pretrained ResNet adapted to accept all 13 Sentinel-2 bands.

**Label budgets:** 10, 50, and 250 images per class  
**Random seeds:** 1, 2, 3  
**Primary metric:** Classification accuracy  
**Secondary metrics:** Macro-F1 and per-class recall

Additional investigations include first-layer initialization strategies, spectral band-group comparisons, and computational trade-offs.

## Project Structure

```text
data255_team4_project/
├── src/
│   ├── data.py
│   └── subsets.py
├── scripts/
│   └── check_data.py
├── artifacts/
│   ├── normalization.json
│   └── manifests/
│       ├── train_10_seed1.json
│       ├── train_10_seed2.json
│       ├── train_10_seed3.json
│       ├── train_50_seed1.json
│       ├── train_50_seed2.json
│       ├── train_50_seed3.json
│       ├── train_250_seed1.json
│       ├── train_250_seed2.json
│       └── train_250_seed3.json
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/louisasss/data255_team4_project.git
cd data255_team4_project
```

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

The EuroSAT multispectral dataset is approximately 1.93 GB and is downloaded automatically through TorchGeo when first accessed.

## Data Pipeline

The data pipeline loads EuroSAT multispectral images, selects the requested spectral bands, applies normalization, and generates balanced training subsets.

### Verify the Data Pipeline

Run:

```bash
python -m scripts.check_data
```

The verification script checks:

- RGB and 13-band image shapes and labels.
- Matching samples between RGB and multispectral datasets.
- Validation and test dataset loading.
- Saved subset manifests across label budgets and random seeds.

### Using the Dataset

```python
from src.data import get_dataset

# Training datasets: 50 images per class, seed 1
train_rgb = get_dataset("train", "rgb", 50, 1)
train_ms = get_dataset("train", "all", 50, 1)

# Validation and test datasets
val_rgb = get_dataset("val", "rgb")
test_rgb = get_dataset("test", "rgb")
```

Each dataset returns normalized image tensors and class labels.

- **RGB image shape:** `(3, 64, 64)`
- **Multispectral image shape:** `(13, 64, 64)`
- **Labels:** Integers from 0 to 9

## Reproducibility

The project uses fixed TorchGeo dataset splits, balanced label-budget sampling, and predefined random seeds to support reproducible comparisons.

Training subsets are recorded in `artifacts/manifests/`, and training-only normalization statistics are stored in `artifacts/normalization.json`.

The held-out test set is reserved for final evaluation and is not used for model selection or hyperparameter tuning.

## References

- Helber, P., et al. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.*
- Neumann, M., et al. (2019). *In-Domain Representation Learning for Remote Sensing.*
- Stewart, A. J., et al. (2022). *TorchGeo: Deep Learning with Geospatial Data.*
- Wang, Y., et al. (2023). *SSL4EO-S12.*