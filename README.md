# Beyond RGB: Using Extra Satellite Bands for Land-Use Classification

DATA255 Deep Learning Final Project — Team 4

## Project Overview

This project investigates whether using all 13 Sentinel-2 satellite bands improves land-use classification compared to standard RGB images, particularly when labeled training data is limited.

We use the EuroSAT multispectral dataset and ImageNet-pretrained ResNet models to compare classification performance across different label budgets.

## Team Members

- Sang Ah Lee
- Natalie Leung
- Kazi Samina Maraj Mumu
- Louisa Stumpf

## Dataset

**Dataset:** EuroSAT Multispectral (EuroSAT-MS)

**Source:** https://zenodo.org/records/7711810

- Approximately 27,000 images
- 10 land-use classes
- Image dimensions: 64 × 64
- RGB inputs: B4, B3, B2
- Multispectral inputs: All 13 Sentinel-2 bands

We use the predefined TorchGeo train/validation/test splits.

## Experimental Setup

We compare:

1. Reference baseline: Random Forest and/or Logistic Regression
2. Strong baseline: ImageNet-pretrained RGB ResNet
3. Main investigation: ImageNet-pretrained 13-band ResNet

**Label budgets:** 10, 50, and 250 images per class

**Random seeds:** 1, 2, 3

**Primary metric:** Classification accuracy

**Secondary metrics:** Macro-F1 and per-class recall

Additional experiments investigate first-layer initialization strategies, different spectral band groups, and pretrained remote-sensing models where feasible.

## Project Structure

```text
data255_team4_project/
├── src/
│   ├── data.py
│   └── subsets.py
├── scripts/
│   └── check_data.py
├── configs/
├── results/
├── requirements.txt
├── README.md
└── .gitignore
```

The structure will be updated as implementation progresses.

## Installation

Clone the repository:

```bash
git clone https://github.com/louisasss/data255_team4_project.git
cd data255_team4_project
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Project

Training, evaluation, and experiment commands will be added as the implementation is completed.

## Project Status

**In development**

Current work includes dataset preprocessing, model implementation, reference baselines, experiment infrastructure, and evaluation utilities.

## References

- Helber et al. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.
- Neumann et al. (2019). In-Domain Representation Learning for Remote Sensing.
- Stewart et al. (2022). TorchGeo: Deep Learning with Geospatial Data.
- Wang et al. (2023). SSL4EO-S12.