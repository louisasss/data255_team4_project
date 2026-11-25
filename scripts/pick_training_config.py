import argparse
import csv
import time
from pathlib import Path
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from src.data import get_dataset
from src.metrics import compute_metrics
from src.models import build_model, count_parameters

FIXED = {
    "model": "resnet18",
    "pretrained": True,
    "fine_tuning": "all layers",
    "optimizer": "AdamW",
    "weight_decay": 0.0001,
    "batch_size": 64,
    "max_epochs": 10,
    "augmentation": "random horizontal and vertical flips (train only)",
    "checkpoint_rule": "epoch with the best validation accuracy",
    "init_strategy_for_13_bands": "mean_rgb_rescaled",
}

LEARNING_RATES = [0.0001, 0.0003, 0.001]

MODELS_TO_TRY = [
    ("rgb", 3, None),
    ("all", 13, FIXED["init_strategy_for_13_bands"]),
]

SEED = 1

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def augment(images):
    # flips do not change the land-use class
    if torch.rand(1).item() < 0.5:
        images = torch.flip(images, dims=[3])
    if torch.rand(1).item() < 0.5:
        images = torch.flip(images, dims=[2])
    return images

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, seen = 0.0, 0

    for images, labels in loader:
        images = augment(images).to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        seen += labels.size(0)

    return total_loss / seen

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, seen = 0.0, 0
    y_true, y_pred = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        total_loss += loss_fn(outputs, labels).item() * labels.size(0)
        seen += labels.size(0)
        y_true += labels.cpu().tolist()
        y_pred += outputs.argmax(dim=1).cpu().tolist()

    metrics = compute_metrics(y_true, y_pred)
    return total_loss / seen, metrics["accuracy"], metrics["macro_f1"]

def run_one(bands, in_channels, init_strategy, lr, train_set, val_set, epochs, device):
    torch.manual_seed(SEED)
    model = build_model(in_channels, init_strategy, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=FIXED["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_set, batch_size=FIXED["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=FIXED["batch_size"], shuffle=False, num_workers=0)
    best_val_acc, best_val_f1, best_epoch = 0.0, 0.0, -1
    start = time.time()

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, loss_fn, device)

        # keep the epoch with the best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc, best_val_f1, best_epoch = val_acc, val_f1, epoch
        print(f"  epoch {epoch:2d}   train_loss {train_loss:.4f}   val_loss {val_loss:.4f}   "
              f"val_acc {val_acc:.4f}   val_f1 {val_f1:.4f}")
    return {
        "bands": bands,
        "init_strategy": init_strategy,
        "lr": lr,
        "best_val_acc": round(best_val_acc, 4),
        "best_val_macro_f1": round(best_val_f1, 4),
        "best_epoch": best_epoch,
        "epochs_run": epochs,
        "train_minutes": round((time.time() - start) / 60, 2),
        "parameters": count_parameters(model),
        "device": device,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=FIXED["max_epochs"])
    args = parser.parse_args()
    device = get_device()
    epochs = 2 if args.quick else args.epochs
    print("Device:", device)
    rows = []
    
    for bands, in_channels, init_strategy in MODELS_TO_TRY:
        if args.quick:
            #10 images per class and 200 validation images
            train_set = get_dataset("train", bands, 10, SEED)
            full_val = get_dataset("val", bands)
            val_set = Subset(full_val, list(range(0, len(full_val), 27)))
        else:
            #full training split and full validation split
            train_set = get_dataset("train", bands)
            val_set = get_dataset("val", bands)

        for lr in LEARNING_RATES:
            print(f"\nbands={bands}  init={init_strategy}  lr={lr}")
            row = run_one(bands, in_channels, init_strategy, lr, train_set, val_set, epochs, device)
            rows.append(row)

    #pick the learning rate with the best average of rgb and 13-band validation accuracy
    print("\nResults (validation accuracy):")
    best_lr, best_score = None, -1.0
    for lr in LEARNING_RATES:
        scores = [row["best_val_acc"] for row in rows if row["lr"] == lr]
        average = sum(scores) / len(scores)
        print(f"  lr={lr}: rgb={scores[0]:.4f}  all={scores[1]:.4f}  average={average:.4f}")
        if average > best_score:
            best_lr, best_score = lr, average

    print(f"\nChosen learning rate: {best_lr} (average validation accuracy {best_score:.4f})")
    suffix = "_quick" if args.quick else ""
    results_path = Path(f"results/lr_search{suffix}.csv")
    config_path = Path(f"configs/train_config{suffix}.yaml")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    train_config = dict(FIXED)
    train_config["max_epochs"] = epochs
    train_config["lr"] = best_lr
    train_config["selected_by"] = "best average validation accuracy of rgb and 13-band models, full training data, seed 1"
    train_config["learning_rates_tried"] = LEARNING_RATES
    with config_path.open("w") as f:
        yaml.safe_dump(train_config, f, sort_keys=False)
    print("Saved", results_path)
    print("Saved", config_path)

if __name__ == "__main__":
    main()