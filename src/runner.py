import random
import time
import yaml
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.metrics import compute_metrics
from src.models import count_parameters

REQUIRED_TRAIN_KEYS = ["optimizer", "lr", "weight_decay", "batch_size", "max_epochs"]


def set_seed(seed: int) -> None:
    """Reset all training random number generators to `seed`, so run is repeatable."""
    # 1. Python's random module
    random.seed(seed)
    # 2. NumPy
    np.random.seed(seed)
    # 3. PyTorch (also seeds MPS/CUDA)
    torch.manual_seed(seed)


def get_device() -> str:
    """pick mps/cuda/cpu"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_train_config(path="configs/train_config.yaml") -> dict:
    """Read training recipe from config file"""
    with open(path) as f:
        cfg = yaml.safe_load(f) #turn YAML file into python dict

    missing = [k for k in REQUIRED_TRAIN_KEYS if k not in cfg] #check all required keys are there
    
    if missing:
        raise ValueError(f"{path} is missing keys: {missing}") #raise error if required key is missing
    
    if cfg["optimizer"] != "AdamW":
        raise ValueError(f"Runner only supports AdamW, got {cfg['optimizer']}") #raise error if optimizer not equal AdamW
    
    return {k: cfg[k] for k in REQUIRED_TRAIN_KEYS} #only return needed keys, leave out descriptive fields


def augment(images: torch.Tensor) -> torch.Tensor:
    """Randomly flip each image in the batch horizontally and/or vertically."""
    images = images.clone() #make a copy so we don't change original batch in place
    batch_size = images.shape[0] #extract the number of images from batch (batch, channels, height, width)

    # horizontal flip: one coin flip per image
    flip_h = torch.rand(batch_size) < 0.5 # draws one random number between 0 and 1 for each image, turns number into True/False
    images[flip_h] = torch.flip(images[flip_h], dims=[3]) #select only images marked True, mirror them left to right (dims 3 = width), put flipped version back in original position (False stay where they are)

    # vertical flip: a separate coin flip per image
    flip_v = torch.rand(batch_size) < 0.5
    images[flip_v] = torch.flip(images[flip_v], dims=[2]) #select images marked True in flip_v, mirror them top to bottom (dims 2 = height), put back in place

    return images


def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    """train epoch on every batch"""
    model.train() #switch ResNet layers' behavior to training mode
    total_loss, seen = 0.0, 0 #set up running totals to compute avg loss

    for images, labels in loader: #go through DataLoader one batch at a time to give shape (images = (64, 13, 64, 64) and labels = (64, ))
        images = augment(images).to(device) #flip batch first, then move to GPU (only augment training data)
        labels = labels.to(device) #labels on same device as model outputs

        outputs = model(images) #forward pass (outputs shape: (64, 10), one score per class per image)
        loss = loss_fn(outputs, labels) #measure how wrong predicitons are

        #training 
        optimizer.zero_grad() #clear gradients from previous batch
        loss.backward() #compute how each weight should change to reduce loss
        optimizer.step() #update weights (AdamW)

        total_loss += loss.item() * labels.size(0) #multiply loss by batch size for total loss of that batch (compensates for uneven batch sizes)
        seen += labels.size(0) #counts how many images processed so far by this epoch
    return total_loss / seen #return avg loss per image across this epoch


@torch.no_grad() #decorator, turns off gradient tracking for function
def evaluate(model, loader, loss_fn, device) -> dict:
    """score model on validation (or test) data"""
    model.eval() #switch ResNet mode to evaluation
    total_loss, seen = 0.0, 0 # set up running totals for computing avg loss
    y_true, y_pred = [], [] #set up empty lists to store y values

    for images, labels in loader: #go through DataLoader one batch at a time to give shape
        images = images.to(device) #move batch to GPU
        labels = labels.to(device) #labels on same device as model outputs

        outputs = model(images) #forward pass
        total_loss += loss_fn(outputs, labels).item() * labels.size(0) #multiply loss by batch size for total loss of that batch (compensates for uneven batch sizes)
        seen += labels.size(0) #counts how many images processed so far by this epoch

        y_true += labels.cpu().tolist() #collect true labels across all batches; move results off GPU, and turn them into python list (expected input for compute_metrics())
        y_pred += outputs.argmax(dim=1).cpu().tolist() #prediction = class with highest score from 10 scores per image in outputs, collect predictions across all batches; move results off GPU, and turn them into python list (expected input for compute_metrics())

    metrics = compute_metrics(y_true, y_pred) #return confusion_matrix + dictionary with accuracy, macro_f1, and recall,class' for each of the 10 classes per image
    metrics["loss"] = total_loss / seen #add avg val loss to dictionary
    return metrics


def train_and_evaluate(model, train_data, val_data, config, train_cfg=None, checkpoint_dir="checkpoints") -> dict:
    """
    Train one run, keep the best-validation checkpoint, return its result row.
    Called by adapter:
        1. build DataLoaders, optimizer, and loss function
        2. loop over epochs and at each epoch: 
            - train (augment each batch)
            - evaluate on validation
            - checkpoint if validation accuracy is best so far
        3. return result row, save to CSV
    """
    #setup
    if train_cfg is None:
        train_cfg = load_train_config() #if config file missing, load training config
    device = get_device() #set device GPU
    set_seed(config.seed) #read and set seed value from ExperimentConfig

    g = torch.Generator() #create random number generator
    g.manual_seed(config.seed) #set random number generator seed to config value
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=train_cfg["batch_size"],
        shuffle=True, #shuffle order
        generator=g #generator only used in train
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=train_cfg["batch_size"],
        shuffle=False #keep fixed order so evaluation stays consistent
    )

    model = model.to(device) #move model to GPU, before optimizer is created
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]) #create AdamW optimizer using config values
    loss_fn = torch.nn.CrossEntropyLoss() #set loss function

    checkpoint_path = Path(checkpoint_dir) / f"{config.run_id}.pt" #save best weights to checkpoints/ as individual files per run_id
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True) #create folder if missing

    #training loop with val-based checkpointing
    best_acc, best_epoch, best_metrics = -1.0, -1, None #set up variables to store current best values
    start = time.perf_counter() #record training start time

    for epoch in range(train_cfg["max_epochs"]): #loop through epochs
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device) #train 
        val = evaluate(model, val_loader, loss_fn, device) #validation
        print(f"{config.run_id} epoch {epoch}: train_loss {train_loss:.4f} "
              f"val loss {val['loss']:.4f} val_acc {val['accuracy']:.4f}")

        if val["accuracy"] > best_acc: #checkpoint rule: record epoch if it beats current best score; if tie, earlier epoch kept
            best_acc, best_epoch, best_metrics = val["accuracy"], epoch, val #store accuracy, epoch, and metrics values in running variables
            torch.save(model.state_dict(), checkpoint_path) #only save weights

    train_time = time.perf_counter() - start #measure time taken by whole loop, train time includes per-epoch validation

    #build result row
    best_metrics.pop("confusion_matrix") #remove confusion matrix (array, can't go in CSV row)
    best_metrics["val_loss"] = best_metrics.pop("loss") #rename 'loss' for clarity

    return {
        "run_id": config.run_id,
        "model": "resnet18", #hard-coded for now, change once ExperimentConfig is updated.
        "bands": config.bands,
        "budget": config.budget,
        "seed": config.seed,
        "eval_split": config.eval_split,
        "best_val_epoch": best_epoch,
        **best_metrics, #unpacks dictionary into CSV row
        "train_time_s": train_time,
        "params": count_parameters(model),
        "device": device,
        "checkpoint": str(checkpoint_path)
    }

def write_row(row: dict, path) -> None:
    """Save row to CSV"""
    path = Path(path) #accept plain string or path and convert to Path
    path.parent.mkdir(parents=True, exist_ok=True) #create results/runs/ if it doesn't exist

    with path.open("w", newline="") as f: #write to path file, overwrites existing file so no duplicates are added
        writer = csv.DictWriter(f, fieldnames=list(row.keys())) #write dictionary as one CSV row, keys = column names
        writer.writeheader() #write column header row
        writer.writerow(row) #write values


if __name__ == "__main__":
    from src.data import get_stub_dataset
    from src.experiment_grid import ExperimentConfig
    from src.main_comparison import prepare_run

    config = ExperimentConfig("rgb", 10, 1)
    no_data = lambda *args, **kwargs: None

    # 1. the adapter seeds the model: same seed -> same starting weights
    w1 = prepare_run(config, dataset_loader=no_data)["model"].fc.weight
    w2 = prepare_run(config, dataset_loader=no_data)["model"].fc.weight
    assert torch.equal(w1, w2), "adapter does not seed model init"

    # 2. end-to-end run on stub data, 2 epochs
    model = prepare_run(config, dataset_loader=no_data)["model"]
    train_cfg = {**load_train_config(), "max_epochs": 2}
    row = train_and_evaluate(model, get_stub_dataset(100, 3), get_stub_dataset(50, 3),
                             config, train_cfg=train_cfg, checkpoint_dir="checkpoints/smoke")

    # 3. the row has the columns compare_results.py needs
    required = {"model", "bands", "budget", "seed", "accuracy", "macro_f1"}
    assert required <= set(row), f"missing columns: {required - set(row)}"

    print("runner smoke test passed")

