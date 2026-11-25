import random
import numpy as np
import torch
import yaml
from src.metrics import compute_metrics

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


def train_and_evaluate(model, train_data, val_data, config) -> dict:   # Sang Ah's contract
    """
    Called by adapter:
        1. build DataLoaders, optimizer, and loss function
        2. loop over epochs and at each epoch: 
            - train (augment each batch)
            - evaluate on validation
            - checkpoint if validation accuracy is best so far
        3. return result row, save to CSV
    """
    pass


def write_row(row: dict, path) -> None:
    """Save row to CSV"""
    pass


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from src.data import get_stub_dataset
    from src.models import build_model

    set_seed(1)
    model = build_model(3, pretrained=False).to(get_device())
    loader = DataLoader(get_stub_dataset(100, 3), batch_size=64, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(3):
        print(epoch, train_one_epoch(model, loader, optimizer, loss_fn, get_device()))
    set_seed(1)
    a = torch.randn(3)
    set_seed(1)
    b = torch.randn(3)
    set_seed(2)
    c = torch.randn(3)
    assert torch.equal(a, b) and not torch.equal(a, c)
    print("success")
    print(get_device())

    val_loader = DataLoader(get_stub_dataset(50, 3), batch_size=64)
    m = evaluate(model, val_loader, loss_fn, get_device())
    print(m["accuracy"], m["macro_f1"], m["loss"])
    print(load_train_config())
