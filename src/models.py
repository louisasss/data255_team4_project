import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

NUM_CLASSES = 10

def load_resnet18(pretrained):
    #imagenet weights or random weights
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        model = resnet18(weights=None)

    #replace the 1000-class imagenet head with a 10-class head
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def build_model(in_channels, init_strategy=None, pretrained=True):
    if in_channels != 3:
        raise ValueError(f"in_channels must be 3, got {in_channels}")
    return load_resnet18(pretrained)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
