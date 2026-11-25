import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from src.data import RGB_IDX

NUM_CLASSES = 10
INIT_STRATEGIES = ["random_new", "mean_rgb_rescaled", "learned_1x1"]

def load_resnet18(pretrained):
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def make_13_band_conv(old_conv, init_strategy):
    new_conv = nn.Conv2d(13, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        old_weight = old_conv.weight

        #band starts as the average rgb filter
        if init_strategy == "mean_rgb_rescaled":
            mean_filter = old_weight.mean(dim=1)
            for band in range(13):
                new_conv.weight[:, band] = mean_filter
        for rgb_channel in range(3):
            band = RGB_IDX[rgb_channel]
            new_conv.weight[:, band] = old_weight[:, rgb_channel]

        #13 inputs so scale down to keep the output size similar
        if init_strategy == "mean_rgb_rescaled":
            new_conv.weight *= 3 / 13

    return new_conv

class BandMixerResNet(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.mixer = nn.Conv2d(13, 3, kernel_size=1, bias=False)
        self.resnet = resnet

        #start by just picking b4, b3, b2 so it begins like the rgb model
        with torch.no_grad():
            self.mixer.weight.zero_()
            for rgb_channel in range(3):
                band = RGB_IDX[rgb_channel]
                self.mixer.weight[rgb_channel, band, 0, 0] = 1.0

    def forward(self, x):
        x = self.mixer(x)
        return self.resnet(x)

def build_model(in_channels, init_strategy=None, pretrained=True):
    if in_channels not in (3, 13):
        raise ValueError(f"in_channels must be 3 or 13, got {in_channels}")

    model = load_resnet18(pretrained)

    #rgb: the pretrained model already takes 3 channels
    if in_channels == 3:
        return model

    #13 bands from scratch: new random first layer
    if not pretrained:
        model.conv1 = nn.Conv2d(13, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    if init_strategy not in INIT_STRATEGIES:
        raise ValueError(f"init_strategy must be one of {INIT_STRATEGIES}, got {init_strategy}")
    if init_strategy == "learned_1x1":
        return BandMixerResNet(model)
    model.conv1 = make_13_band_conv(model.conv1, init_strategy)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
