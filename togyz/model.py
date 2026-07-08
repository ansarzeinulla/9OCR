"""Model factory and checkpoint helpers (plain torch.save, no fastai pickles)."""

import torch
import torch.nn as nn
from torchvision import models

from .classes import CLASSES
from .preprocess import TARGET_H, TARGET_W


def build_model(num_classes: int = len(CLASSES)) -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path, model, epoch, val_acc, optimizer=None, classes=None):
    torch.save(
        {
            "model_state": model.state_dict(),
            "classes": classes or CLASSES,
            "input_size": [TARGET_H, TARGET_W],
            "epoch": epoch,
            "val_acc": val_acc,
            "optimizer_state": optimizer.state_dict() if optimizer else None,
        },
        path,
    )


def load_checkpoint(path, device=None) -> tuple[nn.Module, dict]:
    device = device or auto_device()
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(len(ckpt["classes"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt
