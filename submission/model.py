import torch
import torch.nn as nn
from torchvision import models, transforms


def get_model():
    # Load arsitektur ConvNeXt-Tiny tanpa bobot online
    model = models.convnext_tiny(weights=None)

    # Akses layer Linear pada indeks ke-2 dari Sequential classifier
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 5)

    return model


def get_transform():
    # Preprocessing validasi standar
    return transforms.Compose([
        transforms.Resize(236),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])