"""Biologically appropriate image augmentations for tomato leaf mining analysis."""

from typing import Tuple
import torch
import torchvision.transforms as T


def get_baseline_transforms(
    is_training: bool = True,
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """Create torchvision transform pipeline for MobileNetV3 baseline.
    
    Training augmentations are conservative to preserve fine serpentine mine structures
    and avoid introducing synthetic puncture/chewing artifacts.
    """
    if is_training:
        return T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        return T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


def get_region_transforms(
    is_training: bool = True,
    image_size: Tuple[int, int] = (224, 224),
) -> T.Compose:
    """Transforms for region-based pest evidence crops."""
    return get_baseline_transforms(is_training=is_training, image_size=image_size)

