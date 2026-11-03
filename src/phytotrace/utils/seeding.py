"""Deterministic seeding for reproducible ML experiments."""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch (including MPS/CPU).
    
    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

