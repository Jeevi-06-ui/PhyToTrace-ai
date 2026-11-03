"""Safe model checkpointing and experiment persistence."""

from pathlib import Path
from typing import Any, Dict, Optional
import torch

from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path

logger = get_logger("phytotrace.checkpointing")


def save_checkpoint(
    state_dict: Dict[str, Any],
    epoch: int,
    metrics: Dict[str, float],
    save_path: Path,
    is_best: bool = False,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a model checkpoint with reproducibility metadata.
    
    Args:
        state_dict: Model state_dict.
        epoch: Current epoch number.
        metrics: Metric dictionary (e.g. {'val_loss': 0.25, 'macro_f1': 0.88}).
        save_path: Destination path.
        is_best: Whether this is the best performing checkpoint.
        extra_metadata: Additional configuration or hyperparameters.
        
    Returns:
        Path of saved file.
    """
    save_path = resolve_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "epoch": epoch,
        "model_state_dict": state_dict,
        "metrics": metrics,
        "is_best": is_best,
        "metadata": extra_metadata or {},
    }
    
    torch.save(payload, save_path)
    logger.info("Saved checkpoint (Epoch %d, Best=%s) to %s", epoch, is_best, save_path)
    return save_path


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load model checkpoint safely."""
    checkpoint_path = resolve_path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Bare state dict
        model.load_state_dict(checkpoint)
        
    logger.info("Loaded checkpoint from %s", checkpoint_path)
    return checkpoint

