"""Apple Silicon MPS and CPU device management for PhytoTrace AI."""

import os
import torch
from phytotrace.utils.logging import get_logger

logger = get_logger("phytotrace.device")


def get_device(preference: str = "auto") -> torch.device:
    """Select the optimal computing device.
    
    Priority:
    1. If preference is 'mps' and Apple MPS is available, returns torch.device('mps').
    2. If preference is 'cpu', returns torch.device('cpu').
    3. If preference is 'auto', checks MPS availability; if functional, returns 'mps', else 'cpu'.
    
    Never uses CUDA since the environment is Apple Silicon macOS.
    
    Args:
        preference: 'auto', 'mps', or 'cpu'. Defaults to environment var PHYTOTRACE_DEVICE or 'auto'.
        
    Returns:
        torch.device instance.
    """
    pref = os.getenv("PHYTOTRACE_DEVICE", preference).lower().strip()
    
    if pref == "cpu":
        logger.info("Device explicitly configured to CPU.")
        return torch.device("cpu")
        
    # Check Apple Silicon Metal Performance Shaders (MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        try:
            # Perform a quick smoke test on MPS to guarantee it can allocate and run ops
            test_tensor = torch.zeros((1, 1), device="mps")
            _ = test_tensor + 1.0
            logger.info("Apple Metal Performance Shaders (MPS) active and verified on Apple Silicon.")
            return torch.device("mps")
        except Exception as e:
            logger.warning("MPS is reported available but failed initialization smoke test: %s. Falling back to CPU.", e)
            return torch.device("cpu")
            
    logger.info("MPS not available or requested CPU. Falling back to CPU.")
    return torch.device("cpu")


def run_device_smoke_test() -> dict:
    """Run a small forward/backward pass test on the selected device and report metrics."""
    dev = get_device("auto")
    info = {
        "selected_device": str(dev),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "mps_built": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built()),
        "test_forward_backward_pass": False,
        "error": None,
    }
    
    try:
        x = torch.randn(4, 16, device=dev, requires_grad=True)
        linear = torch.nn.Linear(16, 2).to(dev)
        out = linear(x)
        loss = out.sum()
        loss.backward()
        info["test_forward_backward_pass"] = True
    except Exception as exc:
        info["error"] = str(exc)
        logger.error("Device smoke test failed: %s", exc)
        
    return info

