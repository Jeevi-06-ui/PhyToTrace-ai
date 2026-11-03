"""Configuration loader and schema validator for PhytoTrace AI."""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from phytotrace.utils.paths import resolve_path


def load_yaml_config(config_name_or_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file from configs/ directory or absolute/relative path.
    
    Args:
        config_name_or_path: Filename like 'baseline.yaml' or path 'configs/baseline.yaml'.
        
    Returns:
        Dictionary containing parsed configuration.
    """
    path = Path(config_name_or_path)
    if not path.is_file():
        # Check inside configs/ directory
        candidate = resolve_path("configs") / config_name_or_path
        if candidate.is_file():
            path = candidate
        else:
            candidate_yaml = resolve_path("configs") / f"{config_name_or_path}.yaml"
            if candidate_yaml.is_file():
                path = candidate_yaml
            else:
                raise FileNotFoundError(f"Configuration file not found: {config_name_or_path}")
                
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    return data or {}

