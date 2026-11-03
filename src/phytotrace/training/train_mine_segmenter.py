"""Training wrapper for Ultralytics YOLO mine segmentation model."""

from pathlib import Path
from typing import Dict, Optional
from phytotrace.config import load_yaml_config
from phytotrace.device import get_device
from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path

logger = get_logger("phytotrace.train_mine_segmenter")


def train_mine_segmenter_pipeline(
    data_yaml_path: str = "data/annotations/yolo_dataset.yaml",
    epochs: int = 30,
    batch_size: int = 4,
    imgsz: int = 640,
    pretrained_weights: str = "yolo11n-seg.pt",
) -> Dict[str, any]:
    """Train YOLO instance segmentation on leaf-miner polygon annotations.
    
    Verifies data presence before training.
    """
    data_yaml = resolve_path(data_yaml_path)
    if not data_yaml.is_file():
        logger.warning("YOLO dataset YAML not found at %s. Marking status DATA_NOT_READY.", data_yaml)
        return {
            "status": "DATA_NOT_READY",
            "message": f"Annotation file {data_yaml} not found. Please provide CVAT/YOLO polygon annotations.",
        }

    try:
        from ultralytics import YOLO
        device = get_device("auto")
        dev_str = "mps" if device.type == "mps" else "cpu"

        model = YOLO(pretrained_weights)
        logger.info("Starting YOLO mine segmentation training on device %s...", dev_str)
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=dev_str,
            project="models/checkpoints",
            name="mine_segmenter",
            exist_ok=True,
        )
        return {"status": "trained", "results": str(results)}
    except Exception as e:
        logger.error("Error during YOLO segmentation training: %s", e)
        return {"status": "error", "error": str(e)}

