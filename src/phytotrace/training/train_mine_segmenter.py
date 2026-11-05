"""Training wrapper for Ultralytics YOLO mine segmentation model."""

from pathlib import Path
from typing import Any, Dict, Optional
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
    pretrained_weights: str = "yolov8n-seg.pt",
    run_name: str = "mine_experiment",
) -> Dict[str, Any]:
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

        # Resolve weights path if existing locally
        local_weights = resolve_path(pretrained_weights)
        weights_src = str(local_weights) if local_weights.is_file() else pretrained_weights

        logger.info("Initializing YOLO model with weights '%s'...", weights_src)
        model = YOLO(weights_src)
        logger.info("Starting YOLO mine segmentation training on device %s for %d epochs...", dev_str, epochs)
        
        project_dir = resolve_path("models/checkpoints")
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=dev_str,
            project=str(project_dir),
            name=run_name,
            exist_ok=False,
            workers=0,
            mask_ratio=2,
            overlap_mask=False,
            mosaic=0.0,
        )
        
        save_dir = getattr(results, "save_dir", project_dir / run_name)
        return {
            "status": "trained",
            "save_dir": str(save_dir),
            "results": str(results),
        }
    except Exception as e:
        logger.error("Error during YOLO segmentation training: %s", e)
        return {"status": "error", "error": str(e)}
