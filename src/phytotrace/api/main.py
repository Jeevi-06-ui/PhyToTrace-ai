"""FastAPI local service for PhytoTrace AI inference engine."""

from contextlib import asynccontextmanager
from datetime import datetime
import io
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional
import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import torch

from phytotrace.api.schemas import (
    HealthResponse,
    ModelComponentStatus,
    ModelsStatusResponse,
    PredictionResponse,
    VersionResponse,
)
from phytotrace.device import get_device
from phytotrace.inference.pipeline import InferencePipeline
from phytotrace.utils.logging import get_logger

logger = get_logger("phytotrace.api")

# Global pipeline instance (lazy loaded or initialized at startup)
pipeline_instance: Optional[InferencePipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance
    try:
        pipeline_instance = InferencePipeline()
        logger.info("Initialized PhytoTrace InferencePipeline successfully.")
    except Exception as e:
        logger.error("Failed to initialize pipeline on startup: %s", e)
    yield


app = FastAPI(
    title="PhytoTrace AI Diagnostic Service",
    description="Explainable AI for Mixed Leaf-Miner Infestation Detection, Damage Assessment and Progression Monitoring in Tomato Crops",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_pipeline() -> InferencePipeline:
    global pipeline_instance
    if pipeline_instance is None:
        pipeline_instance = InferencePipeline()
    return pipeline_instance


@app.get("/health", response_model=HealthResponse)
def health_check():
    dev = get_device("auto")
    mps_avail = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    return HealthResponse(
        status="healthy",
        device=str(dev),
        mps_available=mps_avail,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/version", response_model=VersionResponse)
def version():
    return VersionResponse(
        api_version="0.1.0",
        model_version="PhytoTrace-v0.1.0-alpha",
        torch_version=torch.__version__,
    )


@app.get("/models", response_model=ModelsStatusResponse)
def get_models_status():
    pipe = get_pipeline()
    statuses = [
        ModelComponentStatus(
            name="Baseline 3-Class MobileNetV3",
            is_trained=pipe.baseline_model is not None,
            weights_path="models/checkpoints/baseline/baseline_mobilenetv3_best.pt",
            description="Screens healthy vs leaf-miner vs other damage.",
        ),
        ModelComponentStatus(
            name="Leaf Surface Segmenter (Lightweight U-Net)",
            is_trained=pipe.leaf_segmenter.is_trained,
            weights_path="models/checkpoints/leaf_segmenter/leaf_unet_best.pt",
            description="Segments leaf canopy to compute visible leaf pixel area.",
        ),
        ModelComponentStatus(
            name="Mine Instance Segmenter (YOLO-seg)",
            is_trained=pipe.mine_segmenter.is_trained,
            weights_path="models/checkpoints/mine_segmenter/best.pt",
            description="Localizes and segments leaf-mining damage lesions.",
        ),
        ModelComponentStatus(
            name="Pest-Evidence Classifier (Tuta / Liriomyza)",
            is_trained=pipe.species_wrapper.is_trained,
            weights_path="models/checkpoints/species/species_evidence_best.pt",
            description="Region-based multi-label evidence model with uncertainty abstention.",
        ),
    ]
    return ModelsStatusResponse(models=statuses)


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename uploaded.")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {ext}. Supported: .jpg, .jpeg, .png, .webp",
        )

    pipe = get_pipeline()
    contents = await file.read()
    
    # Check max file size (25MB)
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds maximum 25 MB limit.")

    # Decode image from buffer
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image. File may be corrupted.")

    img_id = Path(file.filename).stem
    result = pipe.predict(image_bgr, image_id=img_id)

    # Strip out non-serializable OpenCV image array before returning JSON
    result.pop("_annotated_image", None)
    return result


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(files: List[UploadFile] = File(...)):
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Batch limit is 20 images per request.")

    pipe = get_pipeline()
    results = []

    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        img_id = Path(file.filename).stem if file.filename else "unknown"
        res = pipe.predict(image_bgr, image_id=img_id)
        res.pop("_annotated_image", None)
        results.append(res)

    return results
