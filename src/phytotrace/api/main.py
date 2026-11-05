"""FastAPI production service for PhytoTrace AI diagnostic engine."""

from contextlib import asynccontextmanager
from datetime import datetime
import io
import json
import os
from uuid import uuid4
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional
import cv2
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from phytotrace.api.scan_input import read_scan_upload
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
import numpy as np
import torch

from phytotrace.api.schemas import (
    FarmOverviewResponse,
    HealthResponse,
    InspectionListResponse,
    InspectionRecordResponse,
    ModelComponentStatus,
    ModelsStatusResponse,
    PlantTimelineResponse,
    PredictionResponse,
    SampleImageInfo,
    VersionResponse,
)
from phytotrace.db.database import InspectionRepository
from phytotrace.device import get_device
from phytotrace.inference.pipeline import InferencePipeline
from phytotrace.inference.severity import compute_damage_severity
from phytotrace.inference.visualization import (
    create_side_by_side_comparison,
    render_leaf_overlay,
    render_mine_overlay,
)
from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path

logger = get_logger("phytotrace.api")

# Global instances
pipeline_instance: Optional[InferencePipeline] = None
repo_instance: Optional[InspectionRepository] = None

STORAGE_BASE = resolve_path(os.getenv("PHYTOTRACE_LOCAL_SCAN_STORAGE", "data/inspections"))
STORAGE_RAW = STORAGE_BASE / "raw"
STORAGE_MASKS = STORAGE_BASE / "masks"
STORAGE_OVERLAYS = STORAGE_BASE / "overlays"


def ensure_storage_dirs() -> None:
    STORAGE_RAW.mkdir(parents=True, exist_ok=True)
    STORAGE_MASKS.mkdir(parents=True, exist_ok=True)
    STORAGE_OVERLAYS.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance, repo_instance
    try:
        if os.getenv("PHYTOTRACE_ENV") != "production":
            ensure_storage_dirs()
        repo_instance = None if os.getenv("PHYTOTRACE_ENV") == "production" else InspectionRepository(os.getenv("PHYTOTRACE_SQLITE_PATH", "data/phytotrace.db"))
        pipeline_instance = InferencePipeline()
        logger.info("Initialized PhytoTrace InferencePipeline and SQLite Repository successfully.")
    except Exception as e:
        logger.error("Failed to initialize pipeline on startup: %s", e)
    yield


app = FastAPI(
    title="PhytoTrace AI Diagnostic Service",
    description="Explainable AI for Tomato Leaf Boundary, Leaf-Mining Lesion Localization, and Severity Assessment",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("PHYTOTRACE_CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def restrict_legacy_inspection_routes(request, call_next):
    path = request.url.path
    if os.getenv("PHYTOTRACE_ENV") == "production" and (
        path == "/predict" or path.startswith(("/api/predict", "/api/inspections", "/api/farms", "/api/plants"))
    ):
        return JSONResponse({"detail": "Use the authenticated or guest v1 API."}, status_code=404)
    return await call_next(request)


from phytotrace.api.v1.router import router as v1_router
app.include_router(v1_router)


def get_pipeline() -> InferencePipeline:
    global pipeline_instance
    if pipeline_instance is None:
        pipeline_instance = InferencePipeline()
    return pipeline_instance


def get_repo() -> InspectionRepository:
    global repo_instance
    if repo_instance is None:
        repo_instance = InspectionRepository(os.getenv("PHYTOTRACE_SQLITE_PATH", "data/phytotrace.db"))
    return repo_instance


def _build_image_urls(insp_id: str) -> Dict[str, str]:
    return {
        "original": f"/api/inspections/{insp_id}/images/original",
        "annotated": f"/api/inspections/{insp_id}/images/annotated",
        "leaf_mask": f"/api/inspections/{insp_id}/images/leaf_mask",
        "mine_mask": f"/api/inspections/{insp_id}/images/mine_mask",
        "leaf_overlay": f"/api/inspections/{insp_id}/images/leaf_overlay",
        "mine_overlay": f"/api/inspections/{insp_id}/images/mine_overlay",
        "side_by_side": f"/api/inspections/{insp_id}/images/side_by_side",
    }


@app.get("/api/health", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse, include_in_schema=False)
def health_check():
    dev = get_device("auto")
    mps_avail = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    return HealthResponse(
        status="healthy",
        device=str(dev),
        mps_available=mps_avail,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/api/version", response_model=VersionResponse)
@app.get("/version", response_model=VersionResponse, include_in_schema=False)
def version():
    return VersionResponse(
        api_version="0.1.0",
        model_version="PhytoTrace-v0.1.0-alpha",
        torch_version=torch.__version__,
    )


@app.get("/api/models", response_model=ModelsStatusResponse)
@app.get("/models", response_model=ModelsStatusResponse, include_in_schema=False)
def get_models_status():
    pipe = get_pipeline()
    statuses = [
        ModelComponentStatus(
            name="Baseline 3-Class MobileNetV3-Small",
            is_trained=pipe.baseline_model is not None,
            weights_path="models/checkpoints/baseline/baseline_mobilenetv3_best.pt",
            description="Fine-tuned Stage B backbone screening healthy vs leaf-miner vs other damage.",
        ),
        ModelComponentStatus(
            name="Leaf Surface Segmenter",
            is_trained=pipe.leaf_segmenter.is_trained,
            weights_path="models/checkpoints/leaf_segmenter/leaf_unet_best.pt",
            description="Apple MPS accelerated neural model segmenting leaf canopy boundary (Best Val Dice: 0.7923).",
        ),
        ModelComponentStatus(
            name="Mine Instance Segmenter",
            is_trained=pipe.mine_segmenter.is_trained,
            weights_path="models/checkpoints/mine_segmenter/best.pt",
            description="Deep neural instance segmenter localizing mine lesion polygons.",
        ),
        ModelComponentStatus(
            name="Pest-Evidence Classifier (Tuta / Liriomyza)",
            is_trained=pipe.species_wrapper.is_trained,
            weights_path="models/checkpoints/species/species_evidence_best.pt",
            description="Evidence classification gated with uncertainty abstention until Phase 5 label expansion.",
        ),
    ]
    return ModelsStatusResponse(models=statuses)


@app.post("/api/predict", response_model=PredictionResponse)
@app.post("/predict", response_model=PredictionResponse, include_in_schema=False)
async def predict_and_inspect(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    farm_id: str = Form("FARM_MAIN"),
    plant_id: str = Form("PLANT_001"),
    leaf_id: str = Form("LEAF_01"),
    notes: str = Form(""),
    crop_box: Optional[str] = Form(None),
):
    """Run unified AI inference, compute damage severity, record inspection, and return full diagnostics."""
    upload_file = file or image
    if upload_file is None:
        raise HTTPException(
            status_code=400,
            detail="No image uploaded. Please provide an image file in 'image' or 'file' form field.",
        )
    file = upload_file

    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename uploaded.")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {ext}. Supported: .jpg, .jpeg, .png, .webp",
        )

    image_bgr, parsed_crop_box = await read_scan_upload(file, crop_box)
    ensure_storage_dirs()
    pipe = get_pipeline()
    repo = get_repo()

    img_stem = Path(file.filename).stem
    insp_id = f"INSP_{uuid4().hex}"
    pred = await run_in_threadpool(pipe.predict, image_bgr, image_id=img_stem, crop_box=parsed_crop_box)

    annotated_bgr = pred.pop("_annotated_image", None)
    leaf_mask = pred.pop("_leaf_mask", None)
    mine_mask = pred.pop("_mine_mask", None)

    # Determine severity tier
    dmg_pct = pred.get("damage_percentage")
    if dmg_pct is None:
        severity_tier = "untrusted_canopy"
    elif dmg_pct == 0.0:
        severity_tier = "none"
    elif dmg_pct < 5.0:
        severity_tier = "low_infestation"
    elif dmg_pct < 20.0:
        severity_tier = "moderate_infestation"
    else:
        severity_tier = "severe_infestation"

    # Save images to disk
    orig_path = STORAGE_RAW / f"{insp_id}_orig.jpg"
    ann_path = STORAGE_OVERLAYS / f"{insp_id}_annotated.jpg"
    leaf_mask_path = STORAGE_MASKS / f"{insp_id}_leaf.png"
    mine_mask_path = STORAGE_MASKS / f"{insp_id}_mine.png"
    leaf_overlay_path = STORAGE_OVERLAYS / f"{insp_id}_leaf_overlay.jpg"
    mine_overlay_path = STORAGE_OVERLAYS / f"{insp_id}_mine_overlay.jpg"
    side_by_side_path = STORAGE_OVERLAYS / f"{insp_id}_side_by_side.jpg"

    cv2.imwrite(str(orig_path), image_bgr)

    if annotated_bgr is not None:
        cv2.imwrite(str(ann_path), annotated_bgr)
        sbs = create_side_by_side_comparison(
            original_bgr=image_bgr,
            annotated_bgr=annotated_bgr,
            decision_summary=pred.get("decision", ""),
            damage_percentage=dmg_pct,
            model_version=pred.get("model_version", "PhytoTrace"),
        )
        cv2.imwrite(str(side_by_side_path), sbs)
    else:
        cv2.imwrite(str(ann_path), image_bgr)
        cv2.imwrite(str(side_by_side_path), image_bgr)

    if leaf_mask is not None:
        cv2.imwrite(str(leaf_mask_path), leaf_mask)
        leaf_overlay = render_leaf_overlay(image_bgr, leaf_mask)
        cv2.imwrite(str(leaf_overlay_path), leaf_overlay)
    else:
        h, w = image_bgr.shape[:2]
        cv2.imwrite(str(leaf_mask_path), np.zeros((h, w), dtype=np.uint8))
        cv2.imwrite(str(leaf_overlay_path), image_bgr)

    if mine_mask is not None:
        cv2.imwrite(str(mine_mask_path), mine_mask)
        mine_overlay = render_mine_overlay(image_bgr, mine_mask)
        cv2.imwrite(str(mine_overlay_path), mine_overlay)
    else:
        h, w = image_bgr.shape[:2]
        cv2.imwrite(str(mine_mask_path), np.zeros((h, w), dtype=np.uint8))
        cv2.imwrite(str(mine_overlay_path), image_bgr)

    # Check for longitudinal progression compared to previous inspection of this plant/leaf
    timeline = repo.get_plant_timeline(plant_id)
    progression = None
    prev = next((r for r in reversed(timeline) if r.get("leaf_id") == leaf_id), None)
    if prev is not None and dmg_pct is not None and prev.get("damage_percentage") is not None:
        prev_pct = prev["damage_percentage"]
        delta = round(dmg_pct - prev_pct, 2)
        progression = {
            "previous_inspection_id": prev.get("id"),
            "previous_date": prev.get("created_at"),
            "previous_damage_percentage": prev_pct,
            "progression_delta": delta,
            "status": "expanding" if delta > 1.0 else ("receding" if delta < -1.0 else "stable"),
        }

    # Store in database
    db_record = {
        "id": insp_id,
        "farm_id": farm_id,
        "plant_id": plant_id,
        "leaf_id": leaf_id,
        "created_at": datetime.now().isoformat(),
        "original_filename": file.filename,
        "original_image_path": str(orig_path),
        "annotated_image_path": str(ann_path),
        "leaf_mask_path": str(leaf_mask_path),
        "mine_mask_path": str(mine_mask_path),
        "quality_status": pred.get("quality_status", "acceptable"),
        "leaf_area_pixels": pred.get("leaf_area_pixels"),
        "damage_detected": pred.get("damage_detected"),
        "damage_area_pixels": pred.get("damage_area_pixels"),
        "damage_percentage": dmg_pct,
        "severity_tier": severity_tier,
        "decision": pred.get("decision", "inference_error"),
        "uncertainty_status": pred.get("uncertainty_status"),
        "explanation": pred.get("explanation"),
        "safety_notice": pred.get("safety_notice"),
        "model_version": pred.get("model_version"),
        "processing_time_ms": pred.get("processing_time_ms", 0),
        "damage_regions": pred.get("damage_regions", []),
        "failure_classification": pred.get("failure_classification"),
        "diagnostic_telemetry": pred.get("diagnostic_telemetry", {}),
        "rejected_candidates": pred.get("rejected_candidates", []),
        "notes": notes,
    }
    repo.create_inspection(db_record)

    img_urls = _build_image_urls(insp_id)

    return PredictionResponse(
        inspection_id=insp_id,
        image_id=img_stem,
        farm_id=farm_id,
        plant_id=plant_id,
        leaf_id=leaf_id,
        model_version=pred.get("model_version", "PhytoTrace-v0.1.0"),
        status=pred.get("status", "ANALYSIS_SUCCESS"),
        quality_status=pred.get("quality_status", "acceptable"),
        damage_detected=pred.get("damage_detected"),
        damage_regions=pred.get("damage_regions", []),
        leaf_area_pixels=pred.get("leaf_area_pixels"),
        damage_area_pixels=pred.get("damage_area_pixels"),
        damage_percentage=dmg_pct,
        severity_tier=severity_tier,
        tuta_evidence=pred.get("tuta_evidence"),
        liriomyza_evidence=pred.get("liriomyza_evidence"),
        diagnosis_name=pred.get("diagnosis_name"),
        diagnosis_accuracy=pred.get("diagnosis_accuracy"),
        pest_name=pred.get("pest_name"),
        species_breakdown=pred.get("species_breakdown"),
        mixed_infestation_status=pred.get("mixed_infestation_status", "not_evaluated"),
        uncertainty_status=pred.get("uncertainty_status", "normal"),
        decision=pred.get("decision", "inference_error"),
        explanation=pred.get("explanation", ""),
        safety_notice=pred.get("safety_notice", ""),
        limitations=pred.get("limitations", []),
        processing_time_ms=pred.get("processing_time_ms", 0),
        roi_mode=pred.get("roi_mode", "automatic"),
        roi_label=pred.get("roi_label", "Automated canopy scan"),
        crop_box=pred.get("crop_box"),
        image_urls=img_urls,
        progression=progression,
        failure_classification=pred.get("failure_classification"),
        diagnostic_telemetry=pred.get("diagnostic_telemetry"),
        rejected_candidates=pred.get("rejected_candidates"),
    )


@app.get("/api/inspections", response_model=InspectionListResponse)
def list_inspections(
    farm_id: Optional[str] = Query(None),
    plant_id: Optional[str] = Query(None),
    severity_tier: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    repo = get_repo()
    items = repo.list_inspections(
        farm_id=farm_id,
        plant_id=plant_id,
        severity_tier=severity_tier,
        limit=limit,
        offset=offset,
    )
    total = repo.count_inspections(
        farm_id=farm_id,
        plant_id=plant_id,
        severity_tier=severity_tier,
    )

    enriched = []
    for item in items:
        item["image_urls"] = _build_image_urls(item["id"])
        enriched.append(item)

    return InspectionListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=enriched,
    )


@app.get("/api/inspections/{insp_id}", response_model=InspectionRecordResponse)
def get_inspection_detail(insp_id: str):
    repo = get_repo()
    item = repo.get_inspection(insp_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Inspection '{insp_id}' not found.")
    item["image_urls"] = _build_image_urls(insp_id)
    return item


@app.get("/api/inspections/{insp_id}/images/{img_type}")
def get_inspection_image(insp_id: str, img_type: str):
    repo = get_repo()
    item = repo.get_inspection(insp_id)
    if not item:
        raise HTTPException(status_code=404, detail="Inspection not found.")

    type_to_path = {
        "original": item.get("original_image_path"),
        "annotated": item.get("annotated_image_path"),
        "leaf_mask": item.get("leaf_mask_path"),
        "mine_mask": item.get("mine_mask_path"),
        "leaf_overlay": str(STORAGE_OVERLAYS / f"{insp_id}_leaf_overlay.jpg"),
        "mine_overlay": str(STORAGE_OVERLAYS / f"{insp_id}_mine_overlay.jpg"),
        "side_by_side": str(STORAGE_OVERLAYS / f"{insp_id}_side_by_side.jpg"),
    }

    p_str = type_to_path.get(img_type)
    if not p_str:
        raise HTTPException(status_code=400, detail=f"Invalid image type '{img_type}'.")

    target_path = resolve_path(p_str)
    if not target_path.is_file():
        raise HTTPException(status_code=404, detail=f"Image file for '{img_type}' does not exist on disk.")

    media_type = "image/png" if target_path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(target_path, media_type=media_type)


@app.get("/api/inspections/{insp_id}/report", response_class=PlainTextResponse)
def export_inspection_report(insp_id: str):
    repo = get_repo()
    item = repo.get_inspection(insp_id)
    if not item:
        raise HTTPException(status_code=404, detail="Inspection not found.")

    report_lines = [
        f"# PhytoTrace AI Diagnostic Inspection Report: {item['id']}",
        f"**Date:** {item['created_at']}",
        f"**Farm:** {item['farm_id']} | **Plant:** {item['plant_id']} | **Leaf:** {item['leaf_id']}",
        f"**Model Version:** {item['model_version']}",
        "",
        "## Diagnostic Summary",
        f"- **Primary Decision:** `{item['decision']}`",
        f"- **Severity Tier:** `{item['severity_tier'].upper()}`",
        f"- **Visible Damage Percentage:** {item['damage_percentage'] if item['damage_percentage'] is not None else 'N/A'}%",
        f"- **Leaf Canopy Area:** {item['leaf_area_pixels']} px",
        f"- **Mined Area:** {item['damage_area_pixels']} px",
        f"- **Damage Detected:** {item['damage_detected']}",
        f"- **Uncertainty Status:** `{item['uncertainty_status']}`",
        "",
        "## Agronomic Explanation",
        f"> {item['explanation']}",
        "",
        "## Advisory Safety Notice",
        f"> {item['safety_notice']}",
        "",
        "## Individual Lesion Regions",
    ]

    regions = item.get("damage_regions", [])
    if not regions:
        report_lines.append("No mining lesion regions localized.")
    else:
        for r in regions:
            report_lines.append(
                f"- **Region {r.get('region_id')}:** Bounding box {r.get('bbox')}, "
                f"Confidence: {r.get('confidence')}, Area: {r.get('area_pixels')} px"
            )

    report_lines.extend([
        "",
        "---",
        "*PhytoTrace AI — Explainable Pest Damage Monitoring on Tomato Crops.*",
    ])

    return "\n".join(report_lines)


@app.get("/api/farms", response_model=FarmOverviewResponse)
def get_farm_overview():
    repo = get_repo()
    return repo.get_farm_overview()


@app.get("/api/plants/{plant_id}/timeline", response_model=PlantTimelineResponse)
def get_plant_timeline(plant_id: str):
    repo = get_repo()
    records = repo.get_plant_timeline(plant_id)

    enriched = []
    latest_dmg = None
    for r in records:
        r["image_urls"] = _build_image_urls(r["id"])
        enriched.append(r)
        if r.get("damage_percentage") is not None:
            latest_dmg = r["damage_percentage"]

    overall = "stable"
    if len(records) >= 2:
        first_dmg = records[0].get("damage_percentage") or 0.0
        last_dmg = records[-1].get("damage_percentage") or 0.0
        if last_dmg - first_dmg > 2.0:
            overall = "expanding"
        elif last_dmg - first_dmg < -2.0:
            overall = "receding"

    return PlantTimelineResponse(
        plant_id=plant_id,
        total_inspections=len(records),
        latest_damage_pct=latest_dmg,
        overall_progression=overall,
        timeline=enriched,
    )


@app.get("/api/samples", response_model=List[SampleImageInfo])
def get_sample_images():
    """Returns curated representative sample images for 1-click live testing."""
    samples = [
        SampleImageInfo(
            filename="5558326.jpg",
            label="5558326 (Symptomatic Leaf Miner)",
            category="leaf_miner_damage",
            description="Tomato leaf with distinct serpentine leaf miner damage tracts.",
            url="/api/samples/5558326.jpg",
        ),
        SampleImageInfo(
            filename="5595360.jpg",
            label="5595360 (Multi-Region Miner)",
            category="leaf_miner_damage",
            description="Extensive leaf miner damage across upper and lower leaflets.",
            url="/api/samples/5595360.jpg",
        ),
        SampleImageInfo(
            filename="BLK_3_1002_PL010_H.JPG",
            label="PL010_H (Confirmed Healthy)",
            category="healthy",
            description="Verified negative healthy leaf with zero mining damage.",
            url="/api/samples/BLK_3_1002_PL010_H.JPG",
        ),
        SampleImageInfo(
            filename="BLK_3_1002_PL021_H.JPG",
            label="PL021_H (Healthy Canopy)",
            category="healthy",
            description="Intact tomato leaf leaflet with clean surface.",
            url="/api/samples/BLK_3_1002_PL021_H.JPG",
        ),
    ]
    if os.getenv("PHYTOTRACE_ENV") == "production":
        return []
    return [sample for sample in samples if resolve_path(f"data/raw/original_photos/{sample.filename}").is_file()]


@app.get("/api/samples/{filename}")
def serve_sample_image(filename: str):
    if filename not in {sample.filename for sample in get_sample_images()}:
        raise HTTPException(status_code=404, detail="Sample image not found.")
    p = resolve_path(f"data/raw/original_photos/{filename}")
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/research/benchmarks")
def get_research_benchmarks():
    """Retrieve precomputed computer vision benchmark telemetry and candidate filter logs."""
    data_path = resolve_path("reports/debug_segmentation/benchmarks_diagnostic_data.json")
    if not data_path.is_file():
        raise HTTPException(status_code=404, detail="Benchmark diagnostics not found.")
    with open(data_path, "r") as f:
        data = json.load(f)
    return data


@app.get("/api/research/benchmark-images/{filename}")
def serve_benchmark_image(filename: str):
    """Serve intermediate debug image artifacts (heatmaps, masks, raw detections)."""
    safe_name = Path(filename).name
    p = resolve_path(f"reports/debug_segmentation/{safe_name}")
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Benchmark image not found.")
    media_type = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(p, media_type=media_type)
