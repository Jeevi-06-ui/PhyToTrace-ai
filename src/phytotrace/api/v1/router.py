"""PhytoTrace AI — FastAPI Versioned Enterprise API (v1).

Full endpoints for real model analysis, evidence mapping, Firebase auth context,
MongoDB Atlas persistence, Weather Intelligence, and Gemini Assistant.
"""

from base64 import b64encode
from datetime import datetime, timezone
from threading import Lock
import os
from time import monotonic
from uuid import uuid4

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from phytotrace.api.scan_input import read_leaf_polygon, read_scan_upload
from pydantic import BaseModel, Field

from phytotrace.auth.firebase import UserContext, get_current_user, require_authenticated_user
from phytotrace.db.mongo import get_db
from phytotrace.inference.pipeline import InferencePipeline
from phytotrace.services.assistant import get_assistant_service
from phytotrace.services.weather import WeatherUnavailable, get_weather_service
from phytotrace.storage import get_storage
from phytotrace.utils.logging import get_logger

logger = get_logger("phytotrace.api.v1")

router = APIRouter(prefix="/api/v1", tags=["PhytoTrace v1 Enterprise API"])

# Pipeline singleton holder
_pipeline_instance: Optional[InferencePipeline] = None


def get_v1_pipeline() -> InferencePipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = InferencePipeline()
    return _pipeline_instance


# ---------------------------------------------------------------------------
# Pydantic Request / Response Models
# ---------------------------------------------------------------------------
class SystemHealthResponse(BaseModel):
    status: str
    version: str
    device: str
    models_active: Dict[str, bool]


class AssistantChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", min_length=1, max_length=96)
    inspection_id: Optional[str] = None
    language: str = "en"


class AssistantChatResponse(BaseModel):
    session_id: str
    message: str
    source: str
    language: str
    citations: List[str]


class WeatherCropAdviceRequest(BaseModel):
    latitude: float = Field(default=12.9716, ge=-90, le=90)
    longitude: float = Field(default=77.5946, ge=-180, le=180)
    location_name: str = "Bengaluru, Karnataka"
    crop: str = "Tomato"
    planting_date: Optional[str] = None
    growth_stage: str = "flowering"
    soil_type: str = "red_loam"
    irrigation_method: str = "drip"
    farm_area: Optional[float] = None
    farm_area_unit: str = "acres"
    farm_area_m2: Optional[float] = None
    recent_irrigation: Optional[str] = None
    fertilizer_history: Optional[str] = None
    soil_moisture_pct: Optional[float] = Field(default=None, ge=0, le=100)
    soil_ph: Optional[float] = Field(default=None, ge=3.0, le=10.0)
    language: str = "en"
    save_to_profile: bool = False


# ---------------------------------------------------------------------------
# 1. System Health & Model Status
# ---------------------------------------------------------------------------
@router.get("/health", response_model=SystemHealthResponse)
def get_v1_health():
    pipe = get_v1_pipeline()
    return SystemHealthResponse(
        status="healthy",
        version="PhytoTrace-v1.0.0-MVP",
        device=str(pipe.device),
        models_active={
            "unet_leaf_segmenter": pipe.leaf_segmenter.is_trained,
            "yolo_mine_detector": pipe.mine_segmenter.is_trained,
            "mobilenet_baseline": pipe.baseline_model is not None,
        },
    )


@router.get("/models/status")
def get_v1_models_status():
    pipe = get_v1_pipeline()
    return {
        "device": str(pipe.device),
        "models": [
            {
                "id": "leaf_unet",
                "name": "Canopy Boundary Segmenter",
                "checkpoint": os.getenv("PHYTOTRACE_LEAF_CHECKPOINT", "models/checkpoints/leaf_segmenter/leaf_unet_best.pt"),
                "loaded": pipe.leaf_segmenter.is_trained,
            },
            {
                "id": "mine_yolov8n",
                "name": "Damage Instance Localizer",
                "checkpoint": os.getenv("PHYTOTRACE_MINE_CHECKPOINT", "models/checkpoints/mine_segmenter/best.pt"),
                "loaded": pipe.mine_segmenter.is_trained,
            },
            {
                "id": "baseline_mobilenetv3",
                "name": "MobileNetV3 Supplementary Screener",
                "checkpoint": "models/checkpoints/baseline/baseline_mobilenetv3_best.pt",
                "loaded": pipe.baseline_model is not None,
                "macro_f1": 0.5667,
            },
        ],
    }


# ---------------------------------------------------------------------------
# 2. Real AI Analysis & Evidence Map
# ---------------------------------------------------------------------------
_guest_windows: dict[tuple[str, str], tuple[float, int]] = {}
_rate_lock = Lock()


def rate_limit(request: Request, operation: str, limit: int):
    now = monotonic()
    peer = request.client.host if request.client else "unknown"
    key = (peer, operation)
    with _rate_lock:
        for expired in [k for k, (deadline, _) in _guest_windows.items() if deadline <= now]:
            del _guest_windows[expired]
        if key not in _guest_windows and len(_guest_windows) >= 4096:
            raise HTTPException(429, "Service busy. Try again in one minute.")
        deadline, count = _guest_windows.get(key, (now + 60, 0))
        if count >= limit:
            raise HTTPException(429, "Request limit reached. Try again in one minute.", headers={"Retry-After": "60"})
        _guest_windows[key] = (deadline, count + 1)


@router.post("/analyze")
async def analyze_leaf_photograph(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    farm_id: str = Form("GREENHOUSE_MAIN"),
    plant_id: str = Form("PLANT_001"),
    leaf_id: str = Form("LEAF_01"),
    notes: Optional[str] = Form(""),
    crop_box: Optional[str] = Form(None),
    leaf_polygon: Optional[str] = Form(None),
    user: UserContext = Depends(get_current_user),
):
    """Run real U-Net and YOLOv8 inference with A/B/C deterministic evidence mapping."""
    import cv2
    from phytotrace.inference.visualization import render_leaf_overlay, render_mine_overlay

    rate_limit(request, "scan", 12)
    upload = file or image
    if upload is None:
        raise HTTPException(400, "Upload a tomato leaf photograph.")
    img_bgr, parsed_crop_box = await read_scan_upload(upload, crop_box)
    polygon = read_leaf_polygon(leaf_polygon, img_bgr)
    if polygon is not None and parsed_crop_box is not None:
        raise HTTPException(422, "Choose one focus method.")
    pipe = get_v1_pipeline()
    result = await run_in_threadpool(
        pipe.predict, img_bgr, image_id=upload.filename, crop_box=parsed_crop_box, leaf_polygon=polygon,
    )
    leaf_mask = result.pop("_leaf_mask", None)
    mine_mask = result.pop("_mine_mask", None)
    annotated = result.pop("_annotated_image", None)
    images = {"original": (".jpg", img_bgr)}
    if annotated is not None:
        images["annotated"] = (".jpg", annotated)
    if leaf_mask is not None:
        images["leaf_mask"] = (".png", leaf_mask)
        images["leaf_overlay"] = (".jpg", render_leaf_overlay(img_bgr, leaf_mask))
    if mine_mask is not None:
        images["mine_mask"] = (".png", mine_mask)
        images["mine_overlay"] = (".jpg", render_mine_overlay(img_bgr, mine_mask))

    def encode_assets():
        return {name: (ext, cv2.imencode(ext, raster)[1].tobytes())
                for name, (ext, raster) in images.items()}

    assets = await run_in_threadpool(encode_assets)
    record = {
        **result, "farm_id": farm_id, "plant_id": plant_id, "leaf_id": leaf_id,
        "notes": notes, "severity_tier": "not_assessed",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if user.is_guest:
        record["image_urls"] = {
            name: f"data:image/{'jpeg' if ext == '.jpg' else 'png'};base64,{b64encode(data).decode('ascii')}"
            for name, (ext, data) in assets.items()
        }
        return JSONResponse(record, headers={"Cache-Control": "no-store"})

    return await run_in_threadpool(save_authenticated_scan, user, record, assets)


def save_authenticated_scan(user, record, assets):
    from phytotrace.storage.cloudinary_storage import CloudinaryStorage

    try:
        db = get_db()
        storage = get_storage()
    except Exception as exc:
        logger.warning("Cloud persistence initialization failed (%s). Falling back to direct resilient delivery.", exc)
        scan_id = f"INSP_{uuid4().hex}"
        record.update(id=scan_id, inspection_id=scan_id)
        record["image_urls"] = {
            name: f"data:image/{'jpeg' if ext == '.jpg' else 'png'};base64,{b64encode(data).decode('ascii')}"
            for name, (ext, data) in assets.items()
        }
        return JSONResponse(record, headers={"Cache-Control": "no-store"})

    if not db.is_mongo or not isinstance(storage, CloudinaryStorage) or not storage.is_active:
        logger.warning(
            "Cloud persistence unavailable (mongo=%s, cloudinary=%s). Falling back to resilient delivery.",
            getattr(db, "is_mongo", False),
            getattr(storage, "is_active", False),
        )
        scan_id = f"INSP_{uuid4().hex}"
        record.update(id=scan_id, inspection_id=scan_id)
        try:
            db.save_inspection(record, owner_uid=user.uid)
        except Exception as e:
            logger.warning("Local save failed: %s", e)
        record["image_urls"] = {
            name: f"data:image/{'jpeg' if ext == '.jpg' else 'png'};base64,{b64encode(data).decode('ascii')}"
            for name, (ext, data) in assets.items()
        }
        return JSONResponse(record, headers={"Cache-Control": "no-store"})

    scan_id = f"INSP_{uuid4().hex}"
    keys = {name: f"inspections/{scan_id}/{name}{ext}" for name, (ext, _) in assets.items()}
    uploaded = []
    try:
        for name, (_, data) in assets.items():
            key = keys[name]
            storage.upload_private(data, key)
            uploaded.append(key)
        record.update(id=scan_id, inspection_id=scan_id, asset_keys=keys)
        db.save_inspection(record, owner_uid=user.uid)
    except Exception as exc:
        for key in uploaded:
            try:
                storage.delete_private(key)
            except Exception:
                pass
        logger.error("Cloudinary/MongoDB save failed: %s. Using resilient delivery.", exc)
        record["image_urls"] = {
            name: f"data:image/{'jpeg' if ext == '.jpg' else 'png'};base64,{b64encode(data).decode('ascii')}"
            for name, (ext, data) in assets.items()
        }
        return JSONResponse(record, headers={"Cache-Control": "no-store"})

    return deliver_inspection(record, storage)


def deliver_inspection(record, storage=None):
    record = dict(record)
    keys = record.pop("asset_keys", {})
    if keys:
        storage = storage or get_storage()
        record["image_urls"] = {name: storage.private_url(key) for name, key in keys.items()}
        record["thumbnail_url"] = record["image_urls"].get("annotated", record["image_urls"].get("original"))
    return record


@router.get("/analyses/{analysis_id}")
def get_analysis_by_id(analysis_id: str, user: UserContext = Depends(require_authenticated_user)):
    db = get_db()
    insp = db.get_inspection(analysis_id, owner_uid=user.uid)
    if not insp:
        raise HTTPException(status_code=404, detail="Inspection analysis not found.")
    return deliver_inspection(insp)


@router.delete("/analyses/{analysis_id}", status_code=204)
def delete_analysis(analysis_id: str, user: UserContext = Depends(require_authenticated_user)):
    db = get_db()
    record = db.get_inspection(analysis_id, owner_uid=user.uid)
    if not record:
        raise HTTPException(404, "Inspection analysis not found.")
    storage = get_storage()
    for key in record.get("asset_keys", {}).values():
        if not storage.delete_private(key):
            raise HTTPException(503, "Image deletion is incomplete. Retry this request.")
    db.db.inspections.delete_one({"id": analysis_id, "owner_uid": user.uid})


@router.get("/analyses/{analysis_id}/regions")
def get_analysis_regions(analysis_id: str, user: UserContext = Depends(require_authenticated_user)):
    db = get_db()
    insp = db.get_inspection(analysis_id, owner_uid=user.uid)
    if not insp:
        raise HTTPException(status_code=404, detail="Inspection analysis not found.")
    return {
        "analysis_id": analysis_id,
        "total_regions": len(insp.get("damage_regions", [])),
        "regions": insp.get("damage_regions", []),
    }


# ---------------------------------------------------------------------------
# 3. User Identity & Profile
# ---------------------------------------------------------------------------
@router.get("/users/me")
def get_user_profile(user: UserContext = Depends(get_current_user)):
    if user.is_guest:
        return {**user.model_dump(), "preferred_language": "en", "preferences": {}}
    db = get_db()
    profile = db.upsert_user(user.uid, user.email or "", user.display_name) if db.is_mongo else {}
    prefs = db.get_user_preferences(user.uid)
    return {
        "uid": user.uid,
        "email": user.email,
        "display_name": profile.get("display_name", user.display_name),
        "is_guest": user.is_guest,
        "role": user.role,
        "preferred_language": profile.get("preferred_language", "en"),
        "preferences": prefs,
    }


@router.get("/users/preferences")
def get_user_preferences_endpoint(user: UserContext = Depends(require_authenticated_user)):
    db = get_db()
    return {"preferences": db.get_user_preferences(user.uid)}


@router.put("/users/preferences")
def update_user_preferences_endpoint(
    req: Dict[str, Any],
    user: UserContext = Depends(require_authenticated_user),
):
    db = get_db()
    if not db.is_mongo:
        raise HTTPException(503, "Account preferences are unavailable.")
    return db.save_user_preferences(user.uid, req)


# ---------------------------------------------------------------------------
# 4. Inspections & History
# ---------------------------------------------------------------------------
@router.get("/inspections")
def list_v1_inspections(
    farm_id: Optional[str] = None,
    plant_id: Optional[str] = None,
    severity_tier: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: UserContext = Depends(get_current_user),
):
    if user.is_guest:
        return {"total": 0, "limit": limit, "offset": offset, "items": []}
    db = get_db()
    records = db.list_inspections(
        farm_id=farm_id,
        plant_id=plant_id,
        severity_tier=severity_tier,
        owner_uid=user.uid,
        limit=limit,
        offset=offset,
    )
    records["items"] = [deliver_inspection(record) for record in records["items"]]
    return records



@router.get("/farms")
def farm_overview(user: UserContext = Depends(get_current_user)):
    empty = {"total_inspections": 0, "total_farms": 0, "total_plants": 0,
             "average_damage_pct": None, "active_infestations": 0, "severity_distribution": {}, "farms": []}
    if user.is_guest:
        return empty
    db = get_db()
    if not db.is_mongo:
        raise HTTPException(503, "Account records are unavailable.")
    farms = list(db.db.inspections.aggregate([
        {"$match": {"owner_uid": user.uid}},
        {"$group": {"_id": "$farm_id", "plants": {"$addToSet": "$plant_id"},
                    "inspections": {"$sum": 1}, "damage_total": {"$sum": "$damage_percentage"},
                    "measurements": {"$sum": {"$cond": [{"$isNumber": "$damage_percentage"}, 1, 0]}},
                    "accepted": {"$sum": {"$cond": [{"$eq": ["$damage_detected", True]}, 1, 0]}}}},
    ]))
    measures = sum(farm["measurements"] for farm in farms)
    return {
        **empty, "total_inspections": sum(farm["inspections"] for farm in farms),
        "total_farms": len(farms), "total_plants": sum(len(farm["plants"]) for farm in farms),
        "average_damage_pct": sum(farm["damage_total"] for farm in farms) / measures if measures else None,
        "active_infestations": sum(farm["accepted"] for farm in farms),
        "farms": [{"farm_id": farm["_id"], "inspections": farm["inspections"], "plants": len(farm["plants"]),
                   "average_damage_pct": farm["damage_total"] / farm["measurements"] if farm["measurements"] else None}
                  for farm in farms],
    }


# ---------------------------------------------------------------------------
# 5. Weather Intelligence & Agronomic Guidance
# ---------------------------------------------------------------------------
@router.get("/weather/reverse-geocode")
async def reverse_geocode_location(
    request: Request,
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    lang: str = Query("en"),
):
    rate_limit(request, "weather_geocode", 30)
    return await get_weather_service().reverse_geocode(latitude=lat, longitude=lon, language=lang)


@router.get("/weather/geocode")
async def forward_geocode_location(
    request: Request,
    q: str = Query(..., min_length=2),
    lang: str = Query("en"),
):
    rate_limit(request, "weather_geocode", 30)
    results = await get_weather_service().forward_geocode(query=q, language=lang)
    return {"results": results}


@router.get("/weather/current")
async def get_current_weather(
    request: Request,
    lat: float = Query(12.9716, ge=-90, le=90),
    lon: float = Query(77.5946, ge=-180, le=180),
    location: str = Query("Bengaluru, Karnataka"),
    lang: str = Query("en"),
):
    rate_limit(request, "weather", 20)
    try:
        return await get_weather_service().get_weather(
            latitude=lat,
            longitude=lon,
            location_name=location,
            language=lang,
        )
    except WeatherUnavailable as exc:
        raise HTTPException(503, "Live weather is temporarily unavailable.") from exc


@router.post("/weather/crop-advice")
async def get_weather_crop_advice(
    req: WeatherCropAdviceRequest,
    request: Request,
    user: UserContext = Depends(get_current_user),
):
    rate_limit(request, "weather", 20)
    profile_dict = req.model_dump()
    try:
        data = await get_weather_service().get_weather(
            latitude=req.latitude,
            longitude=req.longitude,
            location_name=req.location_name,
            farmer_profile=profile_dict,
            language=req.language,
        )
    except WeatherUnavailable as exc:
        raise HTTPException(503, "Live weather is temporarily unavailable.") from exc

    # Persist in MongoDB Atlas for authenticated users if requested
    if not user.is_guest and (req.save_to_profile or profile_dict.get("farm_area")):
        try:
            db = get_db()
            db.save_user_preferences(user.uid, {
                "weather_profile": profile_dict,
                "preferred_location": {
                    "latitude": req.latitude,
                    "longitude": req.longitude,
                    "location_name": req.location_name,
                },
                "language": req.language,
            })
            db.save_weather_snapshot({
                "location": req.location_name,
                "latitude": req.latitude,
                "longitude": req.longitude,
                "temperature": data.get("current", {}).get("temperature_c"),
                "humidity": data.get("current", {}).get("humidity_pct"),
                "source": data.get("source"),
            })
        except Exception as e:
            logger.warning("Could not persist weather preferences to MongoDB: %s", e)

    return {
        "location": req.location_name,
        "crop": req.crop,
        "growth_stage": req.growth_stage,
        "soil_type": req.soil_type,
        "irrigation_method": req.irrigation_method,
        "field_context": profile_dict,
        "current_weather": data.get("current"),
        "agronomic_guidance": data.get("agronomic_guidance"),
        "source": data.get("source"),
        "provider_status": data.get("provider_status"),
    }


# ---------------------------------------------------------------------------
# 6. Personalized PhytoTrace Agri Assistant
# ---------------------------------------------------------------------------
@router.post("/assistant/chat", response_model=AssistantChatResponse)
async def chat_with_agri_assistant(
    request: Request,
    req: AssistantChatRequest,
    user: UserContext = Depends(get_current_user),
):
    rate_limit(request, "assistant", 8)
    if req.inspection_id and user.is_guest:
        raise HTTPException(401, "Sign in to attach a saved inspection.")
    db = None if user.is_guest else get_db()
    inspection_ctx = db.get_inspection(req.inspection_id, owner_uid=user.uid) if db and req.inspection_id else None
    if req.inspection_id and not inspection_ctx:
        raise HTTPException(404, "Inspection analysis not found.")
    history = db.get_assistant_messages(req.session_id, user.uid) if db else []
    res = await get_assistant_service().generate_response(
        user_message=req.message,
        session_history=history,
        inspection_context=inspection_ctx,
        language=req.language,
    )
    if db:
        db.save_assistant_message(req.session_id, {"role": "user", "text": req.message}, user.uid)
        db.save_assistant_message(req.session_id, {"role": "assistant", "text": res["text"]}, user.uid)

    return AssistantChatResponse(
        session_id=req.session_id or "default",
        message=res["text"],
        source=res["source"],
        language=res["language"],
        citations=res.get("citations", []),
    )

