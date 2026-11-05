"""FastAPI request and response schemas for PhytoTrace AI."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    device: str
    mps_available: bool
    timestamp: str


class ModelComponentStatus(BaseModel):
    name: str
    is_trained: bool
    weights_path: Optional[str]
    description: str


class ModelsStatusResponse(BaseModel):
    models: List[ModelComponentStatus]


class RegionEvidenceOutput(BaseModel):
    region_id: str
    label: Optional[str] = None
    centroid: Optional[List[int]] = None
    leaf_area_pct: Optional[float] = None
    polygons: Optional[List[List[List[int]]]] = None
    damage_type: Optional[str] = None
    description: Optional[str] = None
    uncertainty: Optional[str] = None
    recommended_observation: Optional[str] = None
    bbox: List[int]
    area_pixels: int
    confidence: float
    polygon: List[List[int]]
    pest_evidence: str
    tuta_score: Optional[float] = None
    liriomyza_score: Optional[float] = None


class PredictionResponse(BaseModel):
    inspection_id: Optional[str] = None
    image_id: str
    farm_id: Optional[str] = None
    plant_id: Optional[str] = None
    leaf_id: Optional[str] = None
    model_version: str
    status: Optional[str] = "ANALYSIS_SUCCESS"
    quality_status: str
    damage_detected: Optional[bool]
    damage_regions: List[RegionEvidenceOutput]
    leaf_area_pixels: Optional[int]
    damage_area_pixels: Optional[int]
    damage_percentage: Optional[float]
    severity_tier: Optional[str] = "none"
    tuta_evidence: Optional[float] = None
    liriomyza_evidence: Optional[float] = None
    diagnosis_name: Optional[str] = None
    diagnosis_accuracy: Optional[float] = None
    pest_name: Optional[str] = None
    species_breakdown: Optional[Dict[str, Any]] = None
    mixed_infestation_status: str
    uncertainty_status: str
    decision: str
    explanation: str
    safety_notice: str
    limitations: List[str]
    processing_time_ms: int
    roi_mode: Optional[str] = "automatic"
    roi_label: Optional[str] = "Automated canopy scan"
    crop_box: Optional[List[int]] = None
    image_urls: Optional[Dict[str, str]] = None
    progression: Optional[Dict[str, Any]] = None
    failure_classification: Optional[str] = None
    diagnostic_telemetry: Optional[Dict[str, Any]] = None
    rejected_candidates: Optional[List[Dict[str, Any]]] = None


class VersionResponse(BaseModel):
    api_version: str
    model_version: str
    torch_version: str


class InspectionRecordResponse(BaseModel):
    id: str
    farm_id: str
    plant_id: str
    leaf_id: str
    created_at: str
    original_filename: str
    quality_status: str
    leaf_area_pixels: Optional[int] = None
    damage_detected: Optional[bool]
    damage_area_pixels: Optional[int] = None
    damage_percentage: Optional[float] = None
    severity_tier: str
    decision: str
    uncertainty_status: Optional[str] = None
    explanation: Optional[str] = None
    safety_notice: Optional[str] = None
    model_version: str
    processing_time_ms: Optional[int] = 0
    damage_regions: List[Dict[str, Any]] = []
    notes: Optional[str] = ""
    progression_delta: Optional[float] = None
    progression_status: Optional[str] = None
    image_urls: Optional[Dict[str, str]] = None


class InspectionListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[InspectionRecordResponse]


class FarmSummary(BaseModel):
    farm_id: str
    inspections: int
    plants: int
    average_damage_pct: float


class FarmOverviewResponse(BaseModel):
    total_inspections: int
    total_farms: int
    total_plants: int
    average_damage_pct: float
    active_infestations: int
    severity_distribution: Dict[str, int]
    farms: List[FarmSummary]


class PlantTimelineResponse(BaseModel):
    plant_id: str
    total_inspections: int
    latest_damage_pct: Optional[float] = None
    overall_progression: Optional[str] = None
    timeline: List[InspectionRecordResponse]


class SampleImageInfo(BaseModel):
    filename: str
    label: str
    category: str
    description: str
    url: str
