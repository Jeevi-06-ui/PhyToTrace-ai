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
    bbox: List[int]
    area_pixels: int
    confidence: float
    polygon: List[List[int]]
    pest_evidence: str
    tuta_score: Optional[float]
    liriomyza_score: Optional[float]


class PredictionResponse(BaseModel):
    image_id: str
    model_version: str
    quality_status: str
    damage_detected: Optional[bool]
    damage_regions: List[RegionEvidenceOutput]
    leaf_area_pixels: Optional[int]
    damage_area_pixels: Optional[int]
    damage_percentage: Optional[float]
    tuta_evidence: Optional[float]
    liriomyza_evidence: Optional[float]
    mixed_infestation_status: str
    uncertainty_status: str
    decision: str
    explanation: str
    safety_notice: str
    limitations: List[str]
    processing_time_ms: int


class VersionResponse(BaseModel):
    api_version: str
    model_version: str
    torch_version: str

