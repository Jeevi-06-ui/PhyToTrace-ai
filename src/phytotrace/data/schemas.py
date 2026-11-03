"""Metadata and annotation schemas for PhytoTrace AI."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class GrowthStage(str, Enum):
    VEGETATIVE = "vegetative"
    FLOWERING = "flowering"
    FRUITING = "fruiting"
    HARVESTING = "harvesting"
    UNKNOWN = "unknown"


class ImageViewType(str, Enum):
    WHOLE_LEAF = "whole_leaf"
    CLOSE_UP_MINE = "close_up_mine"
    CANOPY_CONTEXT = "canopy_context"
    DORSAL = "dorsal"
    VENTRAL = "ventral"


class ImageQuality(str, Enum):
    SHARP = "sharp"
    SLIGHTLY_BLURRY = "slightly_blurry"
    POOR_LIGHTING = "poor_lighting"
    GLARE = "glare"
    SEVERE_BLUR = "severe_blur"


class VisibleDamageType(str, Enum):
    NONE = "none"
    LEAF_MINER_MINE = "leaf_miner_mine"
    CHEWING = "chewing"
    BLIGHT = "blight"
    PUNCTURES = "punctures"
    CURLING = "curling"
    OTHER = "other"


class SuspectedPest(str, Enum):
    NONE = "none"
    TUTA_ABSOLUTA = "tuta_absoluta"
    LIRIOMYZA_SPP = "liriomyza_spp"
    MIXED = "mixed"
    OTHER = "other"
    UNKNOWN = "unknown"


class ConfirmedPest(str, Enum):
    NONE = "none"
    TUTA_ABSOLUTA = "tuta_absoluta"
    LIRIOMYZA_SPP = "liriomyza_spp"
    MIXED = "mixed"
    UNCONFIRMED = "unconfirmed"


class PestConfirmationMethod(str, Enum):
    NONE = "none"
    LARVAL_MICROSCOPY = "larval_microscopy"
    ADULT_MORPHOLOGY = "adult_morphology"
    EXPERT_AGRONOMIST = "expert_agronomist"
    MOLECULAR_PCR = "molecular_pcr"
    PHEROMONE_TRAP_CORRELATION = "pheromone_trap_correlation"


class LabelVerificationStatus(str, Enum):
    UNVERIFIED_FIELD_GUESS = "unverified_field_guess"
    EXPERT_REVIEWED = "expert_reviewed"
    LAB_CONFIRMED = "lab_confirmed"


class MixedInfestationStatus(str, Enum):
    SINGLE_PEST = "single_pest"
    MIXED_CONFIRMED = "mixed_confirmed"
    MIXED_SUSPECTED = "mixed_suspected"
    NONE = "none"
    NOT_EVALUATED = "not_evaluated"


class AnnotationStatus(str, Enum):
    UNANNOTATED = "unannotated"
    IN_PROGRESS = "in_progress"
    ANNOTATED = "annotated"
    VERIFIED = "verified"


class ImageMetadataRecord(BaseModel):
    """Schema for a single row in the dataset metadata catalog."""
    image_id: str = Field(..., description="Unique alphanumeric identifier e.g. IMG_0001")
    original_filename: str = Field(..., description="Original camera filename")
    image_path: str = Field(..., description="Relative path within project, e.g. data/raw/IMG_0001.jpg")
    collector_id: str = Field(..., description="Pseudonymized collector ID")
    collection_date: str = Field(..., description="Date in YYYY-MM-DD")
    farm_id: str = Field(..., description="Farm or site identifier")
    field_id: str = Field("F1", description="Field identifier")
    row_id: str = Field("R1", description="Row identifier")
    plant_id: str = Field(..., description="Individual plant identifier")
    leaf_id: str = Field(..., description="Individual leaf identifier")
    collection_session_id: str = Field(..., description="Session identifier")
    tomato_variety: str = Field("unknown", description="Cultivar or variety")
    growth_stage: GrowthStage = GrowthStage.UNKNOWN
    image_view_type: ImageViewType = ImageViewType.WHOLE_LEAF
    image_quality: ImageQuality = ImageQuality.SHARP
    visible_damage_type: VisibleDamageType = VisibleDamageType.NONE
    suspected_pest: SuspectedPest = SuspectedPest.UNKNOWN
    confirmed_pest: ConfirmedPest = ConfirmedPest.UNCONFIRMED
    pest_confirmation_method: PestConfirmationMethod = PestConfirmationMethod.NONE
    expert_reviewer_id: str = Field("none", description="Pseudonymized expert ID")
    label_verification_status: LabelVerificationStatus = LabelVerificationStatus.UNVERIFIED_FIELD_GUESS
    mixed_infestation_status: MixedInfestationStatus = MixedInfestationStatus.NOT_EVALUATED
    annotation_status: AnnotationStatus = AnnotationStatus.UNANNOTATED
    notes: Optional[str] = Field("", description="Field observations")
    consent_or_permission: str = Field("yes", description="Facility/farm permission status")
    usage_license: str = Field("CC-BY-4.0", description="Data license")

