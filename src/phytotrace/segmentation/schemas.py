"""Schemas and data models for Phase 3 segmentation annotation and mask management."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class SegmentationClass(str, Enum):
    LEAF = "leaf"
    MINE_DAMAGE = "mine_damage"


class AnnotationStatus(str, Enum):
    UNANNOTATED = "unannotated"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"
    FLAGGED = "flagged"


class ReviewerDecision(str, Enum):
    """Formal status of human expert review on ambiguous or conflict annotations."""
    PENDING_REVIEW = "pending_review"
    CONFIRMED_NEGATIVE = "confirmed_negative"
    CONFIRMED_MINE = "confirmed_mine"
    NEEDS_CORRECTION = "needs_correction"
    UNCERTAIN_EXCLUDE = "uncertain_exclude"


@dataclass
class PolygonAnnotation:
    """Represents a single 2D polygon annotation on an image."""
    points: List[Tuple[float, float]]  # List of (x, y) float coordinates
    label: SegmentationClass
    area_pixels: float = 0.0
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # (xmin, ymin, width, height)
    attributes: Dict[str, str] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """A valid polygon must have at least 3 vertices and positive area."""
        return len(self.points) >= 3 and self.area_pixels >= 0


@dataclass
class ImageSegmentationCatalogRecord:
    """Record tracking segmentation annotation readiness for a single photograph."""
    image_id: str
    original_filename: str
    plant_id: str
    farm_id: str
    visible_damage_type: str
    leaf_mask_status: AnnotationStatus = AnnotationStatus.UNANNOTATED
    mine_annotation_status: AnnotationStatus = AnnotationStatus.UNANNOTATED
    annotation_reviewer: str = "none"
    num_damage_regions: int = 0
    image_quality: str = "sharp"
    ready_for_training: bool = False
    is_verified_negative: bool = False
    leaf_mask_path: Optional[str] = None
    mine_mask_path: Optional[str] = None
    reviewer_decision: str = "pending_review"
    reviewer_notes: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, any]:
        return {
            "image_id": self.image_id,
            "original_filename": self.original_filename,
            "plant_id": self.plant_id,
            "farm_id": self.farm_id,
            "visible_damage_type": self.visible_damage_type,
            "leaf_mask_status": self.leaf_mask_status.value if isinstance(self.leaf_mask_status, AnnotationStatus) else str(self.leaf_mask_status),
            "mine_annotation_status": self.mine_annotation_status.value if isinstance(self.mine_annotation_status, AnnotationStatus) else str(self.mine_annotation_status),
            "annotation_reviewer": self.annotation_reviewer,
            "reviewer_decision": self.reviewer_decision.value if isinstance(self.reviewer_decision, ReviewerDecision) else str(self.reviewer_decision),
            "reviewer_notes": self.reviewer_notes,
            "num_damage_regions": self.num_damage_regions,
            "image_quality": self.image_quality,
            "ready_for_training": self.ready_for_training,
            "is_verified_negative": self.is_verified_negative,
            "leaf_mask_path": self.leaf_mask_path or "",
            "mine_mask_path": self.mine_mask_path or "",
            "notes": self.notes,
        }
