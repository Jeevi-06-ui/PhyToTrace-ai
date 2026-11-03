"""Metadata management and tabular validation for PhytoTrace AI."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pydantic import ValidationError

from phytotrace.data.schemas import ImageMetadataRecord
from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path

logger = get_logger("phytotrace.metadata")

REQUIRED_COLUMNS = [
    "image_id",
    "original_filename",
    "image_path",
    "collector_id",
    "collection_date",
    "farm_id",
    "field_id",
    "row_id",
    "plant_id",
    "leaf_id",
    "collection_session_id",
    "tomato_variety",
    "growth_stage",
    "image_view_type",
    "image_quality",
    "visible_damage_type",
    "suspected_pest",
    "confirmed_pest",
    "pest_confirmation_method",
    "expert_reviewer_id",
    "label_verification_status",
    "mixed_infestation_status",
    "annotation_status",
    "notes",
    "consent_or_permission",
    "usage_license",
]


class MetadataManager:
    """Manages the dataset metadata CSV, ensuring schema conformity and referential integrity."""

    def __init__(self, metadata_path: Path):
        self.metadata_path = resolve_path(metadata_path)
        self.df: pd.DataFrame = pd.DataFrame()
        if self.metadata_path.exists():
            self.load()

    def load(self) -> pd.DataFrame:
        """Load metadata CSV into DataFrame."""
        self.df = pd.read_csv(self.metadata_path, dtype=str).fillna("")
        return self.df

    def save(self) -> None:
        """Write current DataFrame to disk."""
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.metadata_path, index=False)
        logger.info("Saved metadata with %d records to %s", len(self.df), self.metadata_path)

    def validate_all_records(self) -> Tuple[bool, List[str]]:
        """Validate every record against the Pydantic ImageMetadataRecord schema.
        
        Returns:
            (is_valid, list_of_error_messages)
        """
        errors = []
        if self.df.empty:
            return True, []
            
        missing_cols = set(REQUIRED_COLUMNS) - set(self.df.columns)
        if missing_cols:
            return False, [f"Missing required metadata columns: {sorted(list(missing_cols))}"]

        for idx, row in self.df.iterrows():
            row_dict = row.to_dict()
            try:
                ImageMetadataRecord(**row_dict)
            except ValidationError as ve:
                errors.append(f"Row {idx} (ID: {row_dict.get('image_id', 'unknown')}): {ve}")

        return len(errors) == 0, errors

    def add_record(self, record: ImageMetadataRecord) -> None:
        """Append a validated record."""
        row_dict = record.model_dump()
        new_row = pd.DataFrame([row_dict])
        if self.df.empty:
            self.df = new_row
        else:
            self.df = pd.concat([self.df, new_row], ignore_index=True)

    def get_record(self, image_id: str) -> Optional[Dict[str, str]]:
        """Retrieve record by image_id."""
        if self.df.empty or "image_id" not in self.df.columns:
            return None
        match = self.df[self.df["image_id"] == image_id]
        if not match.empty:
            return match.iloc[0].to_dict()
        return None

