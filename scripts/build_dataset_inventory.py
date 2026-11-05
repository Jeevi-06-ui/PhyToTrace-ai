#!/usr/bin/env python3
"""PhytoTrace AI — Recursive Dataset Inventory & Contact Sheet Generator."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from phytotrace.data.deduplicate import compute_dhash, compute_sha256, detect_duplicates
from phytotrace.data.metadata import REQUIRED_COLUMNS
from phytotrace.data.schemas import (
    AnnotationStatus,
    ConfirmedPest,
    GrowthStage,
    ImageMetadataRecord,
    ImageQuality,
    ImageViewType,
    LabelVerificationStatus,
    MixedInfestationStatus,
    PestConfirmationMethod,
    SuspectedPest,
    VisibleDamageType,
)
from phytotrace.data.validators import check_and_fix_exif_orientation
from phytotrace.preprocessing.quality import assess_image_quality
from phytotrace.utils.paths import resolve_path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def scan_raw_dataset(raw_dir: Path) -> List[Path]:
    """Recursively discover all image files in raw_dir without modifying or deleting anything."""
    raw_dir = resolve_path(raw_dir)
    image_paths = []
    for p in raw_dir.rglob("*"):
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(p)
    return sorted(image_paths)


def parse_filename_metadata(file_path: Path, index_num: int) -> Dict[str, str]:
    """Parse filename structure into initial metadata attributes without guessing species."""
    name = file_path.name
    name_stem = file_path.stem
    rel_path = str(file_path.relative_to(resolve_path(".")))

    # Defaults
    collector_id = "COL_DRIVE_01"
    collection_date = "2026-10-02"
    farm_id = "FARM_UNKNOWN"
    field_id = "F1"
    row_id = "R1"
    plant_id = f"PL_{index_num:03d}"
    leaf_id = "L01"
    session_id = "SESS_DRIVE_01"
    suggested_label = "unlabelled"
    notes = "Imported from Google Drive raw collection."

    # Pattern: BLK_3_1002_PLxxx_H.JPG or BLK_3_1002_PLxxx_NH.JPG
    if name.upper().startswith("BLK_"):
        parts = name_stem.split("_")
        if len(parts) >= 5:
            # BLK, 3, 1002, PLxxx, H/NH
            farm_id = f"BLOCK_{parts[1]}"
            session_id = f"SESS_{parts[2]}"
            plant_id = parts[3]
            leaf_id = "L01"
            tag = parts[4].upper()
            if tag == "H":
                suggested_label = "healthy"
                notes = "Field tag: Healthy leaflet check."
            elif tag == "NH":
                suggested_label = "unlabelled"  # Non-healthy, needs user review for miner vs other
                notes = "Field tag: Non-healthy / damaged leaflet; needs review for leaf miner vs other damage."

    elif name.startswith("55"):
        farm_id = "ARCHIVE_REF"
        session_id = "SESS_ENTOMOLOGY_ARCHIVE"
        plant_id = f"PL_ARCH_{name_stem}"
        suggested_label = "unlabelled"
        notes = "Reference leaf photograph from entomological archive."

    elif name.upper().startswith("IMG_"):
        farm_id = "FARM_CANON_01"
        session_id = "SESS_FIELD_CAMERA"
        plant_id = f"PL_FIELD_{name_stem}"
        suggested_label = "unlabelled"
        notes = "High-resolution camera field capture."

    return {
        "image_id": f"IMG_{index_num:04d}",
        "original_filename": name,
        "image_path": rel_path,
        "collector_id": collector_id,
        "collection_date": collection_date,
        "farm_id": farm_id,
        "field_id": field_id,
        "row_id": row_id,
        "plant_id": plant_id,
        "leaf_id": leaf_id,
        "collection_session_id": session_id,
        "tomato_variety": "unknown",
        "growth_stage": "fruiting",
        "image_view_type": "whole_leaf",
        "image_quality": "sharp",
        "visible_damage_type": suggested_label,
        "suspected_pest": "unknown",
        "confirmed_pest": "unconfirmed",
        "pest_confirmation_method": "none",
        "expert_reviewer_id": "pending_review",
        "label_verification_status": "unverified_field_guess",
        "mixed_infestation_status": "not_evaluated",
        "annotation_status": "unannotated",
        "notes": notes,
        "consent_or_permission": "yes",
        "usage_license": "internal_research",
    }


def generate_contact_sheet(
    image_paths: List[Path],
    metadata_records: List[Dict[str, str]],
    output_path: Path,
    cols: int = 8,
    thumb_size: Tuple[int, int] = (180, 180),
) -> Path:
    """Create a high-density thumbnail contact sheet with filename and label tags."""
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_images = len(image_paths)
    rows = (num_images + cols - 1) // cols
    card_w, card_h = thumb_size[0], thumb_size[1] + 45  # extra space for text banner

    sheet_w = cols * card_w
    sheet_h = rows * card_h
    contact_sheet = Image.new("RGB", (sheet_w, sheet_h), color=(240, 244, 240))
    draw = ImageDraw.Draw(contact_sheet)

    for idx, (img_p, meta) in enumerate(zip(image_paths, metadata_records)):
        row = idx // cols
        col = idx % cols
        x = col * card_w
        y = row * card_h

        # Load and letterbox thumbnail
        try:
            with Image.open(img_p) as pil_im:
                pil_im.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                # Center thumbnail in card
                offset_x = x + (thumb_size[0] - pil_im.width) // 2
                offset_y = y + (thumb_size[1] - pil_im.height) // 2
                contact_sheet.paste(pil_im, (offset_x, offset_y))
        except Exception:
            # Draw gray box if corrupt
            draw.rectangle([x, y, x + thumb_size[0], y + thumb_size[1]], fill=(200, 200, 200))

        # Draw border
        draw.rectangle([x, y, x + card_w - 1, y + card_h - 1], outline=(200, 215, 200), width=1)

        # Draw label banner
        label = meta.get("visible_damage_type", "unlabelled")
        if label == "healthy":
            tag_color = (46, 125, 50)  # Green
        elif label == "leaf_miner_damage":
            tag_color = (230, 81, 0)   # Orange
        elif label == "other_damage":
            tag_color = (198, 40, 40)  # Red
        else:
            tag_color = (100, 110, 100) # Gray

        draw.rectangle([x + 2, y + thumb_size[1] + 2, x + card_w - 3, y + thumb_size[1] + 20], fill=tag_color)
        
        # Text annotations
        draw.text((x + 6, y + thumb_size[1] + 4), f"#{idx+1:02d} {label[:12].upper()}", fill=(255, 255, 255))
        draw.text((x + 6, y + thumb_size[1] + 24), meta.get("plant_id", "PL"), fill=(30, 50, 30))
        draw.text((x + 6, y + thumb_size[1] + 34), meta.get("original_filename", "")[:18], fill=(90, 100, 90))

    contact_sheet.save(output_path, quality=90)
    return output_path


def build_inventory_and_metadata(
    raw_dir: str = "data/raw",
    output_metadata_csv: str = "data/metadata/metadata.csv",
    output_report_json: str = "reports/dataset_inventory.json",
    output_report_csv: str = "reports/dataset_inventory.csv",
    output_contact_sheet: str = "reports/thumbnail_contact_sheet.jpg",
) -> Dict[str, any]:
    """Execute complete recursive inventory scan, duplicate detection, and metadata compilation."""
    raw_p = resolve_path(raw_dir)
    image_paths = scan_raw_dataset(raw_p)

    if not image_paths:
        return {
            "status": "EMPTY",
            "message": f"No image files found in {raw_p}",
            "total_images": 0,
        }

    # Load existing metadata if present to avoid overwriting user edits
    existing_meta_p = resolve_path(output_metadata_csv)
    existing_user_labels = {}
    if existing_meta_p.is_file():
        try:
            df_existing = pd.read_csv(existing_meta_p, dtype=str).fillna("")
            for _, r in df_existing.iterrows():
                fname = str(r.get("original_filename", ""))
                if fname:
                    existing_user_labels[fname] = r.to_dict()
        except Exception:
            pass

    # 1. Detailed per-image analysis
    inventory_items = []
    metadata_records = []
    dimensions_tally = {}
    formats_tally = {}
    sizes_bytes = []

    for idx, img_p in enumerate(image_paths, start=1):
        file_size = img_p.stat().st_size
        sizes_bytes.append(file_size)
        fmt = img_p.suffix.lower()
        formats_tally[fmt] = formats_tally.get(fmt, 0) + 1

        # Check dimension and decode
        w, h = 0, 0
        try:
            with Image.open(img_p) as pil_img:
                w, h = pil_img.size
                dims_key = f"{w}x{h}"
                dimensions_tally[dims_key] = dimensions_tally.get(dims_key, 0) + 1
        except Exception:
            pass

        # Hashes
        sha = compute_sha256(img_p)
        dh = compute_dhash(img_p)

        # Metadata record
        parsed = parse_filename_metadata(img_p, index_num=idx)
        
        # If user previously assigned a label in metadata.csv, preserve it!
        if img_p.name in existing_user_labels:
            user_rec = existing_user_labels[img_p.name]
            for col in REQUIRED_COLUMNS:
                if col in user_rec and user_rec[col]:
                    parsed[col] = user_rec[col]

        metadata_records.append(parsed)

        inventory_items.append({
            "index": idx,
            "image_id": parsed["image_id"],
            "filename": img_p.name,
            "relative_path": str(img_p.relative_to(resolve_path("."))),
            "format": fmt,
            "width": w,
            "height": h,
            "aspect_ratio": round(w / max(1, h), 2),
            "size_kb": round(file_size / 1024, 2),
            "sha256": sha,
            "dhash": dh,
            "current_label": parsed["visible_damage_type"],
            "plant_id": parsed["plant_id"],
            "farm_id": parsed["farm_id"],
        })

    # 2. Duplicate Detection
    exact_dups, near_dups = detect_duplicates(image_paths, near_dup_threshold=4)

    # 3. Label Breakdown
    labels_series = pd.Series([m["visible_damage_type"] for m in metadata_records])
    class_distribution = labels_series.value_counts().to_dict()

    # 4. Save metadata.csv
    df_meta = pd.DataFrame(metadata_records)
    out_meta_p = resolve_path(output_metadata_csv)
    out_meta_p.parent.mkdir(parents=True, exist_ok=True)
    df_meta.to_csv(out_meta_p, index=False)

    # 5. Save Inventory CSV and JSON
    out_csv_p = resolve_path(output_report_csv)
    out_json_p = resolve_path(output_report_json)
    out_csv_p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inventory_items).to_csv(out_csv_p, index=False)

    inventory_report = {
        "timestamp": datetime.now().isoformat(),
        "total_images": len(image_paths),
        "total_size_mb": round(sum(sizes_bytes) / (1024 * 1024), 2),
        "file_formats": formats_tally,
        "unique_resolutions_count": len(dimensions_tally),
        "top_resolutions": dict(sorted(dimensions_tally.items(), key=lambda x: x[1], reverse=True)[:5]),
        "exact_duplicate_clusters": len(exact_dups),
        "near_duplicate_clusters": len(near_dups),
        "independent_plants_count": len(set(df_meta["plant_id"].dropna())),
        "independent_farms_count": len(set(df_meta["farm_id"].dropna())),
        "current_label_distribution": class_distribution,
        "metadata_csv_path": str(out_meta_p),
        "inventory_csv_path": str(out_csv_p),
    }

    with open(out_json_p, "w", encoding="utf-8") as f:
        json.dump(inventory_report, f, indent=2)

    # 6. Generate thumbnail contact sheet
    out_sheet_p = resolve_path(output_contact_sheet)
    generate_contact_sheet(image_paths, metadata_records, out_sheet_p, cols=8)
    inventory_report["contact_sheet_path"] = str(out_sheet_p)

    return inventory_report


def main():
    parser = argparse.ArgumentParser(description="Scan and catalog downloaded tomato leaf photographs.")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Directory of raw photos")
    parser.add_argument("--output_metadata", type=str, default="data/metadata/metadata.csv")
    args = parser.parse_args()

    print("=" * 70)
    print("🌿 PHYTOTRACE AI — RECURSIVE DATASET SCAN & INVENTORY GENERATOR")
    print("=" * 70)

    report = build_inventory_and_metadata(
        raw_dir=args.raw_dir,
        output_metadata_csv=args.output_metadata,
    )

    print(f"Total Images Found         : {report['total_images']}")
    print(f"Total Size on Disk         : {report['total_size_mb']} MB")
    print(f"File Formats               : {report['file_formats']}")
    print(f"Unique Aspect Ratios/Dims  : {report['unique_resolutions_count']}")
    print(f"Exact Duplicates (SHA-256) : {report['exact_duplicate_clusters']}")
    print(f"Near Duplicates (dHash)    : {report['near_duplicate_clusters']}")
    print(f"Independent Plants Count   : {report['independent_plants_count']}")
    print(f"Independent Farm Sites     : {report['independent_farms_count']}")
    print("-" * 70)
    print("Current Label Distribution (Before Labelling Review):")
    for lbl, count in report["current_label_distribution"].items():
        print(f"  • {lbl:<20}: {count} images")
    print("-" * 70)
    print(f"✅ Metadata Catalog Written : {report['metadata_csv_path']}")
    print(f"✅ Detailed Inventory CSV   : {report['inventory_csv_path']}")
    print(f"✅ Thumbnail Contact Sheet  : {report['contact_sheet_path']}")
    print("=" * 70)


if __name__ == "__main__":
    main()

