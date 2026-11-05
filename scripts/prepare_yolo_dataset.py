#!/usr/bin/env python3
"""PhytoTrace AI — Phase 3 YOLO Segmentation Dataset Preparation CLI.

Exports verified polygon annotations from CVAT COCO into standard Ultralytics YOLO segmentation format:
- data/yolo_segmentation/images/{train,val}
- data/yolo_segmentation/labels/{train,val}
- data/annotations/yolo_dataset.yaml
- Empty label files for confirmed negative (healthy/no-mine) samples
- Strict exclusion of the frozen held-out test partition
"""

import argparse
import sys
from pathlib import Path

from phytotrace.segmentation.yolo_exporter import export_yolo_segmentation_dataset
from phytotrace.utils.paths import resolve_path


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO segmentation dataset from verified CVAT annotations.")
    parser.add_argument("--catalog", type=str, default="data/segmentation/catalog.csv", help="Segmentation catalog CSV")
    parser.add_argument("--coco_json", type=str, default="data/segmentation/annotations/instances_default.json", help="COCO annotations JSON")
    parser.add_argument("--splits_json", type=str, default="data/splits/splits.json", help="Dataset splits JSON")
    parser.add_argument("--photos_dir", type=str, default="data/raw/original_photos", help="Source original photos directory")
    parser.add_argument("--output_dir", type=str, default="data/yolo_segmentation", help="Target YOLO dataset directory")
    parser.add_argument("--yaml_out", type=str, default="data/annotations/yolo_dataset.yaml", help="Path to output yolo_dataset.yaml")
    args = parser.parse_args()

    print("=" * 70)
    print("🌿 PHYTOTRACE AI — PREPARE YOLO SEGMENTATION DATASET")
    print("=" * 70)

    try:
        stats = export_yolo_segmentation_dataset(
            catalog_path=resolve_path(args.catalog),
            coco_json_path=resolve_path(args.coco_json),
            splits_json_path=resolve_path(args.splits_json),
            photos_dir=resolve_path(args.photos_dir),
            output_dir=resolve_path(args.output_dir),
            yaml_output_path=resolve_path(args.yaml_out),
            class_id=0,
            class_name="mine_damage",
        )
    except Exception as e:
        print(f"\n❌ Dataset preparation failed: {e}")
        sys.exit(1)

    print("\nDataset Export Statistics:")
    print(f"• Train Split Total Images        : {stats['train']['total_images']}")
    print(f"  - Positive Images (with mines)  : {stats['train']['positive_images']}")
    print(f"  - Verified Negative Images (0)  : {stats['train']['verified_negative_images']}")
    print(f"  - Total Mine Polygons in Train  : {stats['train']['total_polygons']}")
    print(f"• Val Split Total Images          : {stats['val']['total_images']}")
    print(f"  - Positive Images (with mines)  : {stats['val']['positive_images']}")
    print(f"  - Verified Negative Images (0)  : {stats['val']['verified_negative_images']}")
    print(f"  - Total Mine Polygons in Val    : {stats['val']['total_polygons']}")
    print(f"• Quarantined Frozen Test Images  : {stats['excluded_test_images']} (Strictly isolated)")
    print(f"• Excluded Unready Images         : {stats['excluded_unready_images']}")
    print("-" * 70)
    print(f"✅ Dataset YAML successfully created at: {resolve_path(args.yaml_out)}")
    print(f"✅ YOLO dataset root: {resolve_path(args.output_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

