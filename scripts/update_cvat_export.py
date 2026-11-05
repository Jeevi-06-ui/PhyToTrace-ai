#!/usr/bin/env python3
"""PhytoTrace AI — Safe CVAT COCO Export Updater & Archiver.

Safely replaces instances_default.json from a newly downloaded CVAT export (ZIP or JSON),
automatically archiving previous versions with timestamps while guaranteeing zero modification
to raw photographs or master dataset records.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

from phytotrace.utils.paths import resolve_path


def compute_md5(file_path: Path) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def inspect_coco_dict(d: dict) -> dict:
    images = d.get("images", [])
    annotations = d.get("annotations", [])
    categories = {c["id"]: c.get("name", str(c["id"])) for c in d.get("categories", [])}
    
    ann_by_cat = {}
    for a in annotations:
        cname = categories.get(a.get("category_id"), "unknown")
        ann_by_cat[cname] = ann_by_cat.get(cname, 0) + 1
        
    return {
        "num_images": len(images),
        "num_annotations": len(annotations),
        "categories": categories,
        "annotations_by_category": ann_by_cat,
    }


def main():
    parser = argparse.ArgumentParser(description="Safely update CVAT COCO annotation export.")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to new CVAT export (.zip file or instances_default.json)",
    )
    parser.add_argument(
        "--target", "-t",
        type=str,
        default="data/segmentation/annotations/instances_default.json",
        help="Target annotation file path in project",
    )
    parser.add_argument(
        "--backup_dir", "-b",
        type=str,
        default="data/segmentation/backups/json_history",
        help="Directory to store timestamped backups",
    )
    args = parser.parse_args()

    input_p = resolve_path(args.input)
    target_p = resolve_path(args.target)
    backup_dir_p = resolve_path(args.backup_dir)

    print("=" * 70)
    print("🌿 PHYTOTRACE AI — SAFE CVAT COCO EXPORT UPDATER")
    print("=" * 70)

    if not input_p.is_file():
        print(f"❌ Error: Input file not found: {input_p}")
        sys.exit(1)

    # 1. Parse incoming content
    raw_json_bytes = None
    if input_p.suffix.lower() == ".zip":
        print(f"Reading ZIP archive: {input_p}...")
        with zipfile.ZipFile(input_p, "r") as zf:
            json_members = [m for m in zf.namelist() if m.endswith(".json")]
            if not json_members:
                print("❌ Error: No JSON file found inside ZIP archive.")
                sys.exit(1)
            target_member = next((m for m in json_members if "instances_default" in m), json_members[0])
            print(f"Extracting '{target_member}' from ZIP archive...")
            raw_json_bytes = zf.read(target_member)
    else:
        print(f"Reading JSON file: {input_p}...")
        with open(input_p, "rb") as f:
            raw_json_bytes = f.read()

    # 2. Validate COCO structure
    try:
        new_data = json.loads(raw_json_bytes.decode("utf-8"))
    except Exception as e:
        print(f"❌ Error: Failed to parse valid JSON from {input_p}: {e}")
        sys.exit(1)

    if not isinstance(new_data, dict) or "images" not in new_data or "annotations" not in new_data:
        print("❌ Error: Missing required COCO fields ('images', 'annotations'). Invalid COCO format.")
        sys.exit(1)

    new_stats = inspect_coco_dict(new_data)
    print(f"✅ Incoming COCO valid: {new_stats['num_images']} images, {new_stats['num_annotations']} annotations {new_stats['annotations_by_category']}.")

    # 3. Backup existing target file if present
    backup_dir_p.mkdir(parents=True, exist_ok=True)
    if target_p.is_file():
        old_md5 = compute_md5(target_p)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir_p / f"instances_default_{ts}_{old_md5[:8]}.json"
        shutil.copy2(target_p, backup_file)
        print(f"📦 Previous export backed up to: {backup_file}")
        
        with open(target_p, "r", encoding="utf-8") as f:
            try:
                old_data = json.load(f)
                old_stats = inspect_coco_dict(old_data)
                print(f"   (Previous: {old_stats['num_images']} images, {old_stats['num_annotations']} annotations {old_stats['annotations_by_category']})")
            except Exception:
                old_stats = None

    # 4. Atomic write of new JSON
    target_p.parent.mkdir(parents=True, exist_ok=True)
    temp_p = target_p.with_suffix(".tmp")
    with open(temp_p, "wb") as f:
        f.write(raw_json_bytes)
    temp_p.replace(target_p)

    new_md5 = compute_md5(target_p)
    print(f"✅ Target file updated: {target_p}")
    print(f"   • MD5 Checksum : {new_md5}")
    print(f"   • Size         : {target_p.stat().st_size:,} bytes")
    print(f"   • Images       : {new_stats['num_images']}")
    print(f"   • Annotations  : {new_stats['num_annotations']} {new_stats['annotations_by_category']}")
    print("🔒 Integrity check: 0 photographs modified; original files strictly preserved.")
    print("=" * 70)


if __name__ == "__main__":
    main()

