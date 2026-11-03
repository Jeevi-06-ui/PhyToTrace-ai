#!/usr/bin/env python3
"""PhytoTrace AI — Duplicate and Near-Duplicate Detection Script."""

import argparse
import json
from pathlib import Path

from phytotrace.data.deduplicate import detect_duplicates
from phytotrace.utils.paths import resolve_path


def main():
    parser = argparse.ArgumentParser(description="Scan dataset for exact and perceptual near duplicates.")
    parser.add_argument("--image_dir", type=str, default="data/raw", help="Directory of images to scan")
    parser.add_argument("--threshold", type=int, default=4, help="Hamming distance threshold for near duplicates")
    args = parser.parse_args()

    img_dir = resolve_path(args.image_dir)
    print("=" * 60)
    print("🌿 PHYTOTRACE AI — DEDUPLICATION AUDIT")
    print("=" * 60)

    if not img_dir.is_dir():
        print(f"Directory not found: {img_dir}")
        return

    images = [
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    print(f"Scanning {len(images)} images in {img_dir}...")

    exact_dups, near_clusters = detect_duplicates(images, near_dup_threshold=args.threshold)

    print(f"Exact Duplicate Groups Found : {len(exact_dups)}")
    for sha, paths in exact_dups.items():
        print(f"  - Hash {sha[:12]}... : {[p.name for p in paths]}")

    print(f"Near-Duplicate Clusters Found: {len(near_clusters)}")
    for idx, cluster in enumerate(near_clusters):
        print(f"  - Cluster #{idx+1} : {[p.name for p in cluster]}")

    report = {
        "scanned_directory": str(img_dir),
        "total_images": len(images),
        "exact_duplicates": {k: [str(p) for p in v] for k, v in exact_dups.items()},
        "near_duplicate_clusters": [[str(p) for p in c] for c in near_clusters],
    }

    report_p = resolve_path("reports/duplicates_report.json")
    report_p.parent.mkdir(parents=True, exist_ok=True)
    with open(report_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Audit report saved to {report_p}.")
    print("=" * 60)


if __name__ == "__main__":
    main()

